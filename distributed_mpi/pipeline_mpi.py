"""
Step C: the same model selection as pipeline_select.py, but with MPI.

Distributed-memory design (every process has its OWN memory, data moves only
through messages):

  rank 0   : loads, cleans and engineers features (serial, reuses
             serial/pipeline.py), then prepares the work for every rank
  bcast    : the held-out test rows are broadcast to all ranks
  send     : each rank receives only the training rows its models need
             (point-to-point messages), so nobody needs the 2M-row table
  compute  : every rank (rank 0 too) trains its own models
  gather   : results come back to rank 0, which picks the best model

Models are assigned by "longest job first" (greedy) using depth x trees as
the cost estimate. Same seeds and same row samples as pipeline_select.py, so
the best model and its RMSE must be identical.

Usage (from distributed_mpi/):
    mpirun -n 4 python3 pipeline_mpi.py --data ../data/dataset.csv --subsample 50000
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse, csv, sys, time
from pathlib import Path

import numpy as np
from mpi4py import MPI
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "serial"))     # serial/pipeline.py

GRID = [(depth, trees) for depth in (6, 8, 10, 12) for trees in (20, 40, 60)]


def assign(jobs, size):
    """Greedy longest-job-first assignment of jobs (i, depth, trees) to ranks."""
    loads = [0] * size
    plan = [[] for _ in range(size)]
    for job in sorted(jobs, key=lambda j: -(j[1] * j[2])):
        r = loads.index(min(loads))
        plan[r].append(job)
        loads[r] += job[1] * job[2]
    return plan


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="../data/dataset.csv")
    ap.add_argument("--subsample", type=int, default=100_000)
    ap.add_argument("--results", type=str, default="../benchmarks/selection_log.csv")
    args = ap.parse_args()

    t_start = time.perf_counter()
    payload_test = None
    if rank == 0:
        import pipeline as serial            # serial/pipeline.py
        df = serial.load_data(args.data)
        df = serial.clean(df)
        df = serial.engineer_features(df)
        X = df.drop(columns=["target"]).to_numpy(dtype=np.float64)
        y = df["target"].to_numpy(dtype=np.float64)
        n = len(df)
        train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2, random_state=42)
        test_idx = test_idx[:100_000]
        payload_test = (X[test_idx], y[test_idx])
        t_prep = time.perf_counter() - t_start

    comm.Barrier()
    t0 = time.perf_counter()

    # 1) test rows to everyone
    Xt, yt = comm.bcast(payload_test, root=0)

    # 2) each rank gets only the training rows for its own models
    if rank == 0:
        jobs = [(i, d, t) for i, (d, t) in enumerate(GRID)]
        plan = assign(jobs, size)
        n_sub = min(args.subsample, len(train_idx))
        mine = None
        for r in range(size):
            payload = []
            for i, d, t in plan[r]:
                rows = np.random.default_rng(1000 + i).choice(train_idx, size=n_sub, replace=False)
                payload.append((i, d, t, X[rows], y[rows]))
            if r == 0:
                mine = payload
            else:
                comm.send(payload, dest=r, tag=1)
    else:
        mine = comm.recv(source=0, tag=1)
    t_comm = time.perf_counter() - t0

    # 3) compute
    local = []
    for i, d, t, Xs, ys in mine:
        model = RandomForestRegressor(n_estimators=t, max_depth=d, max_features=0.5,
                                      n_jobs=1, random_state=i)
        model.fit(Xs, ys)
        preds = model.predict(Xt)
        local.append((i, d, t, float(np.sqrt(np.mean((preds - yt) ** 2)))))

    # 4) results back to rank 0
    gathered = comm.gather(local, root=0)
    comm.Barrier()
    t_sel = time.perf_counter() - t0

    if rank == 0:
        results = sorted(r for part in gathered for r in part)
        best = min(results, key=lambda r: r[3])
        print(f"MPI MODEL SELECTION ({size} processes, {len(GRID)} models)")
        print(f"Rows processed      : {n:,}")
        print(f"Prep time (rank 0 only, serial) : {t_prep:.3f} s")
        print(f"Communication time  : {t_comm:.3f} s (test bcast + sending rows)")
        print(f"Selection time      : {t_sel:.3f} s")
        print(f"TOTAL TIME (Tp)     : {time.perf_counter() - t_start:.3f} s")
        print(f"Best model          : depth={best[1]}, trees={best[2]}, RMSE={best[3]:.4f}")

        out = Path(args.results)
        out.parent.mkdir(parents=True, exist_ok=True)
        new = not out.exists()
        with open(out, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["phase", "workers", "total_time_s", "rows", "rmse"])
            w.writerow(["mpi", size, f"{t_sel:.4f}", n, f"{best[3]:.4f}"])


if __name__ == "__main__":
    main()
