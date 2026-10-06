"""
Step B: heavy, embarrassingly parallel work = model selection.

Data prep (load -> clean -> features) is reused from pipeline_mp.py. Then 12
RandomForest models with different settings are trained and scored on the same
held-out rows, and the best one is picked. Every model is an independent job,
so the workers always have real CPU work to do.

  --workers 1  : plain serial loop in one process (the baseline)
  --workers N  : N processes share the data through fork and take models
                 one by one (dynamic load balancing)

Same random seeds everywhere, so every worker count must pick the same best
model with the same RMSE.

Usage (from parallel_multiprocessing/):
    python3 pipeline_select.py --workers 4 --data ../data/dataset.csv
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse, time
import multiprocessing as mp
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import pipeline_mp as base

GRID = [(depth, trees) for depth in (6, 8, 10, 12) for trees in (20, 40, 60)]

# Shared with the workers through fork (nothing is copied or pickled)
_X = None
_Y = None
_TRAIN = None
_TEST = None


def _fit_eval(job):
    i, depth, trees, n_sub = job
    rng = np.random.default_rng(1000 + i)
    rows = rng.choice(_TRAIN, size=n_sub, replace=False)
    model = RandomForestRegressor(n_estimators=trees, max_depth=depth,
                                  max_features=0.5, n_jobs=1, random_state=i)
    model.fit(_X[rows], _Y[rows])
    preds = model.predict(_X[_TEST])
    rmse = float(np.sqrt(np.mean((preds - _Y[_TEST]) ** 2)))
    return i, depth, trees, rmse


def run_selection(workers, ctx, n_sub):
    jobs = [(i, d, t, n_sub) for i, (d, t) in enumerate(GRID)]
    # Biggest models first: keeps the workers evenly busy until the end
    jobs.sort(key=lambda j: -(j[1] * j[2]))
    if workers == 1:
        results = [_fit_eval(j) for j in jobs]
    else:
        with ctx.Pool(workers) as pool:
            results = list(pool.imap_unordered(_fit_eval, jobs, chunksize=1))
    return sorted(results)          # back to model-number order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="../data/dataset.csv")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--subsample", type=int, default=100_000,
                    help="training rows used by each model")
    ap.add_argument("--results", type=str, default="../benchmarks/selection_log.csv")
    args = ap.parse_args()

    global _X, _Y, _TRAIN, _TEST
    ctx = mp.get_context("fork")
    w = args.workers

    t_start = time.perf_counter()
    t0 = time.perf_counter()
    df = base.load_data(args.data)
    df = base.clean(df, ctx, w)
    df = base.engineer_features(df)
    t_prep = time.perf_counter() - t0

    _X = df.drop(columns=["target"]).to_numpy(dtype=np.float64)
    _Y = df["target"].to_numpy(dtype=np.float64)
    n = len(df)
    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2, random_state=42)
    _TRAIN, _TEST = train_idx, test_idx[:100_000]

    t0 = time.perf_counter()
    results = run_selection(w, ctx, min(args.subsample, len(_TRAIN)))
    t_sel = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start

    best = min(results, key=lambda r: r[3])
    print(f"MODEL SELECTION ({w} workers, {len(GRID)} models)")
    print(f"Rows processed      : {n:,}")
    print(f"Prep time (load+clean+features) : {t_prep:.3f} s")
    print(f"Selection time      : {t_sel:.3f} s")
    print(f"TOTAL TIME (Tp)     : {t_total:.3f} s")
    print(f"Best model          : depth={best[1]}, trees={best[2]}, RMSE={best[3]:.4f}")

    base.record_result("selection", w, t_sel, n, best[3], args.results)


if __name__ == "__main__":
    main()
