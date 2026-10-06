"""
Multiprocessing pipeline v2 (Step A): same as pipeline_mp.py, but the CSV file
is also READ in parallel. The file is cut into byte ranges, each worker parses
its own range, and the pieces are glued together.

Needs pipeline_mp.py in the same folder (it reuses clean/features/train).
Usage (from parallel_multiprocessing/):
    python3 pipeline_mp2.py --workers 4 --data ../data/dataset.csv
"""
import argparse, io, os, time
import multiprocessing as mp
import pandas as pd

import pipeline_mp as base      # also sets 1 BLAS thread per process


def _read_chunk(job):
    path, i, n, header_len, size, cols = job
    start = header_len + i * (size - header_len) // n
    end = header_len + (i + 1) * (size - header_len) // n
    with open(path, "rb") as f:
        f.seek(start - 1)
        f.readline()                 # skip the line that belongs to the previous worker
        pos = f.tell()
        if pos >= end:
            return None
        data = f.read(end - pos)
        if not data.endswith(b"\n"):
            data += f.readline()     # finish the line that crosses our boundary
    return pd.read_csv(io.BytesIO(data), header=None, names=cols,
                       dtype={"category": "category"})


def load_data_parallel(path, ctx, workers):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        header = f.readline()
    cols = header.decode().strip().split(",")
    jobs = [(path, i, workers, len(header), size, cols) for i in range(workers)]
    with ctx.Pool(workers) as pool:
        parts = pool.map(_read_chunk, jobs)
    return pd.concat([p for p in parts if p is not None], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="../data/dataset.csv")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--results", type=str, default="../benchmarks/results.csv")
    args = ap.parse_args()

    ctx = mp.get_context("fork")
    w = args.workers

    t_start = time.perf_counter()
    t0 = time.perf_counter(); df = load_data_parallel(args.data, ctx, w); t_load = time.perf_counter() - t0
    t0 = time.perf_counter(); df = base.clean(df, ctx, w); t_clean = time.perf_counter() - t0
    t0 = time.perf_counter(); df = base.engineer_features(df); t_feat = time.perf_counter() - t0
    t0 = time.perf_counter(); _, rmse = base.train(df, ctx, w); t_train = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start

    print(f"MULTIPROCESSING PIPELINE v2 ({w} workers, parallel load)")
    print(f"Rows processed      : {len(df):,}")
    print(f"Load time           : {t_load:.3f} s")
    print(f"Clean time          : {t_clean:.3f} s")
    print(f"Feature engineering : {t_feat:.3f} s")
    print(f"Train time          : {t_train:.3f} s")
    print(f"TOTAL TIME (Tp)     : {t_total:.3f} s")
    print(f"Test RMSE           : {rmse:.4f}   (serial: 5.1424)")

    base.record_result("multiprocessing_pload", w, t_total, len(df), rmse, args.results)


if __name__ == "__main__":
    main()
