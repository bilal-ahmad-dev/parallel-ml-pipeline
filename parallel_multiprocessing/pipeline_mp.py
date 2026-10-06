"""
Multiprocessing pipeline (Week 1, Days 3-5).

Same stages and same result as serial/pipeline.py, but the heavy parts run on
several worker processes:

  load      : serial (disk bound, little to gain)
  clean     : row hashes computed in parallel, duplicates found globally,
              medians computed on the full data (exact same result as serial)
  features  : serial (only ~4% of the serial time)
  train     : each worker computes X'X and X'y for its slice of the training
              rows; the partial sums are added and the normal equations solved.
              This gives the same model as LinearRegression.

Usage (from parallel_multiprocessing/):
    python3 pipeline_mp.py --workers 4 --data ../data/dataset.csv
"""
import os

# One BLAS thread per process. Without this, 12 workers x 12 BLAS threads
# would fight over the same cores and the timings would be meaningless.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse, csv, time
import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Data shared with the workers through fork (copy-on-write, nothing is pickled)
_DF = None
_X = None
_Y = None


# ----------------------------------------------------------------- workers
def _hash_chunk(bounds):
    lo, hi = bounds
    return pd.util.hash_pandas_object(_DF.iloc[lo:hi], index=False).to_numpy()


def _partial_normal_eq(idx):
    Xc = _X[idx]
    yc = _Y[idx]
    return Xc.T @ Xc, Xc.T @ yc


# ------------------------------------------------------------------ stages
def load_data(path):
    return pd.read_csv(path)


def clean(df, ctx, workers):
    global _DF
    _DF = df
    n = len(df)
    bounds = list(zip(np.linspace(0, n, workers + 1).astype(int)[:-1],
                      np.linspace(0, n, workers + 1).astype(int)[1:]))
    with ctx.Pool(workers) as pool:
        parts = pool.map(_hash_chunk, bounds)
    hashes = np.concatenate(parts)
    dup = pd.Series(hashes).duplicated().to_numpy()   # keeps first occurrence
    df = df[~dup]

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def engineer_features(df):
    df = pd.get_dummies(df, columns=["category"], prefix="cat", dtype=float)
    numeric_cols = [c for c in df.columns if c.startswith("feature_")]
    for col in numeric_cols:
        mean, std = df[col].mean(), df[col].std()
        df[col] = (df[col] - mean) / (std if std > 0 else 1.0)
    return df


def train(df, ctx, workers):
    global _X, _Y
    feature_cols = [c for c in df.columns if c != "target"]
    # Drop one dummy column: the 4 dummies sum to 1 and clash with the
    # intercept. Predictions stay identical, the equations become solvable.
    feature_cols.remove([c for c in feature_cols if c.startswith("cat_")][-1])

    n, k = len(df), len(feature_cols)
    X = np.empty((n, k + 1))
    X[:, :k] = df[feature_cols].to_numpy(dtype=np.float64)
    X[:, k] = 1.0                       # intercept column
    y = df["target"].to_numpy(dtype=np.float64)
    _X, _Y = X, y

    # Same split as the serial version (same random_state, same n)
    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2, random_state=42)

    chunks = np.array_split(train_idx, workers)
    with ctx.Pool(workers) as pool:
        parts = pool.map(_partial_normal_eq, chunks)

    XtX = sum(p[0] for p in parts)
    Xty = sum(p[1] for p in parts)
    w = np.linalg.solve(XtX, Xty)

    preds = X[test_idx] @ w
    rmse = float(np.sqrt(np.mean((preds - y[test_idx]) ** 2)))
    return w, rmse


# ----------------------------------------------------------------- results
def serial_median(path):
    if not os.path.isfile(path):
        return None
    times = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["phase"] == "serial":
                times.append(float(row["total_time_s"]))
    return float(np.median(times)) if times else None


def record_result(phase, workers, total, rows, rmse, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    file_exists = os.path.isfile(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["phase", "workers", "total_time_s", "rows", "rmse"])
        writer.writerow([phase, workers, f"{total:.4f}", rows, f"{rmse:.4f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="../data/dataset.csv")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--results", type=str, default="../benchmarks/results.csv")
    args = ap.parse_args()

    ctx = mp.get_context("fork")   # Python 3.14 defaults to forkserver on Linux

    t_start = time.perf_counter()
    t0 = time.perf_counter(); df = load_data(args.data); t_load = time.perf_counter() - t0
    t0 = time.perf_counter(); df = clean(df, ctx, args.workers); t_clean = time.perf_counter() - t0
    t0 = time.perf_counter(); df = engineer_features(df); t_feat = time.perf_counter() - t0
    t0 = time.perf_counter(); w, rmse = train(df, ctx, args.workers); t_train = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start

    print(f"MULTIPROCESSING PIPELINE ({args.workers} workers)")
    print(f"Rows processed      : {len(df):,}")
    print(f"Load time           : {t_load:.3f} s")
    print(f"Clean time          : {t_clean:.3f} s")
    print(f"Feature engineering : {t_feat:.3f} s")
    print(f"Train time          : {t_train:.3f} s")
    print(f"TOTAL TIME (Tp)     : {t_total:.3f} s")
    print(f"Test RMSE           : {rmse:.4f}   (serial: 5.1424)")

    t1 = serial_median(args.results)
    if t1:
        speedup = t1 / t_total
        print(f"Speedup vs serial   : {speedup:.2f}x  (T1 median = {t1:.2f} s)")
        print(f"Efficiency          : {speedup / args.workers:.1%}")

    record_result("multiprocessing", args.workers, t_total, len(df), rmse, args.results)


if __name__ == "__main__":
    main()
