"""
Generate a synthetic regression dataset for the Parallel ML Data Processing project.

Why synthetic data: we fully control the size (so we can scale it up to make serial
processing visibly slow, or scale it for weak-scaling tests later), and it is
reproducible for everyone on the team (fixed random seed).

Usage:
    python3 generate_data.py --rows 2000000 --out dataset.csv
"""
import argparse
import numpy as np
import pandas as pd


def generate(n_rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    x1 = rng.normal(50, 15, n_rows)
    x2 = rng.normal(20, 5, n_rows)
    x3 = rng.uniform(0, 100, n_rows)
    x4 = rng.normal(0, 1, n_rows)
    x5 = rng.exponential(2.0, n_rows)
    category = rng.choice(["A", "B", "C", "D"], size=n_rows, p=[0.4, 0.3, 0.2, 0.1])

    noise = rng.normal(0, 5, n_rows)
    target = (2.0 * x1 - 1.5 * x2 + 0.3 * x3 + 4.0 * x4 - 0.8 * x5 + noise)

    df = pd.DataFrame({
        "feature_1": x1,
        "feature_2": x2,
        "feature_3": x3,
        "feature_4": x4,
        "feature_5": x5,
        "category": category,
        "target": target,
    })

    n_missing = int(n_rows * 0.02)
    missing_idx = rng.choice(n_rows, size=n_missing, replace=False)
    df.loc[missing_idx, "feature_3"] = np.nan

    n_dupes = int(n_rows * 0.01)
    dupe_rows = df.sample(n=n_dupes, random_state=seed)
    df = pd.concat([df, dupe_rows], ignore_index=True)

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2_000_000, help="number of base rows to generate")
    ap.add_argument("--out", type=str, default="dataset.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = generate(args.rows, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows (including injected duplicates/missing values) to {args.out}")
