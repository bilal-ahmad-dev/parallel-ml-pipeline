"""
Serial baseline pipeline (Week 1, Days 1-2).
Stages: load -> clean -> feature engineering -> train -> report timing.
Usage: python3 pipeline.py --data ../data/dataset.csv
"""
import argparse, os, time, csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def load_data(path):
    return pd.read_csv(path)


def clean(df):
    df = df.drop_duplicates()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def engineer_features(df):
    df = pd.get_dummies(df, columns=["category"], prefix="cat")
    numeric_cols = [c for c in df.columns if c.startswith("feature_")]
    for col in numeric_cols:
        mean, std = df[col].mean(), df[col].std()
        df[col] = (df[col] - mean) / (std if std > 0 else 1.0)
    return df


def train(df):
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    return model, rmse


def record_result(stage_times, out_path="../benchmarks/results.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    file_exists = os.path.isfile(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["phase", "workers", "total_time_s", "rows", "rmse"])
        writer.writerow([
            "serial", 1, f"{stage_times['total']:.4f}",
            stage_times["rows"], f"{stage_times['rmse']:.4f}",
        ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="../data/dataset.csv")
    args = ap.parse_args()

    t_start = time.perf_counter()
    t0 = time.perf_counter(); df = load_data(args.data); t_load = time.perf_counter() - t0
    t0 = time.perf_counter(); df = clean(df); t_clean = time.perf_counter() - t0
    t0 = time.perf_counter(); df = engineer_features(df); t_feat = time.perf_counter() - t0
    t0 = time.perf_counter(); model, rmse = train(df); t_train = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start

    print("SERIAL BASELINE PIPELINE")
    print(f"Rows processed      : {len(df):,}")
    print(f"Load time           : {t_load:.3f} s")
    print(f"Clean time          : {t_clean:.3f} s")
    print(f"Feature engineering : {t_feat:.3f} s")
    print(f"Train time          : {t_train:.3f} s")
    print(f"TOTAL TIME (T1)     : {t_total:.3f} s")
    print(f"Test RMSE           : {rmse:.4f}")

    record_result({"total": t_total, "rows": len(df), "rmse": rmse})


if __name__ == "__main__":
    main()
