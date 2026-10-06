"""
Benchmark driver: runs the serial and multiprocessing pipelines several times,
interleaved (so slow drifts of the machine hit every configuration equally),
then reports median / best times, speedup, efficiency and saves a plot.

Put this file in benchmarks/ and run from there:
    python3 benchmark.py
    python3 benchmark.py --reps 5 --workers 1 2 4 8 12
Close heavy apps and plug in the charger first.
"""
import argparse, csv, os, re, statistics, subprocess, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"

PATTERNS = {
    "load":     r"Load time\s*:\s*([\d.]+)",
    "clean":    r"Clean time\s*:\s*([\d.]+)",
    "features": r"Feature engineering\s*:\s*([\d.]+)",
    "train":    r"Train time\s*:\s*([\d.]+)",
    "total":    r"TOTAL TIME \(\w+\)\s*:\s*([\d.]+)",
}


def label(c):
    return "serial" if c == "serial" else f"{c} workers"


def run_once(config, data):
    env = os.environ.copy()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[v] = "1"
    if config == "serial":
        cmd = [sys.executable, "pipeline.py", "--data", str(data)]
        cwd = ROOT / "serial"
    else:
        cmd = [sys.executable, "pipeline_mp.py", "--workers", str(config),
               "--data", str(data), "--results", str(BENCH / "bench_mp_log.csv")]
        cwd = ROOT / "parallel_multiprocessing"
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr)
        sys.exit(f"Run failed: {' '.join(cmd)}")
    return {k: float(re.search(p, proc.stdout).group(1)) for k, p in PATTERNS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, 12])
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "dataset.csv")
    args = ap.parse_args()

    configs = ["serial"] + args.workers
    print("Warm-up run (not counted)...")
    run_once(args.workers[0], args.data)

    raw = {c: [] for c in configs}
    for rep in range(args.reps):
        for c in configs:
            r = run_once(c, args.data)
            raw[c].append(r)
            print(f"rep {rep + 1}/{args.reps}  {label(c):<11} "
                  f"total={r['total']:6.2f}s  load={r['load']:5.2f}s")

    with open(BENCH / "bench_raw.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "rep", *PATTERNS])
        for c in configs:
            for i, r in enumerate(raw[c], 1):
                w.writerow([label(c), i, *[r[k] for k in PATTERNS]])

    med = lambda c, k: statistics.median(x[k] for x in raw[c])
    best = lambda c, k: min(x[k] for x in raw[c])

    t_serial = med("serial", "total")
    t_one = med(1, "total") if 1 in args.workers else None

    rows = []
    for c in args.workers:
        t = med(c, "total")
        s_total = t_serial / t
        s_par = (t_one / t) if t_one else float("nan")
        rows.append({
            "workers": c, "median_s": t, "best_s": best(c, "total"),
            "load_s": med(c, "load"), "speedup_vs_serial": s_total,
            "speedup_vs_1worker": s_par, "efficiency": s_par / c,
        })

    print(f"\nSerial baseline: median {t_serial:.2f} s, best {best('serial', 'total'):.2f} s")
    print(f"{'workers':>7} {'median':>8} {'best':>7} {'load':>6} "
          f"{'vs serial':>10} {'vs 1 worker':>12} {'efficiency':>11}")
    for r in rows:
        print(f"{r['workers']:>7} {r['median_s']:>7.2f}s {r['best_s']:>6.2f}s {r['load_s']:>5.2f}s "
              f"{r['speedup_vs_serial']:>9.2f}x {r['speedup_vs_1worker']:>11.2f}x "
              f"{r['efficiency']:>10.1%}")

    with open(BENCH / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ws = [r["workers"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(ws, [r["speedup_vs_serial"] for r in rows], "o-", label="vs serial baseline")
    a1.plot(ws, [r["speedup_vs_1worker"] for r in rows], "s-", label="vs 1 worker (pure parallel)")
    a1.plot(ws, ws, "k--", alpha=0.4, label="ideal")
    a1.set_xlabel("workers"); a1.set_ylabel("speedup"); a1.set_title("Speedup (median of runs)")
    a1.legend(); a1.grid(alpha=0.3)
    a2.plot(ws, [r["efficiency"] * 100 for r in rows], "o-", color="tab:red")
    a2.set_xlabel("workers"); a2.set_ylabel("efficiency (%)")
    a2.set_title("Efficiency = speedup vs 1 worker / workers"); a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(BENCH / "speedup.png", dpi=150)
    print("\nSaved: bench_raw.csv, summary.csv, speedup.png in benchmarks/")


if __name__ == "__main__":
    main()
