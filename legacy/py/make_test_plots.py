#!/usr/bin/env python3
"""
Generate figures from the tester CSV produced by run_qubo_tests_from_report.py (or run_qubo_tests.py).

Required columns in the CSV:
  case, constraints_ok, vars, best_energy, max_residual

Outputs (PNG files):
  - fig_feasibility_by_case.png
  - fig_fail_buckets.png
  - fig_energy_vs_vars.png
  - fig_residual_by_case.png
Optional (if --tsp-sweep CSVs provided):
  - fig_penalty_sweep_tsp.png
Also writes:
  - test_summary.csv  (per-case feasibility & residuals)
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Tester output CSV (e.g., tests_local.csv)")
    ap.add_argument("--out-dir", default=".", help="Directory to write figures")
    ap.add_argument("--title-suffix", default="", help="Optional title suffix (e.g., ' (local)')")
    ap.add_argument("--tsp-sweep", nargs="*", default=[],
                    help="Optional list of CSVs from runs with different penalty multipliers (in order x1, x10, x100...)")
    args = ap.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    # -------- 1) Feasibility rate by case --------
    if {"case", "constraints_ok"}.issubset(df.columns):
        rate = df.groupby("case")["constraints_ok"].mean().reset_index()
        plt.figure(figsize=(8,4))
        plt.bar(rate["case"], rate["constraints_ok"])
        plt.ylim(0, 1)
        plt.ylabel("Feasibility rate")
        plt.title(f"Solver Feasibility by Case{args.title_suffix}")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(outdir / "fig_feasibility_by_case.png", dpi=200)
        plt.close()
    else:
        print("[warn] Missing columns for feasibility plot: need {case, constraints_ok}")

    # -------- 2) Failure reasons by residual bucket --------
    if {"case", "constraints_ok", "max_residual"}.issubset(df.columns):
        bins = [0, 1e-9, 1.0, 2.0, 1e9]
        labels = ["0", "(0,1]", "(1,2]", ">2"]
        tmp = df.copy()
        tmp["bucket"] = pd.cut(tmp["max_residual"].abs(), bins=bins, labels=labels, include_lowest=True)
        fail = tmp[tmp["constraints_ok"] == False]
        if not fail.empty:
            cts = fail.groupby(["case", "bucket"], observed=False).size().unstack(fill_value=0)[labels]
            plt.figure(figsize=(9,5))
            bottom = None
            for lab in labels:
                vals = cts[lab]
                if bottom is None:
                    plt.bar(cts.index, vals, label=lab)
                    bottom = vals
                else:
                    plt.bar(cts.index, vals, bottom=bottom, label=lab)
                    bottom = bottom + vals
            plt.title("Why Rows Failed: Max Residual Buckets" + args.title_suffix)
            plt.xlabel("Case")
            plt.ylabel("Row count")
            plt.xticks(rotation=20)
            plt.legend(title="Max Residual")
            plt.tight_layout()
            plt.savefig(outdir / "fig_fail_buckets.png", dpi=200)
            plt.close()
        else:
            print("[info] No failures present; skipping residual-buckets plot.")
    else:
        print("[warn] Missing columns for fail-buckets plot: need {case, constraints_ok, max_residual}")

    # -------- 3) Energy vs variables --------
    if {"vars", "best_energy", "case", "constraints_ok"}.issubset(df.columns):
        cases = sorted(df["case"].dropna().unique())
        markers = {True: "o", False: "x"}
        plt.figure(figsize=(8,5))
        for c in cases:
            for ok in [True, False]:
                sub = df[(df["case"] == c) & (df["constraints_ok"] == ok)]
                if len(sub) == 0:
                    continue
                plt.scatter(sub["vars"], sub["best_energy"],
                            label=f"{c} ({'OK' if ok else 'NO'})",
                            marker=markers[ok], alpha=0.85)
        plt.xlabel("Variables")
        plt.ylabel("QUBO energy (arb. units)")
        plt.title("Energy vs Problem Size" + args.title_suffix)
        # Dedup legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        seen = set(); new_h, new_l = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                new_h.append(h); new_l.append(l); seen.add(l)
        plt.legend(new_h, new_l, bbox_to_anchor=(1.02,1), loc="upper left")
        plt.tight_layout()
        plt.savefig(outdir / "fig_energy_vs_vars.png", dpi=200)
        plt.close()
    else:
        print("[warn] Missing columns for energy-vs-vars plot: need {vars, best_energy, case, constraints_ok}")

    # -------- 4) Residual magnitude by case --------
    if {"case", "max_residual"}.issubset(df.columns):
        res = df.groupby("case")["max_residual"].mean().reset_index()
        plt.figure(figsize=(8,4))
        plt.bar(res["case"], res["max_residual"])
        plt.ylabel("Mean max residual")
        plt.title("Constraint Residual Magnitude by Case" + args.title_suffix)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(outdir / "fig_residual_by_case.png", dpi=200)
        plt.close()
    else:
        print("[warn] Missing columns for residual-by-case plot: need {case, max_residual}")

    # -------- 5) Optional: penalty sweep (TSP) --------
        if args.tsp_sweep:
            sweeps = []
            for path in args.tsp_sweep:
                p = Path(path)
                if not p.exists():
                    print(f"[warn] sweep file not found: {p}")
                    continue
                d = pd.read_csv(p)
                if "case" in d.columns and "constraints_ok" in d.columns:
                    d = d[d["case"].str.contains("TSP", na=False)]
                    frac = d["constraints_ok"].mean() if len(d) else float("nan")
                    pm = float(d["penalty_mult"].iloc[0]) if "penalty_mult" in d.columns and len(d) else float("nan")
                    sweeps.append((pm, frac))
        if sweeps:
            sweeps.sort(key=lambda x: x[0])
            xs = [pm for pm, _ in sweeps]
            ys = [fr for _, fr in sweeps]
            plt.figure(figsize=(7,4))
            plt.plot(xs, ys, marker="o")
            plt.xscale("log")
            plt.ylim(0, 1)
            plt.xlabel("Penalty multiplier (log scale)")
            plt.ylabel("Feasibility rate (TSP)")
            plt.title("Penalty Multiplier Sweep (TSP)")
            plt.tight_layout()
            plt.savefig(outdir / "fig_penalty_sweep_tsp.png", dpi=200)
            plt.close()

    # -------- Summary CSV for slide notes --------
    have_ok = "constraints_ok" in df.columns
    have_res = "max_residual" in df.columns
    rows = []
    for c, g in df.groupby("case"):
        rows.append({
            "case": c,
            "rows": len(g),
            "feasibility_rate": g["constraints_ok"].mean() if have_ok else None,
            "mean_max_residual": g["max_residual"].mean() if have_res else None,
        })
    pd.DataFrame(rows).to_csv(outdir / "test_summary.csv", index=False)
    print("Wrote figures to", outdir.resolve())
    print("Wrote summary  :", (outdir / 'test_summary.csv').resolve())

if __name__ == "__main__":
    main()
