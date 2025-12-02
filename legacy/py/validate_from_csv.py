#!/usr/bin/env python3
"""
validate_from_csv.py

Reads a CSV that contains an LLM-produced QUBO JSON column and attempts
to compile each row into a QUBO using `qubo_validator.compile_qubo`.

Outputs:
- <out_prefix>_report.csv         : row-by-row pass/fail with reasons and term counts (+ fixes audit)
- <out_prefix>_summary.csv        : success rate aggregated by case
- <out_prefix>_success_rate.png   : bar chart of success rates

Flags:
--strict  : evaluate without normalization/quadratization (stricter scoring);
            constraints must be linear; objective must be <= quadratic; RHS must be numeric.

This script is *backward compatible* with older `qubo_validator.py` that lacks the `strict`
argument. If the `strict` parameter is not present, it will be omitted automatically.
"""

import argparse
import csv
import json
import ast
import inspect
from typing import Optional, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt

from qubo_validator import compile_qubo, QUBOCompileError  # noqa: E402


def detect_json_col(cols, preferred: Optional[str] = None) -> Optional[str]:
    """Heuristically find the JSON column."""
    if preferred and preferred in cols:
        return preferred

    lowered = {c: c.strip().lower().replace("\xa0", " ") for c in cols}

    # Common exact names first
    for c, lc in lowered.items():
        if lc in {"raw json", "raw_json", "raw output", "raw_output"}:
            return c

    # Any column that mentions 'json'
    for c, lc in lowered.items():
        if "json" in lc:
            return c

    return None


def clean_code_fences(s: str) -> str:
    """Strip ```json ...``` fences if present."""
    s = str(s).strip()
    if s.startswith("```"):
        # Remove all backticks then drop a leading 'json' language tag if any
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    return s.strip()


def normalize_objective(obj_str: str) -> str:
    """
    Ensure objective has a sense prefix (minimize|maximize).
    - If already "minimize: <expr>" or "maximize: <expr>" => keep it.
    - If just "<expr>" => default to "minimize: <expr>".
    - If an object-like string => try to unwrap.
    """
    s = obj_str.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    if ":" in s:
        sense, expr = s.split(":", 1)
        expr = expr.strip()
        if expr.startswith("{") and expr.endswith("}"):
            expr = expr[1:-1].strip()
        return f"{sense.strip()}: {expr}"
    if s:
        return f"minimize: {s}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=str, help="Input CSV with a JSON column")
    ap.add_argument("--json-col", type=str, default=None,
                    help="Column name containing the JSON string (e.g., 'Raw JSON')")
    ap.add_argument("--case-col", type=str, default="Case",
                    help="Column for case/category (e.g., 'Case' or 'benchmark')")
    ap.add_argument("--prompt-col", type=str, default="Prompt",
                    help="Column for the natural language prompt (optional; only recorded)")
    ap.add_argument("--model-tag", type=str, default="",
                    help="Optional tag to include in outputs (e.g., 'gpt-4')")
    ap.add_argument("--out-prefix", type=str, default="qubo_compilation",
                    help="Prefix for output files")
    ap.add_argument("--strict", action="store_true",
                    help="Evaluate without normalization/quadratization (stricter scoring)")
    args = ap.parse_args()

    # Load input CSV
    df = pd.read_csv(args.csv_path, dtype=str, engine="python", on_bad_lines="skip")

    # Locate JSON column
    json_col = detect_json_col(df.columns, preferred=args.json_col)
    if json_col is None:
        raise SystemExit(f"Could not find a JSON column. Available columns: {list(df.columns)}")

    case_col = args.case_col if args.case_col in df.columns else None

    # Backward-compat for validators without a 'strict' parameter
    HAS_STRICT = 'strict' in inspect.signature(compile_qubo).parameters

    results = []
    for idx, row in df.iterrows():
        raw = row.get(json_col, None)
        case = row.get(case_col, None) if case_col else None

        if not isinstance(raw, str) or not raw.strip():
            results.append({
                "index": idx, "case": case, "ok": False, "reason": "No JSON in row",
                "num_vars_declared": None, "num_vars_qubo": None,
                "num_quad_terms": None, "num_lin_terms": None, "num_slack_vars": None,
                "fixes": None, "model": args.model_tag
            })
            continue

        s = clean_code_fences(raw)

        # Parse JSON (try JSON first, then Python literal)
        try:
            try:
                qubo = json.loads(s)
            except Exception:
                qubo = ast.literal_eval(s)
            if not isinstance(qubo, dict):
                raise ValueError("Parsed value is not an object")
        except Exception as e:
            results.append({
                "index": idx, "case": case, "ok": False, "reason": f"Parse error: {e}",
                "num_vars_declared": None, "num_vars_qubo": None,
                "num_quad_terms": None, "num_lin_terms": None, "num_slack_vars": None,
                "fixes": None, "model": args.model_tag
            })
            continue

        # Normalize objective sense if missing
        if isinstance(qubo.get("objective"), str):
            qubo["objective"] = normalize_objective(qubo["objective"])

        # Compile
        try:
            kwargs = {"strict": args.strict} if HAS_STRICT else {}
            comp = compile_qubo(qubo, **kwargs)

            if comp.get("ok"):
                results.append({
                    "index": idx, "case": case, "ok": True, "reason": None,
                    "num_vars_declared": len(qubo.get("variables", [])) if isinstance(qubo.get("variables", []), list) else None,
                    "num_vars_qubo": len(comp.get("variables", []) or []),
                    "num_lin_terms": len(comp.get("linear", {}) or {}),
                    "num_quad_terms": len(comp.get("quadratic", {}) or {}),
                    "num_slack_vars": len(comp.get("added_slack", []) or []),
                    "fixes": ";".join(comp.get("fixes", [])) if comp.get("fixes") else "",
                    "model": args.model_tag
                })
            else:
                results.append({
                    "index": idx, "case": case, "ok": False, "reason": comp.get("reason"),
                    "num_vars_declared": len(qubo.get("variables", [])) if isinstance(qubo.get("variables", []), list) else None,
                    "num_vars_qubo": None, "num_quad_terms": None, "num_lin_terms": None, "num_slack_vars": None,
                    "fixes": ";".join(comp.get("fixes", [])) if comp.get("fixes") else "",
                    "model": args.model_tag
                })
        except QUBOCompileError as e:
            results.append({
                "index": idx, "case": case, "ok": False, "reason": f"{e}",
                "num_vars_declared": len(qubo.get("variables", [])) if isinstance(qubo.get("variables", []), list) else None,
                "num_vars_qubo": None, "num_quad_terms": None, "num_lin_terms": None, "num_slack_vars": None,
                "fixes": "", "model": args.model_tag
            })
        except Exception as e:
            results.append({
                "index": idx, "case": case, "ok": False, "reason": f"Unexpected error: {type(e).__name__}: {e}",
                "num_vars_declared": len(qubo.get("variables", [])) if isinstance(qubo.get("variables", []), list) else None,
                "num_vars_qubo": None, "num_quad_terms": None, "num_lin_terms": None, "num_slack_vars": None,
                "fixes": "", "model": args.model_tag
            })

    # Save per-row report
    out_df = pd.DataFrame(results)
    report_path = f"{args.out_prefix}_report.csv"
    out_df.to_csv(report_path, index=False)

    # Aggregate success by case
    if "case" in out_df.columns:
        agg = out_df.groupby("case")["ok"].mean().reset_index().sort_values("ok", ascending=False)
    else:
        tmp = out_df.copy()
        tmp["case"] = tmp["case"].fillna("Unknown")
        agg = tmp.groupby("case")["ok"].mean().reset_index().sort_values("ok", ascending=False)

    summary_path = f"{args.out_prefix}_summary.csv"
    agg.to_csv(summary_path, index=False)

    # Simple success-rate chart
    plt.figure(figsize=(8, 4))
    plt.bar(agg["case"], agg["ok"])
    plt.ylim(0, 1)
    plt.title("Compilation Success Rate by Case" + (" (STRICT)" if args.strict else ""))
    plt.xlabel("Case")
    plt.ylabel("Success Rate")
    plt.xticks(rotation=30)
    plt.tight_layout()
    chart_path = f"{args.out_prefix}_success_rate.png"
    plt.savefig(chart_path, dpi=200)

    print(f"Wrote: {report_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {chart_path}")


if __name__ == "__main__":
    main()
