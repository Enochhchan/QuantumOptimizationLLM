#!/usr/bin/env python3
"""
Run tests from the *compilation report* by joining rows back to the source CSV
to fetch the Raw JSON, compile to QUBO, solve, and verify feasibility.

Usage (local solver):
  python run_qubo_tests_from_report.py \
    qubo_compilation_gpt4_report.csv \
    llm_qubo_results_detailed_with_fidelity_gpt4_without_loop_with_json.csv \
    --json-col "Raw JSON" \
    --out tests_local.csv

Optional flags:
  --row N             Only test a single source row index
  --strict            Compile in strict mode (if your validator supports it)
  --penalty-mult K    Multiply each constraint penalty by K (default 1.0)
  --dwave {none,hybrid,qpu}
"""

import argparse
import json
import ast
import re
import sys
import random
import inspect
from typing import Dict, Any, Tuple, List

import pandas as pd
import dimod
from dimod.reference.samplers import ExactSolver

# Optional D-Wave (graceful if unavailable)
HAVE_DWAVE = False
try:
    from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
    HAVE_DWAVE = True
except Exception:
    pass

# Import validator
import importlib.util, importlib.machinery
VALIDATOR_PATH = "qubo_validator.py"
spec = importlib.util.spec_from_file_location("qubo_validator", VALIDATOR_PATH)
qv = importlib.machinery.SourceFileLoader("qubo_validator", VALIDATOR_PATH).load_module()

# Back-compat: detect if compile_qubo has a 'strict' parameter
HAS_STRICT = 'strict' in inspect.signature(qv.compile_qubo).parameters


# ---------------------- utilities ----------------------

def _parse_json_field(s: str) -> Dict[str, Any]:
    """Parse JSON possibly wrapped in ```json fences or Python-literal style."""
    s = str(s).strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    try:
        return json.loads(s)
    except Exception:
        return ast.literal_eval(s)

def _normalize_objective(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure 'objective' has a sense prefix."""
    if isinstance(obj.get("objective"), str) and ":" not in obj["objective"]:
        t = obj["objective"].strip("{} ")
        if t:
            obj["objective"] = f"minimize: {t}"
    return obj

def _scale_penalties(qubo: dict, mult: float) -> dict:
    """Multiply each constraint 'penalty' by mult (insert if missing)."""
    if mult == 1.0:
        return qubo
    q = dict(qubo)
    new_cons = []
    for c in (q.get("constraints") or []):
        c = dict(c)
        p = c.get("penalty", None)
        if p is None:
            c["penalty"] = mult
        else:
            try:
                c["penalty"] = float(p) * mult
            except Exception:
                pass
        new_cons.append(c)
    q["constraints"] = new_cons
    return q

def _build_bqm(compiled: Dict[str, Any]) -> dimod.BinaryQuadraticModel:
    """Convert compiled dict (linear/quadratic/offset) into a dimod BQM."""
    Q = {}
    for v, h in compiled["linear"].items():
        Q[(v, v)] = Q.get((v, v), 0.0) + float(h)
    for (a, b), J in compiled["quadratic"].items():
        Q[(a, b)] = Q.get((a, b), 0.0) + float(J)
    return dimod.BinaryQuadraticModel.from_qubo(Q, offset=float(compiled.get("offset", 0.0)))

def _energy_of(bqm: dimod.BinaryQuadraticModel, sample: Dict[str, int]) -> float:
    return bqm.energy(sample)

def _greedy_descent(bqm: dimod.BinaryQuadraticModel, restarts: int = 100) -> dimod.SampleSet:
    """Multi-restart greedy bit-flip descent (no external deps)."""
    vars_ = list(bqm.variables)
    best_sample = None
    best_energy = float('inf')
    for _ in range(restarts):
        x = {v: random.randint(0,1) for v in vars_}
        improved = True
        while improved:
            improved = False
            random.shuffle(vars_)
            for v in vars_:
                x[v] ^= 1
                e_new = _energy_of(bqm, x)
                x[v] ^= 1
                e_old = _energy_of(bqm, x)
                if e_new < e_old - 1e-12:
                    x[v] ^= 1
                    improved = True
        e = _energy_of(bqm, x)
        if e < best_energy:
            best_energy = e
            best_sample = x.copy()
    return dimod.SampleSet.from_samples_bqm([best_sample], bqm)

def _solve_bqm(bqm: dimod.BinaryQuadraticModel):
    n = len(bqm.variables)
    if n <= 20:
        return ExactSolver().sample(bqm).first
    return _greedy_descent(bqm, restarts=100).first

def _solve_dwave(bqm: dimod.BinaryQuadraticModel, sampler_kind: str = "hybrid"):
    if not HAVE_DWAVE:
        raise RuntimeError("dwave-system not installed or no credentials configured")
    if sampler_kind == "hybrid":
        return LeapHybridSampler().sample(bqm).first
    else:
        sampler = EmbeddingComposite(DWaveSampler())
        maxJ = max([abs(J) for (_, _), J in bqm.quadratic.items()] + [1.0])
        chain = 2.0 * maxJ
        return sampler.sample(bqm, chain_strength=chain, num_reads=2000).first

def _evaluate_constraints(qubo_json: Dict[str, Any], assignment: Dict[str, int], strict: bool) -> List[Tuple[str, bool, float]]:
    """Re-evaluate original constraints on a given assignment. Returns [(label, ok, residual)]."""
    results = []
    constraints = qubo_json.get("constraints", []) or []
    for i, c in enumerate(constraints):
        expr = c.get("expression", "")
        converted, applied = qv._try_convert_not_equal(expr)
        if applied:
            expr = converted
        lhs, op, rhs = qv._split_relation(expr)
        if op is None:
            results.append((f"Constraint {i}", False, float('inf')))
            continue
        lhs_norm, _ = qv._normalize_expr(lhs) if not strict else (lhs.strip(), [])
        mono_lhs, const_lhs = qv._parse_polynomial(lhs_norm)
        try:
            rhs_val = float(rhs)
        except Exception:
            if strict:
                results.append((f"Constraint {i}", False, float('inf')))
                continue
            rhs_val = qv._safe_eval_num(qv._normalize_expr(rhs)[0])
        # Flip >= to <= convention
        if op == ">=":
            mono_lhs = [(-coef, vars_) for (coef, vars_) in mono_lhs]
            const_lhs = -const_lhs
            rhs_val = -rhs_val
            op = "<="

        lhs_val = const_lhs
        for coef, vs in mono_lhs:
            prod = 1.0
            for v in vs:
                prod *= float(assignment.get(v, 0))
            lhs_val += coef * prod

        if op in ("=", "=="):
            res = abs(lhs_val - rhs_val)
            ok = res <= 1e-6
            results.append((f"Constraint {i} (=)", ok, res))
        elif op == "<=":
            res = lhs_val - rhs_val
            ok = res <= 1e-6
            results.append((f"Constraint {i} (<=)", ok, res))
        else:
            results.append((f"Constraint {i} (?)", False, float('inf')))
    return results


# ---------------------- main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report_csv", type=str, help="validate_from_csv *_report.csv")
    ap.add_argument("source_csv", type=str, help="Original CSV containing the Raw JSON")
    ap.add_argument("--json-col", type=str, default="Raw JSON", help="JSON column name in source CSV")
    ap.add_argument("--row", type=int, default=None, help="Only test a specific source row index")
    ap.add_argument("--strict", action="store_true", help="Use strict compilation (if supported by validator)")
    ap.add_argument("--penalty-mult", type=float, default=1.0, help="Multiply all constraint penalties by this factor")
    ap.add_argument("--dwave", choices=["none","hybrid","qpu"], default="none", help="Use D-Wave sampler")
    ap.add_argument("--out", type=str, default=None, help="Write per-row results to this CSV")
    args = ap.parse_args()

    rep = pd.read_csv(args.report_csv)
    src = pd.read_csv(args.source_csv, dtype=str, engine="python", on_bad_lines="skip")

    if "index" not in rep.columns:
        print("Report is missing 'index' column.", file=sys.stderr)
        sys.exit(2)

    merged = rep.merge(src.reset_index().rename(columns={"index":"index_src"}),
                       left_on="index", right_on="index_src", how="left")

    # Filter to a single row if requested; otherwise test rows that compiled ok in the report
    if args.row is not None:
        merged = merged[merged["index"] == args.row]
    else:
        if "ok" in merged.columns:
            merged = merged[merged["ok"] == True]

    rows_out: List[Dict[str, Any]] = []

    for _, r in merged.iterrows():
        idx = int(r["index"])
        case = r.get("Case", "Unknown")
        raw = r.get(args.json_col, "")
        if not isinstance(raw, str) or not raw.strip():
            print(f"[row {idx} | {case}] No JSON — skipping")
            continue

        # Parse JSON from source row
        try:
            qubo = _parse_json_field(raw)
            if not isinstance(qubo, dict):
                raise ValueError("Parsed JSON is not an object")
        except Exception as e:
            print(f"[row {idx} | {case}] Parse error: {e}")
            continue

        qubo = _normalize_objective(qubo)
        qubo = _scale_penalties(qubo, args.penalty_mult)

        # Compile (respect validator's signature for strict)
        try:
            kwargs = {"strict": args.strict} if HAS_STRICT else {}
            comp = qv.compile_qubo(qubo, **kwargs)
        except Exception as e:
            print(f"[row {idx} | {case}] Compile error: {e}")
            continue

        if not comp.get("ok"):
            print(f"[row {idx} | {case}] Compile FAIL (unexpected): {comp.get('reason')}")
            continue

        # Build and solve
        bqm = _build_bqm(comp)
        print(f"[row {idx} | {case}] vars={len(bqm.variables)} lin={len(comp['linear'])} quad={len(comp['quadratic'])}")
        try:
            if args.dwave == "none":
                sample = _solve_bqm(bqm)
            else:
                sample = _solve_dwave(bqm, sampler_kind=args.dwave)
        except Exception as e:
            print(f"  Solver error: {e}")
            continue

        energy = sample.energy
        assignment = {str(v): int(sample.sample[v]) for v in bqm.variables}
        checks = _evaluate_constraints(qubo, assignment, strict=args.strict)
        ok = all(flag for (_, flag, _) in checks) if checks else True

        print(f"  best_energy={energy:.6f} constraints_ok={ok}")
        for label, flag, residual in checks:
            mark = "✓" if flag else "✗"
            print(f"   - {label}: {mark} (residual={residual:.3g})")

        rows_out.append({
            "index": idx,
            "case": case,
            "vars": len(bqm.variables),
            "lin_terms": len(comp["linear"]),
            "quad_terms": len(comp["quadratic"]),
            "slack_vars": len(comp.get("added_slack", [])),
            "best_energy": float(energy),
            "constraints_ok": bool(ok),
            "max_residual": max((abs(r) for (_,_,r) in checks), default=0.0),
            "sum_residual": sum((abs(r) for (_,_,r) in checks)) if checks else 0.0,
            "strict": bool(args.strict),
            "penalty_mult": float(args.penalty_mult),
            "sampler": ("dwave_" + args.dwave) if args.dwave != "none" else "local",
        })

    if args.out and rows_out:
        pd.DataFrame(rows_out).to_csv(args.out, index=False)
        print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
