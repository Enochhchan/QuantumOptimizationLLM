
import argparse, json, ast, re, sys, random
from typing import Dict, Any, Tuple, List

import dimod
from dimod.reference.samplers import ExactSolver

# Optional D-Wave (works only if account has access)
HAVE_DWAVE = False
try:
    from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
    HAVE_DWAVE = True
except Exception:
    pass

import pandas as pd

# Import our validator
import importlib.util, importlib.machinery
VALIDATOR_PATH = "qubo_validator.py"
spec = importlib.util.spec_from_file_location("qubo_validator", VALIDATOR_PATH)
qv = importlib.machinery.SourceFileLoader("qubo_validator", VALIDATOR_PATH).load_module()

def _parse_json_field(s: str) -> Dict[str, Any]:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    try:
        return json.loads(s)
    except Exception:
        return ast.literal_eval(s)

def _normalize_objective(obj: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(obj.get("objective"), str) and ":" not in obj["objective"]:
        t = obj["objective"].strip("{} ")
        if t:
            obj["objective"] = f"minimize: {t}"
    return obj

def _build_bqm(compiled: Dict[str, Any]) -> dimod.BinaryQuadraticModel:
    Q = {}
    for v, h in compiled["linear"].items():
        Q[(v, v)] = Q.get((v, v), 0.0) + float(h)
    for (a, b), J in compiled["quadratic"].items():
        Q[(a, b)] = Q.get((a, b), 0.0) + float(J)
    return dimod.BinaryQuadraticModel.from_qubo(Q, offset=float(compiled.get("offset", 0.0)))

def _energy_of(bqm: dimod.BinaryQuadraticModel, sample: Dict[str, int]) -> float:
    return bqm.energy(sample)

def _greedy_descent(bqm: dimod.BinaryQuadraticModel, restarts: int = 50) -> dimod.SampleSet:
    """Simple greedy bit-flip descent with multiple random restarts (no external deps)."""
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
                # evaluate delta by flipping v
                x[v] ^= 1
                e_new = _energy_of(bqm, x)
                x[v] ^= 1
                e_old = _energy_of(bqm, x)
                if e_new < e_old - 1e-12:
                    x[v] ^= 1  # accept flip
                    improved = True
        e = _energy_of(bqm, x)
        if e < best_energy:
            best_energy = e
            best_sample = x.copy()
    return dimod.SampleSet.from_samples_bqm([best_sample], bqm)

def _evaluate_constraints(qubo_json: Dict[str, Any], assignment: Dict[str, int], strict: bool) -> List[Tuple[str, bool, float]]:
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

def _solve_bqm(bqm: dimod.BinaryQuadraticModel):
    n = len(bqm.variables)
    if n <= 20:
        return ExactSolver().sample(bqm).first
    # Built-in greedy fallback
    return _greedy_descent(bqm, restarts=100).first

def _solve_dwave(bqm: dimod.BinaryQuadraticModel, sampler_kind: str = "hybrid"):
    if not HAVE_DWAVE:
        raise RuntimeError("dwave-system not installed or no credentials configured")
    if sampler_kind == "hybrid":
        sampler = LeapHybridSampler()
        ss = sampler.sample(bqm)
        return ss.first
    else:
        sampler = EmbeddingComposite(DWaveSampler())
        maxJ = max([abs(J) for (_, _), J in bqm.quadratic.items()] + [1.0])
        chain = 2.0 * maxJ
        ss = sampler.sample(bqm, chain_strength=chain, num_reads=2000)
        return ss.first
    
def _scale_penalties(qubo: dict, mult: float) -> dict:
    if mult == 1.0:
        return qubo
    q = dict(qubo)
    new_cons = []
    for c in (q.get("constraints") or []):
        c = dict(c)
        p = c.get("penalty", None)
        if p is None:
            # if no penalty specified, set one (will still be compared to validator default)
            c["penalty"] = mult
        else:
            try:
                c["penalty"] = float(p) * mult
            except Exception:
                # if non-numeric, leave as-is
                pass
        new_cons.append(c)
    q["constraints"] = new_cons
    return q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=str, help="Input CSV with JSON column (e.g., the original with Raw JSON)")
    ap.add_argument("--json-col", type=str, default=None, help="Name of JSON column (e.g., 'Raw JSON')")
    ap.add_argument("--case-col", type=str, default="Case", help="Case column")
    ap.add_argument("--row", type=int, default=None, help="Only test a specific row index")
    ap.add_argument("--strict", action="store_true", help="Use strict compilation")
    ap.add_argument("--dwave", choices=["none","hybrid","qpu"], default="none", help="Use D-Wave sampler (requires account access)")
    ap.add_argument("--penalty-mult", type=float, default=1.0,
                help="Multiply every constraint penalty by this factor before compiling")
    ap.add_argument("--out", type=str, default=None,
                help="Write per-row results to this CSV")

    args = ap.parse_args()

    df = pd.read_csv(args.csv_path, dtype=str, engine="python", on_bad_lines="skip")
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip().lower()
    json_col = args.json_col
    if not json_col:
        for c in df.columns:
            if "json" in norm(c):
                json_col = c
                break
    if not json_col:
        print("Could not find a JSON column. Try --json-col 'Raw JSON'.", file=sys.stderr)
        sys.exit(2)

    indices = [args.row] if args.row is not None else list(range(len(df)))
    rows_out = []

    for idx in indices:
        row = df.iloc[idx]
        raw = row.get(json_col, "")
        case = row.get(args.case_col, "Unknown")
        if not isinstance(raw, str) or not raw.strip():
            print(f"[row {idx} | {case}] No JSON — skipping")
            continue
        try:
            qubo = _parse_json_field(raw)
            if not isinstance(qubo, dict):
                raise ValueError("Parsed JSON is not an object")
        except Exception as e:
            print(f"[row {idx} | {case}] Parse error: {e}")
            continue

        qubo = _normalize_objective(qubo)

        qubo = _scale_penalties(qubo, args.penalty_mult)
        comp = qv.compile_qubo(qubo, strict=args.strict)
        if not comp.get("ok"):
            print(f"[row {idx} | {case}] Compile FAIL: {comp.get('reason')}")
            continue

        bqm = _build_bqm(comp)
        print(f"[row {idx} | {case}] vars={len(bqm.variables)} lin={len(comp['linear'])} quad={len(comp['quadratic'])} slack={len(comp.get('added_slack', []))}")
        if args.dwave == "none":
            sample = _solve_bqm(bqm)
        else:
            sample = _solve_dwave(bqm, sampler_kind=args.dwave)

        energy = sample.energy
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
            "penalty_mult": float(getattr(args, "penalty_mult", 1.0)),
            "sampler": ("dwave_" + args.dwave) if args.dwave != "none" else "local"
        })
        assignment = {str(v): int(sample.sample[v]) for v in bqm.variables}
        checks = _evaluate_constraints(qubo, assignment, strict=args.strict)

        ok = all(flag for (_, flag, _) in checks) if checks else True
        print(f"  best_energy={energy:.6f} constraints_ok={ok}")
        for label, flag, residual in checks:
            mark = "✓" if flag else "✗"
            print(f"   - {label}: {mark} (residual={residual:.3g})")
        if args.out:
            import pandas as pd
            pd.DataFrame(rows_out).to_csv(args.out, index=False)
            print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
