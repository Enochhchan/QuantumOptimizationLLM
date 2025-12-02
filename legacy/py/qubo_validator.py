
import re
import ast
import operator as _op
from typing import Dict, Tuple, Optional, List, Any

class QUBOCompileError(Exception):
    pass

Term = Tuple[str, Optional[str]]
Lin = Dict[str, float]
Quad = Dict[Tuple[str,str], float]

def _canonical_var(name: str) -> str:
    return re.sub(r'\s+', '_', str(name).strip())

def _extract_var_names(var_field) -> List[str]:
    names = []
    if isinstance(var_field, list):
        for v in var_field:
            if isinstance(v, str):
                names.append(_canonical_var(v))
            elif isinstance(v, dict) and "name" in v:
                names.append(_canonical_var(str(v["name"])))
    return names

def _normalize_expr(expr: str) -> Tuple[str, List[str]]:
    """
    Lightweight normalizer to handle common LLM artifacts (semantics-preserving).
    Returns (normalized_string, tags_applied).
    """
    tags: List[str] = []
    if expr is None:
        return expr, tags
    s = str(expr)

    # Map Unicode lookalikes to ASCII
    before = s
    s = (s.replace('≤', '<=')
           .replace('≥', '>=')
           .replace('≠', '!=')
           .replace('−', '-')   # unicode minus
           .replace('×', '*')
           .replace('·', '*'))
    if s != before:
        tags.append("unicode_ops")

    # Remove commas that sometimes appear in numbers/terms
    before = s
    s = s.replace(',', ' ')
    if s != before:
        tags.append("commas_removed")

    # number^2 -> squared constant (e.g., 6^2 -> 36)
    def _const_square(m):
        n = float(m.group(1))
        out = n * n
        return str(int(out)) if abs(out - int(out)) < 1e-12 else str(out)
    before = s
    s = re.sub(r'\b(\d+(?:\.\d+)?)\s*\^\s*2\b', _const_square, s)
    if s != before:
        tags.append("const_pow2")

    # variable^2 -> variable (binary idempotence)
    before = s
    s = re.sub(r'\b([A-Za-z_][A-Za-z0-9_ ]*)\s*\^\s*2\b', lambda m: m.group(1), s)
    if s != before:
        tags.append("var_pow2")

    # "- VAR" -> "-1*VAR"
    before = s
    s = re.sub(r'-\s+([A-Za-z_][A-Za-z0-9_ ]*)', r'-1*\1', s)
    if s != before:
        tags.append("minus_space_to_coef")

    # Logical NOT on a single variable: !x or x! -> (1 - x)
    before = s
    s2 = re.sub(r'!\s*([A-Za-z_][A-Za-z0-9_ ]*)', lambda m: f"(1 - {m.group(1).strip()})", s)
    s2 = re.sub(r'([A-Za-z_][A-Za-z0-9_ ]*)\s*!', lambda m: f"(1 - {m.group(1).strip()})", s2)
    if s2 != s:
        tags.append("logical_not")
        s = s2

    # Strip (Var) and (123) when they wrap a single token
    before = s
    s = re.sub(r'\(\s*([A-Za-z_][A-Za-z0-9_ ]*)\s*\)', r'\1', s)
    s = re.sub(r'\(\s*([-+]?\d+(?:\.\d+)?)\s*\)', r'\1', s)
    if s != before:
        tags.append("strip_singleton_parens")

    # Drop remaining lone parentheses (we don't support grouped polynomials)
    before = s
    s = s.replace("(", " ").replace(")", " ")
    if s != before:
        tags.append("drop_lone_parens")

    # Collapse spaces
    s = re.sub(r'\s+', ' ', s).strip()

    return s, tags

def _try_convert_not_equal(expr: str) -> Tuple[Optional[str], bool]:
    """
    If a constraint is exactly 'A != B', convert to 'A + B = 1'. Assumes binary variables.
    Returns (converted_expr_or_none, applied_flag).
    """
    s = str(expr).strip()
    m = re.fullmatch(r'([A-Za-z_][A-Za-z0-9_ ]*)\s*!=\s*([A-Za-z_][A-Za-z0-9_ ]*)', s)
    if m:
        a = m.group(1).strip()
        b = m.group(2).strip()
        return f"{a} + {b} = 1", True
    return None, False

def _split_relation(expr: str) -> Tuple[str, Optional[str], Optional[str]]:
    m = re.search(r'(<=|>=|==|=)', expr)
    if not m:
        return expr, None, None
    op = m.group(1)
    parts = re.split(r'(?:<=|>=|==|=)', expr, maxsplit=1)
    if len(parts) != 2:
        raise QUBOCompileError(f"Cannot split relation in expression: {expr}")
    lhs, rhs = parts[0].strip(), parts[1].strip()
    return lhs, op, rhs

def _split_terms(expr: str) -> List[str]:
    s = re.sub(r'\s+', ' ', expr.strip())
    s = s.replace('-', '+-')
    parts = [p.strip() for p in s.split('+') if p.strip()]
    return parts

def _parse_term_general(part: str) -> Tuple[float, List[str]]:
    """
    Parse a term like: "3*A*B*C", "A*B", "-2.5*A", "10", "-City0*City1*City2"
    Returns (coefficient, [vars]) where vars is empty for a constant.
    """
    coef = 1.0
    factors = [f.strip() for f in part.split('*') if f.strip()]
    if not factors:
        return 0.0, []

    if len(factors) == 1 and re.fullmatch(r'[-+]?\d+(\.\d+)?', factors[0]):
        return float(factors[0]), []

    vars_: List[str] = []
    for f in factors:
        # number?
        if re.fullmatch(r'[-+]?\d+(\.\d+)?', f):
            coef *= float(f)
            continue
        # unary '-' before a name like "-A"
        if f.startswith('-') and re.fullmatch(r'[-+]?[A-Za-z_][A-Za-z0-9_ ]*', f):
            coef *= -1.0
            f = f[1:]
        # variable token
        if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_ ]*', f):
            vars_.append(_canonical_var(f))
            continue
        raise QUBOCompileError(f"Unrecognized factor '{f}' in term '{part}'")

    return coef, vars_

def _parse_polynomial(expr: str) -> Tuple[List[Tuple[float, List[str]]], float]:
    """Return list of (coef, [vars]) monomials and constant offset."""
    if expr is None:
        return [], 0.0
    terms = _split_terms(expr)
    monomials: List[Tuple[float, List[str]]] = []
    const = 0.0
    for part in terms:
        coef, vars_ = _parse_term_general(part)
        if len(vars_) == 0:
            const += coef
        else:
            monomials.append((coef, vars_))
    return monomials, const

def _add_to_qubo(linear: Lin, quadratic: Quad, add_lin: Lin, add_quad: Quad, scale: float = 1.0):
    for v, c in add_lin.items():
        linear[v] = linear.get(v, 0.0) + scale * c
    for (a, b), c in add_quad.items():
        key = tuple(sorted((a, b)))
        quadratic[key] = quadratic.get(key, 0.0) + scale * c

def _square_linear(line: Lin, const: float = 0.0) -> Tuple[Lin, Quad, float]:
    L: Lin = {}
    Q: Quad = {}
    off = const * const
    for v, c in line.items():
        L[v] = L.get(v, 0.0) + c * c
    items = list(line.items())
    for i in range(len(items)):
        vi, ci = items[i]
        for j in range(i + 1, len(items)):
            vj, cj = items[j]
            key = tuple(sorted((vi, vj)))
            Q[key] = Q.get(key, 0.0) + 2.0 * ci * cj
    if const != 0.0:
        for v, c in line.items():
            L[v] = L.get(v, 0.0) + 2.0 * const * c
    return L, Q, off

def _safe_eval_num(expr: str) -> float:
    """
    Evaluate a numeric-only expression safely: digits, + - * / and dots.
    Rejects names/symbols (e.g., 'avgTeam1').
    """
    expr = expr.strip()
    if not re.fullmatch(r'[-+*/ 0-9.]+', expr):
        raise ValueError("Non-numeric RHS")
    node = ast.parse(expr, mode='eval').body
    def _eval(n):
        if isinstance(n, ast.Num):
            return float(n.n)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            return +_eval(n.operand) if isinstance(n.op, ast.UAdd) else -_eval(n.operand)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            ops = {ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul, ast.Div: _op.truediv}
            return ops[type(n.op)](_eval(n.left), _eval(n.right))
        raise ValueError("Bad RHS")
    return _eval(node)

class Quadratizer:
    """
    Introduce auxiliary variables to reduce high-degree monomials to linear/quadratic form.
    Uses Rosenberg reduction for z = x*y: add penalty λ*(x*y - 2 z x - 2 z y + 3 z).
    Chains for 3+ variables.
    """
    def __init__(self, linear: Lin, quadratic: Quad, vars_declared: set, base_lambda: float = 10.0):
        self.linear = linear
        self.quadratic = quadratic
        self.vars_declared = vars_declared
        self.base_lambda = max(1.0, float(base_lambda))
        self._aux_counter = 0
        self.added_aux: List[str] = []
        self._memo = {}  # maps canonical factor tuples -> aux name

    def _new_aux(self) -> str:
        self._aux_counter += 1
        name = f"AUX_{self._aux_counter}"
        while name in self.vars_declared:
            self._aux_counter += 1
            name = f"AUX_{self._aux_counter}"
        self.vars_declared.add(name)
        self.added_aux.append(name)
        return name

    def _add_rosenberg_penalty(self, z: str, x: str, y: str, lam: float):
        # λ*(x*y - 2*z*x - 2*z*y + 3*z)
        key_xy = tuple(sorted((x, y)))
        self.quadratic[key_xy] = self.quadratic.get(key_xy, 0.0) + lam
        key_zx = tuple(sorted((z, x)))
        self.quadratic[key_zx] = self.quadratic.get(key_zx, 0.0) - 2.0 * lam
        key_zy = tuple(sorted((z, y)))
        self.quadratic[key_zy] = self.quadratic.get(key_zy, 0.0) - 2.0 * lam
        self.linear[z] = self.linear.get(z, 0.0) + 3.0 * lam

    def product_to_aux(self, factors: List[str], lam: Optional[float] = None) -> str:
        """
        Return a variable name 'u' that represents the product of 'factors'.
        Adds quadratic penalties so that u = product(factors).
        For length 2, creates one auxiliary with penalty.
        For length >=3, chains auxiliaries.
        Reuses auxiliaries for repeated products (order-invariant).
        """
        if lam is None:
            lam = self.base_lambda
        if len(factors) < 2:
            return factors[0] if factors else "CONST_1"

        key = tuple(sorted(factors))
        if key in self._memo:
            return self._memo[key]

        # First pair
        cur = self._new_aux()
        self._add_rosenberg_penalty(cur, key[0], key[1], lam)
        # Chain remaining
        for f in key[2:]:
            nxt = self._new_aux()
            self._add_rosenberg_penalty(nxt, cur, f, lam)
            cur = nxt

        self._memo[key] = cur
        return cur

def compile_qubo(qubo_json: Dict[str, Any], penalty_scale: Optional[float] = None, strict: bool = False) -> Dict[str, Any]:
    fixes_applied: List[str] = []

    # Extract variables
    vars_declared = set(_extract_var_names(qubo_json.get("variables", [])))
    if not vars_declared:
        return {"ok": False, "reason": "No variables declared"}

    # Parse and handle objective
    obj = qubo_json.get("objective")
    if not isinstance(obj, str) or ':' not in obj:
        return {"ok": False, "reason": "Objective missing or malformed"}
    sense, expr = obj.split(':', 1)
    sense = sense.strip().lower()
    expr_raw = expr.strip()

    if sense not in ("minimize", "maximize"):
        return {"ok": False, "reason": "Objective sense must be 'minimize' or 'maximize'"}

    if strict:
        expr_norm, _ = expr_raw, []  # no normalization beyond strip
    else:
        expr_norm, tags = _normalize_expr(expr_raw)
        fixes_applied.extend(tags)

    mono_obj, off_o = _parse_polynomial(expr_norm)

    # Determine penalty scales
    base_obj_scale = sum(abs(coef) for coef, _ in mono_obj)
    if penalty_scale is None:
        penalty_scale = max(1.0, 10.0 * (base_obj_scale if base_obj_scale > 0 else 1.0))
    aux_lambda = 2.0 * penalty_scale  # strong enough to enforce product == aux

    # Build QUBO accumulators
    linear: Lin = {}
    quadratic: Quad = {}
    offset: float = 0.0
    quadrt = Quadratizer(linear, quadratic, vars_declared, base_lambda=aux_lambda)

    # Objective terms
    sign = -1.0 if sense == "maximize" else 1.0
    for coef, varlist in mono_obj:
        coef *= sign
        if len(varlist) == 0:
            offset += coef
        elif len(varlist) == 1:
            v = varlist[0]
            if v not in vars_declared:
                return {"ok": False, "reason": f"Objective uses undeclared variable '{v}'"}
            linear[v] = linear.get(v, 0.0) + coef
        elif len(varlist) == 2:
            a, b = sorted(varlist)
            if a not in vars_declared or b not in vars_declared:
                return {"ok": False, "reason": f"Objective uses undeclared variable in {varlist}"}
            quadratic[(a, b)] = quadratic.get((a, b), 0.0) + coef
        else:
            if strict:
                return {"ok": False, "reason": f"Objective has degree>2 monomial {varlist}"}
            for v in varlist:
                if v not in vars_declared:
                    return {"ok": False, "reason": f"Objective uses undeclared variable '{v}'"}
            u = quadrt.product_to_aux(varlist, lam=aux_lambda)
            linear[u] = linear.get(u, 0.0) + coef

    # Constraints
    added_slack: List[str] = []
    constraints = qubo_json.get("constraints", [])
    if not isinstance(constraints, list):
        return {"ok": False, "reason": "Constraints must be a list"}

    for i, c in enumerate(constraints):
        if not isinstance(c, dict):
            return {"ok": False, "reason": f"Constraint {i} is not a dict"}
        expr = c.get("expression")
        pen = c.get("penalty", penalty_scale)
        try:
            pen = float(pen)
        except Exception:
            return {"ok": False, "reason": f"Constraint {i} penalty is not numeric"}
        if pen <= 0:
            return {"ok": False, "reason": f"Constraint {i} penalty must be > 0"}
        if not isinstance(expr, str):
            return {"ok": False, "reason": f"Constraint {i} expression missing"}

        # Maybe convert 'A != B' -> 'A + B = 1'
        converted, applied = _try_convert_not_equal(expr)
        if applied:
            fixes_applied.append("not_equal_to_xor")
            expr = converted

        lhs, op, rhs = _split_relation(expr)
        if op is None:
            return {"ok": False, "reason": f"Constraint {i} missing relation operator"}

        # Normalize LHS (unless strict)
        if strict:
            lhs_norm = lhs.strip()
        else:
            lhs_norm, tags = _normalize_expr(lhs)
            fixes_applied.extend(tags)

        # Parse LHS
        if op in (">=",):
            mono_lhs, const_lhs = _parse_polynomial(lhs_norm)
            mono_lhs = [(-coef, vars_) for (coef, vars_) in mono_lhs]
            const_lhs = -const_lhs
            # RHS
            try:
                rhs_val = float(rhs)
            except Exception:
                if strict:
                    return {"ok": False, "reason": f"Constraint {i} RHS is not numeric"}
                rhs_val = _safe_eval_num(_normalize_expr(rhs)[0])
            rhs_val = -rhs_val
            op = "<="
        else:
            mono_lhs, const_lhs = _parse_polynomial(lhs_norm)
            try:
                rhs_val = float(rhs)
            except Exception:
                if strict:
                    return {"ok": False, "reason": f"Constraint {i} RHS is not numeric"}
                rhs_val = _safe_eval_num(_normalize_expr(rhs)[0])

        # In strict mode, constraints must be linear
        if strict:
            for _, varlist in mono_lhs:
                if len(varlist) >= 2:
                    return {"ok": False, "reason": f"Constraint {i} contains quadratic terms; only linear supported"}

        # Linearize LHS via auxiliaries for degree>=2 monomials
        lin_map: Lin = {}
        const_lin = const_lhs
        for coef, varlist in mono_lhs:
            if len(varlist) == 0:
                const_lin += coef
            elif len(varlist) == 1:
                v = varlist[0]
                if v not in vars_declared:
                    return {"ok": False, "reason": f"Constraint {i} uses undeclared variable '{v}'"}
                lin_map[v] = lin_map.get(v, 0.0) + coef
            elif len(varlist) >= 2:
                if strict:
                    return {"ok": False, "reason": f"Constraint {i} contains quadratic terms; only linear supported"}
                for v in varlist:
                    if v not in vars_declared:
                        return {"ok": False, "reason": f"Constraint {i} uses undeclared variable '{v}'"}
                u = quadrt.product_to_aux(varlist, lam=aux_lambda)
                lin_map[u] = lin_map.get(u, 0.0) + coef

        if op == "==":
            op = "="

        if op == "=":
            const_term = const_lin - rhs_val
            sq_lin, sq_quad, sq_off = _square_linear(lin_map, const_term)
            _add_to_qubo(linear, quadratic, sq_lin, sq_quad, scale=pen)
            offset += pen * sq_off
        elif op == "<=":
            if abs(rhs_val - int(round(rhs_val))) > 1e-9:
                return {"ok": False, "reason": f"Constraint {i} RHS must be integer for '<='"}
            rhs_int = int(round(rhs_val))
            if rhs_int < 0:
                return {"ok": False, "reason": f"Constraint {i} RHS must be nonnegative for '<='"}
            s_max = max(0, rhs_int)
            nbits = 0
            cap = 1
            while cap <= s_max:
                nbits += 1
                cap <<= 1
            slack_vars = []
            for k in range(nbits):
                sname = f"SLACK_{i}_{k}"
                slack_vars.append(sname)
                vars_declared.add(sname)
            if nbits > 0:
                fixes_applied.append(f"slack_bits[{i}]:{nbits}")
            added_slack.extend(slack_vars)
            lin_total = dict(lin_map)
            for k in range(nbits):
                lin_total[f"SLACK_{i}_{k}"] = lin_total.get(f"SLACK_{i}_{k}", 0.0) + (1 << k)
            const_term = const_lin - rhs_int
            sq_lin, sq_quad, sq_off = _square_linear(lin_total, const_term)
            _add_to_qubo(linear, quadratic, sq_lin, sq_quad, scale=pen)
            offset += pen * sq_off
        else:
            return {"ok": False, "reason": f"Constraint {i} uses unsupported operator '{op}'"}

    aux_count = len(quadrt.added_aux)
    if aux_count > 0:
        fixes_applied.append(f"aux_vars:{aux_count}")

    return {
        "ok": True,
        "reason": None,
        "linear": linear,
        "quadratic": quadratic,
        "offset": offset,
        "variables": sorted(list(vars_declared)),
        "added_slack": added_slack,
        "objective_sense": "minimize",
        "fixes": fixes_applied,
    }
