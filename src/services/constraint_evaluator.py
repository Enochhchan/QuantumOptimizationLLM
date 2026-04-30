from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


class ConstraintEvaluator:
    def __init__(self) -> None:
        self._legacy_module = self._load_legacy_module()

    def evaluate(self, qubo_json: dict[str, Any], assignment: dict[str, int]) -> tuple[bool, float]:
        residuals: list[float] = []
        constraints = qubo_json.get("constraints", []) or []
        for idx, constraint in enumerate(constraints):
            if not isinstance(constraint, dict):
                residuals.append(float("inf"))
                continue
            expr = str(constraint.get("expression", ""))
            converted, applied = self._legacy_module._try_convert_not_equal(expr)
            if applied:
                expr = converted
            lhs, op, rhs = self._legacy_module._split_relation(expr)
            if op is None:
                residuals.append(float("inf"))
                continue

            lhs_norm, _ = self._legacy_module._normalize_expr(lhs)
            monomials, const_term = self._legacy_module._parse_polynomial(lhs_norm)

            try:
                rhs_value = float(rhs)
            except Exception:
                try:
                    rhs_value = self._legacy_module._safe_eval_num(self._legacy_module._normalize_expr(rhs)[0])
                except Exception:
                    residuals.append(float("inf"))
                    continue

            if op == ">=":
                monomials = [(-coef, vars_) for (coef, vars_) in monomials]
                const_term = -const_term
                rhs_value = -rhs_value
                op = "<="

            lhs_value = const_term
            for coef, variables in monomials:
                product = 1.0
                for variable in variables:
                    product *= float(assignment.get(variable, 0))
                lhs_value += coef * product

            if op in ("=", "=="):
                residual = abs(lhs_value - rhs_value)
            elif op == "<=":
                residual = max(0.0, lhs_value - rhs_value)
            else:
                residual = float("inf")
            residuals.append(residual)

        max_residual = max(residuals) if residuals else 0.0
        return max_residual <= 1e-6, max_residual

    @staticmethod
    def _load_legacy_module() -> ModuleType:
        repo_root = Path(__file__).resolve().parents[2]
        module_path = repo_root / "legacy" / "py" / "qubo_validator.py"
        spec = importlib.util.spec_from_file_location("legacy_constraint_validator", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load legacy constraint evaluator from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
