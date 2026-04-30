from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from src.domain.compiled_qubo import CompiledQUBO
from src.errors.compile_error import CompileError


class QUBOCompiler:
    def __init__(self, penalty_scale: float = 1.0) -> None:
        self.penalty_scale = penalty_scale
        self._legacy_module = self._load_legacy_module()

    def compile(self, qubo_json: dict[str, Any]) -> CompiledQUBO:
        payload = self._scale_penalties(qubo_json, self.penalty_scale)
        result = self._legacy_module.compile_qubo(payload, strict=False)
        if not result.get("ok", False):
            raise CompileError(str(result.get("reason", "Unknown compile failure")))

        quadratic_raw = result.get("quadratic", {})
        quadratic: dict[tuple[str, str], float] = {}
        for key, value in quadratic_raw.items():
            if isinstance(key, tuple) and len(key) == 2:
                quadratic[(str(key[0]), str(key[1]))] = float(value)

        added_aux = [fix for fix in result.get("fixes", []) if isinstance(fix, str) and fix.startswith("aux_vars:")]
        return CompiledQUBO(
            linear={str(k): float(v) for k, v in result.get("linear", {}).items()},
            quadratic=quadratic,
            offset=float(result.get("offset", 0.0)),
            variables=[str(v) for v in result.get("variables", [])],
            added_slack=[str(v) for v in result.get("added_slack", [])],
            added_aux=added_aux,
            fixes=[str(v) for v in result.get("fixes", [])],
            objective_sense=str(result.get("objective_sense", "minimize")),
        )

    @staticmethod
    def _scale_penalties(qubo_json: dict[str, Any], multiplier: float) -> dict[str, Any]:
        if multiplier == 1.0:
            return qubo_json
        updated = dict(qubo_json)
        constraints = []
        for constraint in updated.get("constraints", []) or []:
            if not isinstance(constraint, dict):
                constraints.append(constraint)
                continue
            copy_constraint = dict(constraint)
            penalty = copy_constraint.get("penalty")
            if penalty is None:
                copy_constraint["penalty"] = multiplier
            else:
                try:
                    copy_constraint["penalty"] = float(penalty) * multiplier
                except Exception:
                    pass
            constraints.append(copy_constraint)
        updated["constraints"] = constraints
        return updated

    @staticmethod
    def _load_legacy_module() -> ModuleType:
        repo_root = Path(__file__).resolve().parents[2]
        module_path = repo_root / "legacy" / "py" / "qubo_validator.py"
        spec = importlib.util.spec_from_file_location("legacy_qubo_validator", module_path)
        if spec is None or spec.loader is None:
            raise CompileError(f"Unable to load legacy compiler from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
