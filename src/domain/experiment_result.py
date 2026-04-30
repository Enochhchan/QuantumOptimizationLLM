from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.domain.compiled_qubo import CompiledQUBO
from src.domain.prompt import Prompt
from src.domain.solver_result import SolverResult


@dataclass(slots=True)
class ExperimentResult:
    prompt: Prompt
    status: str
    failure_stage: str | None
    error_type: str | None
    latency_seconds: float
    translation_success: bool
    schema_valid: bool
    compile_success: bool
    solve_success: bool
    reverse_prompt: str | None
    fidelity: float | None
    semantic_fidelity: float | None
    num_variables: int | None
    num_constraints: int | None
    complexity_score: float
    raw_json: dict[str, Any] | None = None
    compiled_qubo: CompiledQUBO | None = None
    solver_result: SolverResult | None = None
