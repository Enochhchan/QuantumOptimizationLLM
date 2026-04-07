from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SolverResult:
    solver_name: str
    assignment: dict[str, int]
    energy: float
    feasible: bool
    runtime_seconds: float
    metadata: dict[str, str | float | int | bool] = field(default_factory=dict)
