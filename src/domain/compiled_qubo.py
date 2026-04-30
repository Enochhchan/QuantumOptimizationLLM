from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class CompiledQUBO:
    linear: dict[str, float]
    quadratic: dict[tuple[str, str], float]
    offset: float
    variables: list[str]
    added_slack: list[str] = field(default_factory=list)
    added_aux: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    objective_sense: str = "minimize"
