from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Prompt:
    prompt_id: str
    problem_type: str
    text: str
    tags: list[str] = field(default_factory=list)
    ground_truth_solution: Any | None = None
    ground_truth_value: float | None = None
    reverse_description: str | None = None
    fidelity_score: float | None = None
