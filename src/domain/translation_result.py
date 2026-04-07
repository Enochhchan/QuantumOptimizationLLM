from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TranslationResult:
    raw_response: str | None
    qubo_json: dict[str, Any] | None
    success: bool
    latency_seconds: float
    error: str | None = None
