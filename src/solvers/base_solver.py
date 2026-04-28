from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.compiled_qubo import CompiledQUBO
from src.domain.solver_result import SolverResult


class BaseSolver(ABC):
    @abstractmethod
    def solve(self, compiled_qubo: CompiledQUBO) -> SolverResult:
        raise NotImplementedError
