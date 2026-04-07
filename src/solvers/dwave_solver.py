from __future__ import annotations

from src.domain.compiled_qubo import CompiledQUBO
from src.domain.solver_result import SolverResult
from src.errors.solver_error import SolverError
from src.solvers.base_solver import BaseSolver

try:
    import dimod
    from dwave.system import LeapHybridSampler
except Exception:  # pragma: no cover
    dimod = None  # type: ignore
    LeapHybridSampler = None  # type: ignore


class DWaveSolver(BaseSolver):
    def solve(self, compiled_qubo: CompiledQUBO) -> SolverResult:
        if dimod is None or LeapHybridSampler is None:
            raise SolverError("D-Wave dependencies are not available.")
        raise SolverError("D-Wave solver is intentionally optional in phase 1.")
