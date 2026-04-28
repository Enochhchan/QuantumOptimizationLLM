from __future__ import annotations

import random
import time

from src.domain.compiled_qubo import CompiledQUBO
from src.domain.solver_result import SolverResult
from src.solvers.base_solver import BaseSolver


class LocalSolver(BaseSolver):
    def __init__(self, num_restarts: int = 100) -> None:
        self.num_restarts = num_restarts

    def solve(self, compiled_qubo: CompiledQUBO) -> SolverResult:
        start = time.time()
        variables = list(compiled_qubo.variables)
        if len(variables) <= 20:
            assignment, energy = self._exact_solve(compiled_qubo, variables)
        else:
            assignment, energy = self._greedy_solve(compiled_qubo, variables)

        runtime = round(time.time() - start, 4)
        return SolverResult(
            solver_name="local",
            assignment=assignment,
            energy=energy,
            feasible=False,
            runtime_seconds=runtime,
            metadata={"num_variables": len(variables)},
        )

    def _exact_solve(self, compiled_qubo: CompiledQUBO, variables: list[str]) -> tuple[dict[str, int], float]:
        best_assignment: dict[str, int] | None = None
        best_energy = float("inf")
        count = 1 << len(variables)
        for state in range(count):
            assignment = {var: 1 if (state >> idx) & 1 else 0 for idx, var in enumerate(variables)}
            energy = self._energy(compiled_qubo, assignment)
            if energy < best_energy:
                best_energy = energy
                best_assignment = assignment
        return best_assignment or {var: 0 for var in variables}, best_energy

    def _greedy_solve(self, compiled_qubo: CompiledQUBO, variables: list[str]) -> tuple[dict[str, int], float]:
        best_assignment: dict[str, int] | None = None
        best_energy = float("inf")
        for _ in range(self.num_restarts):
            assignment = {var: random.randint(0, 1) for var in variables}
            improved = True
            while improved:
                improved = False
                random.shuffle(variables)
                for var in variables:
                    current = self._energy(compiled_qubo, assignment)
                    assignment[var] = 1 - assignment[var]
                    candidate = self._energy(compiled_qubo, assignment)
                    if candidate + 1e-12 < current:
                        improved = True
                    else:
                        assignment[var] = 1 - assignment[var]
            energy = self._energy(compiled_qubo, assignment)
            if energy < best_energy:
                best_energy = energy
                best_assignment = dict(assignment)
        return best_assignment or {var: 0 for var in variables}, best_energy

    @staticmethod
    def _energy(compiled_qubo: CompiledQUBO, assignment: dict[str, int]) -> float:
        energy = compiled_qubo.offset
        for variable, coefficient in compiled_qubo.linear.items():
            energy += coefficient * assignment.get(variable, 0)
        for (left, right), coefficient in compiled_qubo.quadratic.items():
            energy += coefficient * assignment.get(left, 0) * assignment.get(right, 0)
        return float(energy)
