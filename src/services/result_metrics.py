from __future__ import annotations

from statistics import median

from src.domain.experiment_result import ExperimentResult


class ResultMetrics:
    def compute_summary(self, results: list[ExperimentResult]) -> dict[str, float]:
        if not results:
            return {
                "num_rows": 0.0,
                "success_rate": 0.0,
                "schema_valid_rate": 0.0,
                "compile_success_rate": 0.0,
                "solve_success_rate": 0.0,
                "median_latency_seconds": 0.0,
            }

        num_rows = len(results)
        success_rate = sum(1 for row in results if row.status == "success") / num_rows
        schema_valid_rate = sum(1 for row in results if row.schema_valid) / num_rows
        compile_success_rate = sum(1 for row in results if row.compile_success) / num_rows
        solve_success_rate = sum(1 for row in results if row.solve_success) / num_rows
        latencies = [row.latency_seconds for row in results]
        return {
            "num_rows": float(num_rows),
            "success_rate": success_rate,
            "schema_valid_rate": schema_valid_rate,
            "compile_success_rate": compile_success_rate,
            "solve_success_rate": solve_success_rate,
            "median_latency_seconds": float(median(latencies)),
        }
