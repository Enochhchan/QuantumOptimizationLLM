from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.domain.experiment_result import ExperimentResult


class ResultWriter:
    def write_results(self, results: list[ExperimentResult], csv_path: Path) -> pd.DataFrame:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [self._to_legacy_row(i + 1, result) for i, result in enumerate(results)]
        frame = pd.DataFrame(rows)
        frame.to_csv(csv_path, index=False)
        return frame

    def write_summary(self, summary: dict[str, float], output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "summary.csv"
        pd.DataFrame([summary]).to_csv(path, index=False)
        return path

    @staticmethod
    def _to_legacy_row(index: int, result: ExperimentResult) -> dict[str, object]:
        return {
            "Index": index,
            "Case": result.prompt.problem_type,
            "Prompt": result.prompt.text,
            "Success": bool(result.schema_valid),
            "Failure Type": result.error_type,
            "Latency (s)": result.latency_seconds,
            "Num Variables": result.num_variables,
            "Num Constraints": result.num_constraints,
            "Prompt Length": len(result.prompt.text),
            "Complexity Score": result.complexity_score,
            "Reverse Prompt": result.reverse_prompt,
            "Fidelity": result.fidelity,
            "Semantic Fidelity": result.semantic_fidelity,
            "Compile Success": result.compile_success,
            "Solve Success": result.solve_success,
            "Failure Stage": result.failure_stage,
            "Raw JSON": json.dumps(result.raw_json, ensure_ascii=False) if result.raw_json is not None else None,
        }
