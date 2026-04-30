from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RunConfig:
    prompts_path: str | None
    model_name: str
    limit: int | None
    dry_run: bool
    skip_embeddings: bool
    only_case: str | None
    output_dir: str
    output_file_name: str
    show_plots: bool
    solver_mode: str
    penalty_scale: float
    legacy_output_mode: bool

    def output_csv_path(self) -> Path:
        return Path(self.output_file_name) if self.legacy_output_mode else Path(self.output_dir) / "results" / self.output_file_name
