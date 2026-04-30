from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.domain.prompt import Prompt


class PromptLibrary:
    def resolve_prompts_path(self, explicit_path: str | None) -> Path:
        if explicit_path:
            candidate = Path(explicit_path).expanduser()
            if candidate.is_file():
                return candidate
            raise FileNotFoundError(f"--prompts not found: {candidate}")

        env_path = os.getenv("QUBO_PROMPTS_PATH")
        if env_path:
            candidate = Path(env_path).expanduser()
            if candidate.is_file():
                return candidate
            raise FileNotFoundError(f"QUBO_PROMPTS_PATH not found: {candidate}")

        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            Path.cwd() / "generated_prompts.csv",
            repo_root / "generated_prompts.csv",
            repo_root / "legacy" / "generated_prompts.csv",
            repo_root / "data" / "generated_prompts.csv",
            repo_root / "data" / "prompts" / "generated_prompts.csv",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        searched = "\n".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "Could not find generated prompts CSV. Looked in:\n" + searched + "\n"
            "Fix by passing --prompts <path> or setting QUBO_PROMPTS_PATH."
        )

    def load(self, prompts_path: Path, *, only_case: str | None = None, limit: int | None = None) -> list[Prompt]:
        frame = pd.read_csv(prompts_path)
        required_columns = {"type", "description"}
        if not required_columns.issubset(frame.columns):
            raise ValueError(f"Prompt CSV must contain columns: {sorted(required_columns)}")

        if only_case:
            frame = frame[frame["type"] == only_case]
        if limit is not None:
            frame = frame.head(limit)

        rows: list[Prompt] = []
        for index, row in frame.reset_index(drop=True).iterrows():
            rows.append(
                Prompt(
                    prompt_id=f"prompt-{index + 1}",
                    problem_type=str(row["type"]),
                    text=str(row["description"]),
                )
            )
        return rows
