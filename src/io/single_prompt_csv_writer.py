from __future__ import annotations

from pathlib import Path

import pandas as pd


class SinglePromptCsvWriter:
    def write(self, *, prompt_text: str, prompt_type: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(
            [
                {
                    "type": prompt_type.strip() if prompt_type.strip() else "Custom",
                    "description": prompt_text.strip(),
                }
            ]
        )
        frame.to_csv(output_path, index=False)
        return output_path
