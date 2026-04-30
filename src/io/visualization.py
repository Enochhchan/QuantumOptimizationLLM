from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Visualization:
    def __init__(self, show_plots: bool, output_dir: str, legacy_output_mode: bool) -> None:
        self.show_plots = show_plots
        self.output_dir = Path(output_dir)
        self.legacy_output_mode = legacy_output_mode
        sns.set_theme(style="whitegrid")

    def generate_all(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        plot_dir = Path(".") if self.legacy_output_mode else self.output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        self._bar_success_rate(frame, plot_dir / "success_rate_by_case.png")
        self._failure_distribution(frame, plot_dir / "failure_types.png")
        self._scatter(frame, "Complexity Score", "Success", "success_vs_complexity.png")
        self._scatter(frame, "Complexity Score", "Fidelity", "fidelity_vs_complexity.png")
        self._scatter(frame, "Complexity Score", "Latency (s)", "latency_vs_complexity.png")

    def _bar_success_rate(self, frame: pd.DataFrame, output_path: Path) -> None:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=frame, x="Case", y="Success", estimator=np.mean, errorbar=None)
        plt.ylim(0, 1)
        plt.title("Success Rate by Problem Type")
        plt.tight_layout()
        plt.savefig(output_path)
        self._finalize()

    def _failure_distribution(self, frame: pd.DataFrame, output_path: Path) -> None:
        plt.figure(figsize=(10, 6))
        order = frame["Failure Type"].fillna("None").value_counts().index
        sns.countplot(data=frame, x="Failure Type", order=order)
        plt.title("Failure Types")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path)
        self._finalize()

    def _scatter(self, frame: pd.DataFrame, x_col: str, y_col: str, file_name: str) -> None:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=frame, x=x_col, y=y_col, hue="Case")
        plt.tight_layout()
        plt.savefig((Path(".") if self.legacy_output_mode else self.output_dir / "plots") / file_name)
        self._finalize()

    def _finalize(self) -> None:
        if self.show_plots:
            plt.show()
        else:
            plt.close()
