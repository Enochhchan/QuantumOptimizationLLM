from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.domain.run_config import RunConfig
from src.io.prompt_library import PromptLibrary
from src.io.result_writer import ResultWriter
from src.io.single_prompt_csv_writer import SinglePromptCsvWriter
from src.io.visualization import Visualization
from src.pipeline.experiment_runner import ExperimentRunner
from src.services.constraint_evaluator import ConstraintEvaluator
from src.services.fidelity_calculator import FidelityCalculator
from src.services.llm_client import LLMClient
from src.services.qubo_compiler import QUBOCompiler
from src.services.qubo_schema import QUBOSchema
from src.services.qubo_translator import QUBOTranslator
from src.services.result_metrics import ResultMetrics
from src.services.reverse_translator import ReverseTranslator
from src.services.schema_validator import SchemaValidator
from src.solvers.dwave_solver import DWaveSolver
from src.solvers.local_solver import LocalSolver


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NL to QUBO translation experiment.")
    parser.add_argument("--prompts", default=None, help="Path to generated_prompts.csv (or set QUBO_PROMPTS_PATH).")
    parser.add_argument(
        "--prompt-text",
        default=None,
        help="Single natural-language prompt to run. If set, a one-row prompt CSV is generated automatically.",
    )
    parser.add_argument(
        "--prompt-type",
        default="Custom",
        help="Problem type label used when --prompt-text is provided.",
    )
    parser.add_argument(
        "--prompt-csv-path",
        default=None,
        help="Optional output path for the generated one-row prompts CSV when --prompt-text is used.",
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4"), help="OpenAI model name (or set OPENAI_MODEL).")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N prompts.")
    parser.add_argument("--dry-run", action="store_true", help="Skip OpenAI calls and use deterministic dry-run output.")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding-based fidelity computations.")
    parser.add_argument("--only-case", default=None, help="Run only a specific problem type.")
    parser.add_argument("--output-dir", default="artifacts", help="Root directory for refactored output artifacts.")
    parser.add_argument(
        "--output-file",
        default="llm_qubo_results_detailed_with_fidelity_gpt4_without_loop.csv",
        help="Detailed results CSV filename.",
    )
    parser.add_argument("--solver", choices=["local", "dwave"], default="local", help="Solver backend.")
    parser.add_argument("--penalty-scale", type=float, default=1.0, help="Multiplier for constraint penalties.")
    parser.add_argument("--legacy-output-mode", action="store_true", help="Write outputs to repo root, matching old behavior.")
    parser.add_argument("--show-plots", action="store_true", help="Display plots in a GUI window.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    prompts_path = args.prompts
    if args.prompt_text:
        writer = SinglePromptCsvWriter()
        default_generated = (
            Path("single_prompt_generated.csv")
            if args.legacy_output_mode
            else Path(args.output_dir) / "results" / "single_prompt_generated.csv"
        )
        generated_path = Path(args.prompt_csv_path) if args.prompt_csv_path else default_generated
        generated_csv = writer.write(
            prompt_text=args.prompt_text,
            prompt_type=args.prompt_type,
            output_path=generated_path,
        )
        prompts_path = str(generated_csv)
        print(f"[pipeline] generated one-row prompt CSV: {generated_csv}")

    config = RunConfig(
        prompts_path=prompts_path,
        model_name=args.model,
        limit=args.limit,
        dry_run=args.dry_run,
        skip_embeddings=args.skip_embeddings,
        only_case=args.only_case,
        output_dir=args.output_dir,
        output_file_name=args.output_file,
        show_plots=args.show_plots,
        solver_mode=args.solver,
        penalty_scale=args.penalty_scale,
        legacy_output_mode=args.legacy_output_mode,
    )

    llm_client = LLMClient(model_name=config.model_name, dry_run=config.dry_run)
    prompt_library = PromptLibrary()
    schema_validator = SchemaValidator(qubo_schema=QUBOSchema())
    translator = QUBOTranslator(llm_client=llm_client)
    reverse_translator = ReverseTranslator(llm_client=llm_client)
    compiler = QUBOCompiler(penalty_scale=config.penalty_scale)
    solver = LocalSolver() if config.solver_mode == "local" else DWaveSolver()
    fidelity_calculator = FidelityCalculator(enable_embeddings=not config.skip_embeddings)
    constraint_evaluator = ConstraintEvaluator()
    result_writer = ResultWriter()
    result_metrics = ResultMetrics()
    visualization = Visualization(show_plots=config.show_plots, output_dir=config.output_dir, legacy_output_mode=config.legacy_output_mode)

    runner = ExperimentRunner(
        config=config,
        prompt_library=prompt_library,
        translator=translator,
        schema_validator=schema_validator,
        compiler=compiler,
        solver=solver,
        reverse_translator=reverse_translator,
        fidelity_calculator=fidelity_calculator,
        constraint_evaluator=constraint_evaluator,
        result_writer=result_writer,
        result_metrics=result_metrics,
        visualization=visualization,
    )
    results, csv_path = runner.run()
    num_success = sum(1 for row in results if row.status == "success")
    print(f"[pipeline] complete: {num_success}/{len(results)} successful")
    print(f"[pipeline] detailed results: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
