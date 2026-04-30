from __future__ import annotations

import time
from pathlib import Path

from src.domain.experiment_result import ExperimentResult
from src.domain.run_config import RunConfig
from src.io.prompt_library import PromptLibrary
from src.io.result_writer import ResultWriter
from src.io.visualization import Visualization
from src.services.constraint_evaluator import ConstraintEvaluator
from src.services.fidelity_calculator import FidelityCalculator
from src.services.qubo_compiler import QUBOCompiler
from src.services.qubo_translator import QUBOTranslator
from src.services.result_metrics import ResultMetrics
from src.services.reverse_translator import ReverseTranslator
from src.services.schema_validator import SchemaValidator
from src.solvers.base_solver import BaseSolver


class ExperimentRunner:
    def __init__(
        self,
        *,
        config: RunConfig,
        prompt_library: PromptLibrary,
        translator: QUBOTranslator,
        schema_validator: SchemaValidator,
        compiler: QUBOCompiler,
        solver: BaseSolver,
        reverse_translator: ReverseTranslator,
        fidelity_calculator: FidelityCalculator,
        constraint_evaluator: ConstraintEvaluator,
        result_writer: ResultWriter,
        result_metrics: ResultMetrics,
        visualization: Visualization,
    ) -> None:
        self.config = config
        self.prompt_library = prompt_library
        self.translator = translator
        self.schema_validator = schema_validator
        self.compiler = compiler
        self.solver = solver
        self.reverse_translator = reverse_translator
        self.fidelity_calculator = fidelity_calculator
        self.constraint_evaluator = constraint_evaluator
        self.result_writer = result_writer
        self.result_metrics = result_metrics
        self.visualization = visualization

    def run(self) -> tuple[list[ExperimentResult], Path]:
        prompts_path = self.prompt_library.resolve_prompts_path(self.config.prompts_path)
        prompts = self.prompt_library.load(
            prompts_path,
            only_case=self.config.only_case,
            limit=self.config.limit,
        )

        results: list[ExperimentResult] = []
        for index, prompt in enumerate(prompts, start=1):
            print(f"[pipeline] processing #{index} ({prompt.problem_type})")
            started = time.time()

            translation = self.translator.translate(prompt)
            if not translation.success or translation.qubo_json is None:
                results.append(
                    ExperimentResult(
                        prompt=prompt,
                        status="error",
                        failure_stage="translation",
                        error_type="Invalid JSON" if translation.success else "LLM Fail",
                        latency_seconds=round(time.time() - started, 4),
                        translation_success=False,
                        schema_valid=False,
                        compile_success=False,
                        solve_success=False,
                        reverse_prompt=None,
                        fidelity=None,
                        semantic_fidelity=None,
                        num_variables=None,
                        num_constraints=None,
                        complexity_score=len(prompt.text) * 0.01,
                        raw_json=None,
                    )
                )
                continue

            qubo_json = translation.qubo_json
            schema_valid, schema_error = self.schema_validator.validate(qubo_json)
            if not schema_valid:
                results.append(
                    ExperimentResult(
                        prompt=prompt,
                        status="error",
                        failure_stage="validation",
                        error_type=schema_error or "Schema Validation",
                        latency_seconds=round(time.time() - started, 4),
                        translation_success=True,
                        schema_valid=False,
                        compile_success=False,
                        solve_success=False,
                        reverse_prompt=None,
                        fidelity=None,
                        semantic_fidelity=None,
                        num_variables=None,
                        num_constraints=None,
                        complexity_score=len(prompt.text) * 0.01,
                        raw_json=qubo_json,
                    )
                )
                continue

            try:
                compiled = self.compiler.compile(qubo_json)
            except Exception as exc:
                results.append(
                    ExperimentResult(
                        prompt=prompt,
                        status="error",
                        failure_stage="compile",
                        error_type=f"Compile Error: {exc}",
                        latency_seconds=round(time.time() - started, 4),
                        translation_success=True,
                        schema_valid=True,
                        compile_success=False,
                        solve_success=False,
                        reverse_prompt=None,
                        fidelity=None,
                        semantic_fidelity=None,
                        num_variables=len(qubo_json.get("variables", [])),
                        num_constraints=len(qubo_json.get("constraints", [])),
                        complexity_score=(len(qubo_json.get("variables", [])) + len(qubo_json.get("constraints", [])) + len(prompt.text) * 0.01),
                        raw_json=qubo_json,
                    )
                )
                continue

            try:
                solver_result = self.solver.solve(compiled)
                feasible, max_residual = self.constraint_evaluator.evaluate(qubo_json, solver_result.assignment)
                solver_result.feasible = feasible
                solver_result.metadata["max_constraint_residual"] = max_residual
            except Exception as exc:
                results.append(
                    ExperimentResult(
                        prompt=prompt,
                        status="error",
                        failure_stage="solve",
                        error_type=f"Solver Error: {exc}",
                        latency_seconds=round(time.time() - started, 4),
                        translation_success=True,
                        schema_valid=True,
                        compile_success=True,
                        solve_success=False,
                        reverse_prompt=None,
                        fidelity=None,
                        semantic_fidelity=None,
                        num_variables=len(compiled.variables),
                        num_constraints=len(qubo_json.get("constraints", [])),
                        complexity_score=(len(compiled.variables) + len(qubo_json.get("constraints", [])) + len(prompt.text) * 0.01),
                        raw_json=qubo_json,
                        compiled_qubo=compiled,
                    )
                )
                continue

            reverse_prompt = self.reverse_translator.reverse_translate(qubo_json)
            fidelity = self.fidelity_calculator.compute_basic(prompt.text, reverse_prompt)
            semantic_fidelity = self.fidelity_calculator.compute_embedding(prompt.text, reverse_prompt)

            results.append(
                ExperimentResult(
                    prompt=prompt,
                    status="success",
                    failure_stage=None,
                    error_type=None,
                    latency_seconds=round(time.time() - started, 4),
                    translation_success=True,
                    schema_valid=True,
                    compile_success=True,
                    solve_success=True,
                    reverse_prompt=reverse_prompt,
                    fidelity=fidelity,
                    semantic_fidelity=semantic_fidelity,
                    num_variables=len(qubo_json.get("variables", [])),
                    num_constraints=len(qubo_json.get("constraints", [])),
                    complexity_score=(len(qubo_json.get("variables", [])) + len(qubo_json.get("constraints", [])) + len(prompt.text) * 0.01),
                    raw_json=qubo_json,
                    compiled_qubo=compiled,
                    solver_result=solver_result,
                )
            )

        output_csv = self.config.output_csv_path()
        frame = self.result_writer.write_results(results, output_csv)
        summary_dir = Path(".") if self.config.legacy_output_mode else Path(self.config.output_dir) / "results"
        summary = self.result_metrics.compute_summary(results)
        self.result_writer.write_summary(summary, summary_dir)
        self.visualization.generate_all(frame)
        return results, output_csv
