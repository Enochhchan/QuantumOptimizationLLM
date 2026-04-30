from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.domain.prompt import Prompt
from src.io.prompt_library import PromptLibrary
from src.services.llm_client import LLMClient
from src.services.qubo_schema import QUBOSchema
from src.services.qubo_translator import QUBOTranslator
from src.services.schema_validator import SchemaValidator


def test_prompt_library_load_and_filter():
    library = PromptLibrary()
    prompts_path = library.resolve_prompts_path("legacy/generated_prompts.csv")
    rows = library.load(prompts_path, only_case="Knapsack", limit=3)
    assert len(rows) == 3
    assert all(row.problem_type == "Knapsack" for row in rows)


def test_dry_run_translation_produces_parseable_json():
    client = LLMClient(model_name="gpt-4", dry_run=True)
    translator = QUBOTranslator(llm_client=client)
    prompt = Prompt(prompt_id="p1", problem_type="Knapsack", text="Pick one item out of two.")
    result = translator.translate(prompt)
    assert result.success
    assert result.qubo_json is not None
    assert "variables" in result.qubo_json
    assert "constraints" in result.qubo_json
    assert "objective" in result.qubo_json


def test_schema_validator_detects_missing_objective():
    validator = SchemaValidator(qubo_schema=QUBOSchema())
    valid, error = validator.validate({"variables": ["x0"], "constraints": []})
    assert not valid
    assert error is not None


def test_cli_dry_run_end_to_end():
    command = [
        sys.executable,
        "src/llm_to_qubo_gpt4.py",
        "--prompts",
        "legacy/generated_prompts.csv",
        "--dry-run",
        "--limit",
        "2",
        "--skip-embeddings",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    output = Path("artifacts/results/llm_qubo_results_detailed_with_fidelity_gpt4_without_loop.csv")
    assert output.exists()
