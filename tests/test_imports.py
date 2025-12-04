# tests/test_imports.py
from pathlib import Path

def test_llm_to_qubo_gpt4_file_exists():
    """Sanity check: main GPT-4 module file should exist in src/."""
    assert (Path("src") / "llm_to_qubo_gpt4.py").exists()
