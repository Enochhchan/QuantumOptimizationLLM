# tests/test_imports.py
def test_can_import_main_modules():
    from src import llm_to_qubo_gpt4  # noqa: F401
