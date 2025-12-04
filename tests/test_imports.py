# tests/test_imports.py
def test_can_import_main_modules():
    import llm_to_qubo  # noqa: F401
    import llm_to_qubo_gpt4  # noqa: F401
