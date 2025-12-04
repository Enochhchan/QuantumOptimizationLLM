# tests/test_llm_qubo_experiment.py

# Import directly from the root-level Python file
from llm_to_qubo_gpt4 import validate_qubo_json, semantic_similarity


def test_validate_qubo_json_valid():
    """A well-formed QUBO JSON should be considered valid."""
    qubo = {
        "variables": ["x1", "x2"],
        "constraints": [
            {"type": "equality", "expression": "x1 + x2 = 1", "penalty": 5}
        ],
        "objective": "minimize: x1 + 2 * x2"
    }
    assert validate_qubo_json(qubo) is True


def test_validate_qubo_json_missing_field():
    """Missing 'objective' should make the JSON invalid."""
    qubo = {
        "variables": ["x1", "x2"],
        "constraints": [
            {"type": "equality", "expression": "x1 + x2 = 1", "penalty": 5}
        ],
    }
    assert validate_qubo_json(qubo) is False


def test_semantic_similarity_range():
    """Semantic similarity returns a value between 0 and 1."""
    a = "minimize x1 + x2"
    b = "minimize x1 plus x2"
    score = semantic_similarity(a, b)
    assert 0 <= score <= 1
    assert score > 0.3  # loose check that similar strings give some similarity
