from __future__ import annotations

from typing import Any

from src.services.qubo_schema import QUBOSchema

try:
    import jsonschema
except Exception:  # pragma: no cover
    jsonschema = None  # type: ignore


class SchemaValidator:
    def __init__(self, qubo_schema: QUBOSchema) -> None:
        self._schema = qubo_schema.get_schema()

    def validate(self, qubo_json: dict[str, Any]) -> tuple[bool, str | None]:
        if not isinstance(qubo_json, dict):
            return False, "Invalid JSON object"

        if jsonschema is not None:
            try:
                jsonschema.validate(instance=qubo_json, schema=self._schema)
            except Exception as exc:
                return False, f"Schema validation error: {exc}"
        else:
            required = {"variables", "constraints", "objective"}
            missing = required.difference(qubo_json.keys())
            if missing:
                return False, f"Missing fields: {', '.join(sorted(missing))}"

        objective = qubo_json.get("objective")
        if not isinstance(objective, str):
            return False, "Objective must be a string"
        if ":" not in objective:
            return False, "Objective must include sense prefix: minimize: or maximize:"
        sense = objective.split(":", 1)[0].strip().lower()
        if sense not in {"minimize", "maximize"}:
            return False, "Objective sense must be minimize or maximize"

        variables = self._extract_vars(qubo_json.get("variables", []))
        if not variables:
            return False, "No variables declared"
        declared = set(variables)

        constraints = qubo_json.get("constraints", [])
        if not isinstance(constraints, list):
            return False, "Constraints must be a list"
        for i, constraint in enumerate(constraints):
            if not isinstance(constraint, dict):
                return False, f"Constraint {i} is not an object"
            expr = constraint.get("expression")
            if not isinstance(expr, str):
                return False, f"Constraint {i} has no expression"
            referenced = set(self._extract_var_references(expr))
            undeclared = referenced.difference(declared)
            if undeclared:
                return False, f"Constraint {i} references undeclared variables: {', '.join(sorted(undeclared))}"

        return True, None

    @staticmethod
    def _extract_vars(variables_field: list[Any]) -> list[str]:
        names: list[str] = []
        for item in variables_field:
            if isinstance(item, str):
                names.append(item.strip())
            elif isinstance(item, dict) and isinstance(item.get("name"), str):
                names.append(item["name"].strip())
        return [name for name in names if name]

    @staticmethod
    def _extract_var_references(expression: str) -> list[str]:
        import re

        tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expression)
        keywords = {"minimize", "maximize"}
        return [token for token in tokens if not token.isdigit() and token.lower() not in keywords]
