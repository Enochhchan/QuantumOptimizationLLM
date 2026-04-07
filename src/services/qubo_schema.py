from __future__ import annotations


class QUBOSchema:
    def get_schema(self) -> dict:
        return {
            "type": "object",
            "required": ["variables", "constraints", "objective"],
            "properties": {
                "variables": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"type": "string"},
                            {
                                "type": "object",
                                "required": ["name"],
                                "properties": {"name": {"type": "string"}},
                                "additionalProperties": True,
                            },
                        ]
                    },
                },
                "constraints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["expression"],
                        "properties": {
                            "type": {"type": "string"},
                            "expression": {"type": "string"},
                            "penalty": {"type": ["number", "integer"]},
                        },
                        "additionalProperties": True,
                    },
                },
                "objective": {"type": "string"},
            },
            "additionalProperties": True,
        }
