from __future__ import annotations

import json
from typing import Any

from src.services.llm_client import LLMClient


class ReverseTranslator:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def reverse_translate(self, qubo_payload: dict[str, Any]) -> str | None:
        system_prompt = (
            "You are a QUBO-to-text explainer. Translate this JSON QUBO into a natural language optimization problem description."
        )
        try:
            return self.llm_client.generate(system_prompt=system_prompt, user_prompt=json.dumps(qubo_payload))
        except Exception:
            return None
