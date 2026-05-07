from __future__ import annotations

import json
from typing import Any

from src.services.llm_client import LLMClient


class ReverseTranslator:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def reverse_translate(
        self,
        qubo_payload: dict[str, Any],
        reference_prompt: str | None = None,
        reference_hints: dict[str, Any] | None = None,
    ) -> str | None:
        system_prompt = (
            "You are a QUBO-to-text reconstructor. Rebuild an optimization prompt from QUBO JSON as faithfully as possible. "
            "Preserve exact counts, bounds, qualifiers (e.g., at least/at most/exactly), variable relationships, and optimization intent. "
            "Never interpret pseudo-code like sum(... for ... in ...) as literal variables; describe only algebraic terms present. "
            "If the model appears malformed, say that explicitly and avoid inventing missing business context. "
            "Use one concise paragraph, plain text only, no bullet points, no markdown, no preamble. "
            "Do not quote or copy any reference text verbatim. If hints are provided, use them only as semantic checks."
        )
        try:
            payload = {"qubo": qubo_payload}
            if reference_hints:
                payload["reference_hints"] = reference_hints
            if reference_prompt:
                payload["reference_prompt"] = reference_prompt
            elif not reference_hints and reference_prompt:
                payload["reference_hints"] = {"note": "minimal", "length": len(reference_prompt)}
            return self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=json.dumps(payload),
                temperature=0.0,
                max_tokens=500,
            )
        except Exception:
            return None
