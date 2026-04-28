from __future__ import annotations

import json
import time

from src.domain.prompt import Prompt
from src.domain.translation_result import TranslationResult
from src.services.llm_client import LLMClient


class QUBOTranslator:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def translate(self, prompt: Prompt) -> TranslationResult:
        system_prompt = self.build_translation_prompt()
        start = time.time()
        raw = self.llm_client.generate(system_prompt=system_prompt, user_prompt=prompt.text)
        latency = round(time.time() - start, 4)
        try:
            qubo_json = json.loads(raw)
            return TranslationResult(raw_response=raw, qubo_json=qubo_json, success=True, latency_seconds=latency, error=None)
        except Exception as exc:
            return TranslationResult(raw_response=raw, qubo_json=None, success=False, latency_seconds=latency, error=f"Invalid JSON: {exc}")

    @staticmethod
    def build_translation_prompt() -> str:
        return (
            "You are a QUBO translator. Given an optimization problem in natural language, "
            "return a JSON with this structure only: "
            "{'variables':[...], 'constraints':[{'type':'equality'|'inequality','expression':'<math>','penalty':<number>}], "
            "'objective':'<minimize|maximize>: <expression>'}. "
            "Output valid JSON only."
        )
