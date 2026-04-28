from __future__ import annotations

import json
import os
from typing import Any

from dotenv import find_dotenv, load_dotenv

from src.errors.translation_error import TranslationError

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


class LLMClient:
    def __init__(self, model_name: str, dry_run: bool = False) -> None:
        load_dotenv(find_dotenv())
        self.model_name = model_name
        self.dry_run = dry_run
        self._client = None
        if not dry_run:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise TranslationError("OPENAI_API_KEY is missing.")
            if OpenAI is None:
                raise TranslationError("openai package is not installed.")
            self._client = OpenAI(api_key=api_key)

    def generate(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.3, max_tokens: int = 600) -> str:
        if self.dry_run:
            return self._dry_run_response(system_prompt)

        if self._client is None:
            raise TranslationError("LLM client not initialized.")

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if not content:
                raise TranslationError("LLM response content is empty.")
            return content
        except Exception as exc:
            raise TranslationError(str(exc)) from exc

    @staticmethod
    def _dry_run_response(system_prompt: str) -> str:
        if "QUBO-to-text explainer" in system_prompt:
            return "Select one item from x0 and x1 while minimizing weighted cost."
        payload: dict[str, Any] = {
            "variables": ["x0", "x1"],
            "constraints": [{"type": "equality", "expression": "x0 + x1 = 1", "penalty": 10}],
            "objective": "minimize: x0 + 2*x1",
        }
        return json.dumps(payload)
