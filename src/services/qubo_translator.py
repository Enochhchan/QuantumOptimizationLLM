from __future__ import annotations

import ast
import json
import re
import time
from typing import Any

from src.domain.prompt import Prompt
from src.domain.translation_result import TranslationResult
from src.services.llm_client import LLMClient


class QUBOTranslator:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def translate(self, prompt: Prompt) -> TranslationResult:
        system_prompt = self.build_translation_prompt()
        start = time.time()
        raw = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=prompt.text,
            temperature=0.0,
            max_tokens=900,
        )
        latency = round(time.time() - start, 4)
        try:
            qubo_json = self._extract_json_object(raw)
            return TranslationResult(raw_response=raw, qubo_json=qubo_json, success=True, latency_seconds=latency, error=None)
        except Exception as exc:
            try:
                repaired_raw = self._repair_json_with_llm(raw)
                qubo_json = self._extract_json_object(repaired_raw)
                return TranslationResult(raw_response=repaired_raw, qubo_json=qubo_json, success=True, latency_seconds=latency, error=None)
            except Exception as repair_exc:
                return TranslationResult(
                    raw_response=raw,
                    qubo_json=None,
                    success=False,
                    latency_seconds=latency,
                    error=f"Invalid JSON: {exc}; repair failed: {repair_exc}",
                )

    @staticmethod
    def build_translation_prompt() -> str:
        return (
            "You are a QUBO translator. Convert the optimization request into a mathematically coherent QUBO-ready model. "
            "Return one RFC8259 JSON object only, with double quotes, no markdown, no prose. "
            "Schema: "
            "{\"variables\":[...],\"constraints\":[{\"type\":\"equality\"|\"inequality\",\"expression\":\"<math>\",\"penalty\":<number>}],"
            "\"objective\":\"<minimize|maximize>: <expression>\"}. "
            "Rules: (1) Every symbol used in objective/constraints must be in variables. "
            "(2) Use explicit binary decision variables (for assignment problems prefer names like x_driver_delivery). "
            "(3) Encode each natural-language requirement as at least one concrete algebraic constraint with numeric RHS. "
            "(4) Do not invent placeholder aggregate variables (for example total_travel_time) unless tied to equations. "
            "(5) Keep penalties positive and scaled so constraints are enforceable."
        )

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        raw = str(text or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
            if raw.endswith("```"):
                raw = raw[:-3].strip()

        try:
            parsed = QUBOTranslator._loads_with_escape_repair(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        try:
            parsed = QUBOTranslator._literal_eval_relaxed(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = raw[start : end + 1]
            try:
                parsed = QUBOTranslator._loads_with_escape_repair(snippet)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                try:
                    parsed = QUBOTranslator._literal_eval_relaxed(snippet)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass

        raise ValueError("Could not parse valid JSON object from LLM response.")

    @staticmethod
    def _loads_with_escape_repair(candidate: str) -> Any:
        try:
            return json.loads(candidate)
        except Exception:
            # Repair lone backslashes (e.g. \l) that violate JSON escaping rules.
            repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", candidate)
            return json.loads(repaired)

    @staticmethod
    def _literal_eval_relaxed(candidate: str) -> Any:
        # Handle JavaScript-like literals often emitted by LLMs.
        normalized = re.sub(r"\btrue\b", "True", candidate)
        normalized = re.sub(r"\bfalse\b", "False", normalized)
        normalized = re.sub(r"\bnull\b", "None", normalized)
        # Remove trailing commas before ] or }.
        normalized = re.sub(r",\s*([}\]])", r"\1", normalized)
        return ast.literal_eval(normalized)

    def _repair_json_with_llm(self, raw_response: str) -> str:
        repair_system_prompt = (
            "You repair malformed JSON. Convert the provided text into one valid JSON object only. "
            "Use double quotes, no markdown, no comments, no extra text."
        )
        return self.llm_client.generate(
            system_prompt=repair_system_prompt,
            user_prompt=raw_response,
            temperature=0.0,
            max_tokens=900,
        )
