"""clients/llm_judge_client.py — LLM-as-a-Judge client wrapper for NVIDIA chat completions with robust JSON parsing."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

import config

logger = logging.getLogger(__name__)


@dataclass
class JudgeEvaluationResult:
    ok: bool
    score: float | None = None
    rationale: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    raw_output: str | None = None
    model_used: str | None = None
    error: str | None = None


def _clean_json_text(text: str) -> str:
    """Extract raw JSON from code blocks or raw text."""
    cleaned = text.strip()
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if match:
            return match.group(1).strip()
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return cleaned[first_brace : last_brace + 1]
    return cleaned


class LLMJudgeClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        primary_model: str | None = None,
        fallback_model: str | None = None,
    ) -> None:
        self.base_url = (base_url or config.LLM_JUDGE_BASE_URL).rstrip("/")
        self.api_key = api_key or config.LLM_JUDGE_API_KEY
        self.primary_model = primary_model or config.LLM_JUDGE_MODEL
        self.fallback_model = fallback_model or config.REFORMULATION_MODEL or "nvidia/nemotron-3.5-lightning-30b-a3b"

    async def evaluate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> JudgeEvaluationResult:
        """Call LLM judge and strictly parse JSON output with fallback model retry."""
        if not self.api_key:
            return JudgeEvaluationResult(
                ok=False,
                error="LLM_JUDGE_API_KEY is not set",
            )

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        models_to_try = [self.primary_model]
        if self.fallback_model and self.fallback_model != self.primary_model:
            models_to_try.append(self.fallback_model)

        last_error: str | None = None
        for model in models_to_try:
            try:
                raw_response = await self._call_model(messages, model=model, temperature=temperature, max_tokens=max_tokens)
                if not raw_response:
                    continue

                cleaned_json = _clean_json_text(raw_response)
                try:
                    parsed = json.loads(cleaned_json)
                    if isinstance(parsed, dict):
                        score_val = parsed.get("score") or parsed.get("rating") or parsed.get("confidence")
                        try:
                            score_float = float(score_val) if score_val is not None else None
                        except (ValueError, TypeError):
                            score_float = None

                        rationale_val = (
                            parsed.get("rationale")
                            or parsed.get("reason")
                            or parsed.get("explanation")
                            or parsed.get("comments")
                        )

                        return JudgeEvaluationResult(
                            ok=True,
                            score=score_float,
                            rationale=str(rationale_val) if rationale_val is not None else None,
                            data=parsed,
                            raw_output=raw_response,
                            model_used=model,
                        )
                except json.JSONDecodeError as j_err:
                    logger.warning("Judge output from model %s was not valid JSON: %s\nOutput: %s", model, j_err, raw_response[:300])
                    last_error = f"Malformed JSON from {model}: {j_err}"

            except Exception as exc:
                logger.warning("LLM Judge call to model %s failed: %s", model, exc)
                last_error = str(exc)

        return JudgeEvaluationResult(
            ok=False,
            error=last_error or "All judge models failed",
        )

    async def _call_model(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Synchronous chat completion executed in a worker thread with fast failover."""
        def _sync() -> str:
            # Fast 15s timeout and no internal SDK retry loop to allow fast fallback
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=15.0,
                max_retries=0,
            )
            extra_body = None
            if "nemotron" in model:
                extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if extra_body:
                kwargs["extra_body"] = extra_body

            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""

        return await asyncio.to_thread(_sync)


default_judge_client = LLMJudgeClient()
