"""LLM backend — supports HuggingFace Inference API and local transformers."""

from __future__ import annotations

import json
import logging
import re

from news_agent.config import Settings

logger = logging.getLogger(__name__)


class LLM:
    """Thin wrapper around a text-generation model."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: object | None = None

    async def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        if self._settings.llm_backend == "local":
            return self._generate_local(prompt, max_tokens)
        return await self._generate_api(prompt, max_tokens)

    # ------------------------------------------------------------------
    # HuggingFace Inference API (free, no GPU)
    # ------------------------------------------------------------------
    async def _generate_api(self, prompt: str, max_tokens: int) -> str:
        from huggingface_hub import InferenceClient

        if self._client is None:
            self._client = InferenceClient(
                model=self._settings.hf_model,
                token=self._settings.hf_token or None,
            )

        client: InferenceClient = self._client  # type: ignore[assignment]
        response = client.text_generation(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.3,
            do_sample=True,
            return_full_text=False,
        )
        return response

    # ------------------------------------------------------------------
    # Local transformers (requires GPU + pip install .[local])
    # ------------------------------------------------------------------
    def _generate_local(self, prompt: str, max_tokens: int) -> str:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local backend requires transformers + torch. "
                "Install with: pip install '.[local]'"
            ) from exc

        if self._client is None:
            logger.info("Loading model %s locally…", self._settings.hf_model)
            tokenizer = AutoTokenizer.from_pretrained(self._settings.hf_model)
            model = AutoModelForCausalLM.from_pretrained(
                self._settings.hf_model,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self._client = (tokenizer, model)

        tokenizer, model = self._client  # type: ignore[misc]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                do_sample=True,
            )
        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)


def extract_json(text: str) -> dict | list | None:
    """Best-effort extraction of a JSON object/array from LLM output."""
    # Try to find JSON block in markdown fences
    match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", text)
    if match:
        text = match.group(1)
    # Try to find raw JSON
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break
    return None
