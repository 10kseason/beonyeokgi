from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import logging
import requests

logger = logging.getLogger("vc-translator.llm")


LANGUAGE_NAMES: Dict[str, str] = {
    "ko": "Korean",
    "ja": "Japanese",
    "zh": "Chinese",
}


@dataclass
class LLMTranslatorConfig:
    backend: str = "ollama"
    timeout_sec: float = 8.0
    system_prompt: str = (
        "You are a professional translator. Translate any {language_name} text you receive into "
        "natural, conversational English. Only return the translated English text without additional "
        "comments or explanations."
    )
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = ""
    lmstudio_url: str = "http://localhost:1234/v1"
    lmstudio_model: str = ""
    temperature: float = 0.2


class LLMTranslator:
    """Simple REST client for Ollama/LM Studio based translation."""

    def __init__(self, cfg: LLMTranslatorConfig) -> None:
        self.cfg = cfg
        self._session = requests.Session()

    def translate(self, text: str, source_language: str) -> str:
        normalized = (source_language or "").split("-")[0].lower()
        if not text.strip():
            return ""
        language_name = LANGUAGE_NAMES.get(normalized, normalized or "unknown language")
        prompt = self.cfg.system_prompt.format(language_name=language_name)
        content = f"Please translate the following {language_name} text into natural English and return only the English translation.\n\n{text.strip()}"
        backend = (self.cfg.backend or "ollama").strip().lower()
        if backend == "lmstudio":
            return self._translate_lmstudio(prompt, content)
        return self._translate_ollama(prompt, content)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------
    def _translate_ollama(self, prompt: str, content: str) -> str:
        url = f"{self.cfg.ollama_url.rstrip('/')}/api/generate"
        model = self.cfg.ollama_model or ""
        if not model:
            logger.warning("LLM translator (Ollama) model not configured; skipping translation")
            return content
        payload = {
            "model": model,
            "prompt": f"{prompt}\n\nUser: {content}\nTranslator:",
            "stream": False,
            "options": {"temperature": max(0.0, float(self.cfg.temperature))},
        }
        try:
            resp = self._session.post(url, json=payload, timeout=self.cfg.timeout_sec)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("LLM translator (Ollama) request failed: %s", exc, exc_info=True)
            return content
        translated = data.get("response") or data.get("text") or ""
        return translated.strip() or content

    def _translate_lmstudio(self, prompt: str, content: str) -> str:
        base = self.cfg.lmstudio_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        url = f"{base}/chat/completions"
        model = self.cfg.lmstudio_model or ""
        if not model:
            logger.warning("LLM translator (LM Studio) model not configured; skipping translation")
            return content
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            "temperature": max(0.0, float(self.cfg.temperature)),
        }
        try:
            resp = self._session.post(url, json=payload, timeout=self.cfg.timeout_sec)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("LLM translator (LM Studio) request failed: %s", exc, exc_info=True)
            return content
        choices = data.get("choices")
        if not choices:
            return content
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not message:
            return content
        translated = message.get("content", "")
        return translated.strip() or content
