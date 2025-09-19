from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Sequence, Tuple

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
        self._history: Deque[Tuple[str, str]] = deque(maxlen=2)

    def translate(self, text: str, source_language: str) -> str:
        normalized = (source_language or "").split("-")[0].lower()
        stripped = text.strip()
        if not stripped:
            return ""
        language_name = LANGUAGE_NAMES.get(normalized, normalized or "unknown language")
        prompt = self.cfg.system_prompt.format(language_name=language_name)
        history_snapshot = list(self._history)
        backend = (self.cfg.backend or "ollama").strip().lower()
        if backend == "lmstudio":
            translated = self._translate_lmstudio(prompt, stripped, language_name, history_snapshot, stripped)
        else:
            translated = self._translate_ollama(prompt, stripped, language_name, history_snapshot, stripped)
        translated = translated.strip()
        if translated:
            self._remember_history(stripped, translated)
            return translated
        return stripped

    def reset_history(self) -> None:
        self._history.clear()

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------
    def _translate_ollama(
        self,
        prompt: str,
        current_text: str,
        language_name: str,
        history: Sequence[Tuple[str, str]],
        fallback: str,
    ) -> str:
        url = f"{self.cfg.ollama_url.rstrip('/')}/api/generate"
        model = self.cfg.ollama_model or ""
        if not model:
            logger.warning("LLM translator (Ollama) model not configured; skipping translation")
            return fallback
        context_lines = self._format_history_lines(history)
        parts: List[str] = []
        if context_lines:
            parts.append("Context from the most recent translations:")
            parts.extend(context_lines)
            parts.append("")
        parts.append(
            "Translate the following "
            f"{language_name} text into natural English and respond only with the English translation:"
        )
        parts.append("")
        parts.append(current_text)
        content = "\n".join(parts)
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
            return fallback
        translated = data.get("response") or data.get("text") or ""
        return translated.strip() or fallback

    def _translate_lmstudio(
        self,
        prompt: str,
        current_text: str,
        language_name: str,
        history: Sequence[Tuple[str, str]],
        fallback: str,
    ) -> str:
        base = self.cfg.lmstudio_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        url = f"{base}/chat/completions"
        model = self.cfg.lmstudio_model or ""
        if not model:
            logger.warning("LLM translator (LM Studio) model not configured; skipping translation")
            return fallback
        messages = self._build_messages(prompt, language_name, history, current_text)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": max(0.0, float(self.cfg.temperature)),
        }
        try:
            resp = self._session.post(url, json=payload, timeout=self.cfg.timeout_sec)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("LLM translator (LM Studio) request failed: %s", exc, exc_info=True)
            return fallback
        choices = data.get("choices")
        if not choices:
            return fallback
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not message:
            return fallback
        translated = message.get("content", "")
        return translated.strip() or fallback

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _remember_history(self, source: str, translated: str) -> None:
        clean_source = source.strip()
        clean_translated = translated.strip()
        if not clean_source or not clean_translated:
            return
        if self._history and self._history[-1][0] == clean_source:
            self._history[-1] = (clean_source, clean_translated)
        else:
            self._history.append((clean_source, clean_translated))

    @staticmethod
    def _format_history_lines(history: Sequence[Tuple[str, str]]) -> List[str]:
        lines: List[str] = []
        for idx, (source, translated) in enumerate(history, start=1):
            src = source.strip()
            tgt = translated.strip()
            if not src or not tgt:
                continue
            lines.append(f"{idx}. Source: {src}")
            lines.append(f"   Translation: {tgt}")
        return lines

    def _build_messages(
        self,
        system_prompt: str,
        language_name: str,
        history: Sequence[Tuple[str, str]],
        current_text: str,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for source, translated in history:
            src = source.strip()
            tgt = translated.strip()
            if not src or not tgt:
                continue
            messages.append({"role": "user", "content": src})
            messages.append({"role": "assistant", "content": tgt})
        user_content = (
            "Translate the following "
            f"{language_name} text into natural English and respond only with the English translation:\n\n"
            f"{current_text}"
        )
        messages.append({"role": "user", "content": user_content})
        return messages
