from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


LANGUAGE_MODELS: Dict[str, str] = {
    "ko": "Helsinki-NLP/opus-mt-ko-en",
    "ja": "Helsinki-NLP/opus-mt-ja-en",
    "zh": "Helsinki-NLP/opus-mt-zh-en",
}


@dataclass
class TranslatorConfig:
    model_name: str
    device: Optional[str] = None


class Seq2SeqTranslator:
    """Lazy wrapper around Hugging Face seq2seq translation models."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.cfg = TranslatorConfig(model_name=model_name, device=device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if device:
            self._device = torch.device(device)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
        ).to(self._device)
        with torch.no_grad():
            generated = self._model.generate(**inputs, max_length=512, num_beams=4)
        output = self._tokenizer.decode(generated[0], skip_special_tokens=True)
        return output.strip()


_TRANSLATOR_CACHE: Dict[Tuple[str, Optional[str]], Seq2SeqTranslator] = {}


def get_translator(language: str, device: str | None = None) -> Optional[Seq2SeqTranslator]:
    """Return a cached translator for the requested language, if available."""

    normalized = (language or "").split("-")[0].lower()
    model_name = LANGUAGE_MODELS.get(normalized)
    if not model_name:
        return None
    cache_key = (normalized, device)
    translator = _TRANSLATOR_CACHE.get(cache_key)
    if translator is None:
        translator = Seq2SeqTranslator(model_name=model_name, device=device)
        _TRANSLATOR_CACHE[cache_key] = translator
    return translator


class KoEnTranslator(Seq2SeqTranslator):
    """Backwards-compatible wrapper for the Korean → English translator."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        super().__init__(model_name or LANGUAGE_MODELS["ko"], device=device)
