from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass
class TranslatorConfig:
    model_name: str = "Helsinki-NLP/opus-mt-ko-en"
    device: Optional[str] = None


class KoEnTranslator:
    """Translate Korean text to English using a small seq2seq model."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        model_name = model_name or TranslatorConfig.model_name
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
