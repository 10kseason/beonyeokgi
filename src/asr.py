from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

try:
    from .translator import KoEnTranslator
except Exception:  # pragma: no cover - optional dependency
    KoEnTranslator = None  # type: ignore


@dataclass
class ASRConfig:
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    task: str = "translate"
    language: str = "ko"
    beam_size: int = 1
    input_sr: int = 16000


class ASR:
    def __init__(
        self,
        model: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        task: str = "translate",
        language: str = "ko",
        beam_size: int = 1,
        input_sr: int = 16000,
    ) -> None:
        self.cfg = ASRConfig(
            model=model,
            device=device,
            compute_type=compute_type,
            task=task,
            language=language,
            beam_size=beam_size,
            input_sr=input_sr,
        )
        self.model = WhisperModel(
            self.cfg.model,
            device=self.cfg.device,
            compute_type=self.cfg.compute_type,
        )
        self._translator: Optional["KoEnTranslator"] = None
        self._use_post_translate = False
        if (
            KoEnTranslator is not None
            and self.cfg.task.lower() == "translate"
            and (self.cfg.language is None or self.cfg.language.lower() in {"ko", "korean"})
        ):
            try:
                self._translator = KoEnTranslator()
                self._use_post_translate = True
            except Exception as exc:
                print(
                    "[Translator] Failed to initialize Ko->En translator: "
                    f"{exc}. Falling back to Whisper output."
                )
        elif self.cfg.task.lower() == "translate" and KoEnTranslator is None:
            print(
                "[Translator] torch/transformers not available; falling back to Whisper output."
            )

    @staticmethod
    def _bytes_to_float32_mono(pcm_bytes: bytes) -> np.ndarray:
        audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        if audio_i16.size == 0:
            return np.zeros(0, dtype=np.float32)
        audio_f32 = (audio_i16.astype(np.float32) / 32768.0).copy()
        return audio_f32

    def _resample_to_16k(self, audio: np.ndarray) -> np.ndarray:
        src = int(self.cfg.input_sr)
        if src <= 0 or audio.size == 0:
            return audio.astype(np.float32, copy=False)
        tgt = 16000
        if src == tgt:
            return audio.astype(np.float32, copy=False)
        # Fast path for integer decimation (e.g., 48000->16000)
        if src % tgt == 0:
            factor = src // tgt
            return audio[::factor].astype(np.float32, copy=False)
        # General case: linear interpolation
        new_len = int(round(audio.size * (tgt / float(src))))
        if new_len <= 1:
            return audio.astype(np.float32, copy=False)
        x_old = np.linspace(0, 1, num=audio.size, endpoint=False, dtype=np.float64)
        x_new = np.linspace(0, 1, num=new_len, endpoint=False, dtype=np.float64)
        resampled = np.interp(x_new, x_old, audio.astype(np.float64))
        return resampled.astype(np.float32)

    def transcribe_translate(self, pcm_bytes: bytes) -> str:
        data_f32 = self._bytes_to_float32_mono(pcm_bytes)
        data_f32 = self._resample_to_16k(data_f32)
        if data_f32.size == 0:
            return ""
        task = "transcribe" if self._use_post_translate else self.cfg.task
        segments, _ = self.model.transcribe(
            data_f32,
            language=self.cfg.language,
            task=task,
            beam_size=self.cfg.beam_size,
            vad_filter=True,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        if not text:
            return ""
        if self._use_post_translate and self._translator is not None:
            try:
                return self._translator.translate(text)
            except Exception as exc:
                print(
                    "[Translator] Failed to translate text via Helsinki-NLP model: "
                    f"{exc}. Returning original transcription."
                )
        return text
