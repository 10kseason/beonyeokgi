from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

try:
    from .translator import KoEnTranslator
except Exception:  # pragma: no cover - optional dependency
    KoEnTranslator = None  # type: ignore

from .utils import remove_korean_fillers


@dataclass
class ASRConfig:
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    task: str = "translate"
    language: str = "ko"
    beam_size: int = 1
    input_sr: int = 16000
    temperature: float = 0.0
    condition_on_previous_text: bool = True


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
        temperature: float = 0.0,
        condition_on_previous_text: bool = True,
    ) -> None:
        self.cfg = ASRConfig(
            model=model,
            device=device,
            compute_type=compute_type,
            task=task,
            language=language,
            beam_size=beam_size,
            input_sr=input_sr,
            temperature=temperature,
            condition_on_previous_text=condition_on_previous_text,
        )
        self.cfg.condition_on_previous_text = bool(self.cfg.condition_on_previous_text)
        self.model = WhisperModel(
            self.cfg.model,
            device=self.cfg.device,
            compute_type=self.cfg.compute_type,
        )
        self._translator: Optional["KoEnTranslator"] = None
        self._translator_init_failed = False
        self._use_post_translate = False
        self._force_transcribe = False
        self._base_task = task
        self.set_language(language)

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
        task = "transcribe" if (self._use_post_translate or self._force_transcribe) else self._base_task
        segments, _ = self.model.transcribe(
            data_f32,
            language=self.cfg.language,
            task=task,
            beam_size=self.cfg.beam_size,
            vad_filter=True,
            temperature=float(self.cfg.temperature),
            condition_on_previous_text=self.cfg.condition_on_previous_text,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        if not text:
            return ""
        if self._use_post_translate or (self.cfg.language and str(self.cfg.language).lower().startswith("ko")):
            text = remove_korean_fillers(text)
        if self._use_post_translate and not self._force_transcribe and self._translator is not None:
            try:
                return self._translator.translate(text)
            except Exception as exc:
                print(
                    "[Translator] Failed to translate text via Helsinki-NLP model: "
                    f"{exc}. Returning original transcription."
                )
        return text

    def set_language(self, language: Optional[str]) -> None:
        normalized = (language or "").strip().lower()
        self.cfg.language = normalized or None
        self._refresh_post_translate_flag()

    def set_decoding_options(self, beam_size: int, temperature: float) -> None:
        self.cfg.beam_size = max(1, int(beam_size))
        self.cfg.temperature = float(max(0.0, temperature))

    def set_force_transcribe(self, enabled: bool) -> None:
        value = bool(enabled)
        if value == self._force_transcribe:
            return
        self._force_transcribe = value
        if value:
            self._use_post_translate = False
        else:
            self._refresh_post_translate_flag()

    def _refresh_post_translate_flag(self) -> None:
        normalized = (self.cfg.language or "").strip().lower() if self.cfg.language else ""
        wants_translator = (
            self._base_task.lower() == "translate"
            and not self._force_transcribe
            and normalized in {"ko", "ko-kr", "korean"}
        )
        if wants_translator and KoEnTranslator is not None:
            if self._translator is None and not self._translator_init_failed:
                try:
                    self._translator = KoEnTranslator()
                except Exception as exc:
                    print(
                        "[Translator] Failed to initialize Ko->En translator: "
                        f"{exc}. Falling back to Whisper output."
                    )
                    self._translator = None
                    self._translator_init_failed = True
            self._use_post_translate = self._translator is not None
        else:
            if wants_translator and KoEnTranslator is None and not self._translator_init_failed:
                print(
                    "[Translator] torch/transformers not available; falling back to Whisper output."
                )
                self._translator_init_failed = True
            self._use_post_translate = False
