from __future__ import annotations

import asyncio
import audioop
import logging
import math
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from queue import Empty
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .audio_io import MicReader
from .asr import ASR
from .preprocess import AudioPreprocessor
from .llm_translator import LLMTranslator, LLMTranslatorConfig
from .kokoro_subtitles import KokoroSubtitleStreamer, SubtitleStreamConfig
from .tts_kokoro import KokoroTTS
from .utils import contains_cjk, detect_max_cuda_vram_gb, parse_sd_device
from .vad import VADSegmenter
from .voice_changer_client import VoiceChangerClient, VoiceChangerConfig


logger = logging.getLogger("vc-translator.pipeline")

try:  # Optional dependency for explicit translation fallback
    from .translator import get_translator
except Exception:  # pragma: no cover - optional dependency
    get_translator = None  # type: ignore


@dataclass(frozen=True)
class LanguageOption:
    code: str
    label: str
    target: str


LANGUAGE_OPTIONS: Sequence[LanguageOption] = (
    LanguageOption("ko", "Korean -> EN", "en"),
    LanguageOption("ja", "Japanese -> EN", "en"),
    LanguageOption("zh", "Chinese -> EN", "en"),
)


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    chunk_min_ms: int
    chunk_max_ms: int
    hop_ms: int
    beam_size: int
    temperature: float


@dataclass(frozen=True)
class ForcedSegmentConfig:
    threshold_dbfs: float
    min_segment_ms: int
    sustained_loud_ms: int
    max_buffer_ms: int


class ForcedSegmenter:
    def __init__(self, frame_ms: int, cfg: ForcedSegmentConfig) -> None:
        self.frame_ms = max(1, int(frame_ms))
        self.cfg = ForcedSegmentConfig(
            threshold_dbfs=float(cfg.threshold_dbfs),
            min_segment_ms=max(self.frame_ms, int(cfg.min_segment_ms)),
            sustained_loud_ms=max(self.frame_ms, int(cfg.sustained_loud_ms)),
            max_buffer_ms=max(self.frame_ms, int(cfg.max_buffer_ms)),
        )
        self.reset()

    def reset(self) -> None:
        self._buffer = bytearray()
        self._buffer_ms = 0
        self._speech_ms = 0
        self._loud_run_ms = 0
        self._silence_run_ms = 0
        self._active = False

    def push(self, frame: bytes) -> Optional[bytes]:
        if not frame:
            return None
        rms = audioop.rms(frame, 2)
        level = -120.0 if rms <= 0 else 20.0 * math.log10(rms / 32768.0)
        is_loud = level >= self.cfg.threshold_dbfs
        if not self._active:
            if not is_loud:
                return None
            self._active = True
        self._buffer.extend(frame)
        self._buffer_ms += self.frame_ms
        if is_loud:
            self._speech_ms += self.frame_ms
            self._loud_run_ms += self.frame_ms
            self._silence_run_ms = 0
        else:
            self._silence_run_ms += self.frame_ms
            self._loud_run_ms = 0
        if self._buffer_ms >= self.cfg.max_buffer_ms:
            return self._emit(force=True)
        if self._loud_run_ms >= self.cfg.sustained_loud_ms:
            return self._emit(force=True)
        if not is_loud and self._speech_ms >= self.cfg.min_segment_ms:
            return self._emit(force=False)
        if not is_loud and self._silence_run_ms >= self.frame_ms * 2:
            self.reset()
        return None

    def _emit(self, force: bool) -> Optional[bytes]:
        if self._speech_ms < self.cfg.min_segment_ms and not force:
            self.reset()
            return None
        segment = bytes(self._buffer)
        self.reset()
        return segment


PRESETS: Dict[str, Preset] = {
    "latency": Preset(
        key="latency",
        label="Low latency",
        chunk_min_ms=1000,
        chunk_max_ms=1200,
        hop_ms=250,
        beam_size=3,
        temperature=0.0,
    ),
    "accuracy": Preset(
        key="accuracy",
        label="High accuracy",
        chunk_min_ms=1400,
        chunk_max_ms=1600,
        hop_ms=320,
        beam_size=5,
        temperature=0.0,
    ),
}


class SharedState:
    def __init__(
        self,
        language: str,
        preset_key: str,
        input_device: Optional[object],
        output_device: Optional[object],
        input_label: str,
        output_label: str,
        kokoro_device: Optional[object],
        kokoro_label: str,
        compute_mode: str,
        llm_translate: bool,
    ) -> None:
        self._lock = threading.Lock()
        self._requested_language = language
        self._active_language = language
        self._requested_preset = preset_key
        self._active_preset = preset_key
        self._requested_input_device = input_device
        self._active_input_device = input_device
        self._requested_output_device = output_device
        self._active_output_device = output_device
        self._input_label = input_label or ""
        self._output_label = output_label or ""
        self._requested_kokoro_device = kokoro_device
        self._active_kokoro_device = kokoro_device
        self._kokoro_label = kokoro_label or ""
        self._requested_compute_mode = (compute_mode or "auto").strip().lower()
        self._active_compute_mode = self._requested_compute_mode
        self._requested_llm_translate = bool(llm_translate)
        self._active_llm_translate = bool(llm_translate)
        self._generation = 0
        self.latency_ms = 0.0

    # ------------------------------------------------------------------
    # Requests from UI thread
    # ------------------------------------------------------------------
    def request_language(self, language: str) -> None:
        with self._lock:
            if language != self._requested_language:
                self._requested_language = language
                self._generation += 1

    def request_preset(self, preset_key: str) -> None:
        with self._lock:
            if preset_key != self._requested_preset:
                self._requested_preset = preset_key
                self._generation += 1

    def request_input_device(self, device: Optional[object], label: str) -> None:
        with self._lock:
            if device != self._requested_input_device:
                self._requested_input_device = device
                self._generation += 1
            self._input_label = label or ""

    def request_output_device(self, device: Optional[object], label: str) -> None:
        with self._lock:
            if device != self._requested_output_device:
                self._requested_output_device = device
                self._generation += 1
            self._output_label = label or ""

    def request_kokoro_device(self, device: Optional[object], label: str) -> None:
        with self._lock:
            if device != self._requested_kokoro_device:
                self._requested_kokoro_device = device
                self._generation += 1
            self._kokoro_label = label or ""

    def request_compute_mode(self, mode: str) -> None:
        normalized = (mode or "auto").strip().lower()
        with self._lock:
            if normalized != self._requested_compute_mode:
                self._requested_compute_mode = normalized
                self._generation += 1

    def request_llm_translate(self, enabled: bool) -> None:
        value = bool(enabled)
        with self._lock:
            if value != self._requested_llm_translate:
                self._requested_llm_translate = value
                self._generation += 1

    # ------------------------------------------------------------------
    # Pipeline thread interactions
    # ------------------------------------------------------------------
    def get_configuration(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "requested_language": self._requested_language,
                "active_language": self._active_language,
                "requested_preset": self._requested_preset,
                "active_preset": self._active_preset,
                "requested_input": self._requested_input_device,
                "active_input": self._active_input_device,
                "requested_output": self._requested_output_device,
                "active_output": self._active_output_device,
                "requested_kokoro": self._requested_kokoro_device,
                "active_kokoro": self._active_kokoro_device,
                "requested_compute_mode": self._requested_compute_mode,
                "active_compute_mode": self._active_compute_mode,
                "requested_llm_translate": self._requested_llm_translate,
                "active_llm_translate": self._active_llm_translate,
                "generation": self._generation,
            }

    def set_active_language(self, language: str) -> None:
        with self._lock:
            self._active_language = language

    def set_active_preset(self, preset_key: str) -> None:
        with self._lock:
            self._active_preset = preset_key

    def set_active_input_device(self, device: Optional[object]) -> None:
        with self._lock:
            self._active_input_device = device

    def set_active_output_device(self, device: Optional[object]) -> None:
        with self._lock:
            self._active_output_device = device

    def set_active_kokoro_device(self, device: Optional[object]) -> None:
        with self._lock:
            self._active_kokoro_device = device

    def set_active_compute_mode(self, mode: str) -> None:
        normalized = (mode or "auto").strip().lower()
        with self._lock:
            self._active_compute_mode = normalized

    def set_active_llm_translate(self, enabled: bool) -> None:
        value = bool(enabled)
        with self._lock:
            self._active_llm_translate = value

    def set_labels(
        self,
        input_label: Optional[str] = None,
        output_label: Optional[str] = None,
        kokoro_label: Optional[str] = None,
    ) -> None:
        with self._lock:
            if input_label is not None:
                self._input_label = input_label
            if output_label is not None:
                self._output_label = output_label
            if kokoro_label is not None:
                self._kokoro_label = kokoro_label

    def update_latency(self, latency_ms: float) -> None:
        with self._lock:
            self.latency_ms = float(max(0.0, latency_ms))

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "language": self._requested_language,
                "preset": self._requested_preset,
                "latency_ms": self.latency_ms,
                "input_label": self._input_label,
                "output_label": self._output_label,
                "kokoro_label": self._kokoro_label,
                "compute_mode": self._requested_compute_mode,
                "compute_mode_active": self._active_compute_mode,
                "llm_translate": self._requested_llm_translate,
                "llm_translate_active": self._active_llm_translate,
            }

    def get_active_language(self) -> str:
        with self._lock:
            return self._active_language

    def get_active_preset(self) -> str:
        with self._lock:
            return self._active_preset

    def get_active_input_device(self) -> Optional[object]:
        with self._lock:
            return self._active_input_device

    def get_active_output_device(self) -> Optional[object]:
        with self._lock:
            return self._active_output_device

    def get_active_kokoro_device(self) -> Optional[object]:
        with self._lock:
            return self._active_kokoro_device

    def get_active_compute_mode(self) -> str:
        with self._lock:
            return self._active_compute_mode

    def get_llm_translate_requested(self) -> bool:
        with self._lock:
            return self._requested_llm_translate

    def get_llm_translate_active(self) -> bool:
        with self._lock:
            return self._active_llm_translate

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    stripped = text.strip()
    if not stripped:
        return []
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(stripped) if part.strip()]
    return parts if parts else [stripped]


class TranslatorPipeline:
    def __init__(
        self,
        cfg: Dict[str, Any],
        state: SharedState,
        save_devices: Callable[[Optional[object], Optional[object], Optional[object], Optional[bool]], None],
    ) -> None:
        self.cfg = cfg
        self.state = state
        self._save_devices = save_devices
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._mic: Optional[MicReader] = None
        self._tts: Optional[KokoroTTS] = None
        self._voice_changer: Optional[VoiceChangerClient] = None
        self._latency_history: deque[float] = deque(maxlen=50)
        self._current_input_device = state.get_active_input_device()
        self._current_output_device = state.get_active_output_device()
        self._current_kokoro_device = state.get_active_kokoro_device()
        self._kokoro_output_override: Optional[object] = None
        self._translator_cache: Dict[str, Optional[Any]] = {}
        self._llm_translator: Optional[LLMTranslator] = None
        self._llm_translator_key: Optional[tuple] = None
        translator_cfg = self.cfg.get("translator")
        if isinstance(translator_cfg, dict):
            self._translator_settings = translator_cfg
        else:
            self._translator_settings = {}
            self.cfg["translator"] = self._translator_settings
        self._language_targets = {option.code: option.target for option in LANGUAGE_OPTIONS}
        self._subtitle_streamer: Optional[KokoroSubtitleStreamer] = None
        self._subtitle_queue: deque[Tuple[str, Optional[float], Optional[str], Optional[str]]] = deque()
        self._last_subtitle_text: Optional[str] = None
        self._kokoro_busy_until = 0.0
        self._forced_segmenter: Optional[ForcedSegmenter] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._thread_main, name="translator-pipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(lambda: None)
            except RuntimeError:
                pass
        mic = self._mic
        if mic is not None:
            mic.stop()
            self._forced_segmenter = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    # ------------------------------------------------------------------
    # Internal workers
    # ------------------------------------------------------------------
    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run_async())
        except Exception:
            logger.exception("Translator pipeline crashed")

    async def _run_async(self) -> None:
        self._loop = asyncio.get_running_loop()
        device_cfg = self.cfg.get("device", {}) or {}
        kokoro_cfg = self.cfg.get("kokoro", {}) or {}
        input_sr = int(device_cfg.get("input_samplerate") or 48000)
        output_sr = int(device_cfg.get("output_samplerate") or 48000)
        input_device = device_cfg.get("input_device")
        output_device = device_cfg.get("output_device")
        kokoro_passthrough = kokoro_cfg.get("passthrough_input_device")
        self._kokoro_output_override = self._normalize_output_override(
            kokoro_cfg.get("output_device")
        )
        parsed_output_device = parse_sd_device(output_device)
        self._current_input_device = input_device
        self._current_output_device = parsed_output_device
        self._current_kokoro_device = kokoro_passthrough

        detected_vram_gb = detect_max_cuda_vram_gb()

        preset_key = self.state.get_active_preset()
        preset = PRESETS.get(preset_key, PRESETS["latency"])

        mic = MicReader(
            sr=input_sr,
            block_ms=self.cfg.get("vad", {}).get("frame_ms", 30),
            input_device=input_device,
        )
        self._mic = mic

        vad_cfg = self.cfg.get("vad", {}) or {}
        chunk_enabled = bool(vad_cfg.get("chunk_enable", False))
        if chunk_enabled:
            chunk_min_ms = int(vad_cfg.get("chunk_min_ms", preset.chunk_min_ms))
            chunk_max_ms = int(vad_cfg.get("chunk_max_ms", preset.chunk_max_ms))
            chunk_hop_ms = int(vad_cfg.get("chunk_hop_ms", preset.hop_ms))
        else:
            chunk_min_ms = None
            chunk_max_ms = None
            chunk_hop_ms = None
        listen_max_sec = vad_cfg.get("listen_max_sec")
        max_utt_sec = float(listen_max_sec if listen_max_sec is not None else vad_cfg.get("max_utterance_sec", 9.0))
        silence_end_ms = int(vad_cfg.get("listen_silence_ms", vad_cfg.get("silence_end_ms", 400)))

        force_cfg = vad_cfg.get("force", {}) or {}
        if isinstance(force_cfg, dict) and force_cfg.get("enable"):
            self._forced_segmenter = ForcedSegmenter(
                frame_ms=int(vad_cfg.get("frame_ms", 30)),
                cfg=ForcedSegmentConfig(
                    threshold_dbfs=float(force_cfg.get("rms_speech_threshold_dbfs", -33.0)),
                    min_segment_ms=int(force_cfg.get("min_forced_segment_ms", 2000)),
                    sustained_loud_ms=int(force_cfg.get("sustained_loud_ms", 7000)),
                    max_buffer_ms=int(force_cfg.get("max_buffer_ms", 20000)),
                ),
            )
        else:
            self._forced_segmenter = None

        vad = VADSegmenter(
            sr=input_sr,
            frame_ms=int(vad_cfg.get("frame_ms", 30)),
            aggressiveness=int(vad_cfg.get("aggressiveness", 2)),
            min_speech_sec=float(vad_cfg.get("min_speech_sec", 0.30)),
            max_utt_sec=max_utt_sec,
            silence_end_ms=silence_end_ms,
            chunk_min_ms=chunk_min_ms,
            chunk_max_ms=chunk_max_ms,
            chunk_hop_ms=chunk_hop_ms,
        )

        preproc = AudioPreprocessor(target_sr=16000, highpass_hz=90.0, lowpass_hz=7200.0)

        asr_cfg = self.cfg.get("asr", {}) or {}

        raw_device = asr_cfg.get("device")
        if isinstance(raw_device, str):
            device_clean = raw_device.strip()
            asr_device = device_clean or "cuda"
        elif raw_device:
            asr_device = str(raw_device)
        else:
            asr_device = "cuda"

        raw_compute_type = asr_cfg.get("compute_type")
        if isinstance(raw_compute_type, str):
            compute_clean = raw_compute_type.strip()
            asr_compute_type = compute_clean or "float16"
        elif raw_compute_type:
            asr_compute_type = str(raw_compute_type)
        else:
            asr_compute_type = "float16"

        device_lower = asr_device.lower() if isinstance(asr_device, str) else str(asr_device).lower()
        compute_lower = (
            asr_compute_type.lower()
            if isinstance(asr_compute_type, str)
            else str(asr_compute_type).lower()
        )
        if device_lower.startswith("cuda") and compute_lower == "int8":
            asr_compute_type = "int8_float16"

        asr = ASR(
            model=asr_cfg.get("whisper_model", "large-v3-turbo"),
            device=asr_device,
            compute_type=asr_compute_type,
            task=asr_cfg.get("task", "translate"),
            language=asr_cfg.get("language", "ko"),
            beam_size=preset.beam_size,
            input_sr=preproc.target_sr,
            temperature=preset.temperature,
            condition_on_previous_text=bool(asr_cfg.get("condition_on_previous_text", True)),
        )
        asr.set_force_transcribe(self.state.get_llm_translate_active())
        self.state.set_active_language(asr_cfg.get("language", "ko"))
        self.state.set_active_preset(preset.key)

        vc_client = self._create_voice_changer()
        self._voice_changer = vc_client

        kokoro_device_value = kokoro_cfg.get("device")
        if isinstance(kokoro_device_value, str):
            kokoro_device_clean = kokoro_device_value.strip().lower()
        else:
            kokoro_device_clean = str(kokoro_device_value).strip().lower() if kokoro_device_value else ""
        if kokoro_device_clean in {"", "auto"}:
            if detected_vram_gb >= 12.0:
                kokoro_cfg["device"] = "cuda"
                if "use_half" not in kokoro_cfg:
                    kokoro_cfg["use_half"] = True
                logger.info("Detected %.1f GiB VRAM; enabling Kokoro CUDA backend", detected_vram_gb)

        tts = KokoroTTS(
            model=kokoro_cfg.get("model", "hexgrad/Kokoro-82M"),
            speaker=str(kokoro_cfg.get("speaker", "") or ""),
            backend=str(kokoro_cfg.get("backend", "auto") or "auto"),
            device=str(kokoro_cfg.get("device", "auto") or "auto"),
            use_half=bool(kokoro_cfg.get("use_half", True)),
            onnx_model=kokoro_cfg.get("onnx_model") or None,
            onnx_providers=kokoro_cfg.get("onnx_providers"),
            execution_provider=kokoro_cfg.get("execution_provider") or None,
            pace=float(self.cfg.get("tts", {}).get("pace", 1.0) or 1.0),
            volume_db=float(self.cfg.get("tts", {}).get("volume_db", 0.0) or 0.0),
            out_sr=output_sr,
            output_device=(
                self._kokoro_output_override
                if self._kokoro_output_override is not None
                else parsed_output_device
            ),
            passthrough_input_device=kokoro_passthrough,
            voice_changer=vc_client,
            short_threshold_ms=float(kokoro_cfg.get("short_threshold_ms", 500.0)),
            min_batch_ms=float(kokoro_cfg.get("min_batch_ms", 900.0)),
            max_batch_ms=float(kokoro_cfg.get("max_batch_ms", 1200.0)),
            medium_min_ms=float(kokoro_cfg.get("medium_min_ms", 800.0)),
            medium_max_ms=float(kokoro_cfg.get("medium_max_ms", 1500.0)),
            crossfade_ms=float(kokoro_cfg.get("crossfade_ms", 120.0)),
            tail_flush_ms=float(kokoro_cfg.get("tail_flush_ms", 350.0)),
            short_idle_flush_ms=float(kokoro_cfg.get("short_idle_flush_ms", 650.0)),
            warmup_runs=int(kokoro_cfg.get("warmup_runs", 2)),
            dynamic_pace=bool(kokoro_cfg.get("dynamic_pace", True)),
            dynamic_pace_min=float(kokoro_cfg.get("dynamic_pace_min", 0.35)),
            dynamic_pace_max=float(kokoro_cfg.get("dynamic_pace_max", 1.8)),
            dynamic_pace_tolerance=float(kokoro_cfg.get("dynamic_pace_tolerance", 0.12)),
            dynamic_min_target_ms=float(kokoro_cfg.get("dynamic_min_target_ms", 750.0)),
            dynamic_max_target_ms=float(kokoro_cfg.get("dynamic_max_target_ms", 20000.0)),
            estimate_max_ms=float(kokoro_cfg.get("estimate_max_ms", kokoro_cfg.get("max_estimate_ms", 20000.0))),
        )
        if self._kokoro_output_override is not None:
            logger.info("Routing Kokoro playback to override device %s", self._kokoro_output_override)

        default_output_target = (
            self._kokoro_output_override
            if self._kokoro_output_override is not None
            else parsed_output_device
        )
        if (
            vc_client is not None
            and vc_client.cfg.fallback_output_device in (None, "", "default")
        ):
            normalized_fallback = self._normalize_output_override(default_output_target)
            vc_client.cfg.fallback_output_device = (
                normalized_fallback if normalized_fallback not in (None, "") else None
            )

        self._tts = tts
        self.state.set_active_kokoro_device(kokoro_passthrough)
        self._subtitle_streamer = self._create_subtitle_streamer()
        self._subtitle_queue.clear()
        self._last_subtitle_text = None
        self._kokoro_busy_until = 0.0

        try:
            mic.start()
            logger.info("Translator pipeline started")
            while not self._stop_event.is_set():
                self._apply_configuration(vad, asr, tts)
                self._process_subtitle_queue()
                try:
                    frame = mic.read(timeout=0.1)
                except Empty:
                    self._process_subtitle_queue()
                    continue
                segments = self._collect_segments(vad, frame)
                if not segments:
                    self._process_subtitle_queue()
                    continue
                for raw in segments:
                    await self._handle_segment(raw, input_sr, preproc, asr, tts)
                    if self._stop_event.is_set():
                        break
                self._process_subtitle_queue()
        finally:
            try:
                if tts is not None:
                    tts.flush()
            except Exception:
                logger.debug("Failed to flush TTS on shutdown", exc_info=True)
            streamer = self._subtitle_streamer
            self._subtitle_streamer = None
            self._subtitle_queue.clear()
            self._last_subtitle_text = None
            self._kokoro_busy_until = 0.0
            if streamer is not None:
                try:
                    streamer.close()
                except Exception:
                    logger.debug("Failed to close Kokoro subtitle streamer", exc_info=True)
            mic.stop()
            self._forced_segmenter = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _collect_segments(self, vad: VADSegmenter, frame: bytes) -> List[bytes]:
        forced = self._forced_segmenter
        forced_segment: Optional[bytes] = None
        if forced is not None:
            forced_segment = forced.push(frame)
        segments: List[bytes] = []
        first = vad.push(frame)
        if first is not None:
            segments.append(first)
        while True:
            pending = vad.pop_pending()
            if pending is None:
                break
            segments.append(pending)
        if segments:
            if forced is not None:
                forced.reset()
        elif forced_segment is not None:
            segments.append(forced_segment)
        return segments

    def _enqueue_subtitle(
        self,
        text: str,
        duration_ms: Optional[float],
        source_language: Optional[str],
        target_language: Optional[str],
    ) -> None:
        clean = text.strip()
        if not clean:
            return
        if self._subtitle_streamer is None:
            return
        if clean == (self._last_subtitle_text or ""):
            return
        for existing in self._subtitle_queue:
            if existing[0] == clean:
                return
        self._subtitle_queue.append((clean, duration_ms, source_language, target_language))

    def _process_subtitle_queue(self, *, force_current: bool = False) -> None:
        if not self._subtitle_queue:
            return
        streamer = self._subtitle_streamer
        if streamer is None:
            self._subtitle_queue.clear()
            self._last_subtitle_text = None
            return
        now = time.perf_counter()
        while self._subtitle_queue:
            if not force_current and now < self._kokoro_busy_until:
                break
            text, duration_ms, source_language, target_language = self._subtitle_queue.popleft()
            if not text or text == self._last_subtitle_text:
                force_current = False
                now = time.perf_counter()
                continue
            try:
                streamer.submit(
                    text,
                    duration_ms=duration_ms,
                    source_language=source_language,
                    target_language=target_language,
                )
            except Exception:
                logger.debug("Failed to stream Kokoro subtitle", exc_info=True)
            self._last_subtitle_text = text
            force_current = False
            now = time.perf_counter()

    async def _handle_segment(
        self,
        segment: bytes,
        capture_sr: int,
        preproc: AudioPreprocessor,
        asr: ASR,
        tts: KokoroTTS,
    ) -> None:
        processed_bytes, processed_sr = preproc.process(segment, capture_sr)
        if not processed_bytes:
            return
        samples = len(processed_bytes) // 2
        segment_ms = (samples / float(processed_sr)) * 1000.0 if processed_sr > 0 else 0.0
        asr_start = time.perf_counter()
        text = asr.transcribe_translate(processed_bytes)
        asr_elapsed = (time.perf_counter() - asr_start) * 1000.0
        if not text:
            return
        logger.info("Whisper transcript (%.1f ms): %s", asr_elapsed, text)
        sentences = split_sentences(text)
        if not sentences:
            sentences = [text]
        per_sentence_ms = segment_ms / len(sentences) if segment_ms > 0 and sentences else None
        tts_elapsed_total = 0.0
        active_language = self.state.get_active_language()
        normalized_language = self._normalize_language(active_language)
        target_language = self._language_targets.get(normalized_language)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence = self._ensure_english_text(sentence, active_language)
            if not sentence:
                continue
            if self._should_skip_non_english(sentence, active_language):
                logger.info("Skipping TTS because text is not English after translation: %s", sentence)
                continue
            speak_start = time.perf_counter()
            was_idle = speak_start >= self._kokoro_busy_until
            try:
                tts.synth_to_play(sentence, src_duration_ms=per_sentence_ms)
            except Exception:
                logger.exception("Kokoro playback failed")
                break
            after_speak = time.perf_counter()
            speak_elapsed_ms = (after_speak - speak_start) * 1000.0
            tts_elapsed_total += speak_elapsed_ms
            busy_ms = per_sentence_ms if per_sentence_ms and per_sentence_ms > 0 else speak_elapsed_ms
            if busy_ms <= 0.0:
                busy_ms = speak_elapsed_ms
            busy_ms = max(busy_ms, speak_elapsed_ms, 100.0)
            self._kokoro_busy_until = max(self._kokoro_busy_until, after_speak + busy_ms / 1000.0)
            self._enqueue_subtitle(sentence, per_sentence_ms, active_language, target_language)
            self._process_subtitle_queue(force_current=was_idle)
            logger.info("Kokoro synthesis/playback (%.1f ms): %s", speak_elapsed_ms, sentence)
        total_latency = asr_elapsed + tts_elapsed_total
        self._latency_history.append(total_latency)
        avg_latency = sum(self._latency_history) / len(self._latency_history)
        self.state.update_latency(avg_latency)

    def _normalize_language(self, language: Optional[str]) -> str:
        if not language:
            return ""
        return str(language).split("-")[0].lower()

    def _get_translator(self, language: str):
        if get_translator is None:
            return None
        if language in self._translator_cache:
            return self._translator_cache[language]
        try:
            translator = get_translator(language)
        except Exception:
            logger.warning("Failed to initialize translator for %s", language, exc_info=True)
            translator = None
        self._translator_cache[language] = translator
        return translator

    def _get_llm_translator(self) -> Optional[LLMTranslator]:
        settings = self._translator_settings
        if not settings:
            return None

        cfg = LLMTranslatorConfig()

        backend_value = settings.get("llm_backend", settings.get("backend"))
        if isinstance(backend_value, str) and backend_value.strip():
            cfg.backend = backend_value.strip().lower()

        def _coerce_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        timeout_value = settings.get("timeout_sec", settings.get("timeout"))
        cfg.timeout_sec = _coerce_float(timeout_value, cfg.timeout_sec)
        if cfg.timeout_sec <= 0.0:
            cfg.timeout_sec = LLMTranslatorConfig().timeout_sec
        settings["timeout_sec"] = cfg.timeout_sec

        temperature_value = settings.get("temperature")
        cfg.temperature = _coerce_float(temperature_value, cfg.temperature)
        if cfg.temperature < 0.0:
            cfg.temperature = 0.0
        settings["temperature"] = cfg.temperature

        system_prompt = settings.get("system_prompt")
        if isinstance(system_prompt, str) and system_prompt.strip():
            cfg.system_prompt = system_prompt
        ollama_section = settings.get("ollama")
        if isinstance(ollama_section, dict):
            base = ollama_section.get("base_url")
            if isinstance(base, str) and base.strip():
                cfg.ollama_url = base.strip()
            model = ollama_section.get("model")
            if isinstance(model, str) and model.strip():
                cfg.ollama_model = model.strip()
        base_url = settings.get("ollama_url")
        if isinstance(base_url, str) and base_url.strip():
            cfg.ollama_url = base_url.strip()
        model_name = settings.get("ollama_model")
        if isinstance(model_name, str) and model_name.strip():
            cfg.ollama_model = model_name.strip()

        lmstudio_section = settings.get("lmstudio")
        if isinstance(lmstudio_section, dict):
            base = lmstudio_section.get("base_url")
            if isinstance(base, str) and base.strip():
                cfg.lmstudio_url = base.strip()
            model = lmstudio_section.get("model")
            if isinstance(model, str) and model.strip():
                cfg.lmstudio_model = model.strip()
        base_url = settings.get("lmstudio_url")
        if isinstance(base_url, str) and base_url.strip():
            cfg.lmstudio_url = base_url.strip()
        model_name = settings.get("lmstudio_model")
        if isinstance(model_name, str) and model_name.strip():
            cfg.lmstudio_model = model_name.strip()

        key = (
            cfg.backend,
            cfg.timeout_sec,
            cfg.system_prompt,
            cfg.ollama_url,
            cfg.ollama_model,
            cfg.lmstudio_url,
            cfg.lmstudio_model,
            cfg.temperature,
        )
        if self._llm_translator is None or self._llm_translator_key != key:
            self._llm_translator = LLMTranslator(cfg)
            self._llm_translator_key = key
        return self._llm_translator

    def _reset_llm_history(self) -> None:
        translator = self._llm_translator
        if translator is None:
            return
        try:
            translator.reset_history()
        except Exception:
            logger.debug("Failed to reset LLM translator history", exc_info=True)

    def _ensure_english_text(self, text: str, language: Optional[str]) -> str:
        normalized = self._normalize_language(language)
        if not normalized:
            return text
        target = self._language_targets.get(normalized)
        if target != "en":
            return text
        if not contains_cjk(text):
            return text
        if self.state.get_llm_translate_active():
            llm_translator = self._get_llm_translator()
            if llm_translator is not None:
                try:
                    translated = llm_translator.translate(text, normalized)
                except Exception:
                    logger.warning("LLM translator failed", exc_info=True)
                else:
                    translated = translated.strip()
                    if translated and (translated != text or not contains_cjk(translated)):
                        return translated
        translator = self._get_translator(normalized)
        if translator is None:
            return text
        try:
            translated = translator.translate(text)
        except Exception:
            logger.warning("Translator %s failed", normalized, exc_info=True)
            return text
        return translated or text

    def _should_skip_non_english(self, text: str, language: Optional[str]) -> bool:
        normalized = self._normalize_language(language)
        target = self._language_targets.get(normalized)
        if target != "en":
            return False
        return contains_cjk(text)

    def _apply_configuration(self, vad: VADSegmenter, asr: ASR, tts: KokoroTTS) -> None:
        cfg = self.state.get_configuration()
        if cfg["requested_language"] != cfg["active_language"]:
            language = cfg["requested_language"]
            asr.set_language(language)
            self.cfg.setdefault("asr", {})["language"] = language
            self.state.set_active_language(language)
            self._reset_llm_history()
        if cfg["requested_preset"] != cfg["active_preset"]:
            preset = PRESETS.get(cfg["requested_preset"], PRESETS["latency"])
            vad.configure_chunking(preset.chunk_min_ms, preset.chunk_max_ms, preset.hop_ms)
            asr.set_decoding_options(preset.beam_size, preset.temperature)
            self.cfg.setdefault("asr", {})["beam_size"] = preset.beam_size
            self.cfg.setdefault("asr", {})["temperature"] = preset.temperature
            self.state.set_active_preset(preset.key)
        if cfg["requested_input"] != cfg["active_input"]:
            self._switch_input_device(cfg["requested_input"])
        if cfg["requested_output"] != cfg["active_output"]:
            self._switch_output_device(cfg["requested_output"], tts)
        if cfg["requested_kokoro"] != cfg["active_kokoro"]:
            self._switch_kokoro_passthrough(cfg["requested_kokoro"], tts)
        if cfg["requested_llm_translate"] != cfg["active_llm_translate"]:
            enabled = bool(cfg["requested_llm_translate"])
            self._llm_translator = None
            self._llm_translator_key = None
            self._translator_settings["use_llm"] = enabled
            self.state.set_active_llm_translate(enabled)
            self._persist_devices()
        asr.set_force_transcribe(self.state.get_llm_translate_active())

    def _switch_input_device(self, device: Optional[object]) -> None:
        parsed = parse_sd_device(device)
        if self._mic is None:
            self._current_input_device = parsed
            self.state.set_active_input_device(device)
            return
        self._mic.restart(parsed)
        self._current_input_device = parsed
        self.cfg.setdefault("device", {})["input_device"] = parsed
        self.state.set_active_input_device(device)
        self._persist_devices()

    def _switch_output_device(self, device: Optional[object], tts: KokoroTTS) -> None:
        parsed = parse_sd_device(device)
        self._current_output_device = parsed
        self.cfg.setdefault("device", {})["output_device"] = parsed
        target_device = (
            self._kokoro_output_override
            if self._kokoro_output_override is not None
            else parsed
        )
        tts.set_output_device(target_device)
        if self._voice_changer is not None:
            fallback_device = self._normalize_output_override(target_device)
            self._voice_changer.cfg.fallback_output_device = (
                fallback_device if fallback_device not in (None, "") else None
            )
        self.state.set_active_output_device(device)
        self._persist_devices()

    def _normalize_output_override(self, value: Optional[object]) -> Optional[object]:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped.lower() == "default":
                return None
            return parse_sd_device(stripped)
        return parse_sd_device(value)

    def _switch_kokoro_passthrough(self, device: Optional[object], tts: KokoroTTS) -> None:
        parsed = parse_sd_device(device)
        self._current_kokoro_device = parsed
        self.cfg.setdefault("kokoro", {})["passthrough_input_device"] = parsed
        if hasattr(tts, "set_passthrough_device"):
            try:
                tts.set_passthrough_device(parsed)
            except Exception:
                logger.debug("Failed to switch Kokoro passthrough device", exc_info=True)
        self.state.set_active_kokoro_device(device)
        self._persist_devices()

    def _persist_devices(self) -> None:
        try:
            self._save_devices(
                self._current_input_device,
                self._current_output_device,
                self._current_kokoro_device,
                self.state.get_llm_translate_requested(),
            )
        except Exception:
            logger.debug("Failed to persist devices", exc_info=True)

    def _create_subtitle_streamer(self) -> Optional[KokoroSubtitleStreamer]:
        settings = self.cfg.get("kokoro_subtitles")
        if not isinstance(settings, dict):
            return None
        endpoint = settings.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint.strip():
            return None
        config = SubtitleStreamConfig(endpoint=endpoint.strip())
        method = settings.get("method")
        if isinstance(method, str) and method.strip():
            config.method = method.strip().upper()
        timeout = settings.get("timeout_sec", settings.get("timeout"))
        try:
            config.timeout_sec = float(timeout)
        except (TypeError, ValueError):
            pass
        include_ts = settings.get("include_timestamps")
        if include_ts is not None:
            config.include_timestamps = bool(include_ts)
        retry_limit = settings.get("retry_limit")
        if retry_limit is not None:
            try:
                config.retry_limit = max(0, int(retry_limit))
            except (TypeError, ValueError):
                pass
        retry_backoff = settings.get("retry_backoff_sec")
        if retry_backoff is not None:
            try:
                config.retry_backoff_sec = max(0.0, float(retry_backoff))
            except (TypeError, ValueError):
                pass
        headers_value = settings.get("headers")
        if isinstance(headers_value, dict):
            normalized_headers: Dict[str, str] = {}
            for key, value in headers_value.items():
                try:
                    normalized_headers[str(key)] = str(value)
                except Exception:
                    continue
            config.headers = normalized_headers
        try:
            return KokoroSubtitleStreamer(config)
        except Exception:
            logger.warning("Failed to initialize Kokoro subtitle streamer", exc_info=True)
            return None

    def _create_voice_changer(self) -> Optional[VoiceChangerClient]:
        vc_cfg = self.cfg.get("voice_changer", {}) or {}
        wants_vc = bool(vc_cfg.get("enabled")) or bool(vc_cfg.get("save_original_path")) or bool(
            vc_cfg.get("save_converted_path")
        )
        if not wants_vc:
            return None

        def _maybe_int(value: Any) -> Optional[int]:
            try:
                ivalue = int(value)
            except (TypeError, ValueError):
                return None
            return ivalue if ivalue > 0 else None

        cfg = VoiceChangerConfig(
            enabled=bool(vc_cfg.get("enabled", False)),
            base_url=str(vc_cfg.get("base_url", "http://localhost:18000")),
            endpoint=str(vc_cfg.get("endpoint", "/api/voice-changer/convert_chunk")),
            input_sample_rate=_maybe_int(vc_cfg.get("input_sample_rate")),
            output_sample_rate=_maybe_int(vc_cfg.get("output_sample_rate")),
            timeout_sec=float(vc_cfg.get("timeout_sec", 5.0) or 5.0),
            save_original_path=vc_cfg.get("save_original_path") or None,
            save_converted_path=vc_cfg.get("save_converted_path") or None,
            fallback_endpoint=str(vc_cfg.get("fallback_endpoint", "/api/voice-changer/convert_chunk_bulk")),
            fallback_output_device=vc_cfg.get("fallback_output_device") or None,
            stream_mode=bool(vc_cfg.get("stream_mode", False)),
            stream_chunk_ms=int(vc_cfg.get("stream_chunk_ms", 1000) or 1000),
        )
        try:
            return VoiceChangerClient(cfg)
        except Exception:
            logger.exception("Failed to initialize voice changer client")
            return None










