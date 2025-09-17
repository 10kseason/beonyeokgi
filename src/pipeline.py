from __future__ import annotations

import asyncio
import logging
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
from .tts_kokoro import KokoroTTS
from .utils import contains_hangul, parse_sd_device
from .vad import VADSegmenter
from .voice_changer_client import VoiceChangerClient, VoiceChangerConfig


logger = logging.getLogger("vc-translator.pipeline")

LANGUAGE_OPTIONS: Sequence[Tuple[str, str]] = (
    ("ko", "한국어"),
    ("ja", "일본어"),
    ("zh", "중국어"),
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


PRESETS: Dict[str, Preset] = {
    "latency": Preset(
        key="latency",
        label="지연 우선",
        chunk_min_ms=1000,
        chunk_max_ms=1200,
        hop_ms=250,
        beam_size=3,
        temperature=0.0,
    ),
    "accuracy": Preset(
        key="accuracy",
        label="정확도 우선",
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

    def set_labels(
        self,
        input_label: Optional[str] = None,
        output_label: Optional[str] = None,
    ) -> None:
        with self._lock:
            if input_label is not None:
                self._input_label = input_label
            if output_label is not None:
                self._output_label = output_label

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


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


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
        save_devices: Callable[[Optional[object], Optional[object]], None],
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
        input_sr = int(device_cfg.get("input_samplerate") or 48000)
        output_sr = int(device_cfg.get("output_samplerate") or 48000)
        input_device = device_cfg.get("input_device")
        output_device = device_cfg.get("output_device")
        self._current_input_device = input_device
        self._current_output_device = output_device

        preset_key = self.state.get_active_preset()
        preset = PRESETS.get(preset_key, PRESETS["latency"])

        mic = MicReader(
            sr=input_sr,
            block_ms=self.cfg.get("vad", {}).get("frame_ms", 30),
            input_device=input_device,
        )
        self._mic = mic

        vad_cfg = self.cfg.get("vad", {}) or {}
        vad = VADSegmenter(
            sr=input_sr,
            frame_ms=vad_cfg.get("frame_ms", 30),
            aggressiveness=vad_cfg.get("aggressiveness", 2),
            min_speech_sec=vad_cfg.get("min_speech_sec", 0.30),
            max_utt_sec=vad_cfg.get("max_utterance_sec", 9.0),
            silence_end_ms=vad_cfg.get("silence_end_ms", 400),
            chunk_min_ms=preset.chunk_min_ms,
            chunk_max_ms=preset.chunk_max_ms,
            chunk_hop_ms=preset.hop_ms,
        )

        preproc = AudioPreprocessor(target_sr=16000, highpass_hz=90.0, lowpass_hz=7200.0)

        asr_cfg = self.cfg.get("asr", {}) or {}
        asr = ASR(
            model=asr_cfg.get("whisper_model", "large-v3-turbo"),
            device=asr_cfg.get("device", "cpu"),
            compute_type=asr_cfg.get("compute_type", "int8"),
            task=asr_cfg.get("task", "translate"),
            language=asr_cfg.get("language", "ko"),
            beam_size=preset.beam_size,
            input_sr=preproc.target_sr,
            temperature=preset.temperature,
        )
        self.state.set_active_language(asr_cfg.get("language", "ko"))
        self.state.set_active_preset(preset.key)

        vc_client = self._create_voice_changer()
        self._voice_changer = vc_client

        kokoro_cfg = self.cfg.get("kokoro", {}) or {}
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
            output_device=output_device,
            passthrough_input_device=kokoro_cfg.get("passthrough_input_device"),
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
        )
        self._tts = tts

        try:
            mic.start()
            logger.info("Translator pipeline started")
            while not self._stop_event.is_set():
                self._apply_configuration(vad, asr, tts)
                try:
                    frame = mic.read(timeout=0.1)
                except Empty:
                    continue
                segments = self._collect_segments(vad, frame)
                if not segments:
                    continue
                for raw in segments:
                    await self._handle_segment(raw, input_sr, preproc, asr, tts)
                    if self._stop_event.is_set():
                        break
        finally:
            try:
                if tts is not None:
                    tts.flush()
            except Exception:
                logger.debug("Failed to flush TTS on shutdown", exc_info=True)
            mic.stop()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _collect_segments(self, vad: VADSegmenter, frame: bytes) -> List[bytes]:
        segments: List[bytes] = []
        first = vad.push(frame)
        if first is not None:
            segments.append(first)
        while True:
            pending = vad.pop_pending()
            if pending is None:
                break
            segments.append(pending)
        return segments

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
        sentences = split_sentences(text)
        if not sentences:
            sentences = [text]
        per_sentence_ms = segment_ms / len(sentences) if segment_ms > 0 and sentences else None
        tts_elapsed_total = 0.0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if contains_hangul(sentence):
                logger.info("Skipping TTS because text contains Hangul: %s", sentence)
                continue
            speak_start = time.perf_counter()
            try:
                tts.synth_to_play(sentence, src_duration_ms=per_sentence_ms)
            except Exception:
                logger.exception("Kokoro playback failed")
                break
            tts_elapsed_total += (time.perf_counter() - speak_start) * 1000.0
        total_latency = asr_elapsed + tts_elapsed_total
        self._latency_history.append(total_latency)
        avg_latency = sum(self._latency_history) / len(self._latency_history)
        self.state.update_latency(avg_latency)

    def _apply_configuration(self, vad: VADSegmenter, asr: ASR, tts: KokoroTTS) -> None:
        cfg = self.state.get_configuration()
        if cfg["requested_language"] != cfg["active_language"]:
            language = cfg["requested_language"]
            asr.set_language(language)
            self.cfg.setdefault("asr", {})["language"] = language
            self.state.set_active_language(language)
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
        tts.set_output_device(parsed)
        if self._voice_changer is not None:
            self._voice_changer.cfg.fallback_output_device = parsed if parsed not in ("", None) else None
        self.state.set_active_output_device(device)
        self._persist_devices()

    def _persist_devices(self) -> None:
        try:
            self._save_devices(self._current_input_device, self._current_output_device)
        except Exception:
            logger.debug("Failed to persist devices", exc_info=True)

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
