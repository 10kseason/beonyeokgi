from __future__ import annotations

import importlib
import inspect
import logging
import math
import platform
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import sounddevice as sd

from .utils import float32_to_int16, parse_sd_device, resample_audio

try:  # Optional dependency used when available
    from pydub import AudioSegment
except Exception:  # pragma: no cover - optional dependency
    AudioSegment = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from .voice_changer_client import VoiceChangerClient


logger = logging.getLogger("vc-translator.tts.kokoro")


@dataclass
class KokoroConfig:
    """Runtime configuration for Kokoro TTS."""

    model: str = "hexgrad/Kokoro-82M"
    speaker: str = ""
    backend: str = "auto"
    device: str = "auto"
    use_half: bool = True
    onnx_model: Optional[str] = None
    onnx_providers: Optional[Sequence[str]] = None
    execution_provider: Optional[str] = None
    pace: float = 1.0
    volume_db: float = 0.0
    out_sr: int = 48000
    output_device: Optional[str] = None
    passthrough_input_device: Optional[object] = None
    short_threshold_ms: float = 500.0
    min_batch_ms: float = 900.0
    max_batch_ms: float = 1200.0
    medium_min_ms: float = 800.0
    medium_max_ms: float = 1500.0
    crossfade_ms: float = 120.0
    tail_flush_ms: float = 350.0
    short_idle_flush_ms: float = 650.0
    warmup_runs: int = 2
    dynamic_pace: bool = True
    dynamic_pace_min: float = 0.35
    dynamic_pace_max: float = 1.8
    dynamic_pace_tolerance: float = 0.12
    dynamic_min_target_ms: float = 750.0
    dynamic_max_target_ms: float = 20000.0
    estimate_max_ms: float = 20000.0


@dataclass
class BackendPlan:
    backend: str
    device: str
    use_half: bool
    providers: Optional[Tuple[str, ...]] = None
    execution_provider: Optional[str] = None
    label: str = ""
    latency: float = float("inf")
    runtime: Optional[Callable[[str], Tuple[np.ndarray, int]]] = None


@dataclass
class PlaybackState:
    stream: Optional[sd.OutputStream] = None
    tail: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    samplerate: int = 0
    timer: Optional[threading.Timer] = None
    last_submit: float = 0.0
    device: Optional[object] = None


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def reset(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def stddev(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.count - 1))


DEFAULT_PROBE_TEXT = (
    "Warmup sentence for Kokoro backend latency benchmarking. "
    "This should roughly produce one second of speech."
)
DEFAULT_KOKORO_VOICES: Dict[str, str] = {
    "a": "af_bella",
    "b": "bf_alice",
    "e": "ef_dora",
    "f": "ff_siwis",
    "h": "hf_alpha",
    "i": "if_sara",
    "p": "pf_dora",
    "j": "jf_nezumi",
    "z": "zf_xiaoxiao",
}

class KokoroTTS:
    """Wrapper around Kokoro 82M text-to-speech backends."""

    def __init__(
        self,
        model: str,
        speaker: str = "",
        backend: str = "auto",
        device: str = "auto",
        use_half: bool = True,
        onnx_model: Optional[str] = None,
        onnx_providers: Optional[Sequence[str]] = None,
        execution_provider: Optional[str] = None,
        pace: float = 1.0,
        volume_db: float = 0.0,
        out_sr: int = 48000,
        output_device: Optional[str] = None,
        passthrough_input_device: Optional[object] = None,
        voice_changer: Optional["VoiceChangerClient"] = None,
        short_threshold_ms: float = 500.0,
        min_batch_ms: float = 900.0,
        max_batch_ms: float = 1200.0,
        medium_min_ms: float = 800.0,
        medium_max_ms: float = 1500.0,
        crossfade_ms: float = 120.0,
        tail_flush_ms: float = 350.0,
        short_idle_flush_ms: float = 650.0,
        warmup_runs: int = 2,
        dynamic_pace: bool = True,
        dynamic_pace_min: float = 0.35,
        dynamic_pace_max: float = 1.8,
        dynamic_pace_tolerance: float = 0.12,
        dynamic_min_target_ms: float = 750.0,
        dynamic_max_target_ms: float = 20000.0,
        estimate_max_ms: float = 20000.0,
    ) -> None:
        backend_value = str(backend).lower()
        device_value = str(device).lower() if isinstance(device, str) else device
        providers_value: Optional[Tuple[str, ...]] = None
        if onnx_providers is not None:
            providers_value = tuple(str(p) for p in onnx_providers)
        self.cfg = KokoroConfig(
            model=model,
            speaker=speaker,
            backend=backend_value,
            device=device_value if device_value is not None else "auto",
            use_half=use_half,
            onnx_model=onnx_model,
            onnx_providers=providers_value,
            execution_provider=execution_provider,
            pace=pace,
            volume_db=volume_db,
            out_sr=out_sr,
            output_device=output_device,
            passthrough_input_device=passthrough_input_device,
            short_threshold_ms=float(short_threshold_ms),
            min_batch_ms=float(min_batch_ms),
            max_batch_ms=float(max_batch_ms),
            medium_min_ms=float(medium_min_ms),
            medium_max_ms=float(medium_max_ms),
            crossfade_ms=float(crossfade_ms),
            tail_flush_ms=float(tail_flush_ms),
            short_idle_flush_ms=float(short_idle_flush_ms),
            warmup_runs=int(warmup_runs),
            dynamic_pace=bool(dynamic_pace),
            dynamic_pace_min=float(dynamic_pace_min),
            dynamic_pace_max=float(dynamic_pace_max),
            dynamic_pace_tolerance=float(dynamic_pace_tolerance),
            dynamic_min_target_ms=float(dynamic_min_target_ms),
            dynamic_max_target_ms=float(dynamic_max_target_ms),
            estimate_max_ms=float(estimate_max_ms),
        )
        self.cfg.short_threshold_ms = max(0.0, self.cfg.short_threshold_ms)
        self.cfg.min_batch_ms = max(0.0, self.cfg.min_batch_ms)
        self.cfg.max_batch_ms = max(self.cfg.min_batch_ms, self.cfg.max_batch_ms)
        self.cfg.medium_min_ms = max(self.cfg.short_threshold_ms, self.cfg.medium_min_ms)
        self.cfg.medium_max_ms = max(self.cfg.medium_min_ms, self.cfg.medium_max_ms)
        self.cfg.crossfade_ms = max(0.0, self.cfg.crossfade_ms)
        self.cfg.tail_flush_ms = max(50.0, self.cfg.tail_flush_ms)
        self.cfg.short_idle_flush_ms = max(0.0, self.cfg.short_idle_flush_ms)
        self.cfg.warmup_runs = max(0, int(self.cfg.warmup_runs))
        self.cfg.dynamic_pace = bool(self.cfg.dynamic_pace)
        self.cfg.dynamic_pace_min = max(0.1, float(self.cfg.dynamic_pace_min))
        self.cfg.dynamic_pace_max = max(self.cfg.dynamic_pace_min, float(self.cfg.dynamic_pace_max))
        self.cfg.dynamic_pace_tolerance = max(0.0, float(self.cfg.dynamic_pace_tolerance))
        self.cfg.dynamic_min_target_ms = max(0.0, float(self.cfg.dynamic_min_target_ms))
        self.cfg.dynamic_max_target_ms = max(0.0, float(self.cfg.dynamic_max_target_ms))
        if self.cfg.dynamic_max_target_ms > 0.0 and self.cfg.dynamic_min_target_ms > self.cfg.dynamic_max_target_ms:
            self.cfg.dynamic_max_target_ms = self.cfg.dynamic_min_target_ms
        self.cfg.estimate_max_ms = max(0.0, float(self.cfg.estimate_max_ms))
        if self.cfg.dynamic_max_target_ms > 0.0 and self.cfg.estimate_max_ms > self.cfg.dynamic_max_target_ms:
            self.cfg.estimate_max_ms = self.cfg.dynamic_max_target_ms
        self.voice_changer = voice_changer
        self._runtime: Optional[Callable[[str], Tuple[np.ndarray, int]]] = None
        self._auto_backend = self.cfg.backend == "auto" or (
            isinstance(self.cfg.device, str) and self.cfg.device == "auto"
        )
        self._plans: List[BackendPlan] = []
        self._active_plan_idx = 0
        self._latency_stats = RunningStats()
        self._latency_anomaly_count = 0
        self._short_segments: List[Tuple[str, float]] = []
        self._short_accum_ms = 0.0
        self._last_short_added = 0.0
        self._play_states: Dict[str, PlaybackState] = {}
        self._play_lock = threading.Lock()
        self._fade_samples = int(self.cfg.out_sr * self.cfg.crossfade_ms / 1000.0)
        self._tail_flush_delay = self.cfg.tail_flush_ms / 1000.0
        self._short_idle_threshold = self.cfg.short_idle_flush_ms / 1000.0
        self._probe_text = DEFAULT_PROBE_TEXT
        self._pending_target_ms: Optional[float] = None
        self._last_dynamic_pace: Optional[float] = None

        self.cfg.output_device = self._normalize_device(self.cfg.output_device)
        self.cfg.passthrough_input_device = self._normalize_device(
            self.cfg.passthrough_input_device
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def synth_to_play(self, text: str, src_duration_ms: Optional[float] = None) -> Optional[bool]:
        clean_text = text.strip()
        if not clean_text:
            return None

        now = time.perf_counter()
        if (
            self._short_segments
            and self._short_idle_threshold > 0.0
            and now - self._last_short_added > self._short_idle_threshold
        ):
            self._flush_short_buffer(force=True)

        duration_ms = src_duration_ms if src_duration_ms and src_duration_ms > 0.0 else self._estimate_duration_ms(clean_text)

        if duration_ms < self.cfg.short_threshold_ms:
            self._short_segments.append((clean_text, duration_ms))
            self._short_accum_ms += duration_ms
            self._last_short_added = now
            if (
                self._short_accum_ms >= self.cfg.min_batch_ms
                or self._short_accum_ms >= self.cfg.max_batch_ms
                or len(self._short_segments) >= 4
            ):
                return self._flush_short_buffer(force=True)
            return None

        result: Optional[bool] = None
        if self._short_segments:
            result = self._flush_short_buffer(force=True)

        playback_result = self._synthesize_and_play(clean_text, target_duration_ms=duration_ms)
        return playback_result if playback_result is not None else result

    def flush(self) -> None:
        try:
            self._flush_short_buffer(force=True)
        except Exception:
            logger.debug("Kokoro short buffer flush failed", exc_info=True)
        with self._play_lock:
            for state in self._play_states.values():
                self._flush_tail_locked(state)
                if state.stream is not None:
                    try:
                        state.stream.stop()
                        state.stream.close()
                    except Exception:
                        logger.debug("Failed to close Kokoro stream", exc_info=True)
                state.stream = None
            self._play_states.clear()

    def _flush_short_buffer(self, force: bool = False) -> Optional[bool]:
        if not self._short_segments:
            return None
        total_ms = sum(duration for _, duration in self._short_segments)
        if not force and total_ms < self.cfg.min_batch_ms:
            return None
        combined_text = " ".join(text for text, _ in self._short_segments)
        self._short_segments.clear()
        self._short_accum_ms = 0.0
        self._last_short_added = 0.0
        return self._synthesize_and_play(combined_text, target_duration_ms=total_ms)

    def _estimate_duration_ms(self, text: str) -> float:
        stripped = text.strip()
        if not stripped:
            return 0.0
        tokens = [token for token in stripped.split() if token]
        words = len(tokens)
        estimate = max(words, 1) * 320.0  # ~187 wpm baseline
        dense_chars = len("".join(tokens)) if tokens else len(stripped.replace(" ", ""))
        if dense_chars:
            estimate = max(estimate, dense_chars * 45.0)
        estimate = max(estimate, 240.0)
        max_cap = getattr(self.cfg, "estimate_max_ms", 0.0)
        if max_cap > 0.0:
            estimate = min(estimate, max_cap)
        return float(estimate)

    def _resolve_dynamic_pace(self, text: str) -> float:
        base_pace = max(0.1, float(self.cfg.pace))
        target_ms = self._pending_target_ms
        if not getattr(self.cfg, "dynamic_pace", False) or target_ms is None or target_ms <= 0.0:
            self._last_dynamic_pace = base_pace
            return base_pace
        if self.cfg.dynamic_min_target_ms > 0.0 and target_ms < self.cfg.dynamic_min_target_ms:
            self._last_dynamic_pace = base_pace
            return base_pace
        if self.cfg.dynamic_max_target_ms > 0.0:
            target_ms = min(target_ms, self.cfg.dynamic_max_target_ms)
        estimated_ms = self._estimate_duration_ms(text)
        estimated_ms = max(estimated_ms, 1.0)
        target_ms = max(target_ms, 1.0)
        ratio = estimated_ms / target_ms
        tolerance = self.cfg.dynamic_pace_tolerance
        if tolerance > 0.0 and abs(1.0 - ratio) <= tolerance:
            self._last_dynamic_pace = base_pace
            return base_pace
        pace = base_pace * ratio
        min_pace = self.cfg.dynamic_pace_min
        max_pace = self.cfg.dynamic_pace_max
        pace = max(min_pace, min(max_pace, pace))
        previous = self._last_dynamic_pace
        if pace != base_pace and (previous is None or abs(previous - pace) >= 0.01):
            logger.debug("Kokoro dynamic pace %.2f -> %.2f (target=%.1f ms, est=%.1f ms)", base_pace, pace, target_ms, estimated_ms)
        self._last_dynamic_pace = pace
        return pace

    def _synthesize_and_play(self, text: str, target_duration_ms: Optional[float] = None) -> Optional[bool]:
        runtime = self._ensure_runtime()
        synth_start = time.perf_counter()
        previous_target = self._pending_target_ms
        target_ms = None
        if target_duration_ms is not None and target_duration_ms > 0.0:
            target_ms = float(target_duration_ms)
            max_target = getattr(self.cfg, "dynamic_max_target_ms", 0.0)
            if max_target > 0.0:
                target_ms = min(target_ms, max_target)
        self._pending_target_ms = target_ms
        try:
            try:
                audio_f32, sample_rate = runtime(text)
            except Exception as exc:
                if self._try_recover_from_exception(exc):
                    try:
                        runtime = self._ensure_runtime()
                        audio_f32, sample_rate = runtime(text)
                    except Exception as retry_exc:
                        logger.error("Kokoro synthesis retry failed: %s", retry_exc)
                        return None
                else:
                    logger.error("Kokoro synthesis failed: %s", exc)
                    return None
        finally:
            self._pending_target_ms = previous_target
        synth_elapsed = time.perf_counter() - synth_start
        if audio_f32.size == 0 or sample_rate <= 0:
            return None
        if self.cfg.volume_db:
            gain = float(10 ** (self.cfg.volume_db / 20.0))
            audio_f32 = np.clip(audio_f32 * gain, -1.0, 1.0)
        target_sr = int(self.cfg.out_sr)
        if target_sr > 0 and sample_rate != target_sr:
            audio_f32 = resample_audio(audio_f32, sample_rate, target_sr)
            sample_rate = target_sr
        base_audio = audio_f32.astype(np.float32, copy=False)
        playback_audio = base_audio
        playback_sr = sample_rate

        vc_success: Optional[bool] = None
        fallback_device = None
        if self.voice_changer is not None:
            fallback_device = getattr(self.voice_changer.cfg, "fallback_output_device", None)
            vc_success = False
            use_stream = bool(getattr(self.voice_changer.cfg, "stream_mode", False))
            stream_failed = False
            samples_i16 = float32_to_int16(base_audio)
            if use_stream:
                stream_result = self.voice_changer.convert_stream(samples_i16, sample_rate)
                if stream_result:
                    stream_gen, stream_sr = stream_result
                    try:
                        for idx, chunk in enumerate(stream_gen):
                            chunk_arr = np.asarray(chunk, dtype=np.float32)
                            chunk_sr = int(stream_sr) if stream_sr else sample_rate
                            if chunk_arr.size == 0:
                                continue
                            if chunk_sr != target_sr:
                                chunk_arr = resample_audio(chunk_arr, chunk_sr, target_sr)
                                chunk_sr = target_sr
                            self._write_audio(chunk_arr, chunk_sr)
                            logger.debug("Kokoro streaming chunk #%d len=%d", idx, chunk_arr.size)
                    except Exception as exc:
                        logger.warning("Kokoro streaming playback error: %s", exc)
                        stream_failed = True
                    else:
                        self._handle_latency(synth_elapsed)
                        return True
                else:
                    stream_failed = True
            if not use_stream or stream_failed:
                result = self.voice_changer.convert(samples_i16, sample_rate)
                if result:
                    converted_f32, vc_sr = result
                    playback_audio = np.asarray(converted_f32, dtype=np.float32)
                    playback_sr = int(vc_sr) if vc_sr else sample_rate
                    if playback_sr != target_sr:
                        playback_audio = resample_audio(playback_audio, playback_sr, target_sr)
                        playback_sr = target_sr
                    vc_success = True
                else:
                    vc_success = False

        if self.voice_changer is not None and vc_success is False:
            if fallback_device:
                self._play_audio_to_targets(
                    base_audio,
                    sample_rate,
                    primary_device=fallback_device,
                    include_default=False,
                )
                self._handle_latency(synth_elapsed)
                return False
            playback_audio = base_audio
            playback_sr = sample_rate

        self._play_audio_to_targets(
            playback_audio.astype(np.float32, copy=False),
            playback_sr,
        )
        self._handle_latency(synth_elapsed)
        return vc_success

    def _handle_latency(self, latency: float) -> None:
        self._latency_stats.update(latency)
        stddev = self._latency_stats.stddev()
        if self._latency_stats.count >= 5 and stddev > 0.0:
            threshold = self._latency_stats.mean + 2.0 * stddev
            if latency > threshold:
                self._latency_anomaly_count += 1
            else:
                self._latency_anomaly_count = 0
        else:
            self._latency_anomaly_count = 0
        if self._latency_anomaly_count >= 3:
            if self._switch_to_next_plan():
                self._latency_anomaly_count = 0

    @staticmethod
    def _normalize_device(device: Optional[object]) -> Optional[object]:
        if device is None:
            return None
        if isinstance(device, str):
            stripped = device.strip()
            if not stripped:
                return None
            return stripped
        return device

    def _device_key(self, device: Optional[object]) -> str:
        parsed = parse_sd_device(device if device is not None else self.cfg.output_device)
        return str(parsed) if parsed is not None else "default"

    def _ensure_state_stream(
        self, state: PlaybackState, sample_rate: int, device: Optional[object]
    ) -> None:
        desired_device = device if device is not None else self.cfg.output_device
        parsed_device = parse_sd_device(desired_device)
        if state.stream is not None:
            if state.samplerate != sample_rate or state.device != desired_device:
                try:
                    state.stream.stop()
                    state.stream.close()
                except Exception:
                    logger.debug("Closing Kokoro stream failed", exc_info=True)
                state.stream = None
                state.tail = np.zeros(0, dtype=np.float32)
        if state.stream is None:
            state.device = desired_device
            try:
                state.stream = sd.OutputStream(
                    samplerate=sample_rate,
                    channels=1,
                    dtype="int16",
                    device=parsed_device,
                )
                state.stream.start()
            except Exception as exc:
                state.stream = None
                raise RuntimeError(f"Failed to open Kokoro output stream: {exc}")
            state.samplerate = sample_rate
            state.tail = np.zeros(0, dtype=np.float32)

    def _play_audio_to_targets(
        self,
        audio: np.ndarray,
        sample_rate: int,
        primary_device: Optional[object] = None,
        include_default: bool = True,
    ) -> None:
        if audio.size == 0 or sample_rate <= 0:
            return
        targets: List[Optional[object]] = []
        normalized_primary = self._normalize_device(primary_device)
        if normalized_primary is not None:
            targets.append(normalized_primary)
        if include_default:
            normalized_default = self._normalize_device(self.cfg.output_device)
            targets.append(normalized_default)
        normalized_passthrough = self._normalize_device(self.cfg.passthrough_input_device)
        if normalized_passthrough is not None:
            targets.append(normalized_passthrough)

        seen: set[str] = set()
        for target in targets:
            if target is None and not include_default and normalized_primary is None:
                continue
            key = self._device_key(target)
            if key in seen:
                continue
            seen.add(key)
            self._write_audio(audio, sample_rate, device=target)

    def _write_audio(self, audio: np.ndarray, sample_rate: int, device: Optional[object] = None) -> None:
        if audio.size == 0 or sample_rate <= 0:
            return
        device_key = self._device_key(device)
        audio_f32 = audio.astype(np.float32, copy=False)
        with self._play_lock:
            state = self._play_states.get(device_key)
            if state is None:
                state = PlaybackState()
                self._play_states[device_key] = state
            self._ensure_state_stream(state, sample_rate, device)
            if state.timer is not None:
                state.timer.cancel()
                state.timer = None

            fade_cap = max(0, self._fade_samples)
            if state.tail.size == 0:
                if fade_cap == 0 or audio_f32.size <= fade_cap:
                    state.stream.write(float32_to_int16(audio_f32))
                    state.tail = np.zeros(0, dtype=np.float32)
                else:
                    body_end = audio_f32.size - fade_cap
                    if body_end > 0:
                        state.stream.write(float32_to_int16(audio_f32[:body_end]))
                    state.tail = audio_f32[-fade_cap:].copy()
            else:
                fade_len = min(fade_cap, state.tail.size, audio_f32.size)
                write_chunks: List[np.ndarray] = []
                if state.tail.size > fade_len:
                    write_chunks.append(state.tail[:-fade_len])
                if fade_len > 0:
                    fade_out = np.linspace(1.0, 0.0, fade_len, endpoint=True, dtype=np.float32)
                    fade_in = np.linspace(0.0, 1.0, fade_len, endpoint=True, dtype=np.float32)
                    cross = state.tail[-fade_len:] * fade_out + audio_f32[:fade_len] * fade_in
                    write_chunks.append(cross)
                idx = fade_len
                tail_len = fade_cap if fade_cap and audio_f32.size > fade_cap else 0
                body_end = audio_f32.size - tail_len
                if body_end > idx:
                    write_chunks.append(audio_f32[idx:body_end])
                if write_chunks:
                    merged = np.concatenate(write_chunks)
                    if merged.size > 0:
                        state.stream.write(float32_to_int16(merged))
                if fade_cap > 0 and audio_f32.size > fade_cap:
                    state.tail = audio_f32[-fade_cap:].copy()
                else:
                    state.tail = np.zeros(0, dtype=np.float32)

            state.last_submit = time.perf_counter()
            if state.tail.size > 0 and self._tail_flush_delay > 0.0:
                self._schedule_tail_flush(device_key, state)

    def set_output_device(self, device: Optional[object]) -> None:
        """Update the playback device and reset active streams."""

        with self._play_lock:
            self.cfg.output_device = self._normalize_device(device)
            for state in self._play_states.values():
                if state.timer is not None:
                    state.timer.cancel()
                    state.timer = None
                if state.stream is not None:
                    try:
                        state.stream.stop()
                        state.stream.close()
                    except Exception:
                        logger.debug("Closing Kokoro stream after device change failed", exc_info=True)
                state.stream = None
                state.tail = np.zeros(0, dtype=np.float32)
            self._play_states.clear()

    def set_passthrough_device(self, device: Optional[object]) -> None:
        """Update the optional mirror device and release previous resources."""

        normalized = self._normalize_device(device)
        with self._play_lock:
            previous = self._normalize_device(self.cfg.passthrough_input_device)
            if previous == normalized:
                self.cfg.passthrough_input_device = normalized
                return
            if previous is not None:
                prev_key = self._device_key(previous)
                state = self._play_states.pop(prev_key, None)
                if state is not None:
                    if state.timer is not None:
                        state.timer.cancel()
                        state.timer = None
                    if state.stream is not None:
                        try:
                            state.stream.stop()
                            state.stream.close()
                        except Exception:
                            logger.debug("Closing Kokoro passthrough stream failed", exc_info=True)
            self.cfg.passthrough_input_device = normalized

    def _schedule_tail_flush(self, device_key: str, state: PlaybackState) -> None:
        if state.timer is not None:
            state.timer.cancel()
        delay = max(0.05, self._tail_flush_delay)

        def _flush_tail() -> None:
            with self._play_lock:
                self._flush_tail_locked(state)
                state.timer = None

        timer = threading.Timer(delay, _flush_tail)
        timer.daemon = True
        state.timer = timer
        timer.start()

    def _flush_tail_locked(self, state: PlaybackState) -> None:
        if state.timer is not None:
            state.timer.cancel()
            state.timer = None
        if state.stream is None or state.tail.size == 0:
            state.tail = np.zeros(0, dtype=np.float32)
            return
        fade_len = state.tail.size
        fade_out = np.linspace(1.0, 0.0, fade_len, endpoint=True, dtype=np.float32)
        tail = state.tail * fade_out
        state.tail = np.zeros(0, dtype=np.float32)
        try:
            state.stream.write(float32_to_int16(tail))
        except Exception:
            logger.debug("Tail flush write failed", exc_info=True)

    # ------------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------------
    def _ensure_runtime(self) -> Callable[[str], Tuple[np.ndarray, int]]:
        if self._runtime is not None:
            return self._runtime
        if self._auto_backend:
            self._initialize_auto_runtime()
        else:
            plan = BackendPlan(
                backend=self.cfg.backend,
                device=str(self.cfg.device),
                use_half=self.cfg.use_half,
                providers=tuple(self.cfg.onnx_providers) if self.cfg.onnx_providers else None,
                execution_provider=self.cfg.execution_provider,
                label=f"{self.cfg.backend}:{self.cfg.device}",
            )
            runtime = self._build_runtime_for_plan(plan, persist=True)
            plan.runtime = runtime
            plan.latency = 0.0
            self._plans = [plan]
            self._active_plan_idx = 0
            self._runtime = runtime
            self._warmup_runtime(runtime, self.cfg.warmup_runs)
        if self._runtime is None:
            raise RuntimeError("Failed to initialize Kokoro runtime")
        return self._runtime

    def _initialize_auto_runtime(self) -> None:
        plans = self._generate_auto_plans()
        if not plans:
            raise RuntimeError("No Kokoro backend available. Install PyTorch or ONNX runtime for Kokoro.")
        measured: List[BackendPlan] = []
        for plan in plans:
            try:
                runtime, latency = self._benchmark_plan(plan)
            except Exception as exc:
                logger.warning("Skipping Kokoro backend %s: %s", plan.label or plan.backend, exc)
                continue
            plan.runtime = runtime
            plan.latency = latency
            measured.append(plan)
        if not measured:
            raise RuntimeError("Failed to initialize any Kokoro backend option.")
        measured.sort(key=lambda p: p.latency)
        self._plans = measured
        self._active_plan_idx = 0
        best = measured[0]
        self._apply_plan(best)
        self._runtime = best.runtime
        self._warmup_runtime(self._runtime, self.cfg.warmup_runs)
        label = best.label or f"{best.backend}:{best.device}"
        logger.info("Selected Kokoro backend %s (%.3f s dummy synthesis)", label, best.latency)

    def _generate_auto_plans(self) -> List[BackendPlan]:
        plans: List[BackendPlan] = []
        torch_mod = None
        try:
            torch_mod = importlib.import_module("torch")
        except ImportError:
            torch_mod = None
        ort_mod = None
        ort_providers: List[str] = []
        try:
            ort_mod = importlib.import_module("onnxruntime")
            ort_providers = [str(p) for p in getattr(ort_mod, "get_available_providers", lambda: [])()]
        except Exception:
            ort_mod = None

        provider_set = {p.lower() for p in ort_providers}
        has_dml = "dmlexecutionprovider" in provider_set
        has_rocm_provider = "rocmexecutionprovider" in provider_set
        has_cuda_provider = "cudaexecutionprovider" in provider_set
        has_cpu_provider = "cpuexecutionprovider" in provider_set

        torch_cuda_available = False
        cuda_version = None
        hip_version = None
        if torch_mod is not None:
            try:
                torch_cuda_available = bool(torch_mod.cuda.is_available())
            except Exception:
                torch_cuda_available = False
            cuda_version = getattr(getattr(torch_mod, "version", None), "cuda", None)
            hip_version = getattr(getattr(torch_mod, "version", None), "hip", None)

        has_cuda = bool(torch_cuda_available and cuda_version)
        has_rocm = bool(torch_cuda_available and hip_version)
        system = platform.system().lower()

        if has_cuda:
            plans.append(
                BackendPlan(
                    backend="pytorch",
                    device="cuda",
                    use_half=self.cfg.use_half,
                    label="PyTorch CUDA",
                )
            )
            if has_cuda_provider:
                providers = ["CUDAExecutionProvider"]
                if has_cpu_provider:
                    providers.append("CPUExecutionProvider")
                plans.append(
                    BackendPlan(
                        backend="onnx",
                        device="cuda",
                        use_half=False,
                        providers=tuple(providers),
                        execution_provider="CUDAExecutionProvider",
                        label="ONNX CUDA",
                    )
                )
        elif has_rocm:
            if has_rocm_provider:
                providers = ["ROCMExecutionProvider"]
                if has_cpu_provider:
                    providers.append("CPUExecutionProvider")
                plans.append(
                    BackendPlan(
                        backend="onnx",
                        device="rocm",
                        use_half=False,
                        providers=tuple(providers),
                        execution_provider="ROCMExecutionProvider",
                        label="ONNX ROCm",
                    )
                )
            elif torch_mod is not None:
                plans.append(
                    BackendPlan(
                        backend="pytorch",
                        device="cuda",
                        use_half=False,
                        label="PyTorch ROCm",
                    )
                )
        elif system == "windows" and has_dml:
            providers = ["DmlExecutionProvider"]
            if has_cpu_provider:
                providers.append("CPUExecutionProvider")
            plans.append(
                BackendPlan(
                    backend="onnx",
                    device="dml",
                    use_half=False,
                    providers=tuple(providers),
                    execution_provider="DmlExecutionProvider",
                    label="ONNX DirectML",
                )
            )
        elif system == "linux" and has_rocm_provider:
            providers = ["ROCMExecutionProvider"]
            if has_cpu_provider:
                providers.append("CPUExecutionProvider")
            plans.append(
                BackendPlan(
                    backend="onnx",
                    device="rocm",
                    use_half=False,
                    providers=tuple(providers),
                    execution_provider="ROCMExecutionProvider",
                    label="ONNX ROCm",
                )
            )

        if torch_mod is not None:
            plans.append(
                BackendPlan(
                    backend="pytorch",
                    device="cpu",
                    use_half=False,
                    label="PyTorch CPU",
                )
            )
        if ort_mod is not None and has_cpu_provider:
            plans.append(
                BackendPlan(
                    backend="onnx",
                    device="cpu",
                    use_half=False,
                    providers=("CPUExecutionProvider",),
                    execution_provider="CPUExecutionProvider",
                    label="ONNX CPU",
                )
            )
        return plans

    def _build_runtime_for_plan(
        self, plan: BackendPlan, *, persist: bool = False
    ) -> Callable[[str], Tuple[np.ndarray, int]]:
        original = (
            self.cfg.backend,
            self.cfg.device,
            self.cfg.use_half,
            tuple(self.cfg.onnx_providers) if self.cfg.onnx_providers else None,
            self.cfg.execution_provider,
        )
        self.cfg.backend = plan.backend
        self.cfg.device = plan.device
        self.cfg.use_half = plan.use_half
        self.cfg.onnx_providers = plan.providers
        self.cfg.execution_provider = plan.execution_provider
        try:
            if plan.backend == "pytorch":
                runtime = self._build_pytorch_runtime()
            elif plan.backend == "onnx":
                runtime = self._build_onnx_runtime()
            else:
                raise ValueError(f"Unsupported Kokoro backend: {plan.backend}")
        finally:
            if not persist:
                self.cfg.backend, self.cfg.device, self.cfg.use_half, self.cfg.onnx_providers, self.cfg.execution_provider = original
        return runtime

    def _benchmark_plan(self, plan: BackendPlan) -> Tuple[Callable[[str], Tuple[np.ndarray, int]], float]:
        runtime = self._build_runtime_for_plan(plan, persist=False)
        warmup_runs = 1
        for _ in range(warmup_runs):
            try:
                runtime(self._probe_text)
            except Exception:
                break
        start = time.perf_counter()
        runtime(self._probe_text)
        elapsed = time.perf_counter() - start
        return runtime, elapsed

    def _apply_plan(self, plan: BackendPlan) -> None:
        self.cfg.backend = plan.backend
        self.cfg.device = plan.device
        self.cfg.use_half = plan.use_half
        self.cfg.onnx_providers = plan.providers
        self.cfg.execution_provider = plan.execution_provider

    def _switch_to_next_plan(self) -> bool:
        if not self._plans or self._active_plan_idx + 1 >= len(self._plans):
            return False
        self._active_plan_idx += 1
        plan = self._plans[self._active_plan_idx]
        try:
            runtime = plan.runtime or self._build_runtime_for_plan(plan, persist=True)
        except Exception as exc:
            logger.error("Failed to switch Kokoro backend to %s: %s", plan.label or plan.backend, exc)
            return False
        plan.runtime = runtime
        self._apply_plan(plan)
        self._runtime = runtime
        self._warmup_runtime(runtime, self.cfg.warmup_runs)
        label = plan.label or f"{plan.backend}:{plan.device}"
        logger.warning("Switching Kokoro backend to %s after latency degradation", label)
        self._latency_stats.reset()
        return True

    def _warmup_runtime(self, runtime: Callable[[str], Tuple[np.ndarray, int]], runs: int) -> None:
        if runs <= 0:
            return
        for _ in range(runs):
            try:
                runtime(self._probe_text)
            except Exception as exc:
                logger.debug("Kokoro warmup skipped: %s", exc)
                break

    def _try_recover_from_exception(self, exc: Exception) -> bool:
        if not self._is_half_precision_error(exc):
            return False
        if not self._disable_half_precision():
            return False
        logger.warning(
            "Disabling Kokoro half precision after runtime dtype mismatch: %s", exc
        )
        return True

    def _is_half_precision_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        if not message or "half" not in message:
            return False
        indicators = ("same dtype", "dtype", "scalar type", "found input tensor")
        return any(indicator in message for indicator in indicators)

    def _disable_half_precision(self) -> bool:
        changed = False
        if bool(self.cfg.use_half):
            self.cfg.use_half = False
            changed = True
        for plan in self._plans:
            if plan.use_half:
                plan.use_half = False
                plan.runtime = None
                changed = True
        if changed:
            self._runtime = None
        return changed

    # ------------------------------------------------------------------
    # PyTorch backend
    # ------------------------------------------------------------------
    def _build_pytorch_runtime(self) -> Callable[[str], Tuple[np.ndarray, int]]:
        try:
            kokoro = importlib.import_module("kokoro")
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "kokoro package is required for PyTorch backend. Install from the official Kokoro repository."
            ) from exc
        if hasattr(kokoro, "KPipeline") and hasattr(kokoro, "KModel"):
            return self._build_pytorch_runtime_kpipeline(kokoro)
        return self._build_pytorch_runtime_legacy(kokoro)

    def _build_pytorch_runtime_kpipeline(self, kokoro: Any) -> Callable[[str], Tuple[np.ndarray, int]]:
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyTorch is required for Kokoro PyTorch backend.") from exc

        device = self._resolve_torch_device(torch)
        repo_id = self.cfg.model or "hexgrad/Kokoro-82M"
        try:
            model = kokoro.KModel(repo_id=repo_id)
            model = model.to(device)
            model.eval()
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize Kokoro model '{repo_id}': {exc}") from exc

        if bool(self.cfg.use_half) and device.type in {"cuda", "hip"}:
            try:
                model.half()
            except Exception as exc:
                logger.debug("Skipping Kokoro FP16 mode: %s", exc)

        lang_code = self._infer_kokoro_lang_code(self.cfg.speaker)
        speaker = self._normalize_kokoro_speaker(self.cfg.speaker, lang_code)
        try:
            pipeline = kokoro.KPipeline(
                lang_code=lang_code,
                repo_id=repo_id,
                model=model,
                device=device.type,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize Kokoro pipeline for language '{lang_code}': {exc}") from exc

        try:
            with torch.inference_mode():
                pipeline.load_voice(speaker)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Kokoro voice '{speaker}'. Set [kokoro].speaker to a supported voice."
            ) from exc

        sample_rate = 24000

        def runtime(text: str) -> Tuple[np.ndarray, int]:
            clean_text = text.strip()
            if not clean_text:
                return np.zeros(0, dtype=np.float32), sample_rate
            speed = self._resolve_dynamic_pace(clean_text)
            audio_chunks: List[np.ndarray] = []
            with torch.inference_mode():
                for result in pipeline(clean_text, voice=speaker, speed=speed):
                    audio_tensor = getattr(result, "audio", None)
                    if audio_tensor is None:
                        continue
                    if hasattr(audio_tensor, "detach"):
                        audio_np = audio_tensor.detach().cpu().numpy()
                    else:
                        audio_np = np.asarray(audio_tensor)
                    if audio_np.ndim > 1:
                        audio_np = np.mean(audio_np, axis=0)
                    audio_chunks.append(audio_np.astype(np.float32, copy=False))
            if not audio_chunks:
                return np.zeros(0, dtype=np.float32), sample_rate
            combined = np.concatenate(audio_chunks).astype(np.float32, copy=False)
            return combined, sample_rate

        return runtime

    def _resolve_torch_device(self, torch_mod: Any) -> Any:
        device_str = str(self.cfg.device or "auto").strip().lower()
        if device_str in {"auto", "", "none"}:
            if hasattr(torch_mod, "cuda") and callable(getattr(torch_mod.cuda, "is_available", None)) and torch_mod.cuda.is_available():
                return torch_mod.device("cuda")
            mps = getattr(getattr(torch_mod, "backends", None), "mps", None)
            if hasattr(mps, "is_available") and mps.is_available():
                return torch_mod.device("mps")
            return torch_mod.device("cpu")
        try:
            return torch_mod.device(device_str)
        except Exception as exc:
            logger.warning("Invalid Kokoro device '%s'; falling back to CPU (%s)", device_str, exc)
            return torch_mod.device("cpu")

    def _infer_kokoro_lang_code(self, speaker: str) -> str:
        base = (speaker or "").strip()
        if base:
            first = base.split(",", 1)[0].strip()
            if first.endswith(".pt"):
                first = first[:-3]
            if first:
                prefix = first.split("_", 1)[0].lower()
                alias = {
                    "en-us": "a",
                    "en-gb": "b",
                    "es": "e",
                    "fr": "f",
                    "fr-fr": "f",
                    "hi": "h",
                    "it": "i",
                    "pt": "p",
                    "pt-br": "p",
                    "ja": "j",
                    "zh": "z",
                }.get(prefix)
                if alias:
                    return alias
                key = prefix[:1]
                if key in DEFAULT_KOKORO_VOICES:
                    return key
        return "a"

    def _normalize_kokoro_speaker(self, speaker: str, lang_code: str) -> str:
        raw = (speaker or "").strip()
        if raw:
            parts = [part.strip() for part in raw.split(",") if part.strip()]
            normalized_parts = [part[:-3] if part.endswith(".pt") else part for part in parts]
            normalized = ",".join(normalized_parts)
        else:
            normalized = ""
        if not normalized:
            normalized = DEFAULT_KOKORO_VOICES.get(lang_code, DEFAULT_KOKORO_VOICES["a"])
            logger.info(
                "No Kokoro speaker configured; defaulting to voice '%s' for language '%s'.",
                normalized,
                lang_code,
            )
        return normalized

    def _build_pytorch_runtime_legacy(self, kokoro: Any) -> Callable[[str], Tuple[np.ndarray, int]]:
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyTorch is required for Kokoro PyTorch backend.") from exc

        device = self._resolve_torch_device(torch)
        use_half = bool(self.cfg.use_half)
        wants_fp16 = use_half and device.type in {"cuda", "hip"}
        torch_dtype = torch.float16 if wants_fp16 else torch.float32

        loader = self._find_callable(
            [kokoro, getattr(kokoro, "inference", None), getattr(kokoro, "tts", None)],
            ["load_model", "load_tts_model", "load_kokoro", "Kokoro"],
        )
        if loader is None:
            raise RuntimeError("Could not locate Kokoro model loader. Update the kokoro package.")
        model = self._invoke_loader(loader, self.cfg.model, device=device, dtype=torch_dtype)

        voice = self._load_voice(kokoro, self.cfg.speaker)
        tokenizer = self._load_tokenizer(kokoro)
        vocoder = self._load_vocoder(kokoro, device, torch_dtype)

        generator = self._find_callable(
            [kokoro, getattr(kokoro, "tts", None), getattr(kokoro, "inference", None), model],
            ["generate_audio", "generate", "tts", "synthesize", "speak", "infer", "__call__"],
        )
        if generator is None:
            raise RuntimeError("No Kokoro synthesis function available. Check the installed kokoro version.")

        def runtime(text: str) -> Tuple[np.ndarray, int]:
            clean_text = text.strip()
            if not clean_text:
                return np.zeros(0, dtype=np.float32), self.cfg.out_sr
            tokens = self._maybe_tokenize(tokenizer, clean_text, voice)
            pace = self._resolve_dynamic_pace(clean_text)
            with torch.inference_mode():
                return self._invoke_generator(
                    generator,
                    clean_text,
                    tokens,
                    model,
                    voice,
                    vocoder,
                    device,
                    torch_dtype,
                    pace,
                )

        return runtime

    # ------------------------------------------------------------------
    # ONNX backend
    # ------------------------------------------------------------------
    def _build_onnx_runtime(self) -> Callable[[str], Tuple[np.ndarray, int]]:
        try:
            kokoro_onnx = importlib.import_module("kokoro_onnx")
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "kokoro-onnx package is required for ONNX backend. Install from the official Kokoro repository."
            ) from exc

        model_ref = self.cfg.onnx_model or self.cfg.model
        providers = list(self.cfg.onnx_providers or [])
        if self.cfg.execution_provider and self.cfg.execution_provider not in providers:
            providers.insert(0, self.cfg.execution_provider)

        loader = self._find_callable(
            [kokoro_onnx],
            ["Kokoro", "OnnxKokoro", "TTSEngine", "Inference", "load_model", "load_kokoro"],
        )
        if loader is None:
            raise RuntimeError("Unable to locate Kokoro ONNX runtime class or loader.")

        engine = self._invoke_onnx_loader(loader, model_ref, providers)
        generator = self._find_callable(
            [engine, kokoro_onnx],
            ["generate", "tts", "synthesize", "speak", "infer", "generate_audio", "__call__"],
        )
        if generator is None:
            raise RuntimeError("ONNX Kokoro runtime does not expose a synthesis function.")

        def runtime(text: str) -> Tuple[np.ndarray, int]:
            clean_text = text.strip()
            if not clean_text:
                return np.zeros(0, dtype=np.float32), self.cfg.out_sr
            pace = self._resolve_dynamic_pace(clean_text)
            voice = self.cfg.speaker or None
            return self._invoke_generator(
                generator,
                clean_text,
                None,
                engine,
                voice,
                None,
                self.cfg.device,
                None,
                pace,
                backend="onnx",
            )

        return runtime

    # ------------------------------------------------------------------
    # Loader helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _find_callable(modules: Iterable[Any], names: Sequence[str]) -> Optional[Callable[..., Any]]:
        for module in modules:
            if module is None:
                continue
            for name in names:
                attr = getattr(module, name, None)
                if callable(attr):
                    return attr
        return None

    @staticmethod
    def _supports_kw(func: Callable[..., Any], name: str) -> bool:
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            return True
        if name in sig.parameters:
            return True
        for param in sig.parameters.values():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                return True
        return False

    def _invoke_loader(self, loader: Callable[..., Any], model: str, **kwargs: Any) -> Any:
        try:
            filtered = {k: v for k, v in kwargs.items() if self._supports_kw(loader, k)}
            return loader(model, **filtered)
        except TypeError:
            return loader(model)

    def _invoke_onnx_loader(self, loader: Callable[..., Any], model: str, providers: Sequence[str]) -> Any:
        kwargs: dict[str, Any] = {}
        if providers and self._supports_kw(loader, "providers"):
            kwargs["providers"] = list(providers)
        if self._supports_kw(loader, "execution_provider") and self.cfg.execution_provider:
            kwargs["execution_provider"] = self.cfg.execution_provider
        if self._supports_kw(loader, "device"):
            kwargs["device"] = self.cfg.device
        try:
            return loader(model, **kwargs)
        except TypeError:
            return loader(model)

    def _load_voice(self, module: Any, speaker: str) -> Any:
        if not speaker:
            return None
        loader = self._find_callable([module, getattr(module, "tts", None), getattr(module, "inference", None)], [
            "load_voice",
            "load_voicepack",
            "load_speaker",
            "load_voice_model",
        ])
        if loader is None:
            return speaker
        try:
            return loader(speaker)
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Failed to load Kokoro voice '%s': %s", speaker, exc)
            return speaker

    def _load_tokenizer(self, module: Any) -> Optional[Callable[..., Any]]:
        return self._find_callable(
            [module, getattr(module, "tts", None), getattr(module, "inference", None)],
            ["load_tokenizer", "Tokenizer", "load_text_processor", "build_tokenizer"],
        )

    def _load_vocoder(self, module: Any, device: Any, dtype: Any) -> Optional[Any]:
        loader = self._find_callable(
            [module, getattr(module, "tts", None), getattr(module, "inference", None)],
            ["load_vocoder", "load_codec", "load_hifigan", "load_vits_vocoder"],
        )
        if loader is None:
            return None
        kwargs: dict[str, Any] = {}
        if device is not None and self._supports_kw(loader, "device"):
            kwargs["device"] = device
        if dtype is not None and self._supports_kw(loader, "dtype"):
            kwargs["dtype"] = dtype
        try:
            return loader(**kwargs)
        except TypeError:
            return loader()
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Failed to load Kokoro vocoder: %s", exc)
            return None

    def _maybe_tokenize(self, tokenizer: Optional[Callable[..., Any]], text: str, voice: Any) -> Optional[Any]:
        if tokenizer is None:
            return None
        try:
            if voice is not None:
                return tokenizer(text, voice=voice)
        except TypeError:
            pass
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.debug("Kokoro tokenizer with voice failed: %s", exc)
        try:
            return tokenizer(text)
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.debug("Kokoro tokenizer failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Invocation helpers
    # ------------------------------------------------------------------
    def _invoke_generator(
        self,
        generator: Callable[..., Any],
        text: str,
        tokens: Optional[Any],
        model: Any,
        voice: Any,
        vocoder: Any,
        device: Any,
        dtype: Any,
        pace: float,
        backend: str = "pytorch",
    ) -> Tuple[np.ndarray, int]:
        voice_kwargs_keys = ["speaker", "voice", "speaker_id", "voice_name"]
        speed_kwargs_keys = ["speed", "pace", "rate"]
        device_kwargs_keys = ["device", "execution_provider", "provider"]
        dtype_kwargs_keys = ["dtype", "torch_dtype"]
        base_kwargs_list: list[dict[str, Any]] = [{}]
        for key in speed_kwargs_keys:
            base_kwargs_list.append({key: pace})
        kwargs_variants: list[dict[str, Any]] = []
        for base in base_kwargs_list:
            kwargs_variants.append(base.copy())
            if voice is not None:
                for vkey in voice_kwargs_keys:
                    temp = base.copy()
                    temp[vkey] = voice
                    kwargs_variants.append(temp)

        if vocoder is not None:
            for kw in list(kwargs_variants):
                if self._supports_kw(generator, "vocoder"):
                    kw.setdefault("vocoder", vocoder)
                elif self._supports_kw(generator, "decoder"):
                    kw.setdefault("decoder", vocoder)

        if device is not None:
            for kw in list(kwargs_variants):
                for key in device_kwargs_keys:
                    if self._supports_kw(generator, key):
                        kw.setdefault(key, device)

        if dtype is not None:
            for kw in list(kwargs_variants):
                for key in dtype_kwargs_keys:
                    if self._supports_kw(generator, key):
                        kw.setdefault(key, dtype)

        arg_variants: list[Tuple[Any, ...]] = [(text,)]
        if tokens is not None:
            arg_variants.append((tokens,))
        if voice is not None:
            arg_variants.extend(
                [
                    (text, voice),
                    (voice, text),
                ]
            )
            if tokens is not None:
                arg_variants.extend(
                    [
                        (tokens, voice),
                        (voice, tokens),
                    ]
                )

        last_error: Optional[Exception] = None
        bound_self = getattr(generator, "__self__", None)

        for kwargs in kwargs_variants:
            for args in arg_variants:
                # Direct call
                try:
                    result = generator(*args, **kwargs)
                except TypeError as exc:
                    last_error = exc
                except Exception as exc:
                    last_error = exc
                else:
                    audio = self._extract_audio(result)
                    if audio is not None:
                        return audio
                if bound_self is not None:
                    continue
                # Try with model as first argument for free functions
                try:
                    result = generator(model, *args, **kwargs)
                except TypeError as exc:
                    last_error = exc
                except Exception as exc:
                    last_error = exc
                else:
                    audio = self._extract_audio(result)
                    if audio is not None:
                        return audio
                if voice is not None:
                    try:
                        result = generator(model, voice, *args, **kwargs)
                    except TypeError as exc:
                        last_error = exc
                    except Exception as exc:
                        last_error = exc
                    else:
                        audio = self._extract_audio(result)
                        if audio is not None:
                            return audio

        raise RuntimeError(
            f"Kokoro {backend} backend failed to synthesize text. Last error: {last_error}"
        )

    def _extract_audio(self, result: Any) -> Optional[Tuple[np.ndarray, int]]:
        if result is None:
            return None

        def _to_array(data: Any) -> np.ndarray:
            if isinstance(data, np.ndarray):
                arr = data
            elif hasattr(data, "cpu") and hasattr(data, "detach"):
                arr = data.detach().cpu().numpy()
            elif hasattr(data, "numpy"):
                arr = data.numpy()
            elif isinstance(data, (list, tuple)):
                arr = np.asarray(data)
            elif AudioSegment is not None and isinstance(data, AudioSegment):
                samples = np.array(data.get_array_of_samples())
                arr = samples.astype(np.float32) / 32768.0
                return arr
            else:
                raise TypeError(f"Unsupported audio container: {type(data)!r}")
            if arr.ndim > 1:
                arr = np.mean(arr, axis=0)
            return arr.astype(np.float32, copy=False)

        if isinstance(result, tuple):
            if len(result) == 1:
                return self._extract_audio(result[0])
            if len(result) >= 2 and isinstance(result[1], (int, float)):
                audio = _to_array(result[0])
                return audio, int(result[1])

        if isinstance(result, dict):
            for key in ("audio", "waveform", "samples"):
                value = result.get(key)
                if value is None:
                    continue
                try:
                    audio = _to_array(value)
                except Exception:
                    continue
                for sr_key in ("sample_rate", "sampling_rate"):
                    sr_value = result.get(sr_key)
                    if sr_value:
                        return audio, int(sr_value)
                if isinstance(value, dict):
                    for sr_key in ("sample_rate", "sampling_rate"):
                        sr_value = value.get(sr_key)
                        if sr_value:
                            arr = value.get("array") or value.get("samples") or audio
                            audio = _to_array(arr)
                            return audio, int(sr_value)

        if AudioSegment is not None and isinstance(result, AudioSegment):
            audio = np.array(result.get_array_of_samples()).astype(np.float32) / 32768.0
            return audio, int(result.frame_rate)

        # Torch tensor or numpy array directly
        try:
            audio = _to_array(result)
        except Exception:
            return None
        sample_rate = getattr(result, "sample_rate", None) or getattr(result, "sampling_rate", None)
        if not sample_rate:
            sample_rate = self.cfg.out_sr
        return audio, int(sample_rate)

