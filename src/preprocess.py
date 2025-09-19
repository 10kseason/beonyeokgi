from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import logging
import ctypes
import ctypes.util
import shutil
import subprocess

import numpy as np
from pathlib import Path
import pyloudnorm as pyln

from .utils import float32_to_int16, int16_to_float32, resample_audio


logger = logging.getLogger(__name__)


_RNNOISE_WARNED = False


def _load_rnnoise_library() -> Optional[ctypes.CDLL]:
    if ctypes is None:
        return None
    candidates: list[str] = []
    lib_name = ctypes.util.find_library("rnnoise")
    if lib_name:
        candidates.append(lib_name)
    base_dir = Path(__file__).resolve().parent
    for candidate in ("rnnoise.dll", "librnnoise.so", "librnnoise.dylib"):
        candidates.append(str(base_dir / candidate))
        candidates.append(candidate)
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return ctypes.cdll.LoadLibrary(candidate)
        except OSError:
            continue
    return None


_RNNOISE_LIB = _load_rnnoise_library()


class _RNNoiseEffect:
    """Thin ctypes wrapper around the optional RNNoise library."""

    FRAME_SIZE = 480
    SAMPLE_RATE = 48000

    def __init__(self) -> None:
        if _RNNOISE_LIB is None:
            raise RuntimeError("RNNoise library is not available")
        self._lib = _RNNOISE_LIB
        self._lib.rnnoise_create.argtypes = [ctypes.c_void_p]
        self._lib.rnnoise_create.restype = ctypes.c_void_p
        self._lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        self._lib.rnnoise_destroy.restype = None
        self._lib.rnnoise_process_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.rnnoise_process_frame.restype = ctypes.c_float
        try:
            state_ptr = self._lib.rnnoise_create(None)
        except Exception as exc:
            raise RuntimeError("rnnoise_create failed") from exc
        if not state_ptr:
            raise RuntimeError("rnnoise_create returned NULL state")
        self._state = ctypes.c_void_p(state_ptr)
        if hasattr(self._lib, "rnnoise_get_frame_size"):
            try:
                frame_size = int(self._lib.rnnoise_get_frame_size())
            except Exception:
                frame_size = self.FRAME_SIZE
            if frame_size > 0:
                self.frame_size = frame_size
            else:
                self.frame_size = self.FRAME_SIZE
        else:
            self.frame_size = self.FRAME_SIZE
        self.sample_rate = self.SAMPLE_RATE

    def close(self) -> None:
        if getattr(self, "_state", None):
            try:
                self._lib.rnnoise_destroy(self._state)
            finally:
                self._state = None

    def process(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
        if audio.size == 0:
            return audio.astype(np.float32, copy=False), sample_rate
        arr = np.asarray(audio, dtype=np.float32)
        sr = int(sample_rate)
        if sr != self.sample_rate:
            arr = resample_audio(arr, sr, self.sample_rate)
            sr = self.sample_rate
        frame = int(self.frame_size)
        pad = (frame - (arr.size % frame)) % frame
        if pad:
            arr = np.pad(arr, (0, pad), mode="constant")
        out = np.empty_like(arr, dtype=np.float32)
        in_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        state = self._state
        for offset in range(0, arr.size, frame):
            self._lib.rnnoise_process_frame(state, out_ptr + offset, in_ptr + offset)
        if pad:
            out = out[:-pad]
        return out.astype(np.float32, copy=False), sr

class _BiquadFilter:
    """Simple biquad filter with persistent state for mono signals."""

    def __init__(self, filter_type: str, cutoff_hz: float, sample_rate: int, q: float = 0.7071) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        cutoff = max(1.0, float(cutoff_hz))
        self.sample_rate = float(sample_rate)
        self.cutoff = cutoff
        self.q = max(1e-5, float(q))
        self._configure(filter_type)
        self._z1 = 0.0
        self._z2 = 0.0

    def _configure(self, filter_type: str) -> None:
        omega = 2.0 * math.pi * (self.cutoff / self.sample_rate)
        omega = min(max(omega, 1e-6), math.pi - 1e-6)
        sin_omega = math.sin(omega)
        cos_omega = math.cos(omega)
        alpha = sin_omega / (2.0 * self.q)
        if filter_type == "lowpass":
            b0 = (1.0 - cos_omega) * 0.5
            b1 = 1.0 - cos_omega
            b2 = (1.0 - cos_omega) * 0.5
        elif filter_type == "highpass":
            b0 = (1.0 + cos_omega) * 0.5
            b1 = -(1.0 + cos_omega)
            b2 = (1.0 + cos_omega) * 0.5
        else:
            raise ValueError(f"Unsupported filter_type: {filter_type}")
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha
        self._b0 = b0 / a0
        self._b1 = b1 / a0
        self._b2 = b2 / a0
        self._a1 = a1 / a0
        self._a2 = a2 / a0

    def reset(self) -> None:
        self._z1 = 0.0
        self._z2 = 0.0

    def process(self, samples: np.ndarray) -> np.ndarray:
        if samples.size == 0:
            return samples.astype(np.float32, copy=False)
        x = samples.astype(np.float32, copy=False)
        out = np.empty_like(x, dtype=np.float32)
        z1 = self._z1
        z2 = self._z2
        b0, b1, b2, a1, a2 = self._b0, self._b1, self._b2, self._a1, self._a2
        for idx, sample in enumerate(x):
            y = b0 * sample + z1
            z1 = b1 * sample - a1 * y + z2
            z2 = b2 * sample - a2 * y
            out[idx] = y
        self._z1 = z1
        self._z2 = z2
        return out


@dataclass
class LoudnormConfig:
    target_lufs: float = -16.0
    true_peak_db: float = -1.5
    target_lra: float = 11.0
    compressor_threshold_db: float = -23.0
    compressor_attack: float = 0.015
    compressor_release: float = 0.250


class AudioPreprocessor:
    """Apply loudness, filtering, and resampling to microphone audio."""

    def __init__(
        self,
        target_sr: int = 16000,
        highpass_hz: float = 90.0,
        lowpass_hz: float = 7200.0,
        loudnorm: LoudnormConfig | None = None,
    ) -> None:
        self.target_sr = int(target_sr)
        if self.target_sr <= 0:
            raise ValueError("target_sr must be positive")
        self._highpass = _BiquadFilter("highpass", highpass_hz, self.target_sr)
        self._lowpass = _BiquadFilter("lowpass", lowpass_hz, self.target_sr)
        self._loudnorm = loudnorm or LoudnormConfig()
        self._meter = pyln.Meter(self.target_sr)
        self._ffmpeg_path = shutil.which("ffmpeg")
        self._ffmpeg_disabled = False
        self._ffmpeg_warned = False

    def reset(self) -> None:
        self._highpass.reset()
        self._lowpass.reset()

    def process(self, pcm_bytes: bytes, input_sr: int) -> Tuple[bytes, int]:
        if not pcm_bytes:
            return pcm_bytes, self.target_sr
        samples_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        if samples_i16.size == 0:
            return pcm_bytes, self.target_sr
        sr = int(input_sr) if int(input_sr) > 0 else self.target_sr
        if sr != self.target_sr:
            converted = self._try_ffmpeg_resample(samples_i16, sr)
            if converted is not None:
                samples_i16 = converted
                sr = self.target_sr
        audio = int16_to_float32(samples_i16)
        if sr != self.target_sr:
            audio = resample_audio(audio, sr, self.target_sr)
            sr = self.target_sr
        filtered = self._highpass.process(audio)
        filtered = self._lowpass.process(filtered)
        normalized = self._apply_loudnorm(filtered)
        out = float32_to_int16(normalized)
        return out.tobytes(), sr

    def _try_ffmpeg_resample(self, samples: np.ndarray, src_sr: int) -> np.ndarray | None:
        if self._ffmpeg_disabled or self._ffmpeg_path is None:
            return None
        if src_sr <= 0 or src_sr == self.target_sr:
            return None
        cmd = [
            self._ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(int(src_sr)),
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(self.target_sr),
            "pipe:1",
        ]
        try:
            result = subprocess.run(
                cmd,
                input=samples.tobytes(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except FileNotFoundError:
            if not self._ffmpeg_warned:
                logger.warning("ffmpeg executable not found; falling back to internal resampler")
                self._ffmpeg_warned = True
            self._ffmpeg_disabled = True
            return None
        except subprocess.CalledProcessError as exc:
            if not self._ffmpeg_warned:
                err_output = exc.stderr.decode(errors="ignore").strip()
                if err_output:
                    logger.warning("ffmpeg resample failed: %s", err_output)
                else:
                    logger.warning("ffmpeg resample failed with exit code %s", exc.returncode)
                self._ffmpeg_warned = True
            self._ffmpeg_disabled = True
            return None
        output = result.stdout
        if not output:
            if not self._ffmpeg_warned:
                logger.warning("ffmpeg produced no output during resample; using internal resampler")
                self._ffmpeg_warned = True
            return None
        return np.frombuffer(output, dtype=np.int16)

    # ------------------------------------------------------------------
    # Loudness helpers
    # ------------------------------------------------------------------
    def _apply_loudnorm(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio.astype(np.float32, copy=False)
        data = np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)
        try:
            loudness = self._meter.integrated_loudness(data)
            target = pyln.normalize.loudness(data, loudness, self._loudnorm.target_lufs)
        except Exception:
            target = data
        target = self._limit_true_peak(target)
        try:
            lra = self._meter.loudness_range(target)
        except Exception:
            lra = 0.0
        if lra > self._loudnorm.target_lra:
            ratio = min(6.0, max(1.2, lra / self._loudnorm.target_lra))
            compressed = self._compress_dynamic(target, ratio=ratio)
            try:
                loudness_after = self._meter.integrated_loudness(compressed)
                target = pyln.normalize.loudness(
                    compressed, loudness_after, self._loudnorm.target_lufs
                )
            except Exception:
                target = compressed
            target = self._limit_true_peak(target)
        return np.clip(target, -1.0, 1.0).astype(np.float32, copy=False)

    def _limit_true_peak(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio
        try:
            peak = float(np.max(np.abs(self._meter.true_peak(audio))))
        except Exception:
            peak = float(np.max(np.abs(audio)))
        if peak <= 0.0:
            return audio
        peak_db = 20.0 * math.log10(max(peak, 1e-12))
        if peak_db <= self._loudnorm.true_peak_db:
            return audio
        gain = 10.0 ** ((self._loudnorm.true_peak_db - peak_db) / 20.0)
        return np.clip(audio * gain, -1.0, 1.0)

    def _compress_dynamic(self, audio: np.ndarray, ratio: float) -> np.ndarray:
        if audio.size == 0 or ratio <= 1.0:
            return audio
        threshold_db = self._loudnorm.compressor_threshold_db
        threshold = 10.0 ** (threshold_db / 20.0)
        attack = max(1e-4, float(self._loudnorm.compressor_attack))
        release = max(1e-4, float(self._loudnorm.compressor_release))
        attack_coeff = math.exp(-1.0 / (self.target_sr * attack))
        release_coeff = math.exp(-1.0 / (self.target_sr * release))
        env = 0.0
        out = np.empty_like(audio, dtype=np.float32)
        for idx, sample in enumerate(audio):
            rectified = abs(float(sample))
            if rectified > env:
                env = attack_coeff * env + (1.0 - attack_coeff) * rectified
            else:
                env = release_coeff * env + (1.0 - release_coeff) * rectified
            if env <= threshold:
                gain = 1.0
            else:
                gain = (threshold / env) ** (ratio - 1.0)
            out[idx] = float(sample) * gain
        return out







