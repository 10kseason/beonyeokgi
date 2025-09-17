from __future__ import annotations

import os
import re
import wave
from typing import Optional

import numpy as np
from pydub import AudioSegment

_HANGUL_RE = re.compile(r"[\u3130-\u318F\uAC00-\uD7A3]")
_FILLER_RE = re.compile(
    r"(?<![\u3130-\u318F\uAC00-\uD7A3])"  # no Hangul immediately before
    r"((?:[\u3130-\u318F]*음+)|어+|그니까)(?:\s*요)?"  # filler body with optional 요
    r"(?:[\s\u0020\u00A0\.,!?…~]*)",  # optional trailing whitespace/punctuation
    re.IGNORECASE,
)


def parse_sd_device(device: Optional[object]) -> Optional[object]:
    """
    Coerce a sounddevice device argument so that numeric strings become ints.
    Accepts None, int, or string name/index.
    """
    if device is None:
        return None
    if isinstance(device, int):
        return device
    try:
        s = str(device).strip()
    except Exception:
        return device
    if s.isdigit():
        try:
            return int(s)
        except Exception:
            return s
    return s


def audiosegment_from_mp3_bytes(data: bytes) -> AudioSegment:
    """
    Decode MP3 bytes to AudioSegment. Requires ffmpeg in PATH.
    """
    from io import BytesIO

    return AudioSegment.from_file(BytesIO(data), format="mp3")


def normalize_dbfs(seg: AudioSegment, target_dbfs: float) -> AudioSegment:
    if seg.dBFS == float("-inf"):
        return seg
    change = target_dbfs - seg.dBFS
    return seg.apply_gain(change)


def to_int16_np(seg: AudioSegment, out_sr: int) -> np.ndarray:
    seg = seg.set_channels(1).set_frame_rate(out_sr)
    samples = np.array(seg.get_array_of_samples())
    # pydub returns arrays in sample width; here int16
    if samples.dtype != np.int16:
        samples = samples.astype(np.int16)
    return samples


def int16_to_float32(samples: np.ndarray) -> np.ndarray:
    if samples.dtype == np.float32:
        return samples
    return samples.astype(np.float32) / 32768.0


def float32_to_int16(samples: np.ndarray) -> np.ndarray:
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if dst_sr <= 0 or src_sr <= 0 or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    data = audio.astype(np.float32, copy=False)
    if src_sr == dst_sr:
        return data
    ratio = dst_sr / float(src_sr)
    new_len = int(round(data.size * ratio))
    if new_len <= 1:
        return data
    x_old = np.linspace(0, 1, num=data.size, endpoint=False, dtype=np.float64)
    x_new = np.linspace(0, 1, num=new_len, endpoint=False, dtype=np.float64)
    resampled = np.interp(x_new, x_old, data.astype(np.float64))
    return resampled.astype(np.float32)


def write_wav_int16(path: str, samples: np.ndarray, sample_rate: int) -> None:
    if path == "":
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    data = samples.astype(np.int16, copy=False)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(data.tobytes())


def contains_hangul(text: str) -> bool:
    """Return True when the given text contains Hangul characters."""

    if not text:
        return False
    return bool(_HANGUL_RE.search(str(text)))


def remove_korean_fillers(text: str) -> str:
    """Strip common Korean fillers such as 음, 어, and 그니까."""

    if not text:
        return ""
    cleaned = _FILLER_RE.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned, flags=re.UNICODE)
    return cleaned.strip()

