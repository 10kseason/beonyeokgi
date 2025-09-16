from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import logging
import time
import numpy as np
import requests

from .utils import (
    float32_to_int16,
    int16_to_float32,
    resample_audio,
    write_wav_int16,
)

logger = logging.getLogger("vc-translator.vcclient")


@dataclass
class VoiceChangerConfig:
    enabled: bool = False
    base_url: str = "http://localhost:18000"
    endpoint: str = "/api/voice-changer/convert_chunk"
    input_sample_rate: Optional[int] = None
    output_sample_rate: Optional[int] = None
    timeout_sec: float = 5.0
    save_original_path: Optional[str] = None
    save_converted_path: Optional[str] = None
    fallback_endpoint: Optional[str] = "/api/voice-changer/convert_chunk_bulk"
    fallback_output_device: Optional[str] = None
    stream_mode: bool = False
    stream_chunk_ms: int = 1000


class VoiceChangerClient:
    def __init__(self, cfg: VoiceChangerConfig) -> None:
        self.cfg = cfg
        self._session = requests.Session()
        self._rates_checked = False
        self._logged_failures: set[str] = set()
        self._stats_total_chunks = 0
        self._stats_failed_chunks = 0

    def _record_failure(self, key: str, message: str) -> None:
        if key not in self._logged_failures:
            logger.warning("[VoiceChanger] %s", message)
            self._logged_failures.add(key)

    def _ensure_rates(self) -> None:
        if self._rates_checked:
            return
        url = self.cfg.base_url.rstrip('/') + "/api/configuration-manager/configuration"
        try:
            resp = self._session.get(url, timeout=self.cfg.timeout_sec)
            if resp.ok:
                data = resp.json()
                if not self.cfg.input_sample_rate:
                    sr = data.get("input_sample_rate") or data.get("monitor_sample_rate")
                    if sr:
                        self.cfg.input_sample_rate = int(sr)
                if not self.cfg.output_sample_rate:
                    sr = data.get("output_sample_rate") or data.get("monitor_sample_rate")
                    if sr:
                        self.cfg.output_sample_rate = int(sr)
            else:
                self._record_failure("config_http", f"Failed to fetch configuration (HTTP {resp.status_code})")
        except Exception as exc:
            self._record_failure("config_exc", f"Failed to fetch configuration: {exc}")
        if not self.cfg.input_sample_rate or self.cfg.input_sample_rate <= 0:
            self.cfg.input_sample_rate = 16000
        if not self.cfg.output_sample_rate or self.cfg.output_sample_rate <= 0:
            self.cfg.output_sample_rate = self.cfg.input_sample_rate
        self._rates_checked = True

    def _convert_chunk(self, chunk: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        chunk_ms = 0.0
        if sample_rate > 0:
            chunk_ms = (len(chunk) / float(sample_rate)) * 1000.0
        headers = {
            "x-timestamp": str(int(time.time() * 1000))
        }
        payload = chunk.astype(np.float32, copy=False).tobytes()
        files = {
            "waveform": ("waveform.bin", payload, "application/octet-stream")
        }
        endpoints = [self.cfg.endpoint]
        if self.cfg.fallback_endpoint and self.cfg.fallback_endpoint not in endpoints:
            endpoints.append(self.cfg.fallback_endpoint)
        for ep in endpoints:
            url = self.cfg.base_url.rstrip('/') + ep
            start = time.perf_counter()
            try:
                resp = self._session.post(url, files=files, headers=headers, timeout=self.cfg.timeout_sec)
            except Exception as exc:
                self._record_failure(f"{ep}|exception", f"Request to {ep} failed: {exc}")
                continue
            if not resp.ok:
                detail = resp.text.strip()
                self._record_failure(f"{ep}|{resp.status_code}", f"HTTP {resp.status_code} on {ep}: {detail}")
                continue
            if not resp.content:
                self._record_failure(f"{ep}|empty", f"Empty response from {ep}")
                continue
            elapsed = time.perf_counter() - start
            converted = np.frombuffer(resp.content, dtype=np.float32)
            logger.debug("VC chunk via %s: %.1f ms -> %d samples in %.3f s", ep, chunk_ms, converted.size, elapsed)
            return converted
        logger.debug("VC chunk failed after trying endpoints %s (len=%.1f ms)", endpoints, chunk_ms)
        return None

    def convert(self, samples: np.ndarray, sample_rate: int) -> Optional[Tuple[np.ndarray, int]]:
        if samples.size == 0:
            return None
        if self.cfg.save_original_path:
            try:
                write_wav_int16(
                    self.cfg.save_original_path,
                    float32_to_int16(int16_to_float32(samples)),
                    sample_rate,
                )
            except Exception as exc:
                self._record_failure("save_original", f"Failed to save original audio: {exc}")
        if not self.cfg.enabled:
            return None
        self._ensure_rates()
        float_samples = int16_to_float32(samples)
        target_sr = self.cfg.input_sample_rate or sample_rate
        if target_sr != sample_rate:
            float_samples = resample_audio(float_samples, sample_rate, target_sr)
            sample_rate = target_sr
        chunk_start = time.perf_counter()
        converted = self._convert_chunk(float_samples, sample_rate)
        elapsed = time.perf_counter() - chunk_start
        if converted is None:
            self._stats_failed_chunks += 1
            logger.debug("Voice changer convert failed in %.3f s", elapsed)
            return None
        self._stats_total_chunks += 1
        out_sr = self.cfg.output_sample_rate or sample_rate
        if self.cfg.save_converted_path:
            try:
                write_wav_int16(
                    self.cfg.save_converted_path,
                    float32_to_int16(converted),
                    out_sr,
                )
            except Exception as exc:
                self._record_failure("save_converted", f"Failed to save converted audio: {exc}")
        logger.debug("Voice changer convert succeeded in %.3f s (samples=%d)", elapsed, converted.size)
        return converted, out_sr

    def convert_stream(self, samples: np.ndarray, sample_rate: int):
        if samples.size == 0 or not self.cfg.enabled or not self.cfg.stream_mode:
            return None
        self._ensure_rates()
        float_samples = int16_to_float32(samples)
        target_sr = self.cfg.input_sample_rate or sample_rate
        if target_sr != sample_rate:
            float_samples = resample_audio(float_samples, sample_rate, target_sr)
        out_sr = self.cfg.output_sample_rate or sample_rate
        if out_sr <= 0:
            out_sr = sample_rate
        chunk_ms = max(10, int(self.cfg.stream_chunk_ms or 1000))
        chunk_len = max(1, int(target_sr * (chunk_ms / 1000.0)))

        def generator():
            for idx, start in enumerate(range(0, float_samples.size, chunk_len)):
                chunk = float_samples[start:start + chunk_len]
                chunk_timer = time.perf_counter()
                converted = self._convert_chunk(chunk, target_sr)
                elapsed = time.perf_counter() - chunk_timer
                if converted is None:
                    self._stats_failed_chunks += 1
                    logger.debug("VC streaming chunk #%d failed after %.3f s", idx, elapsed)
                    raise RuntimeError("voice changer chunk failed")
                self._stats_total_chunks += 1
                if out_sr != target_sr:
                    converted = resample_audio(converted, target_sr, out_sr)
                logger.debug("VC streaming chunk #%d ok -> %d samples in %.3f s", idx, converted.size, elapsed)
                yield converted

        return generator(), out_sr

    def stats(self) -> Tuple[int, int]:
        return self._stats_total_chunks, self._stats_failed_chunks
