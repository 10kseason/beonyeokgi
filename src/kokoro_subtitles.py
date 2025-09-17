from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger("vc-translator.kokoro.subtitles")


@dataclass
class SubtitleStreamConfig:
    """Configuration for streaming translated subtitles to Kokoro."""

    endpoint: str
    method: str = "POST"
    timeout_sec: float = 2.0
    include_timestamps: bool = True
    retry_limit: int = 1
    retry_backoff_sec: float = 0.5
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class SubtitlePayload:
    text: str
    timestamp_ms: Optional[int]
    duration_ms: Optional[float]
    source_language: Optional[str]
    target_language: Optional[str]
    sequence: int


class KokoroSubtitleStreamer:
    """Background worker that POSTs translated sentences to Kokoro as a stream."""

    def __init__(self, cfg: SubtitleStreamConfig) -> None:
        endpoint = (cfg.endpoint or "").strip()
        if not endpoint:
            raise ValueError("Subtitle endpoint must be provided")
        self.cfg = cfg
        self.cfg.method = (self.cfg.method or "POST").upper()
        self.cfg.timeout_sec = max(0.1, float(self.cfg.timeout_sec or 2.0))
        self.cfg.retry_limit = max(0, int(self.cfg.retry_limit or 0))
        self.cfg.retry_backoff_sec = max(0.0, float(self.cfg.retry_backoff_sec or 0.0))
        self._queue: "Queue[Optional[SubtitlePayload]]" = Queue()
        self._session = requests.Session()
        self._thread = threading.Thread(target=self._worker, name="kokoro-subtitle-stream", daemon=True)
        self._stop_event = threading.Event()
        self._sequence = 0
        self._last_error_signature: Optional[str] = None
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def submit(
        self,
        text: str,
        *,
        duration_ms: Optional[float] = None,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
    ) -> None:
        clean = text.strip()
        if not clean:
            return
        timestamp_ms = int(time.time() * 1000) if self.cfg.include_timestamps else None
        self._sequence += 1
        payload = SubtitlePayload(
            text=clean,
            timestamp_ms=timestamp_ms,
            duration_ms=float(duration_ms) if duration_ms is not None else None,
            source_language=(source_language or "").strip() or None,
            target_language=(target_language or "").strip() or None,
            sequence=self._sequence,
        )
        self._queue.put(payload)

    def close(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        self._queue.put(None)
        if self._thread.is_alive():
            self._thread.join(timeout=1.5)
        try:
            self._session.close()
        except Exception:  # pragma: no cover - best effort cleanup
            pass

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.25)
            except Empty:
                continue
            if item is None:
                break
            try:
                self._send_payload(item)
            except Exception:
                logger.debug("Failed to stream Kokoro subtitle", exc_info=True)
        # Drain remaining items if any
        while True:
            try:
                item = self._queue.get_nowait()
            except Empty:
                break
            if item is None:
                break
            try:
                self._send_payload(item)
            except Exception:
                logger.debug("Failed to stream Kokoro subtitle during drain", exc_info=True)

    def _send_payload(self, payload: SubtitlePayload) -> None:
        data: Dict[str, Any] = {"text": payload.text, "sequence": payload.sequence}
        if payload.timestamp_ms is not None:
            data["timestamp_ms"] = payload.timestamp_ms
        if payload.duration_ms is not None:
            data["duration_ms"] = float(max(0.0, payload.duration_ms))
        if payload.source_language:
            data["source_language"] = payload.source_language
        if payload.target_language:
            data["target_language"] = payload.target_language
        attempt = 0
        last_exc: Optional[BaseException] = None
        while True:
            attempt += 1
            try:
                response = self._session.request(
                    self.cfg.method,
                    self.cfg.endpoint,
                    json=data,
                    headers=self.cfg.headers or None,
                    timeout=self.cfg.timeout_sec,
                )
            except Exception as exc:
                last_exc = exc
                self._log_failure(exc, attempt, payload)
            else:
                if response.ok:
                    if attempt > 1:
                        logger.debug(
                            "Subtitle stream recovered after %d retries (sequence=%d)",
                            attempt - 1,
                            payload.sequence,
                        )
                    return
                last_exc = RuntimeError(f"HTTP {response.status_code}: {response.text.strip()[:200]}")
                self._log_failure(last_exc, attempt, payload)
            if attempt > self.cfg.retry_limit:
                break
            if self.cfg.retry_backoff_sec:
                time.sleep(self.cfg.retry_backoff_sec)
        if last_exc is not None:
            logger.debug("Dropping Kokoro subtitle %d after retries: %s", payload.sequence, last_exc)

    def _log_failure(self, exc: BaseException, attempt: int, payload: SubtitlePayload) -> None:
        signature = f"{type(exc).__name__}:{getattr(exc, 'args', None)}"
        message = (
            f"Kokoro subtitle stream failed on attempt {attempt} (sequence={payload.sequence}): {exc}"
        )
        if signature != self._last_error_signature:
            logger.warning(message)
            self._last_error_signature = signature
        else:
            logger.debug(message)
