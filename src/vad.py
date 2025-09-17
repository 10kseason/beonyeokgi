from __future__ import annotations

import collections
from dataclasses import dataclass

import webrtcvad


@dataclass
class VADConfig:
    sr: int = 16000
    frame_ms: int = 30
    aggressiveness: int = 2
    min_speech_sec: float = 0.30
    max_utt_sec: float = 8.0
    silence_end_ms: int = 300


class VADSegmenter:
    """
    Push frame-by-frame 16-bit PCM bytes and receive full utterance segments.
    Returns None while accumulating; returns bytes when an utterance completes.
    """

    def __init__(
        self,
        sr: int = 16000,
        frame_ms: int = 30,
        aggressiveness: int = 2,
        min_speech_sec: float = 0.30,
        max_utt_sec: float = 8.0,
        silence_end_ms: int = 300,
        chunk_min_ms: int | None = None,
        chunk_max_ms: int | None = None,
        chunk_hop_ms: int | None = None,
    ):
        self.cfg = VADConfig(
            sr=sr,
            frame_ms=frame_ms,
            aggressiveness=aggressiveness,
            min_speech_sec=min_speech_sec,
            max_utt_sec=max_utt_sec,
            silence_end_ms=silence_end_ms,
        )
        self._vad = webrtcvad.Vad(self.cfg.aggressiveness)
        self._pending_segments: "collections.deque[bytes]" = collections.deque()
        self._chunk_buffer = bytearray()
        self._chunk_enabled = False
        self._chunk_min_bytes = 0
        self._chunk_max_bytes = 0
        self._chunk_hop_bytes = 0
        self._reset_state()
        self.configure_chunking(chunk_min_ms, chunk_max_ms, chunk_hop_ms)

    def _reset_state(self, preserve_pending: bool = False):
        self._triggered = False
        self._speech_frames = []  # list[bytes]
        self._accum_ms = 0
        self._silence_ms = 0
        self._frame_bytes = int(self.cfg.sr * self.cfg.frame_ms / 1000) * 2
        self._chunk_buffer = bytearray()
        if not preserve_pending:
            self._pending_segments.clear()

    def push(self, frame_bytes: bytes) -> bytes | None:
        if len(frame_bytes) < self._frame_bytes:
            return self._pop_pending()
        is_speech = self._vad.is_speech(frame_bytes, self.cfg.sr)
        self._accum_ms += self.cfg.frame_ms
        result: bytes | None = None
        if is_speech:
            if not self._triggered:
                self._triggered = True
                self._speech_frames = []
                self._silence_ms = 0
            self._speech_frames.append(frame_bytes)
            self._silence_ms = 0
            if self._chunk_enabled:
                self._chunk_buffer.extend(frame_bytes)
                self._emit_ready_chunks()
        else:
            if self._triggered:
                self._silence_ms += self.cfg.frame_ms
                if self._silence_ms >= self.cfg.silence_end_ms:
                    if self._chunk_enabled:
                        self._flush_chunk_buffer(force=True)
                        self._reset_state(preserve_pending=True)
                        return self._pop_pending()
                    segment = b"".join(self._speech_frames)
                    min_ms = int(self.cfg.min_speech_sec * 1000)
                    self._reset_state()
                    if len(segment) >= int(self.cfg.sr * (min_ms / 1000.0)) * 2:
                        result = segment

        # Force cut if utterance gets too long
        if self._triggered and (self._accum_ms >= int(self.cfg.max_utt_sec * 1000)):
            if self._chunk_enabled:
                self._flush_chunk_buffer(force=True)
                self._reset_state(preserve_pending=True)
                return self._pop_pending()
            segment = b"".join(self._speech_frames)
            self._reset_state()
            if segment:
                result = segment

        if result is not None:
            return result
        return self._pop_pending()

    def pop_pending(self) -> bytes | None:
        return self._pop_pending()

    def configure_chunking(
        self,
        chunk_min_ms: int | None,
        chunk_max_ms: int | None,
        chunk_hop_ms: int | None,
    ) -> None:
        if (
            chunk_min_ms is None
            or chunk_max_ms is None
            or chunk_hop_ms is None
            or chunk_min_ms <= 0
            or chunk_max_ms <= 0
            or chunk_hop_ms <= 0
        ):
            self._chunk_enabled = False
            self._chunk_buffer = bytearray()
            self._chunk_min_bytes = 0
            self._chunk_max_bytes = 0
            self._chunk_hop_bytes = 0
            self._pending_segments.clear()
            return
        chunk_min = max(self.cfg.frame_ms, int(chunk_min_ms))
        chunk_max = max(chunk_min, int(chunk_max_ms))
        hop = max(self.cfg.frame_ms, int(chunk_hop_ms))
        min_bytes = max(
            self._frame_bytes,
            int(self.cfg.sr * (chunk_min / 1000.0)) * 2,
        )
        max_bytes = max(min_bytes, int(self.cfg.sr * (chunk_max / 1000.0)) * 2)
        hop_bytes = max(self._frame_bytes, int(self.cfg.sr * (hop / 1000.0)) * 2)
        self._chunk_min_bytes = min_bytes
        self._chunk_max_bytes = max_bytes
        self._chunk_hop_bytes = hop_bytes
        self._chunk_enabled = True
        self._chunk_buffer = bytearray()
        self._pending_segments.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _emit_ready_chunks(self) -> None:
        if not self._chunk_enabled or self._chunk_max_bytes <= 0:
            return
        while len(self._chunk_buffer) >= self._chunk_max_bytes:
            chunk = bytes(self._chunk_buffer[: self._chunk_max_bytes])
            self._pending_segments.append(chunk)
            hop = min(self._chunk_hop_bytes, self._chunk_max_bytes)
            hop = max(hop, self._frame_bytes)
            hop = min(hop, len(self._chunk_buffer))
            del self._chunk_buffer[:hop]

    def _flush_chunk_buffer(self, force: bool = False) -> None:
        if not self._chunk_enabled or not self._chunk_buffer:
            return
        self._emit_ready_chunks()
        if not self._chunk_buffer:
            return
        if len(self._chunk_buffer) >= self._chunk_min_bytes or force:
            self._pending_segments.append(bytes(self._chunk_buffer))
        self._chunk_buffer = bytearray()

    def _pop_pending(self) -> bytes | None:
        if self._pending_segments:
            return self._pending_segments.popleft()
        return None
