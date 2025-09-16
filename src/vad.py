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
        self._reset_state()

    def _reset_state(self):
        self._triggered = False
        self._speech_frames = []  # list[bytes]
        self._accum_ms = 0
        self._silence_ms = 0
        self._frame_bytes = int(self.cfg.sr * self.cfg.frame_ms / 1000) * 2

    def push(self, frame_bytes: bytes) -> bytes | None:
        if len(frame_bytes) < self._frame_bytes:
            # Ignore partial frames
            return None
        is_speech = self._vad.is_speech(frame_bytes, self.cfg.sr)
        self._accum_ms += self.cfg.frame_ms

        if is_speech:
            if not self._triggered:
                self._triggered = True
                self._speech_frames = []
                self._silence_ms = 0
            self._speech_frames.append(frame_bytes)
            self._silence_ms = 0
        else:
            if self._triggered:
                self._silence_ms += self.cfg.frame_ms
                if self._silence_ms >= self.cfg.silence_end_ms:
                    # End utterance
                    segment = b"".join(self._speech_frames)
                    min_ms = int(self.cfg.min_speech_sec * 1000)
                    self._reset_state()
                    if len(segment) >= int(self.cfg.sr * (min_ms / 1000.0)) * 2:
                        return segment

        # Force cut if utterance gets too long
        if self._triggered and (self._accum_ms >= int(self.cfg.max_utt_sec * 1000)):
            segment = b"".join(self._speech_frames)
            self._reset_state()
            if segment:
                return segment

        return None
