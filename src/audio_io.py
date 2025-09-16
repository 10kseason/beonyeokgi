import queue
import time
from typing import Optional

import numpy as np
import sounddevice as sd
from .utils import parse_sd_device


class MicReader:
    """
    Capture mono 16-bit PCM frames from the default (or named) input device
    at a given sample rate and frame size (ms). Returns raw bytes per read().
    """

    def __init__(self, sr: int = 16000, block_ms: int = 30, input_device: Optional[str] = None):
        self.sr = sr
        self.block_ms = block_ms
        self.block_size = int(sr * block_ms / 1000)
        self.bytes_per_frame = self.block_size * 2  # int16 mono
        self.q: "queue.Queue[bytes]" = queue.Queue(maxsize=64)
        self.stream: Optional[sd.RawInputStream] = None
        self.input_device = input_device

    def _callback(self, indata, frames, time_info, status):  # noqa: D401
        if status:
            # Drop or log status; avoid blocking in callback
            pass
        try:
            self.q.put_nowait(bytes(indata))
        except queue.Full:
            # Drop oldest to keep latency bounded
            try:
                _ = self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(bytes(indata))
            except queue.Full:
                pass

    def start(self):
        if self.stream is not None:
            return
        self.stream = sd.RawInputStream(
            samplerate=self.sr,
            blocksize=self.block_size,
            channels=1,
            dtype="int16",
            callback=self._callback,
            device=parse_sd_device(self.input_device),
        )
        self.stream.start()

    def read(self, timeout: Optional[float] = None) -> bytes:
        """Blocking read of next audio frame (bytes)."""
        return self.q.get(timeout=timeout) if timeout is not None else self.q.get()

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
            finally:
                self.stream.close()
                self.stream = None


def play_pcm_int16(samples: np.ndarray, samplerate: int, output_device: Optional[str] = None):
    """
    Play a numpy array of int16 samples via sounddevice.
    """
    if samples.dtype != np.int16:
        samples = samples.astype(np.int16)
    sd.play(samples, samplerate=samplerate, device=output_device or None, blocking=True)
