from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import logging
import time
import edge_tts
import numpy as np
import sounddevice as sd
from pydub import AudioSegment

from .utils import (
    audiosegment_from_mp3_bytes,
    normalize_dbfs,
    parse_sd_device,
    to_int16_np,
    float32_to_int16,
    resample_audio,
)

if TYPE_CHECKING:
    from .voice_changer_client import VoiceChangerClient

logger = logging.getLogger("vc-translator.tts.edge")


@dataclass
class EdgeConfig:
    voice: str = "en-US-AriaNeural"
    pace: float = 1.0
    volume_db: float = 0.0
    out_sr: int = 48000
    normalize_dbfs: Optional[float] = None
    output_device: Optional[str] = None


class EdgeTTS:
    def __init__(
        self,
        voice: str = "en-US-AriaNeural",
        pace: float = 1.0,
        volume_db: float = 0.0,
        out_sr: int = 48000,
        output_device: Optional[str] = None,
        normalize_dbfs: Optional[float] = None,
        voice_changer: Optional["VoiceChangerClient"] = None,
    ) -> None:
        self.cfg = EdgeConfig(
            voice=voice, pace=pace, volume_db=volume_db, out_sr=out_sr, output_device=output_device,
            normalize_dbfs=normalize_dbfs,
        )
        self.voice_changer = voice_changer

    async def synth_to_play(self, text: str) -> Optional[bool]:
        if not text:
            return None
        rate_pct = int(round((self.cfg.pace - 1.0) * 100))
        rate = f"{rate_pct:+d}%"
        vol_db = int(round(self.cfg.volume_db))
        volume = f"{vol_db:+d}%"

        communicate = edge_tts.Communicate(text=text, voice=self.cfg.voice, rate=rate, volume=volume)
        mp3_bytes = bytearray()
        try:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_bytes.extend(chunk["data"])  # bytes
        except edge_tts.exceptions.NoAudioReceived as exc:
            logger.warning("[EdgeTTS] No audio received from service: %s", exc)
            return False if self.voice_changer is not None else None
        except Exception as exc:
            logger.warning("[EdgeTTS] Edge TTS request failed: %s", exc)
            return False if self.voice_changer is not None else None
        if not mp3_bytes:
            logger.warning("[EdgeTTS] Empty TTS result")
            return False if self.voice_changer is not None else None

        seg: AudioSegment = audiosegment_from_mp3_bytes(bytes(mp3_bytes))
        if self.cfg.normalize_dbfs is not None:
            seg = normalize_dbfs(seg, self.cfg.normalize_dbfs)
        samples = to_int16_np(seg, out_sr=self.cfg.out_sr)
        play_samples = samples
        target_sr = self.cfg.out_sr

        vc_success: Optional[bool] = None
        fallback_device = None
        if self.voice_changer is not None:
            fallback_device = getattr(self.voice_changer.cfg, "fallback_output_device", None)
            vc_success = False
            use_stream = bool(getattr(self.voice_changer.cfg, "stream_mode", False))
            stream_failed = False
            if use_stream:
                stream_result = self.voice_changer.convert_stream(samples, self.cfg.out_sr)
                if stream_result:
                    stream_gen, stream_sr = stream_result
                    try:
                        with sd.OutputStream(
                            samplerate=stream_sr,
                            channels=1,
                            dtype='int16',
                            device=parse_sd_device(self.cfg.output_device),
                        ) as stream:
                            for idx, chunk in enumerate(stream_gen):
                                chunk_i16 = float32_to_int16(chunk)
                                stream.write(chunk_i16)
                                logger.debug("EdgeTTS streaming chunk #%d len=%d", idx, len(chunk_i16))
                    except RuntimeError as exc:
                        logger.warning("EdgeTTS streaming chunk failed: %s", exc)
                        stream_failed = True
                    except Exception as exc:
                        logger.warning("EdgeTTS streaming playback error: %s", exc)
                        stream_failed = True
                    else:
                        return True
                else:
                    stream_failed = True
            if not use_stream or stream_failed:
                vc_start = time.perf_counter()
                result = self.voice_changer.convert(samples, self.cfg.out_sr)
                vc_elapsed = time.perf_counter() - vc_start
                if result:
                    converted_f32, vc_sr = result
                    if vc_sr and vc_sr > 0 and vc_sr != target_sr:
                        converted_play = resample_audio(converted_f32, vc_sr, target_sr)
                    else:
                        converted_play = converted_f32
                    play_samples = float32_to_int16(converted_play)
                    vc_success = True
                    logger.debug("EdgeTTS VC convert succeeded in %.3f s (samples=%d)", vc_elapsed, len(converted_play))
                else:
                    vc_success = False
                    logger.debug("EdgeTTS VC convert failed in %.3f s", vc_elapsed)
        if self.voice_changer is not None and vc_success is False:
            if fallback_device:
                try:
                    sd.play(samples, samplerate=target_sr, device=parse_sd_device(fallback_device))
                    sd.wait()
                except Exception as exc:
                    logger.warning("EdgeTTS fallback playback failed: %s", exc)
                return False
            play_samples = samples

        sd.play(play_samples, samplerate=target_sr, device=parse_sd_device(self.cfg.output_device))
        sd.wait()
        return vc_success
