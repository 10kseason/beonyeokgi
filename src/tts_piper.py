from __future__ import annotations

from dataclasses import dataclass
import io
from typing import Optional, TYPE_CHECKING

import logging
import time
import numpy as np
import sounddevice as sd
from pydub import AudioSegment

from .utils import float32_to_int16, parse_sd_device, resample_audio

if TYPE_CHECKING:
    from .voice_changer_client import VoiceChangerClient

logger = logging.getLogger("vc-translator.tts.piper")


@dataclass
class PiperConfig:
    model_path: str
    out_sr: int = 48000
    pace: float = 1.0
    volume_db: float = 0.0
    output_device: Optional[str] = None


class PiperTTS:
    def __init__(
        self,
        model_path: str,
        out_sr: int = 48000,
        pace: float = 1.0,
        volume_db: float = 0.0,
        output_device: Optional[str] = None,
        voice_changer: Optional["VoiceChangerClient"] = None,
    ) -> None:
        self.cfg = PiperConfig(model_path=model_path, out_sr=out_sr, pace=pace, volume_db=volume_db,
                               output_device=output_device)
        self.voice_changer = voice_changer
        try:
            import piper
            self._piper = piper
        except Exception as e:
            self._piper = None
            self._import_error = e

        self._tts = None
        if self._piper is not None:
            # Lazy load on first use with model path
            pass

    def _ensure_tts(self):
        if self._piper is None:
            raise RuntimeError(
                f"piper-tts not available. Install and provide a valid model. Original error: {self._import_error}"
            )
        if self._tts is None:
            # piper.load supports .onnx or .tar
            self._tts = self._piper.PiperVoice.load(self.cfg.model_path)

    def synth_to_play(self, text: str) -> Optional[bool]:
        if not text:
            return None
        self._ensure_tts()
        wav_bytes = self._tts.synthesize(text, length_scale=max(0.1, 1.0 / max(self.cfg.pace, 0.1)))
        seg = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
        if self.cfg.volume_db:
            seg = seg.apply_gain(self.cfg.volume_db)
        seg = seg.set_channels(1).set_frame_rate(self.cfg.out_sr)
        samples = np.array(seg.get_array_of_samples()).astype(np.int16)
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
                                logger.debug("Piper streaming chunk #%d len=%d", idx, len(chunk_i16))
                    except RuntimeError as exc:
                        logger.warning("Piper streaming chunk failed: %s", exc)
                        stream_failed = True
                    except Exception as exc:
                        logger.warning("Piper streaming playback error: %s", exc)
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
                    logger.debug("Piper VC convert succeeded in %.3f s (samples=%d)", vc_elapsed, len(converted_play))
                else:
                    vc_success = False
                    logger.debug("Piper VC convert failed in %.3f s", vc_elapsed)
        if self.voice_changer is not None and vc_success is False:
            if fallback_device:
                try:
                    sd.play(samples, samplerate=target_sr, device=parse_sd_device(fallback_device))
                    sd.wait()
                except Exception as exc:
                    logger.warning("Piper fallback playback failed: %s", exc)
                return False
            play_samples = samples

        sd.play(play_samples, samplerate=target_sr, device=parse_sd_device(self.cfg.output_device))
        sd.wait()
        return vc_success
