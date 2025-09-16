Realtime Korean��English Speech Translator (Windows)

- Real-time mic capture �� VAD sentence segmentation �� Whisper translate (ko��en) �� TTS playback.
- Optimized for Windows + VB-CABLE to feed TTS into a Discord voice channel.

Quickstart

- Python 3.10?3.12 recommended (examples assume 3.10)
- Windows 10/11 64-bit
- Optional GPU: install CUDA and use faster-whisper with float16

Setup

1) Create venv and install deps:
   - `python -m venv .venv`
   - `.\.venv\Scripts\activate`
   - `pip install --upgrade pip wheel`
   - `pip install -r requirements.txt`
   - `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`  # needed for text translator
2) Audio routing (VB-CABLE):
   - Set default playback to `CABLE Input` (or route with Voicemeeter)
   - In Discord, set Input device to `CABLE Output`
   - Disable echo cancellation / noise suppression / AGC in Discord
3) If using Edge TTS (default), install ffmpeg so pydub can decode MP3:
   - Download ffmpeg and add its `bin` to PATH

Config

Edit `config/settings.toml` to adjust devices, model size, VAD, and TTS voice.
Defaults:
- Input 16 kHz mono; Output 48 kHz
- Whisper `large-v3-turbo`, `cuda` + `float16` when available
- VAD 30 ms frames, aggressiveness 2, silence tail 300 ms
- TTS engine `edge` with `en-US-AriaNeural`

Run

- Activate venv: `.\.venv\Scripts\activate`
- Start: `python -m src.main` (GUI will prompt for input device unless --input-device is provided)
- Or use helper: `run.bat`

CLI Helpers

- List audio devices: `python -m src.main --list-devices`
- List Edge TTS voices: `python -m src.main --list-voices`
- Override common settings at launch:
  - `--engine edge|piper`
  - `--voice en-US-AriaNeural` (Edge)
  - `--piper-model path\to\model.onnx[.tar]`
  - `--input-device "Your Mic"` or index
  - `--output-device "CABLE Input"` or index
  - `--pace 1.1`  `--volume-db -6`
  - Use `--config` to point at a different TOML file

Voice Changer Integration

- Default run saves the raw Edge TTS waveform to `cache/tts_original.wav`.
- Set `[voice_changer].enabled = true` in `config/settings.toml` to pipe each utterance into Ookada's VCClient at `http://localhost:18000` via the `/api/voice-changer/convert_chunk` endpoint.
- When enabled, the converted audio replaces the playback stream and is also written to `cache/tts_converted.wav`. The client auto-detects sample rates from `/api/configuration-manager/configuration`; override `input_sample_rate`/`output_sample_rate` if you need fixed values.
- Tune `timeout_sec`, `base_url`, or `endpoint` to match your VCClient deployment. If the HTTP call fails, the app logs the reason, optionally retries `/api/voice-changer/convert_chunk_bulk`, and falls back to the original TTS audio.
- Optional streaming: set `[voice_changer].stream_mode = true` to send smaller chunks (default 1000 ms, configure with `stream_chunk_ms`) so converted audio starts while later chunks process.

Notes

- Ko?En translation now uses Helsinki-NLP/opus-mt-ko-en via transformers/torch. If those packages are missing, Whisper Turbo will fall back to raw Korean text.
- If audio plays too loud/quiet, tune `[tts].volume_db` and `[stream].normalize_dbfs`.
- Set `[logging].level = "DEBUG"` to enable per-segment ASR/TTS timing logs while tuning performance.
- For CPU-only, set `[asr].device = "cpu"` and `compute_type = "int8"` or `"int8_float16"`.
- For Piper TTS, set `[tts].engine = "piper"` and provide `[tts].piper_model` path to a `.onnx` or `.tar` bundle, then install Piper models separately.

