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
- Whisper `large-v3-turbo` on CPU (`int8` compute) so the ASR step stays responsive without a GPU
- VAD 30 ms frames, aggressiveness 2, silence tail 300 ms
- TTS engine `edge` with `en-US-AriaNeural`; switch to `kokoro` or `piper` for offline voices

Run

- Activate venv: `.\.venv\Scripts\activate`
- Start: `python -m src.main` (GUI will prompt for input device unless --input-device is provided)
- Or use helper: `run.bat`

CLI Helpers

- List audio devices: `python -m src.main --list-devices`
- List Edge TTS voices: `python -m src.main --list-voices`
- Override common settings at launch:
  - `--engine edge|piper|kokoro`
  - `--voice en-US-AriaNeural` (Edge)
  - `--piper-model path\to\model.onnx[.tar]`
  - `--kokoro-model hexgrad/Kokoro-82M`
  - `--kokoro-speaker af_bella`
  - `--kokoro-backend auto|pytorch|onnx`
  - `--kokoro-device auto` / `--kokoro-device cuda` / `--kokoro-device dml` / `--kokoro-device cpu`
  - `--kokoro-provider CUDAExecutionProvider` (repeat for multiple providers)
  - `--input-device "Your Mic"` or index
  - `--output-device "CABLE Input"` or index
  - `--pace 1.1`  `--volume-db -6`
  - Use `--config` to point at a different TOML file

Kokoro 82M GPU TTS
------------------

- Set `[tts].engine = "kokoro"` in `config/settings.toml` to route synthesized speech through Kokoro 82M.
- Backend auto-probing is enabled by default (`[kokoro].backend = "auto"`, `[kokoro].device = "auto"`). At startup the app benchmarks CUDA → ROCm → DirectML → CPU with a one-second dummy sentence on the available runtimes (PyTorch or ONNX) and locks in the fastest option. If the active backend spikes above the running average by more than `2σ` for three consecutive sentences it automatically falls back to the next fastest candidate.
- Device priority matches the hardware:
  - NVIDIA GPUs stick to PyTorch + CUDA (`use_half = true` keeps FP16 inference).
  - AMD on Linux prefers ONNX + ROCm; Windows systems without NVIDIA prefer ONNX + DirectML.
  - CPU-only mode is kept as a final fallback when no GPU runtime is available.
- To pin a specific configuration, override `[kokoro].backend`, `[kokoro].device`, or `[kokoro].onnx_providers` and rerun. CLI overrides such as `--kokoro-backend onnx` or `--kokoro-provider DmlExecutionProvider` are still supported.
- New pacing controls keep playback smooth: the engine batches sub‑0.5 s sentences until the group reaches roughly 0.8–1.2 s, keeps a 100–150 ms crossfade buffer between utterances, and flushes short clips automatically when speech pauses. The tunables (`short_threshold_ms`, `min_batch_ms`, `max_batch_ms`, `crossfade_ms`, etc.) live under `[kokoro]` in the config if you need to tweak them.
- Kokoro playback is skipped automatically if Whisper returns Korean (non-translated) text, preventing Korean sentences from being spoken in the English voice.

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

