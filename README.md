Realtime Speech Translator (Windows)

- Real-time mic capture ??VAD chunking ??Whisper translate (ko/ja/zh ??en) ??Kokoro 82M TTS playback.
- Deterministic 16 kHz preprocessing (loudness normalization, 90 Hz high-pass, 7.2 kHz low-pass) and filler removal for Korean (????Í∑∏Îãà?.
- Single-page desktop UI showing mic/output/Kokoro routing, fixed language pairs (ko/ja/zh ??EN), a latency gauge, and preset selector.
- Optimized for Windows + VB-CABLE to feed TTS into a Discord voice channel.

Quickstart

- Python 3.10?3.12 recommended (examples assume 3.10)
- Windows 10/11 64-bit
- Optional GPU: install CUDA and use faster-whisper with float16

Setup

1) Bootstrap the virtual environment (runs safely multiple times):
   - Double-click `for_vene.bat` or run it from `cmd`. The script creates/updates `.venv`, upgrades `pip/setuptools/wheel`, and installs everything from `requirements.txt` (CPU PyTorch by default).
   - GPU users can edit the `--extra-index-url` line in `requirements.txt` before running the script to target a CUDA/ROCm build from the official PyTorch index.
   - Kokoro PyTorch/ONNX runtimes are now included in `requirements.txt`, so the default TTS backend is ready after this step. Adjust or pin the package versions there if you need specific builds.
2) Audio routing (VB-CABLE):
   - Set default playback to `CABLE Input` (or route with Voicemeeter)
   - In Discord, set Input device to `CABLE Output`
   - Disable echo cancellation / noise suppression / AGC in Discord
3) Optional: install ffmpeg if you plan to use pydub-based features (voice changer previews, etc.):
   - Download ffmpeg and add its `bin` to PATH

Kokoro backend
--------------

- The environment bootstrap installs the official Kokoro runtime packages (PyTorch and ONNX variants) from PyPI. If you prefer a custom build, edit `requirements.txt` before running the setup script or install your desired wheels afterwards.

Config

Edit `config/settings.toml` to adjust devices, model size, presets, and Kokoro tuning.
Defaults:
- Capture at the device sample rate, preprocess to 16 kHz mono with loudnorm (I=-16, TP=-1.5, LRA=11) and 90??200 Hz band-pass before Whisper.
- Whisper `large-v3-turbo` on CPU (`int8` compute) so the ASR step stays responsive without a GPU.
- VAD 30 ms frames, aggressiveness 2, silence tail 300 ms (streaming chunking comes from the preset values).
- Kokoro 82M TTS with 120 ms crossfade and sentence queueing.
- `[app].preset = "latency"` (1000??200 ms chunk, 250 ms hop, beam 3). Switch to `"accuracy"` for longer chunks and higher beam width.

Run

- Activate venv: `.\.venv\Scripts\activate`
- Start: `python -m src.main` (a single-window UI appears showing selected devices, fixed language toggle, preset buttons, and a live latency gauge)
- Or use helper: `run.bat`

CLI Helpers

- List audio devices: `python -m src.main --list-devices`
- List Edge TTS voices: `python -m src.main --list-voices`
- Override common settings at launch:
  - `--input-device "Your Mic"` or index
  - `--output-device "CABLE Input"` or index
  - `--language ko|ja|zh`
  - `--preset latency|accuracy`
  - Use `--config` to point at a different TOML file

Kokoro 82M GPU TTS
------------------

- Kokoro 82M is the only TTS path (no Edge/Piper fallback). `[tts].engine` remains `"kokoro"` for completeness.
- Backend auto-probing is enabled by default (`[kokoro].backend = "auto"`, `[kokoro].device = "auto"`). At startup the app benchmarks CUDA ??ROCm ??DirectML ??CPU with a one-second dummy sentence on the available runtimes (PyTorch or ONNX) and locks in the fastest option. If the active backend spikes above the running average by more than `2?` for three consecutive sentences it automatically falls back to the next fastest candidate.
- Pick a Kokoro voice and set `[kokoro].speaker` (defaults to "af_bella"). Voice files follow the repository naming (`af_*`, `am_*`, `bf_*`, etc.) in `hexgrad/Kokoro-82M/voices`.
- Device priority matches the hardware:
  - NVIDIA GPUs stick to PyTorch + CUDA (`use_half = true` keeps FP16 inference).
  - AMD on Linux prefers ONNX + ROCm; Windows systems without NVIDIA prefer ONNX + DirectML.
  - CPU-only mode is kept as a final fallback when no GPU runtime is available.
- Mirror Kokoro playback to a virtual microphone by setting `[kokoro].passthrough_input_device` to the input device index/name (e.g. the 3rd input device). Audio continues to play on the main output while also feeding the specified input.
- To pin a specific configuration, override `[kokoro].backend`, `[kokoro].device`, or `[kokoro].onnx_providers` and rerun. CLI overrides such as `--kokoro-backend onnx` or `--kokoro-provider DmlExecutionProvider` are still supported.
- Built-in queueing keeps playback smooth: sentences are batched until the group reaches roughly 0.8??.2?s, a fixed 120?ms crossfade blends consecutive utterances, and short clips flush automatically when speech pauses. The tunables (`short_threshold_ms`, `min_batch_ms`, `max_batch_ms`, `crossfade_ms`, etc.) live under `[kokoro]` if you need to tweak them.
- Kokoro playback is skipped automatically if translation fails and non-English (Hangul/Kana/Han) characters remain after the fallback translator pass.
- `[kokoro].passthrough_input_device` can now be set directly from the UI ("Kokoro Ï∂úÎ†•" row) instead of editing the config file.

Voice Changer Integration

- Default run saves the raw Kokoro waveform to `cache/tts_original.wav`.
- Set `[voice_changer].enabled = true` in `config/settings.toml` to pipe each utterance into Ookada's VCClient at `http://localhost:18000` via the `/api/voice-changer/convert_chunk` endpoint.
- When enabled, the converted audio replaces the playback stream and is also written to `cache/tts_converted.wav`. The client auto-detects sample rates from `/api/configuration-manager/configuration`; override `input_sample_rate`/`output_sample_rate` if you need fixed values.
- Tune `timeout_sec`, `base_url`, or `endpoint` to match your VCClient deployment. If the HTTP call fails, the app logs the reason, optionally retries `/api/voice-changer/convert_chunk_bulk`, and falls back to the original TTS audio.
- Optional streaming: set `[voice_changer].stream_mode = true` to send smaller chunks (default 1000 ms, configure with `stream_chunk_ms`) so converted audio starts while later chunks process.

Notes

- Translation to English is enforced for the supported input languages (ko/ja/zh). Whisper handles the primary translation and, if residual CJK text remains, the app falls back to the Helsinki-NLP ko/ja/zh ??en models before speaking.
- If audio plays too loud/quiet, tune `[tts].volume_db` and `[stream].normalize_dbfs`.
- Whisper Ï≤?∑® Í∏∏Ïù¥Î•??òÎ¶¨Í≥??∂Îã§Î©?[vad].listen_max_sec, [vad].listen_silence_ms, chunk_min_ms, chunk_max_msÎ•?Ï°∞Ï†ï???∏Í∑∏Î®ºÌä∏ Î∂ÑÌï† ?úÏ†ê????∂ú ???àÏäµ?àÎã§.
- Set `[logging].level = "DEBUG"` to enable per-segment ASR/TTS timing logs while tuning performance.
- For CPU-only, set `[asr].device = "cpu"` and `compute_type = "int8"` or `"int8_float16"`.
- For Piper TTS, set `[tts].engine = "piper"` and provide `[tts].piper_model` path to a `.onnx` or `.tar` bundle, then install Piper models separately.
- Korean fillers (`??, `??, `Í∑∏Îãà?) are stripped before translation so Whisper/Kokoro only see meaningful speech.



