# Realtime Speech Translator – Technical Overview (English)

## Overview
The application records mono 16-bit PCM audio from a selectable input device, detects speech boundaries, performs Whisper-based automatic speech recognition (ASR) with Korean filler removal and optional Helsinki-NLP or LLM-backed translation, then plays back Kokoro 82M speech synthesis with optional voice conversion. All runtime state is coordinated through a background pipeline that keeps the UI responsive and tracks end-to-end latency.【F:src/audio_io.py†L10-L76】【F:src/vad.py†L19-L177】【F:src/asr.py†L29-L147】【F:src/pipeline.py†L331-L639】【F:src/llm_translator.py†L1-L96】

## Signal Flow
1. **Capture:** `MicReader` uses a `sounddevice.RawInputStream` to queue fixed-size frames from the configured input device, allowing the pipeline thread to read audio without blocking the callback.【F:src/audio_io.py†L10-L76】
2. **Segmentation:** `VADSegmenter` applies WebRTC VAD with configurable frame, aggressiveness, max utterance duration, and optional chunk streaming. When enabled, `ForcedSegmenter` monitors RMS levels to force a cut if VAD misses the end of long or loud phrases.【F:src/vad.py†L19-L177】【F:src/pipeline.py†L405-L545】
3. **Preprocessing:** `AudioPreprocessor` resamples to 16 kHz (ffmpeg when available, otherwise an internal resampler), runs 90 Hz high-pass and 7.2 kHz low-pass biquads, and applies loudness normalization, true-peak limiting, and gentle compression before ASR.【F:src/preprocess.py†L192-L340】
4. **Recognition:** The ASR wrapper converts PCM bytes to float32, resamples, invokes Faster-Whisper with configurable model/beam/temperature, strips Korean fillers, and optionally performs a second pass with Helsinki-NLP translators for ko/ja/zh → en—or, when the GUI’s LLM mode is enabled, via an Ollama/LM Studio backend—if Whisper leaves CJK characters.【F:src/asr.py†L29-L117】【F:src/pipeline.py†L556-L639】【F:src/translator.py†L10-L66】【F:src/llm_translator.py†L1-L96】
5. **Synthesis:** `KokoroTTS` batches short sentences, crossfades overlapping utterances, and can mirror audio to the main output, a passthrough virtual mic, and a fallback device when voice conversion fails.【F:src/tts_kokoro.py†L120-L246】【F:src/tts_kokoro.py†L320-L366】【F:src/tts_kokoro.py†L428-L553】
6. **Voice conversion (optional):** If enabled, the pipeline instantiates `VoiceChangerClient`, which uploads int16 chunks to an Ookada VC Client endpoint, resamples to negotiated rates, supports streaming mode, and writes optional WAV traces.【F:src/pipeline.py†L706-L739】【F:src/voice_changer_client.py†L21-L184】
7. **State/UI:** `SharedState` synchronizes requested vs. active language, preset, devices, and latency; `TranslatorUI` lets the user change devices, presets, compute mode, and Kokoro mirroring while the pipeline applies changes live.【F:src/pipeline.py†L144-L695】【F:src/ui.py†L10-L195】

## Setup
### Dependencies
All Python requirements—including Faster-Whisper, WebRTC VAD wheels, sounddevice, Kokoro runtimes, Edge/Piper TTS, and PyTorch—are pinned in `requirements.txt`. The default extra index installs CPU PyTorch; edit or remove it for CUDA/ROCm builds.【F:requirements.txt†L1-L21】

### Windows bootstrap script
`for_vene.bat` provisions or reuses a `.venv`, upgrades packaging tools, and installs dependencies. It shows the commands needed for manual setup if you prefer PowerShell or cmd steps individually.【F:for_vene.bat†L1-L47】

### Manual setup outline
1. Create and activate a Python 3.10–3.12 virtual environment.
2. `pip install --upgrade pip setuptools wheel`.
3. `pip install -r requirements.txt` (adjust the PyTorch wheel index if targeting GPU builds).【F:for_vene.bat†L22-L45】【F:requirements.txt†L1-L21】

### Optional tools
- Install ffmpeg and add it to `PATH` so Kokoro and pydub can decode MP3 assets and so the preprocessor can use ffmpeg-based resampling.【F:src/preprocess.py†L239-L295】
- VB-CABLE or a similar virtual audio device can be selected through the device pickers for routing Kokoro output to conferencing software.【F:src/main.py†L37-L188】

## Running the Application
1. Activate the virtual environment (`.\.venv\Scripts\activate` on Windows) and start the UI with `python -m src.main` or simply run `run.bat`, which performs the activation automatically.【F:run.bat†L1-L5】【F:src/main.py†L571-L588】
2. On first launch the GUI prompts for microphone and output devices; selections persist to `config/local.toml`. The UI also exposes compute mode toggles, Kokoro passthrough mirroring, and latency feedback.【F:src/main.py†L404-L555】【F:src/pipeline.py†L144-L318】【F:src/ui.py†L10-L195】
3. CLI helpers: `python -m src.main --list-devices` enumerates sound devices, while `--list-voices` fetches Edge TTS voices when the optional package is available.【F:src/main.py†L341-L367】【F:src/main.py†L558-L588】

## Configuration Highlights
Edit `config/settings.toml` (and override in `config/local.toml`) to control runtime behavior:
- `[device]`: default input/output sample rates and device identifiers saved by the UI.【F:config/settings.toml†L1-L5】【F:src/main.py†L217-L305】
- `[asr]`: Whisper model, device/compute type, language lock, and decoding hyperparameters.【F:config/settings.toml†L7-L15】【F:src/asr.py†L29-L147】
- `[vad]` and `[vad.force]`: VAD aggressiveness, silence padding, chunk streaming, and forced segmentation thresholds.【F:config/settings.toml†L17-L30】【F:src/vad.py†L19-L177】【F:src/pipeline.py†L405-L545】
- `[tts]` and `[kokoro]`: Kokoro backend, speaker, batching/crossfade timings, output volume, and optional passthrough input device for virtual-mic mirroring.【F:config/settings.toml†L33-L58】【F:src/tts_kokoro.py†L120-L553】
- `[app]`: default preset (`latency` or `accuracy`) and compute mode selection that maps to CPU or CUDA at runtime.【F:config/settings.toml†L60-L61】【F:src/main.py†L369-L530】
- `[voice_changer]`: Disabled by default—VCC export is inactive until you set `enabled = true`. When activated you can choose endpoints, sample rates, streaming chunk length, and fallback playback device.【F:config/settings.toml†L67-L79】【F:src/pipeline.py†L706-L739】【F:src/voice_changer_client.py†L21-L184】

## Voice Conversion / VCC Notes
Because `voice_changer.enabled` defaults to `false`, the application does not send Kokoro audio to the Ookada VC Client unless you explicitly enable it. When disabled the pipeline simply plays Kokoro audio to the configured outputs; enabling it introduces HTTP conversion requests and optional WAV exports for the raw and converted audio.【F:config/settings.toml†L67-L79】【F:src/tts_kokoro.py†L320-L366】【F:src/voice_changer_client.py†L113-L184】

## Extensibility Tips
- Adjust compute mode with `--compute-mode auto|cpu|cuda`; the pipeline verifies CUDA availability and falls back to CPU when necessary.【F:src/main.py†L369-L530】
- Add new language targets by extending `LANGUAGE_MODELS` and the UI combobox options; the shared state and pipeline already support dynamic language switching.【F:src/translator.py†L10-L73】【F:src/pipeline.py†L144-L655】【F:src/ui.py†L84-L153】
- Integrate alternative TTS backends by adapting `KokoroTTS` or reusing the optional `EdgeTTS` and `PiperTTS` implementations under `src/tts_edge.py` and `src/tts_piper.py`.
