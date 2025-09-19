# 실시간 음성 번역기 – 기술 개요 (한국어)

## 개요
이 애플리케이션은 선택한 입력 장치에서 모노 16비트 PCM 오디오를 받아 말하기 구간을 검출한 뒤, 한국어 군더더기 제거가 포함된 Whisper 기반 음성 인식과 필요 시 Helsinki-NLP 번역을 수행하고, Kokoro 82M 음성 합성 결과를 재생합니다. 음성 변환(VC) 연동은 옵션이며, 모든 상태 관리는 UI와 분리된 파이프라인 스레드가 담당해 전체 지연 시간을 추적합니다.【F:src/audio_io.py†L10-L76】【F:src/vad.py†L19-L177】【F:src/asr.py†L29-L147】【F:src/pipeline.py†L331-L595】

## 신호 흐름
1. **캡처:** `MicReader`가 `sounddevice.RawInputStream`을 이용해 고정 크기 프레임을 큐에 쌓아, 콜백이 막히지 않도록 합니다.【F:src/audio_io.py†L10-L76】
2. **분할:** `VADSegmenter`는 프레임 길이, 민감도, 최대 발화 길이, 청크 스트리밍을 설정할 수 있는 WebRTC VAD를 사용합니다. `ForcedSegmenter`를 켜면 RMS 기반으로 VAD가 놓친 긴 발화를 강제로 잘라냅니다.【F:src/vad.py†L19-L177】【F:src/pipeline.py†L405-L545】
3. **전처리:** `AudioPreprocessor`는 ffmpeg(가능하면) 또는 내부 보간기로 16kHz로 리샘플링한 뒤 90Hz 하이패스와 7.2kHz 로우패스를 거치고, 라우드니스 정규화·트루픽 제한·완만한 컴프레서를 적용합니다.【F:src/preprocess.py†L192-L340】
4. **인식:** ASR 래퍼는 PCM을 float32로 바꾼 뒤 Faster-Whisper를 호출하고, 한국어 군더더기를 제거한 뒤 필요하면 ko/ja/zh → en Helsinki-NLP 모델로 후처리합니다.【F:src/asr.py†L29-L117】【F:src/pipeline.py†L556-L639】【F:src/translator.py†L10-L66】
5. **합성:** `KokoroTTS`는 짧은 문장을 모아 배치 처리하고, 교차 페이드로 이어 붙이며, 메인 출력·패스스루 가상 마이크·VC 실패 시 폴백 장치로 동시에 출력할 수 있습니다.【F:src/tts_kokoro.py†L120-L246】【F:src/tts_kokoro.py†L320-L366】【F:src/tts_kokoro.py†L428-L553】
6. **음성 변환(선택):** 활성화하면 파이프라인이 `VoiceChangerClient`를 생성하여 Ookada VC Client API에 int16 청크를 업로드하고, 샘플레이트 협상·스트리밍·WAV 저장을 처리합니다.【F:src/pipeline.py†L706-L739】【F:src/voice_changer_client.py†L21-L184】
7. **상태/GUI:** `SharedState`가 언어·프리셋·장치·지연 시간을 동기화하고, `TranslatorUI`는 장치/프리셋/연산 모드/Kokoro 미러링을 실시간으로 변경할 수 있는 Tk 인터페이스를 제공합니다.【F:src/pipeline.py†L144-L695】【F:src/ui.py†L10-L195】

## 설치
### 필수 패키지
Faster-Whisper, WebRTC VAD, sounddevice, Kokoro 런타임, Edge/Piper TTS, PyTorch 등 모든 의존성은 `requirements.txt`에 정의되어 있습니다. 기본 extra-index는 CPU용 PyTorch를 설치하므로, GPU 빌드가 필요하면 수정하거나 제거하세요.【F:requirements.txt†L1-L21】

### Windows 자동 구성 스크립트
`for_vene.bat`은 `.venv`를 생성/재사용하고 패키징 도구를 업그레이드한 뒤 요구 패키지를 설치합니다. PowerShell/cmd로 수동 실행하고 싶을 때 참고용 명령도 모두 표시됩니다.【F:for_vene.bat†L1-L47】

### 수동 설치 절차
1. Python 3.10–3.12 가상환경을 만들고 활성화합니다.
2. `pip install --upgrade pip setuptools wheel` 실행.
3. `pip install -r requirements.txt` (GPU용 PyTorch를 원하면 인덱스 URL을 조정).【F:for_vene.bat†L22-L45】【F:requirements.txt†L1-L21】

### 선택 도구
- ffmpeg를 PATH에 추가하면 Kokoro/pydub MP3 디코드와 전처리의 ffmpeg 리샘플링을 사용할 수 있습니다.【F:src/preprocess.py†L239-L295】
- VB-CABLE과 같은 가상 오디오 장치는 GUI 장치 선택기에서 지정해 Discord 등으로 라우팅할 수 있습니다.【F:src/main.py†L37-L188】

## 실행 방법
1. 가상환경을 활성화한 뒤 `python -m src.main` 또는 자동 활성화를 포함한 `run.bat`을 실행합니다.【F:run.bat†L1-L5】【F:src/main.py†L571-L588】
2. 최초 실행 시 마이크와 출력 장치를 선택하며, 이후 값은 `config/local.toml`에 저장됩니다. UI에서는 연산 모드 전환, Kokoro 패스스루 장치, 지연 게이지를 실시간으로 확인할 수 있습니다.【F:src/main.py†L404-L555】【F:src/pipeline.py†L144-L318】【F:src/ui.py†L10-L195】
3. CLI 도구: `python -m src.main --list-devices`로 오디오 장치를 확인하고, `--list-voices`로 Edge TTS 보이스 목록을 조회할 수 있습니다.【F:src/main.py†L341-L367】【F:src/main.py†L558-L588】

## 설정 요약
`config/settings.toml`을 수정하거나 `config/local.toml`로 덮어써 세부 동작을 조정합니다.
- `[device]`: 기본 샘플레이트와 장치 ID. UI가 저장/불러오기 합니다.【F:config/settings.toml†L1-L5】【F:src/main.py†L217-L305】
- `[asr]`: Whisper 모델, 디바이스/정밀도, 언어 고정, 디코딩 옵션.【F:config/settings.toml†L7-L15】【F:src/asr.py†L29-L147】
- `[vad]`, `[vad.force]`: VAD 민감도, 침묵 패드, 청크 스트리밍, 강제 분할 임계값.【F:config/settings.toml†L17-L30】【F:src/vad.py†L19-L177】【F:src/pipeline.py†L405-L545】
- `[tts]`, `[kokoro]`: Kokoro 백엔드/보이스, 배치·크로스페이드 타이밍, 출력 볼륨, 패스스루 장치 설정.【F:config/settings.toml†L33-L58】【F:src/tts_kokoro.py†L120-L553】
- `[app]`: 기본 프리셋(`latency`/`accuracy`)과 연산 모드 지정. 런타임에서 CUDA 가용성에 맞춰 조정됩니다.【F:config/settings.toml†L60-L61】【F:src/main.py†L369-L530】
- `[voice_changer]`: 기본값이 `false`라서 VCC 내보내기가 꺼져 있습니다. `enabled = true`로 바꾸면 엔드포인트, 샘플레이트, 스트리밍 청크 길이, 폴백 출력 장치를 설정할 수 있습니다.【F:config/settings.toml†L67-L79】【F:src/pipeline.py†L706-L739】【F:src/voice_changer_client.py†L21-L184】

## VCC(Voice Changer Client) 안내
`voice_changer.enabled`가 `false`이므로 기본 상태에서는 Kokoro 오디오가 VC 클라이언트로 전송되지 않습니다. 비활성화 시 Kokoro 음성만 바로 재생하며, 활성화하면 HTTP 변환 요청과 원본/변환 WAV 저장이 추가됩니다.【F:config/settings.toml†L67-L79】【F:src/tts_kokoro.py†L320-L366】【F:src/voice_changer_client.py†L113-L184】

## 확장 팁
- `--compute-mode auto|cpu|cuda` 인자를 사용하면 CUDA 지원 여부를 자동 확인해 적절한 모드로 전환합니다.【F:src/main.py†L369-L530】
- 새로운 언어를 추가하려면 `LANGUAGE_MODELS`와 UI 콤보박스를 확장하세요. 공유 상태와 파이프라인은 이미 동적 언어 전환을 지원합니다.【F:src/translator.py†L10-L73】【F:src/pipeline.py†L144-L655】【F:src/ui.py†L84-L153】
- `src/tts_edge.py`, `src/tts_piper.py`의 예시처럼 다른 TTS 백엔드를 연결하거나 `KokoroTTS`를 확장할 수 있습니다.
