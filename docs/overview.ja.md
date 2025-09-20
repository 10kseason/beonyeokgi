# リアルタイム音声翻訳 – 技術概要 (日本語)

リアルタイム音声翻訳 (プレビュー) (Windows)
⚠️ プレビューに関する警告: このプロジェクトは、評価のみを目的とした早期プレビュー版です。仕様の破壊的変更、機能の不足、不安定性が予想されますので、製品版やミッションクリティカルなワークフローでのご使用はお控えください。

ご使用になるには、すべてのドキュメントをお読みください！

CUDAのみを使用したい場合 = for_nvidia-gpu-run.bat を実行し、

requirements.txt を requirements_nvidia.txt に置き換えてください。

そして、設定を変更してください。
local.toml ファイル内

Ini, TOML

[app]
compute_mode = "cuda"
.......
そして

settings.toml ファイル内

Ini, TOML

[asr]
device = "cuda"
compute_type = "float16"
.......
そして

Ini, TOML

[kokoro]
backend = "pytorch"
device = "cuda"

## 概要
本アプリケーションは選択した入力デバイスからモノラル16bit PCM音声を取得し、発話区間を検出してからWhisperベースの音声認識を実行します。韓国語フィラー除去やHelsinki-NLPによるko/ja/zh→英語訳、またはGUIでLLMモードを有効にした際はOllama／LM Studioバックエンドによる翻訳を行い、Kokoro 82Mによる音声合成を再生します。音声変換(VCC)連携は任意で、UIとは別スレッドのパイプラインが全体の状態と遅延を管理します。【F:src/audio_io.py†L10-L76】【F:src/vad.py†L19-L177】【F:src/asr.py†L29-L147】【F:src/pipeline.py†L331-L639】【F:src/llm_translator.py†L1-L96】

## 信号フロー
1. **キャプチャ:** `MicReader` が `sounddevice.RawInputStream` を用いて固定サイズのフレームをキューに蓄積し、コールバックがブロックされないようにします。【F:src/audio_io.py†L10-L76】
2. **区間抽出:** `VADSegmenter` はフレーム長・感度・最大発話時間・チャンクストリーミングを設定できる WebRTC VAD を利用します。`ForcedSegmenter` を有効化すると、RMS レベルを監視して長い発話を強制的に切り出します。【F:src/vad.py†L19-L177】【F:src/pipeline.py†L405-L545】
3. **前処理:** `AudioPreprocessor` は可能なら ffmpeg で16kHzへリサンプリングし、90Hz ハイパスと 7.2kHz ローパスを適用後、ラウドネス正規化・トゥルーピーク制限・緩やかなコンプレッションを行います。【F:src/preprocess.py†L192-L340】
4. **認識:** ASR ラッパーは PCM を float32 に変換して Faster-Whisper を呼び出し、韓国語フィラーを削除した後に CJK 文字が残る場合は Helsinki-NLP 翻訳器、または LLM モード有効時は Ollama／LM Studio バックエンドで英語へ変換します。【F:src/asr.py†L29-L117】【F:src/pipeline.py†L556-L639】【F:src/translator.py†L10-L66】【F:src/llm_translator.py†L1-L96】
5. **合成:** `KokoroTTS` は短い文をバッチ化してクロスフェードしながら再生し、メイン出力・パススルー用仮想マイク・VC失敗時のフォールバックデバイスへ同時にルーティングできます。【F:src/tts_kokoro.py†L120-L246】【F:src/tts_kokoro.py†L320-L366】【F:src/tts_kokoro.py†L428-L553】
6. **音声変換 (任意):** 有効化すると `VoiceChangerClient` が生成され、Ookada VC Client API に int16 チャンクを送信し、サンプルレートの協調やストリーミング、WAV 保存を処理します。【F:src/pipeline.py†L706-L739】【F:src/voice_changer_client.py†L21-L184】
7. **状態とUI:** `SharedState` が言語・プリセット・デバイス・遅延を同期し、`TranslatorUI` がデバイス切替・プリセット・計算モード・Kokoroミラーをリアルタイムに操作できる Tk UI を提供します。【F:src/pipeline.py†L144-L695】【F:src/ui.py†L10-L195】

## セットアップ
### 依存パッケージ
Faster-Whisper、WebRTC VAD、sounddevice、Kokoro ランタイム、Edge/Piper TTS、PyTorch などは `requirements.txt` にまとめてあります。デフォルトの extra-index は CPU 版 PyTorch を指しているので、GPU 版が必要なら編集してください。【F:requirements.txt†L1-L21】

### Windows ブートストラップ
`for_vene.bat` は `.venv` を作成または再利用し、ビルドツールを更新してから依存パッケージをインストールします。手動セットアップの参考になるコマンドもログに表示されます。【F:for_vene.bat†L1-L47】

### 手動セットアップ手順
1. Python 3.10〜3.12 の仮想環境を作成し、アクティブ化します。
2. `pip install --upgrade pip setuptools wheel` を実行します。
3. `pip install -r requirements.txt` を実行し、必要に応じて PyTorch のインデックス URL を変更します。【F:for_vene.bat†L22-L45】【F:requirements.txt†L1-L21】

### オプションツール
- ffmpeg を PATH に追加すると、Kokoro/pydub で MP3 を扱え、前処理の ffmpeg リサンプラーも利用できます。【F:src/preprocess.py†L239-L295】
- VB-CABLE などの仮想オーディオデバイスは GUI のデバイス選択ダイアログから指定でき、配信ソフト等へルーティングできます。【F:src/main.py†L37-L188】

## 実行方法
1. 仮想環境をアクティブ化し、`python -m src.main` を実行するか `run.bat` を利用して自動的に起動します。【F:run.bat†L1-L5】【F:src/main.py†L571-L588】
2. 初回起動時はマイクと出力デバイスを選択し、設定は `config/local.toml` に保存されます。UI では計算モード切替、Kokoro パススルー、遅延ゲージを確認できます。【F:src/main.py†L404-L555】【F:src/pipeline.py†L144-L318】【F:src/ui.py†L10-L195】
3. CLI ヘルパー: `python -m src.main --list-devices` でデバイス一覧を表示し、`--list-voices` で Edge TTS ボイスを取得できます。【F:src/main.py†L341-L367】【F:src/main.py†L558-L588】

## 設定の要点
`config/settings.toml` を編集するか `config/local.toml` で上書きして挙動を調整します。
- `[device]`: 既定のサンプルレートとデバイス ID。UI が保存／復元します。【F:config/settings.toml†L1-L5】【F:src/main.py†L217-L305】
- `[asr]`: Whisper モデル、デバイス／演算形式、言語ロック、デコード設定。【F:config/settings.toml†L7-L15】【F:src/asr.py†L29-L147】
- `[vad]`, `[vad.force]`: VAD 感度、サイレンスパッド、チャンクストリーミング、強制分割の閾値。【F:config/settings.toml†L17-L30】【F:src/vad.py†L19-L177】【F:src/pipeline.py†L405-L545】
- `[tts]`, `[kokoro]`: Kokoro バックエンド／スピーカー、バッチ／クロスフェード設定、出力音量、パススルーデバイス。【F:config/settings.toml†L33-L58】【F:src/tts_kokoro.py†L120-L553】
- `[app]`: 既定プリセット (`latency` / `accuracy`) と計算モード。実行時に CUDA の有無を確認して切り替えます。【F:config/settings.toml†L60-L61】【F:src/main.py†L369-L530】
- `[voice_changer]`: 既定で `false` のため VCC 出力は無効です。`enabled = true` にするとエンドポイントやサンプルレート、ストリームチャンク、フォールバック出力を設定できます。【F:config/settings.toml†L67-L79】【F:src/pipeline.py†L706-L739】【F:src/voice_changer_client.py†L21-L184】

## VCC (Voice Changer Client) に関する注意
`voice_changer.enabled` が `false` のままでは Kokoro 音声は VC クライアントへ送信されません。無効時は Kokoro 音声のみ再生され、有効化すると HTTP 変換リクエストと原音／変換後 WAV の保存が行われます。【F:config/settings.toml†L67-L79】【F:src/tts_kokoro.py†L320-L366】【F:src/voice_changer_client.py†L113-L184】

## 拡張のヒント
- `--compute-mode auto|cpu|cuda` を指定すると CUDA 利用可否を自動判定し、適切なモードに切り替えます。【F:src/main.py†L369-L530】
- `LANGUAGE_MODELS` と UI のコンボボックスを拡張すれば新しい言語に対応できます。共有状態とパイプラインは既に動的な言語切替をサポートしています。【F:src/translator.py†L10-L73】【F:src/pipeline.py†L144-L655】【F:src/ui.py†L84-L153】
- `src/tts_edge.py` や `src/tts_piper.py` を参考に別の TTS バックエンドを追加したり、`KokoroTTS` を拡張できます。
