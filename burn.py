# pip install faster-whisper edge-tts sounddevice webrtcvad pydub numpy
import asyncio, edge_tts, queue, time
import numpy as np
import sounddevice as sd
import webrtcvad
from pydub import AudioSegment
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
VAD_FRAME = 30  # ms
VOICE = "en-US-AriaNeural"  # 원하는 TTS 보이스

# 1) 마이크에서 말 구간만 뽑기
vad = webrtcvad.Vad(2)
audio_q = queue.Queue()

def record_loop(seconds=15):
    buf = []
    def cb(indata, frames, t, status):
        audio_q.put(bytes(indata))
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=int(SAMPLE_RATE*VAD_FRAME/1000),
                           channels=1, dtype='int16', callback=cb):
        start = time.time()
        while time.time()-start < seconds:
            time.sleep(0.01)
            pass
    return b''.join(list(audio_q.queue))

def vad_chunker(raw_bytes):
    frame_len = int(SAMPLE_RATE * VAD_FRAME / 1000) * 2
    frames = [raw_bytes[i:i+frame_len] for i in range(0, len(raw_bytes), frame_len)]
    voiced = []
    speaking = False
    for f in frames:
        if len(f) < frame_len: break
        isv = vad.is_speech(f, SAMPLE_RATE)
        if isv:
            voiced.append(f)
            speaking = True
        elif speaking:
            # 침묵 만나면 종료
            break
    return b"".join(voiced)

# 2) Whisper 로드(번역 모드)
model = WhisperModel("large-v3-turbo", compute_type="float16")  # RTX 4060 handles large-v3-turbo comfortably
def transcribe_translate(wav_np):
    segments, _ = model.transcribe(wav_np, language="ko", task="translate", vad_filter=True)
    txt = " ".join([s.text.strip() for s in segments])
    return txt.strip()

# 3) TTS 재생
async def speak(text):
    if not text: return
    tts = edge_tts.Communicate(text, VOICE)
    await tts.save("/mnt/data/tts.wav")  # 임시 저장
    # 시스템 기본 재생 장치가 CABLE Input이면 아래 한 줄로 충분
    AudioSegment.from_wav("/mnt/data/tts.wav").export("/mnt/data/tts_play.wav", format="wav")
    sd.play(np.array(AudioSegment.from_wav("/mnt/data/tts_play.wav").get_array_of_samples()), 48000)
    sd.wait()

# 4) 메인 루프: 말하면 영어로 읽어줌
async def main():
    print("말하세요. 문장 단위로 영어로 읽어줍니다. (Ctrl+C로 종료)")
    while True:
        raw = record_loop(seconds=6)               # 6초 윈도우
        chunk = vad_chunker(raw)
        if len(chunk) < SAMPLE_RATE*0.3*2:        # 0.3초 미만이면 무시
            continue
        # int16 -> float32
        audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)/32768.0
        text_en = transcribe_translate(audio_np)
        print("EN:", text_en)
        await speak(text_en)

if __name__ == "__main__":
    asyncio.run(main())
