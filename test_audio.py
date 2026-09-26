import urllib.request
from faster_whisper import WhisperModel

print("Downloading test audio...")
urllib.request.urlretrieve("https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav", "test.wav")

print("Loading model...")
model = WhisperModel('pluttodk/hviske-tiske', device='cuda', compute_type='int8_float16')
print("Transcribing...")
segments, info = model.transcribe("test.wav", language="da")

count = 0
for s in segments:
    print(f"OUTPUT: {s.text}")
    count += 1
print(f"Done! {count} segments generated.")
