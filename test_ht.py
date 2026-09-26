from faster_whisper import WhisperModel
import sys

try:
    print("Loading hviske-tiske...")
    model = WhisperModel('pluttodk/hviske-tiske', device='cuda', compute_type='int8_float16')
    print("Model loaded. Transcribing test_da.mp3 with language='da'...")
    segments, info = model.transcribe("test_da.mp3", language="da")
    
    count = 0
    for s in segments:
        print(f"OUTPUT: {s.text}")
        count += 1
        
    print(f"Done! {count} segments generated.")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
