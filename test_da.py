from faster_whisper import WhisperModel
import sys
try:
    print("Loading Danish model...")
    model = WhisperModel('pluttodk/roest-v3-whisper-1.5b-ct2', device='cuda', compute_type='float16')
    print("Success!")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
