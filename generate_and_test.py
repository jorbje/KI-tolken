import asyncio
import edge_tts
from faster_whisper import WhisperModel
import sys

async def main():
    # 1. Generate Danish Audio
    print("Generating test audio...")
    communicate = edge_tts.Communicate("Dette er en formel test. Håber det virker! Hvis ikke, må vi se hvad der er galt.", "da-DK-ChristelNeural")
    await communicate.save("test_da.mp3")
    print("Audio saved.")
    
    # 2. Transcribe
    print("Loading model...")
    model = WhisperModel('pluttodk/hviske-tiske', device='cuda', compute_type='int8_float16')
    print("Transcribing (with prompt)...")
    segments, info = model.transcribe("test_da.mp3", language="da", initial_prompt="Dette er en formel, velskrevet og grammatisk korrekt udskrift, helt uden mundtlige fyldord, tøven eller stammen.")
    print("Output (with prompt):")
    for s in segments:
        print(f" -> {s.text}")

    print("Transcribing (without prompt)...")
    segments, info = model.transcribe("test_da.mp3", language="da")
    print("Output (without prompt):")
    for s in segments:
        print(f" -> {s.text}")
        
    print("Transcribing (without language token)...")
    segments, info = model.transcribe("test_da.mp3")
    print("Output (without language token):")
    for s in segments:
        print(f" -> {s.text}")

if __name__ == "__main__":
    asyncio.run(main())
