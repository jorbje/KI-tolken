from huggingface_hub import snapshot_download

models = [
    "PierreMesure/kb-whisper-large-ct2",
    "pluttodk/roest-v3-whisper-1.5b-ct2"
]

for m in models:
    print(f"Laster ned {m}...")
    snapshot_download(m)
    print(f"{m} ferdig nedlastet.")
