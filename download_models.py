import torch
from huggingface_hub import snapshot_download

# Whisper-modeller i aktiv bruk
models = [
    "NbAiLab/nb-whisper-large",
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    "PierreMesure/kb-whisper-large-ct2"
]

# Sjekk GPU for å velge riktig NLLB-200 modell
vram_gb = 0.0
gpu_name = "CPU / Ukjent"
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
    print(f"[Hardware] Fant {gpu_name} med {vram_gb} GB VRAM.")

if vram_gb >= 14.5:
    nllb_model = "facebook/nllb-200-3.3B"
    print(f"[Hardware] Ytelsesprofil valgt: Forhåndslaster NLLB-200 3.3B (flaggskip)...")
elif vram_gb >= 8.5:
    nllb_model = "facebook/nllb-200-1.3B"
    print(f"[Hardware] Standardprofil valgt: Forhåndslaster NLLB-200 1.3B...")
else:
    nllb_model = "facebook/nllb-200-distilled-600M"
    print(f"[Hardware] Kompaktprofil valgt: Forhåndslaster NLLB-200 600M...")

models.append(nllb_model)

for m in models:
    print(f"Laster ned {m}...")
    try:
        snapshot_download(m)
        print(f"{m} ferdig nedlastet.")
    except Exception as e:
        print(f"Feil ved nedlasting av {m}: {e}")

