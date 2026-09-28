# Prosjektoversikt
**KI-Tolken** er et 100 % lokalt, asynkront Python-system for live-tolking og teksting, tilpasset en frikirkelig kontekst. Det lytter til tale og gir lynraskt tilbake oversettelse som undertekster på en skjerm og opplest tale (TTS) i uavhengige lydkanaler.
Målet er ekstremt lav forsinkelse (low latency). Kjøres på maskin med min. RTX 3060 (12GB VRAM) og 32GB RAM.

# Teknologistakk og arkitektur
*   **Hjernen (`core_pipeline.py`):** Asynkront Python-program for VAD, transkribering, oversettelse, TTS, og WebSockets/REST API.
*   **Skjerm (`skjerm.html`):** Webgrensesnitt for undertekster/pop-on skjerm. Mottar tekst via WebSockets.
*   **Kontrollpanel (`kontrollpanel.html`):** Fjernkontroll for å bytte språk dynamisk, snakker med REST API (aiohttp).
*   **Adminpanel (`admin_panel.html`):** Omfattende konfigurasjon og tolkestyring (lydruting, ordliste, skjerm, etc.).
*   **Modeller:**
    *   **ASR:** `faster-whisper` (`large-v3-turbo` + nordiske modeller). `vad_filter=True`, `condition_on_previous_text=False`.
    *   **NMT:** `facebook/nllb-200-1.3B` (eller `3.3B` på større GPUer) kjører i `float16` for lavt VRAM-forbruk.
    *   **TTS:** `OmniVoice` (lokal multispråklig zero-shot modell i `float16`). Klonede `.pt` prompt-filer brukes.
*   **Lydutgang:** `sounddevice` for ruting til asynkrone lydkanaler/lydkort.

# Gjeldende status og neste steg
*   **Status:** Stabilt med automatisk VRAM-håndtering og GPU-deteksjon (valg mellom 1.3B og 3.3B NLLB-modell basert på tilgjengelig minne).
*   **Stemmer/Språk:** 20 forhåndsklonede språk. Adminpanelet støtter nedlasting av Hugging Face opptak og automatisk generering av stemmeprøver med Google Gemini API, inkludert prøvelytting-funksjon (som cacher `sample.wav`).
*   **Lydruting:** Systemet ruter automatisk til valgte kanaler (`tts_ch1`–`tts_ch16`) og detekterer maskinvare. Begrensninger per maskinvare (f.eks. max 2 for vanlige lydkort) håndteres i panelet.
*   **Tekst-oversettelse:** Maskert oversettelse brukes (via `oversettelse_ordliste.json`) for å sikre korrekt teologisk terminologi uten hallusinasjoner. Også støtte for fonetisk uttale via `uttale_ordliste.json`.
*   **TTS Hastighet:** TTS-motoren slår sammen tekst og bruker dynamisk hastighet (opptil 1.50x) for å forhindre forsinkelse ("gummistrikk-effekt").
*   **Neste steg:** Ekte GPU-batching av lydkanaler i OmniVoice for bedre kapasitet på samme GPU.

# Kjente feil og blindveier (Hva som IKKE fungerer)
*   **Hallusinasjoner i Whisper:** Må bruke `vad_filter=True` og `condition_on_previous_text=False`.
*   **VRAM-krasj/OOM:** Modeller må kjøres i `float16` (`int8_float16` for Whisper) og byttes asynkront med lås (`whisper_lock`). Pass på at `ref_text` oppgis eksplisitt under stemmekloning for å unngå ekstra laste-overhead på VRAM.
*   **Oversettelses-hallusinasjoner (NLLB):** Bruk alltid *maskert oversettelse* (`M99001` koder), aldri rene ord-for-ord oversettelser eller direkte post-filtrering av problemord.
*   **Manglende lyd/tekst pga torchaudio:** `torchaudio.functional` MÅ importeres riktig for at resampling (16kHz til 24kHz) ikke skal krasje i det stille.
*   **Aksent på TTS:** Bruk kun `native` filer for Hugging Face-stemmekloning, ikke opptak med engelsk aksent (`english_accent`).
*   **Windows Loopback (RustDesk):** Bruk `High Definition Audio Device (Headphones)` for fjernstyrings-lyd via WASAPI, ikke USB-lydkortet, for å fange opp nettleserens prøvelytt.
*   **Utvikling vs Produksjon (VIKTIG REGEL):** All utvikling skjer lokalt på Windows, men systemet kjøres i produksjon på en dedikert Linux-PC (Tolke-PC, IP: 192.168.1.33).
    **KRAV TIL AI-ASSISTENT:** Hver gang du gjør en kodeendring i prosjektet, SKAL du automatisk:
    1. Pushe koden til Github (`git add .`, `git commit`, `git push`).
    2. Overføre koden til Tolke-PC-en (via `scp` / eksisterende `.bat`-skript) hvis maskinen er tilgjengelig, og deretter restarte `kitolken.service`.
*   **Forbudt:** Må ikke bruke eldre skytjenester (f.eks. Edge TTS) pga personvern.


