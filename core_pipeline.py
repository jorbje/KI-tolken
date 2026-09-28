import asyncio
import numpy as np
import gc
import sounddevice as sd
import queue
import re
import json
import websockets
import os
import shutil
import warnings
import datetime
import time
import soundfile as sf
from omnivoice import OmniVoice
from aiohttp import web
import urllib.request
import torch
import torchaudio
import torchaudio.functional as F_audio
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from dataclasses import dataclass
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore", message="Estimating duration from bitrate, this may be inaccurate")

# --- DATAKLASSER ---
@dataclass
class AudioPacket:
    data: np.ndarray
    is_final: bool

# --- HUGGING FACE EKTE MENNESKELIGE REFERANSESTEMMER FOR ZERO-SHOT KLONING ---
HF_VOICE_SAMPLES = {
    # Coqui XTTS-v2 studio-referanser (ekte morsmålstalere)
    "de": "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/de_sample.wav",
    "fr": "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/fr_sample.wav",
    "es": "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/es_sample.wav",
    "pt": "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/pt_sample.wav",
    "tr": "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/tr_sample.wav",
    "zh": "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/zh-cn-sample.wav",
    "ja": "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/ja-sample.wav",
    "en": "https://huggingface.co/coqui/XTTS-v2/resolve/main/samples/en_sample.wav",

    # Open Swara (ekte morsmålstalere på Hugging Face - CC-BY-SA 4.0)
    "ru": "https://huggingface.co/datasets/jaymunshi/open-swara/resolve/main/voices/russian/male/russian_male_open_swara_001.wav",
    "pl": "https://huggingface.co/datasets/jaymunshi/open-swara/resolve/main/voices/polish/male/polish_male_open_swara_001.wav",
    "it": "https://huggingface.co/datasets/jaymunshi/open-swara/resolve/main/voices/italian/male/italian_male_open_swara_001.wav",
    "nl": "https://huggingface.co/datasets/jaymunshi/open-swara/resolve/main/voices/dutch/male/dutch_male_open_swara_001.wav",
    "fi": "https://huggingface.co/datasets/jaymunshi/open-swara/resolve/main/voices/finnish/male/finnish_male_open_swara_001.wav",
    "hi": "https://huggingface.co/datasets/jaymunshi/open-swara/resolve/main/voices/hindi/male/hindi_male_open_swara_001.wav",
    "ko": "https://huggingface.co/datasets/jaymunshi/open-swara/resolve/main/voices/korean/male/korean_male_open_swara_001.wav",
    "da": "https://huggingface.co/datasets/jaymunshi/open-swara/resolve/main/voices/danish/male/danish_male_open_swara_001.wav",
    "sv": "https://huggingface.co/datasets/jaymunshi/open-swara/resolve/main/voices/swedish/male/swedish_male_open_swara_001.wav",
    "no": "https://huggingface.co/datasets/jaymunshi/open-swara/resolve/main/voices/norwegian/male/norwegian_male_open_swara_001.wav",
    "sw": "https://huggingface.co/datasets/jaymunshi/open-swara/resolve/main/voices/swahili/male/swahili_male_open_swara_001.wav",
    "ar": "https://huggingface.co/datasets/jaymunshi/open-swara/resolve/main/voices/arabic/male/arabic_male_open_swara_001.wav",
    "el": "https://huggingface.co/datasets/jaymunshi/open-swara/resolve/main/voices/greek/female/greek_female_open_swara_001.wav"
}

# --- FLERSPORS LYDOPPTAK (TALE + TOLK) ---
class MultitrackRecorder:
    """
    Håndterer sanntids flersporsopptak av innkommende tale fra mikrofon og 
    alle aktive tolkespråk generert av OmniVoice TTS.
    Lagrer opptaket som en flerspors WAV-fil (48kHz, 16-bit PCM) for redigering i 
    DAW (Audacity/Reaper), samt separate mono stems for hvert språk.
    """
    def __init__(self, base_dir, sample_rate=48000):
        self.base_dir = base_dir
        self.recordings_dir = os.path.join(self.base_dir, "opptak")
        os.makedirs(self.recordings_dir, exist_ok=True)
        self.sample_rate = sample_rate
        
        self.is_recording = False
        self.start_time = None
        self.recording_id = None
        self.tracks = []
        
        self.mic_chunks = []
        self.mic_samples_count = 0
        self.tts_events = {}
        self.prev_tts_end = {}
        self.last_recording_info = None

    def start_recording(self, tracks_info):
        """Starter flersporsopptak med oppgitte spor."""
        if self.is_recording:
            return {"status": "already_recording", "error": "Opptak pågår allerede"}
            
        now = datetime.datetime.now()
        self.recording_id = f"opptak_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
        self.start_time = time.time()
        self.tracks = tracks_info
        
        self.mic_chunks = []
        self.mic_samples_count = 0
        self.tts_events = {t["id"]: [] for t in self.tracks if t.get("type") == "tts"}
        self.prev_tts_end = {t["id"]: 0 for t in self.tracks if t.get("type") == "tts"}
        
        self.is_recording = True
        track_names = ", ".join([t['name'] for t in self.tracks])
        print(f"[Opptak] Flersporsopptak startet: {self.recording_id} ({len(self.tracks)} spor: {track_names})")
        return {
            "status": "recording",
            "recording_id": self.recording_id,
            "start_time": now.isoformat(),
            "tracks": self.tracks
        }

    def add_mic_chunk(self, chunk):
        """Tapper mikrofon-lyd kontinuerlig fra VAD-tråden."""
        if not self.is_recording:
            return
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1) if chunk.shape[1] > 1 else chunk[:, 0]
        self.mic_chunks.append(chunk.astype(np.float32).copy())
        self.mic_samples_count += len(chunk)

    def add_playback_chunk(self, final_audio):
        """Plasserer ferdig generert TTS-lyd på tidslinjen ved nøyaktig avspillingstidspunkt."""
        if not self.is_recording:
            return
        if final_audio.ndim == 1:
            final_audio = final_audio[:, np.newaxis]
            
        current_sample = self.mic_samples_count
        
        for t in self.tracks:
            if t.get("type") == "tts":
                ch_idx = t.get("ch_idx", 0)
                track_id = t["id"]
                if ch_idx < final_audio.shape[1]:
                    audio = final_audio[:, ch_idx].astype(np.float32).copy()
                    start_sample = max(current_sample, self.prev_tts_end.get(track_id, 0))
                    self.tts_events[track_id].append((start_sample, audio))
                    self.prev_tts_end[track_id] = start_sample + len(audio)

    def stop_recording(self):
        """Stopper opptak, bygger multitrack matrise, eksporterer WAV og stems."""
        if not self.is_recording:
            return {"status": "not_recording", "error": "Ingen opptak pågår"}
            
        self.is_recording = False
        duration_sec = round(time.time() - self.start_time, 2)
        print(f"[Opptak] Opptak stoppet ({duration_sec}s). Behandler flersporslyd...")
        
        if not self.mic_chunks or self.mic_samples_count == 0:
            print("[Opptak] Ingen mikrofondata mottatt under opptak.")
            return {"status": "error", "error": "Ingen mikrofon-lyd ble registrert under opptaket."}
            
        # 1. Sett sammen mikrofon-sporet
        mic_track = np.concatenate(self.mic_chunks)
        max_samples = len(mic_track)
        
        # 2. Finn total lengde på tvers av alle spor (hvis TTS strakk seg litt lenger)
        for track_id, events in self.tts_events.items():
            for s, a in events:
                max_samples = max(max_samples, s + len(a))
                
        # Pad mikrofon dersom TTS overskrider mic
        if len(mic_track) < max_samples:
            mic_track = np.pad(mic_track, (0, max_samples - len(mic_track)), mode='constant')
            
        # 3. Bygg fulle arrays for hvert spor
        constructed_tracks = {}
        for t in self.tracks:
            track_id = t["id"]
            if t.get("type") == "mic":
                constructed_tracks[track_id] = mic_track
            else:
                full_tts = np.zeros(max_samples, dtype=np.float32)
                for s, a in self.tts_events.get(track_id, []):
                    end = s + len(a)
                    if end > max_samples:
                        a = a[:max_samples - s]
                        end = max_samples
                    full_tts[s:end] += a
                constructed_tracks[track_id] = full_tts
                
        # 4. Klargjør og normaliser alle spor til [-1.0, 1.0]
        track_arrays = []
        for t in self.tracks:
            arr = constructed_tracks[t["id"]]
            peak = np.max(np.abs(arr)) if len(arr) > 0 else 0
            if peak > 1.0:
                arr = arr / peak * 0.98
            else:
                arr = np.clip(arr, -1.0, 1.0)
            track_arrays.append(arr)
            
        multitrack_matrix = np.column_stack(track_arrays)
        rec_id = self.recording_id
        num_channels = len(self.tracks)
        
        multitrack_filename = f"{rec_id}_flerspor_{num_channels}kanaler.wav"
        multitrack_path = os.path.join(self.recordings_dir, multitrack_filename)
        
        sf.write(multitrack_path, multitrack_matrix, self.sample_rate, subtype='PCM_16')
        file_size_mb = round(os.path.getsize(multitrack_path) / (1024 * 1024), 2)
        
        # 5. Lagre enkeltspor (stems)
        stem_files = []
        for idx, t in enumerate(self.tracks):
            safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', t['name']).lower()
            stem_filename = f"{rec_id}_spor{idx+1}_{safe_name}.wav"
            stem_path = os.path.join(self.recordings_dir, stem_filename)
            sf.write(stem_path, track_arrays[idx], self.sample_rate, subtype='PCM_16')
            stem_size_mb = round(os.path.getsize(stem_path) / (1024 * 1024), 2)
            stem_files.append({
                "track_index": idx + 1,
                "id": t["id"],
                "name": t["name"],
                "lang": t.get("lang", ""),
                "filename": stem_filename,
                "size_mb": stem_size_mb,
                "download_url": f"/api/recording/download/{stem_filename}",
                "stream_url": f"/api/recording/stream/{stem_filename}"
            })
            
        duration_formatted = f"{int(duration_sec // 60):02d}:{int(duration_sec % 60):02d}"
        info = {
            "id": rec_id,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": duration_sec,
            "duration_formatted": duration_formatted,
            "channels_count": num_channels,
            "multitrack_filename": multitrack_filename,
            "multitrack_size_mb": file_size_mb,
            "multitrack_download_url": f"/api/recording/download/{multitrack_filename}",
            "tracks": stem_files
        }
        
        info_path = os.path.join(self.recordings_dir, f"{rec_id}_info.json")
        try:
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Opptak] Feil ved lagring av metadata: {e}")
            
        self.last_recording_info = info
        print(f"[Opptak] Fullført lagring: {multitrack_filename} ({file_size_mb} MB, {duration_formatted}, {num_channels} spor)")
        return {
            "status": "success",
            "info": info
        }

    def get_status(self):
        """Henter gjeldende opptaksstatus for API."""
        if not self.is_recording:
            return {
                "is_recording": False,
                "last_recording": self.last_recording_info
            }
        elapsed = round(time.time() - self.start_time, 1)
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        return {
            "is_recording": True,
            "recording_id": self.recording_id,
            "elapsed_seconds": elapsed,
            "elapsed_formatted": f"{mins:02d}:{secs:02d}",
            "tracks": self.tracks,
            "sample_rate": self.sample_rate,
            "samples_recorded": self.mic_samples_count,
            "last_recording": self.last_recording_info
        }

    def list_recordings(self):
        """Lister alle tidligere opptak fra opptak/-mappen."""
        recordings = []
        if not os.path.exists(self.recordings_dir):
            return recordings
            
        for fname in os.listdir(self.recordings_dir):
            if fname.endswith("_info.json"):
                info_path = os.path.join(self.recordings_dir, fname)
                try:
                    with open(info_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        recordings.append(data)
                except Exception:
                    pass
                    
        recordings.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return recordings
                    
    def delete_recording(self, rec_id):
        """Sletter et opptak og alle tilhørende filer, og nullstiller last_recording_info hvis det var dette opptaket."""
        safe_id = re.sub(r'[^a-zA-Z0-9_-]', '', str(rec_id))
        deleted = []
        if os.path.exists(self.recordings_dir):
            for fname in os.listdir(self.recordings_dir):
                if fname.startswith(safe_id):
                    fpath = os.path.join(self.recordings_dir, fname)
                    try:
                        os.remove(fpath)
                        deleted.append(fname)
                    except Exception as e:
                        print(f"[Opptak] Kunne ikke slette {fname}: {e}")
                        
        if self.last_recording_info and self.last_recording_info.get("id") == safe_id:
            self.last_recording_info = None
            print(f"[Opptak] Nullstilte sist opptak for slettet id: {safe_id}")
            
        return deleted

# --- HOVEDSYSTEM ---
class AudioPipeline:
    def __init__(self):
        # Asynkrone køer
        self.audio_queue = asyncio.Queue()       
        self.translation_queue = asyncio.Queue() 
        self.tts_queue = asyncio.Queue()         
        self.websocket_queue = asyncio.Queue()   
        self.connected_clients = set() # Holder styr på nettleserne som ser på

        # Konfigurasjon for tilstand og språk (REST API kontrollerer denne)
        self.config = {
            "source": "no", 
            "texting": "no",
            "audio_status": "checking",
            "audio_message": "",
            "system_status": "starting",
            "error_type": "none",
            "error_title": "Starter tolkemotoren...",
            "error_detail": "",
            "mic_level": 0.0,
            "hw_max_channels": 2,
            "active_tts_channels": 2
        }
        
        # Setter opptil 16 potensielle TTS-kanaler
        for i in range(1, 17):
            self.config[f"tts_ch{i}"] = "off"
        # Standard default-språk for kanal 1 og 2
        self.config["tts_ch1"] = "en"
        self.config["tts_ch2"] = "off"

        # VRAM-Lås for å unngå krasj under modellbytte
        self.whisper_lock = asyncio.Lock()

        # --- MODELL-KONFIGURASJONER (Lett å endre via SSH senere) ---
        self.whisper_models = {
            "no": "NbAiLab/nb-whisper-large",
            "en": "deepdml/faster-whisper-large-v3-turbo-ct2", 
            "sv": "PierreMesure/kb-whisper-large-ct2", 
            "da": "deepdml/faster-whisper-large-v3-turbo-ct2"
        }
        
        # Last inn språk-konfigurasjon (dynamisk register for NLLB og TTS)
        self.languages = {}
        try:
            with open("sprak_konfig.json", "r", encoding="utf-8") as f:
                self.languages = json.load(f)
            print(f"[System] Lastet inn språk-konfigurasjon ({len(self.languages)} språk)")
        except Exception as e:
            print(f"[System] Oppretter standard språk-konfigurasjon: {e}")
            self.languages = {
                "no": {"name": "Norsk", "nllb": "nob_Latn", "tts_lang": "en", "voice_folder": "Norsk_stemme_kvinne"},
                "en": {"name": "Engelsk", "nllb": "eng_Latn", "tts_lang": "en", "voice_folder": "engelsk"},
                "sv": {"name": "Svensk", "nllb": "swe_Latn", "tts_lang": "en", "voice_folder": "svensk"},
                "da": {"name": "Dansk", "nllb": "dan_Latn", "tts_lang": "en", "voice_folder": "dansk"},
                "uk": {"name": "Ukrainsk", "nllb": "ukr_Cyrl", "tts_lang": "uk", "voice_folder": "ukrainsk"},
                "es": {"name": "Spansk", "nllb": "spa_Latn", "tts_lang": "es", "voice_folder": "spansk"}
            }
            try:
                with open("sprak_konfig.json", "w", encoding="utf-8") as f:
                    json.dump(self.languages, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.settings_file = os.path.join(self.base_dir, "system_innstillinger.json")

        self.nllb_langs = {k: v["nllb"] for k, v in self.languages.items() if "nllb" in v}
        
        # Undersøk maskinvare (GPU, VRAM) og etabler maskinvareprofil
        self.hw_info = self.detect_hardware()
        self.config["gpu_name"] = self.hw_info["gpu_name"]
        self.config["vram_gb"] = self.hw_info["vram_gb"]
        self.config["hardware_profile"] = self.hw_info["profile_name"]
        self.config["nllb_model_setting"] = "auto"
        self.resolve_model_profile()
        
        # Gjenopprett huskede brukerinnstillinger (tolkespråk, talerspråk, kanaler, modellvalg)
        self.load_settings()
        
        # Tilstand for tolkemodeller (lastes asynkront etter at webserveren er oppe)
        self.models_ready = False
        self.nllb_model = None
        self.nllb_tokenizer = None
        self.whisper = None
        self.omnivoice = None
        self.voice_prompts = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Håndter ordliste for Whisper-kontekst
        norsk_stil = "Dette er en formell, velskrevet og grammatisk korrekt utskrift, helt uten muntlige fyllord, nøling eller stotring. Nøkkelord: "
        self.english_kontekst = "This is a formal, well-written, and grammatically correct transcript, completely free of any spoken filler words, stutters, or hesitation. Keywords: church, Jesus, God, Holy Spirit, hallelujah, amen, grace, salvation, worship."
        self.svensk_kontekst = "Detta är en formell, välskriven och grammatiskt korrekt utskrift, helt utan muntliga fyllnadsord, tvekan eller stammande. Nyckelord: Gudstjänst, predikan, Jesus Kristus, församling, bibeln, lovsång, halleluja, välsignelse, amen."
        self.dansk_kontekst = "Dette er en formel, velskrevet og grammatisk korrekt udskrift, helt uden mundtlige fyldord, tøven eller stammen. Nøgleord: Gudstjeneste, prædiken, Jesus Kristus, menighed, bibelen, lovsang, halleluja, velsignelse, amen."
        
        try:
            with open("ordliste.txt", "r", encoding="utf-8") as f:
                ordliste_innhold = ", ".join([line.strip() for line in f if line.strip()])
            self.kirkelig_kontekst = norsk_stil + ordliste_innhold
            print("[System] Lastet inn spesialord fra ordliste.txt")
        except FileNotFoundError:
            standard_ord = ["Gudstjeneste", "preken", "Jesus Kristus", "menighet", "bibelen", "lovsang", "halleluja", "velsignelse", "amen"]
            self.kirkelig_kontekst = norsk_stil + ", ".join(standard_ord)
            with open("ordliste.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(standard_ord))
            print("[System] Opprettet standard ordliste.txt. Legg til egne ord her senere.")

        # Håndter ordliste for oversettelse (NLLB glossary)
        self.oversettelse_ordliste = {}
        try:
            with open("oversettelse_ordliste.json", "r", encoding="utf-8") as f:
                self.oversettelse_ordliste = json.load(f)
            print(f"[System] Lastet inn oversettelsesordliste ({len(self.oversettelse_ordliste)} begreper)")
        except FileNotFoundError:
            print("[System] Advarsel: Fant ikke oversettelse_ordliste.json")

        # Håndter uttale-ordliste for fonetisk tilpasning av TTS
        try:
            with open("uttale_ordliste.json", "r", encoding="utf-8") as f:
                self.uttale_ordliste = json.load(f)
            print(f"[System] Lastet inn fonetisk uttale-ordliste ({len(self.uttale_ordliste)} ord)")
        except FileNotFoundError:
            self.uttale_ordliste = {"Herren": "Hærren"}
            with open("uttale_ordliste.json", "w", encoding="utf-8") as f:
                json.dump(self.uttale_ordliste, f, indent=4, ensure_ascii=False)
            print("[System] Opprettet standard uttale_ordliste.json")
        except Exception as e:
            print(f"[System] Feil ved lasting av oversettelse_ordliste.json: {e}")

        # --- Radar 3.1 - Dynamisk lydkortsøk og krasj-sikring ---
        self.audio_input_device = None
        self.audio_output_device = None
        self.out_channels = 2
        self.find_audio_devices()

        # --- Opptak av stemmeprøve fra miksebord / mikrofon ---
        self.is_recording_sample = False
        self.sample_recording_buffer = []
        self.last_recorded_sample_path = "recorded_mixer_sample.wav"

        # --- Flerspors Lydopptak (Tale og Tolk) ---
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.multitrack_recorder = MultitrackRecorder(base_dir=base_dir, sample_rate=48000)

    def find_audio_devices(self):
        """Søker etter eksternt lydkort / mikrofon med fallback til interne kort."""
        input_id = None
        output_id = None
        out_ch = 2
        in_name = ""
        try:
            devices = sd.query_devices()
            # 1. Prioriter eksterne USB-lydkort (f.eks Focusrite, Behringer, Audient, Røde etc.)
            for i, dev in enumerate(devices):
                name = dev.get('name', '')
                is_usb = "USB" in name or "iD4" in name or "Audient" in name or "Focusrite" in name or "Behringer" in name
                
                if is_usb:
                    if dev.get('max_input_channels', 0) > 0 and input_id is None:
                        input_id = i
                        in_name = name
                    if dev.get('max_output_channels', 0) > 0 and output_id is None:
                        output_id = i
                        out_ch = dev.get('max_output_channels', 2)
            
            # 2. Hvis ikke funnet eksternt, ta første gyldige inngang/utgang på hovedkortet
            if input_id is None:
                for i, dev in enumerate(devices):
                    if dev.get('max_input_channels', 0) > 0:
                        input_id = i
                        in_name = dev.get('name', '')
                        break
            if output_id is None:
                for i, dev in enumerate(devices):
                    if dev.get('max_output_channels', 0) > 0:
                        output_id = i
                        out_ch = dev.get('max_output_channels', 2)
                        break
        except Exception as e:
            print(f"[System] Feil ved søk etter lydkort: {e}")

        self.audio_input_device = input_id
        self.audio_output_device = output_id
        self.out_channels = out_ch
        self.config["hw_max_channels"] = out_ch

        if input_id is not None:
            self.config["audio_status"] = "ok"
            self.config["audio_message"] = f"Mikrofon tilkoblet ({in_name})"
            return True
        else:
            self.config["audio_status"] = "missing"
            self.config["audio_message"] = "Ingen mikrofon eller lydkort funnet. Sjekk tilkobling."
            return False

    def get_fallback_voice_language(self, exclude_codes=None):
        """Finner et gyldig, aktivert språk med tilgjengelig stemme for lydutgang."""
        if exclude_codes is None:
            exclude_codes = set()
        elif isinstance(exclude_codes, (list, tuple)):
            exclude_codes = set(exclude_codes)
            
        # 1. Kandidater: Aktiverte språk som har registrert stemmeprøve (has_voice/prompt.pt/voice_prompts)
        voice_candidates = []
        for code, info in self.languages.items():
            if code in exclude_codes:
                continue
            if not info.get("enabled", True):
                continue
            folder = info.get("voice_folder", code)
            prompt_file = os.path.join(folder, "prompt.pt")
            has_voice = (
                info.get("has_voice", False)
                or code in getattr(self, "voice_prompts", {})
                or os.path.exists(prompt_file)
            )
            if has_voice:
                voice_candidates.append(code)
                
        # 2. Prioriter naturlige tolkespråk i rekkefølge
        priority_order = ["en", "uk", "es", "sv", "da", "no"]
        for p in priority_order:
            if p in voice_candidates:
                return p
        if voice_candidates:
            return voice_candidates[0]
            
        # 3. Hvis ingen stemmer fantes utenom exclude_codes, se etter ethvert aktivert språk som ikke er ekskludert
        enabled_candidates = [k for k, v in self.languages.items() if v.get("enabled", True) and k not in exclude_codes]
        for p in priority_order:
            if p in enabled_candidates:
                return p
        if enabled_candidates:
            return enabled_candidates[0]
            
        # 4. Hvis alle var i exclude_codes, returner første aktiverte språk
        all_enabled = [k for k, v in self.languages.items() if v.get("enabled", True)]
        if all_enabled:
            return all_enabled[0]
            
        # 5. Siste nødutgang
        return list(self.languages.keys())[0] if self.languages else "en"

    def detect_hardware(self):
        """Undersøker installert GPU og VRAM for å bestemme optimal modellprofil."""
        gpu_name = "Ukjent GPU / CPU"
        vram_gb = 0.0
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                vram_gb = round(props.total_memory / (1024 ** 3), 1)
        except Exception as e:
            print(f"[System] Feil under maskinvaredeteksjon: {e}")

        # Anbefalt profil basert på VRAM:
        # >= 14.5 GB (f.eks. RTX 3090, 4080, 4090 - 16GB / 24GB): NLLB 3.3B + float16
        # 8.5 GB - 14.4 GB (f.eks. RTX 3060, 4070 - 12GB): NLLB 1.3B + int8_float16
        # < 8.5 GB (f.eks. 6GB / 8GB): NLLB 600M + int8_float16
        if vram_gb >= 14.5:
            recommended_model = "facebook/nllb-200-3.3B"
            recommended_compute = "float16"
            profile_name = f"Ytelse ({vram_gb} GB - NLLB 3.3B)"
        elif vram_gb >= 8.5:
            recommended_model = "facebook/nllb-200-1.3B"
            recommended_compute = "int8_float16"
            profile_name = f"Standard ({vram_gb} GB - NLLB 1.3B)"
        else:
            recommended_model = "facebook/nllb-200-distilled-600M"
            recommended_compute = "int8_float16"
            profile_name = f"Kompakt ({vram_gb} GB - NLLB 600M)"

        return {
            "gpu_name": gpu_name,
            "vram_gb": vram_gb,
            "recommended_model": recommended_model,
            "recommended_compute": recommended_compute,
            "profile_name": profile_name
        }

    def resolve_model_profile(self):
        """Avgjør hvilken NLLB-modell og Whisper-presisjon som skal brukes ut fra brukerinnstilling og maskinvare."""
        setting = self.config.get("nllb_model_setting", "auto")
        hw = getattr(self, "hw_info", None) or self.detect_hardware()
        if setting == "3.3B":
            model = "facebook/nllb-200-3.3B"
            compute = "float16"
        elif setting == "1.3B":
            model = "facebook/nllb-200-1.3B"
            compute = "int8_float16"
        elif setting == "600M":
            model = "facebook/nllb-200-distilled-600M"
            compute = "int8_float16"
        else:  # "auto"
            model = hw["recommended_model"]
            compute = hw["recommended_compute"]

        self.config["active_nllb_model"] = model
        self.config["whisper_compute_type"] = compute
        return model, compute

    def load_settings(self):
        """Laster inn lagrede brukerinnstillinger (tolkespråk, talerspråk, kanaler) fra disk."""
        if not os.path.exists(self.settings_file):
            print(f"[System] Ingen {os.path.basename(self.settings_file)} funnet. Oppretter standardinnstillinger...")
            self.save_settings()
            return
            
        try:
            with open(self.settings_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            
            if not isinstance(saved, dict):
                return

            # Valider og overfør lagret antall aktive kanaler
            if "active_tts_channels" in saved and isinstance(saved["active_tts_channels"], int):
                ch = saved["active_tts_channels"]
                if 1 <= ch <= 16:
                    self.config["active_tts_channels"] = ch

            # Valider og overfør talerspråk (må støttes av whisper_models og nllb_langs, og være aktivert)
            if "source" in saved and isinstance(saved["source"], str):
                s = saved["source"].strip().lower()
                if s in self.whisper_models and s in self.nllb_langs and self.languages.get(s, {}).get("enabled", True):
                    self.config["source"] = s
                else:
                    avail = [k for k in self.whisper_models.keys() if k in self.languages and self.languages[k].get("enabled", True)]
                    self.config["source"] = avail[0] if avail else "no"

            # Valider og overfør tekstingsspråk (må finnes i registeret og være aktivert, eller være 'uoversatt')
            if "texting" in saved and isinstance(saved["texting"], str):
                t = saved["texting"].strip().lower()
                if t == "uoversatt" or (t in self.languages and self.languages[t].get("enabled", True)):
                    self.config["texting"] = t
                else:
                    avail_text = [k for k, v in self.languages.items() if v.get("enabled", True)]
                    self.config["texting"] = "no" if "no" in avail_text else (avail_text[0] if avail_text else "no")

            # Valider og overfør tolkespråk for opptil 16 kanaler med fallback til "off" hvis et språk er deaktivert eller slettet
            needs_resave = False
            for i in range(1, 17):
                k = f"tts_ch{i}"
                if k in saved and isinstance(saved[k], str):
                    val = saved[k].strip().lower()
                    if val != "off":
                        is_valid = (val in self.languages and self.languages[val].get("enabled", True))
                        if is_valid:
                            self.config[k] = val
                        else:
                            print(f"[System] Lagret tolkespråk for {k} ('{val}') er deaktivert eller finnes ikke lenger. Settes til 'off' (Av).")
                            self.config[k] = "off"
                            needs_resave = True
                    else:
                        self.config[k] = "off"

            # Modellinnstilling for oversettelse (NLLB-200)
            if "nllb_model_setting" in saved and isinstance(saved["nllb_model_setting"], str):
                val = saved["nllb_model_setting"].strip()
                if val in ["auto", "3.3B", "1.3B", "600M"]:
                    self.config["nllb_model_setting"] = val
            self.resolve_model_profile()

            if "gemini_api_key" in saved and isinstance(saved["gemini_api_key"], str):
                self.config["gemini_api_key"] = saved["gemini_api_key"].strip()

            if needs_resave:
                self.save_settings()

            active_summary = [f"Kanal {i}: {self.config[f'tts_ch{i}']}" for i in range(1, self.config['active_tts_channels'] + 1)]
            print(f"[System] Husket innstillinger fra forrige økt: Talerspråk={self.config['source']}, Teksting={self.config['texting']}, Aktive kanaler={self.config['active_tts_channels']} ({', '.join(active_summary)}), NLLB-modell={self.config.get('active_nllb_model')}")
        except Exception as e:
            print(f"[System] Feil ved innlasting av system_innstillinger.json: {e}")

    def save_settings(self):
        """Lagrer gjeldende språk- og kanalinnstillinger atomisk til disk slik at de huskes ved omstart."""
        try:
            data = {
                "active_tts_channels": self.config.get("active_tts_channels", 2),
                "source": self.config.get("source", "no"),
                "texting": self.config.get("texting", "no"),
                "nllb_model_setting": self.config.get("nllb_model_setting", "auto"),
                "gemini_api_key": self.config.get("gemini_api_key", ""),
            }
            for i in range(1, 17):
                data[f"tts_ch{i}"] = self.config.get(f"tts_ch{i}", "off")

            tmp_file = f"{self.settings_file}.tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, self.settings_file)
            print(f"[System] Brukerinnstillinger (tolkespråk og kanaler) lagret til {os.path.basename(self.settings_file)}")
        except Exception as e:
            print(f"[System] Feil ved lagring av system_innstillinger.json: {e}")

    def set_system_error(self, error_type, title, detail):
        """Setter global feiltilstand og varsler tilkoblede klienter."""
        print(f"[System FEIL - {error_type.upper()}] {title}: {detail}")
        self.config["system_status"] = "error"
        self.config["error_type"] = error_type
        self.config["error_title"] = title
        self.config["error_detail"] = detail
        self.models_ready = False
        if hasattr(self, 'connected_clients') and self.connected_clients:
            asyncio.create_task(self.broadcast_system_status())

    async def broadcast_system_status(self):
        """Kringkaster systemstatus til alle tilkoblede skjermer."""
        if self.connected_clients:
            msg = json.dumps({
                "type": "system_status",
                "status": self.config.get("system_status", "ready"),
                "error_type": self.config.get("error_type", "none"),
                "title": self.config.get("error_title", ""),
                "detail": self.config.get("error_detail", "")
            })
            websockets.broadcast(self.connected_clients, msg)

    async def init_models_async(self):
        """Laster inn KI-modeller asynkront med omfattende feilsøking og feilmeldinger."""
        print("[System] Starter asynkron modellinnlasting...")
        self.config["system_status"] = "starting"
        self.config["error_title"] = "Starter tolkemotoren..."
        self.config["error_detail"] = "Laster inn KI-modeller i minnet..."
        await self.broadcast_system_status()

        # 1. Sjekk diskplass
        try:
            total, used, free = shutil.disk_usage("/")
            free_gb = free / (1024**3)
            if free_gb < 1.0:
                self.set_system_error(
                    error_type="disk",
                    title="Maskinen har gått tom for lagringsplass",
                    detail=f"Harddisken har kun {free_gb:.1f} GB ledig plass. Loggfiler eller midlertidige filer må slettes."
                )
                return
        except Exception as e:
            print(f"[System] Feil ved sjekk av disk: {e}")

        # 2. Sjekk CUDA og skjermkort
        if not torch.cuda.is_available():
            self.set_system_error(
                error_type="cuda",
                title="Skjermkortet (GPU) eller KI-driveren svarer ikke",
                detail=f"NVIDIA GPU ({self.config.get('gpu_name', 'NVIDIA')}) ble ikke funnet eller CUDA-driveren sviktet (DKMS-desynk). Sjekk PCIe-strømkabel eller restart maskinen."
            )
            return

        # 3. Sjekk ledig VRAM
        try:
            free_vram, total_vram = torch.cuda.mem_get_info()
            free_vram_gb = free_vram / (1024**3)
            if free_vram_gb < 3.5:
                self.set_system_error(
                    error_type="vram",
                    title="Maskinens grafikkminne er nesten fullt",
                    detail=f"Kun {free_vram_gb:.1f} GB av {total_vram/(1024**3):.1f} GB VRAM er ledig. En annen prosess bruker minnet. Prøv å trykke 'Start på nytt'."
                )
                return
        except Exception as e:
            print(f"[System] Feil ved VRAM-sjekk: {e}")

        # 4. Last inn stemmefiler (prompts) dynamisk fra sprak_konfig
        self.voice_prompts = {}
        for code, info in self.languages.items():
            folder = info.get("voice_folder", code)
            prompt_file = os.path.join(folder, "prompt.pt")
            if os.path.exists(prompt_file):
                try:
                    self.voice_prompts[code] = torch.load(prompt_file, weights_only=False)
                    info["has_voice"] = True
                except Exception as e:
                    print(f"[System] Kunne ikke laste stemmefil {prompt_file}: {e}")
                    info["has_voice"] = False
            else:
                info["has_voice"] = False

        if not self.voice_prompts:
            self.set_system_error(
                error_type="file",
                title="Ingen stemmefiler funnet på maskinen",
                detail="Fant ingen gyldige prompt.pt-filer i språkmappene (f.eks. Norsk_stemme_kvinne, engelsk, svensk osv.)."
            )
            return

        # 5. Last inn NLLB, Whisper og OmniVoice
        try:
            nllb_model_id = self.config.get("active_nllb_model", "facebook/nllb-200-1.3B")
            whisper_compute = self.config.get("whisper_compute_type", "int8_float16")
            gpu_display = f"{self.config.get('gpu_name', 'GPU')} ({self.config.get('vram_gb', 0)} GB VRAM)"
            print(f"[System] Laster inn NLLB-200 ({nllb_model_id}) for {gpu_display}...")
            self.nllb_model = await asyncio.to_thread(
                lambda: AutoModelForSeq2SeqLM.from_pretrained(nllb_model_id, torch_dtype=torch.float16).to("cuda")
            )
            self.nllb_tokenizer = await asyncio.to_thread(
                lambda: AutoTokenizer.from_pretrained(nllb_model_id, src_lang=self.nllb_langs[self.config["source"]])
            )

            print(f"[System] Laster inn initiell Whisper-modell med compute_type='{whisper_compute}'...")
            self.whisper = await asyncio.to_thread(
                lambda: WhisperModel(self.whisper_models[self.config["source"]], device="cuda", compute_type=whisper_compute)
            )

            print("[System] Laster inn OmniVoice (Universell TTS)...")
            self.omnivoice = await asyncio.to_thread(
                lambda: OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
            )

            print("[System] Alle modeller er klare i VRAM/CPU.")
            self.config["system_status"] = "ready"
            self.config["error_type"] = "none"
            self.config["error_title"] = ""
            self.config["error_detail"] = ""
            self.models_ready = True
            await self.broadcast_system_status()
        except torch.cuda.OutOfMemoryError as e:
            self.set_system_error(
                error_type="vram",
                title="Skjermkortet gikk tomt for minne (VRAM)",
                detail=f"PyTorch CUDA Out of Memory under lasting av modeller: {e}. Prøv å trykke 'Start på nytt' for å frigjøre VRAM."
            )
        except FileNotFoundError as e:
            self.set_system_error(
                error_type="file",
                title="En nødvendig modellfil mangler",
                detail=f"Fant ikke fil/modell: {e.filename or e}."
            )
        except Exception as e:
            self.set_system_error(
                error_type="generic",
                title="Det oppstod en uventet feil under oppstart",
                detail=f"Feil under modell-innlasting: {e}"
            )

    async def swap_models(self, category, new_lang):
        """Sletter gamle modeller og frigjør VRAM før nye lastes inn."""
        print(f"[ModelManager] Bytter {category} til {new_lang}...")
        
        if category == "source":
            # 1. Oppdater NLLB Tokenizer
            ny_src = self.nllb_langs.get(new_lang, "eng_Latn")
            self.nllb_tokenizer.src_lang = ny_src
            
            # 2. Bytt ut Whisper trygt bak en lås
            ny_modell_navn = self.whisper_models.get(new_lang)
            async with self.whisper_lock:
                print("[ModelManager] Frigjør gammel Whisper fra VRAM...")
                if hasattr(self, 'whisper'):
                    del self.whisper
                gc.collect()
                torch.cuda.empty_cache()
                
                # Tvungen pause slik at GPU-en rekker å fysisk frigjøre minnet før vi laster inn neste
                await asyncio.sleep(1)
                
                whisper_compute = self.config.get("whisper_compute_type", "int8_float16")
                print(f"[ModelManager] Laster inn ny Whisper ({ny_modell_navn}) med compute_type='{whisper_compute}'...")
                self.whisper = await asyncio.to_thread(WhisperModel, ny_modell_navn, device="cuda", compute_type=whisper_compute)
                print("[ModelManager] Whisper byttet og klar.")

    async def capture_and_vad(self):
        """Tråd 1: Lytter kontinuerlig og kutter ved pause (Krasj-sikker lytter)."""
        print("[Tråd 1] VAD-arbeider startet...")
        
        MIC_SR = 48000        # Maskinvarens naturlige kvalitet (Audient iD4)
        WHISPER_SR = 16000    # Kvaliteten Whisper krever
        CHUNK_DURATION = 0.1  # 100ms blokker
        CHUNK_SAMPLES = int(MIC_SR * CHUNK_DURATION)
        SILENCE_THRESHOLD = 0.01 
        
        PAUSE_LIMIT = 0.8  # Beholder 0.8s for naturlige pauser
        MAX_SILENT_CHUNKS = int(PAUSE_LIMIT / CHUNK_DURATION)
        MAX_SPEECH_DURATION = 4.0  # Tvinger frem klipp hvis taleren snakker lenge uten pause
        MAX_SPEECH_CHUNKS = int(MAX_SPEECH_DURATION / CHUNK_DURATION)
        
        while True:
            has_card = self.find_audio_devices()
            if not has_card:
                if self.config["audio_status"] != "missing":
                    self.config["audio_status"] = "missing"
                    self.config["audio_message"] = "Lydkort / mikrofon er ikke tilkoblet. Sjekk tilkobling."
                    await self.broadcast_audio_status()
                print("[Tråd 1] Advarsel: Lydkort / mikrofon ikke funnet. Venter... (prøver igjen om 3 sekunder)")
                await asyncio.sleep(3)
                continue
            
            if self.config["audio_status"] != "ok":
                self.config["audio_status"] = "ok"
                self.config["audio_message"] = ""
                await self.broadcast_audio_status()
                print(f"[Tråd 1] Mikrofon funnet (ID: {self.audio_input_device}). Kobler til...")

            mic_queue = queue.Queue()
            
            def audio_callback(indata, frames, time, status):
                if status:
                    pass
                if indata.ndim > 1 and indata.shape[1] > 1:
                    mono_data = indata.mean(axis=1)
                else:
                    mono_data = indata
                mic_queue.put(mono_data.copy().flatten())

            try:
                dev_info = sd.query_devices(self.audio_input_device)
                ch_req = min(2, max(1, dev_info.get('max_input_channels', 1)))

                stream = sd.InputStream(
                    samplerate=MIC_SR,
                    channels=ch_req,
                    dtype='float32', 
                    blocksize=CHUNK_SAMPLES, 
                    callback=audio_callback,
                    device=self.audio_input_device
                )
                
                with stream:
                    print("[Tråd 1] VAD lytter aktivt etter tale...")
                    speech_buffer = np.array([], dtype=np.float32)
                    silent_chunks = 0
                    is_speaking = False
                    
                    while True:
                        try:
                            chunk = mic_queue.get_nowait()
                        except queue.Empty:
                            await asyncio.sleep(0.01)
                            continue

                        rms_volume = np.sqrt(np.mean(chunk**2))
                        # Oppdater mikrofon-nivå for VU-meter i kontrollpanelet
                        self.config["mic_level"] = round(min(1.0, float(rms_volume) / 0.12), 3)

                        # Tapper lyd kontinuerlig dersom opptak av stemmeprøve pågår
                        if getattr(self, "is_recording_sample", False):
                            self.sample_recording_buffer.append(chunk.copy())

                        # Tapper mikrofon-lyd kontinuerlig dersom flersporsopptak pågår
                        if getattr(self, "multitrack_recorder", None) and self.multitrack_recorder.is_recording:
                            self.multitrack_recorder.add_mic_chunk(chunk)

                        if not self.models_ready:
                            await asyncio.sleep(0.02)
                            continue

                        if rms_volume > SILENCE_THRESHOLD:
                            is_speaking = True
                            silent_chunks = 0
                            speech_buffer = np.concatenate((speech_buffer, chunk))
                        elif is_speaking:
                            silent_chunks += 1
                            speech_buffer = np.concatenate((speech_buffer, chunk))
                        
                        if is_speaking:
                            tving_kutt = len(speech_buffer) >= (MAX_SPEECH_CHUNKS * CHUNK_SAMPLES)
                            naturlig_pause = silent_chunks >= MAX_SILENT_CHUNKS
                            
                            if naturlig_pause or tving_kutt:
                                audio_tensor = torch.from_numpy(speech_buffer).float()
                                resampled_tensor = F_audio.resample(audio_tensor, MIC_SR, WHISPER_SR)
                                
                                await self.audio_queue.put(resampled_tensor.numpy())
                                
                                speech_buffer = np.array([], dtype=np.float32)
                                is_speaking = False
                                silent_chunks = 0
            except Exception as e:
                print(f"[Tråd 1] Feil under opptak / lydkort frakoblet: {e}")
                self.config["audio_status"] = "missing"
                self.config["audio_message"] = "Lydkort / mikrofon er ikke tilkoblet. Kontakt Jørgen Bjerke på 99647750."
                await self.broadcast_audio_status()
                await asyncio.sleep(3)

    async def transcribe_worker(self):
        """Tråd 2: Vaskemaskin og Intelligent Splitter (Unngår småbiter)."""
        skjerm_buffer = ""
        oversettelse_buffer = ""
        
        avslutnings_tegn = {'.', '!', '?'}
        pause_tegn = {',', ':', ';'}
        
        vaskemaskin = re.compile(r'\b(eh|ehm|øh|øhm|uh|um|kremt|mm|mhm)\b', re.IGNORECASE)
        
        while True:
            audio_chunk = await self.audio_queue.get()
            if self.config["source"] == "en":
                bruk_kontekst = self.english_kontekst
            elif self.config["source"] == "sv":
                bruk_kontekst = self.svensk_kontekst
            elif self.config["source"] == "da":
                bruk_kontekst = self.dansk_kontekst
            elif self.config["source"] == "no":
                bruk_kontekst = self.kirkelig_kontekst
            else:
                bruk_kontekst = None
            
            async with self.whisper_lock:
                segments, info = await asyncio.to_thread(
                    self.whisper.transcribe,
                    audio_chunk, 
                    beam_size=3,                      # ENDRET: Økt fra 1 til 3 for å unngå at Whisper dropper ord
                    language=self.config["source"], 
                    condition_on_previous_text=False, # ENDRET: Kuttet ut for å hindre at den "henger seg opp" i gamle feil
                    initial_prompt=bruk_kontekst,
                    vad_filter=True,                  # ENDRET: Skrudd PÅ for å luke ut pustelyder Whisper sliter med
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
            for segment in segments:
                tekst = segment.text.strip()
                tekst = vaskemaskin.sub('', tekst)
                tekst = re.sub(r'\s+', ' ', tekst).strip()
                
                if len(tekst) < 2:
                    continue
                    
                skjerm_buffer += " " + tekst
                oversettelse_buffer += " " + tekst
                
                if any(tegn in tekst for tegn in avslutnings_tegn):
                    if len(skjerm_buffer.split()) > 4:
                        skjerm_buffer = skjerm_buffer.strip()
                        texting_lang = self.config.get("texting", "no")
                        if self.config["source"] == texting_lang or texting_lang == "uoversatt":
                            await self.websocket_queue.put(skjerm_buffer)
                        skjerm_buffer = ""
                        
                    if len(oversettelse_buffer.split()) > 3:
                        await self.translation_queue.put(oversettelse_buffer.strip())
                        oversettelse_buffer = ""
                elif any(tegn in tekst for tegn in pause_tegn):
                    if len(oversettelse_buffer.split()) > 6:
                        await self.translation_queue.put(oversettelse_buffer.strip())
                        oversettelse_buffer = ""
                else:
                    # Tvungen utløsning (Fail-safe): Tøm køene hvis de blir for store!
                    if len(oversettelse_buffer.split()) >= 10:
                        await self.translation_queue.put(oversettelse_buffer.strip())
                        oversettelse_buffer = ""
                        
                    if len(skjerm_buffer.split()) >= 12:
                        skjerm_buffer = skjerm_buffer.strip()
                        texting_lang = self.config.get("texting", "no")
                        if self.config["source"] == texting_lang or texting_lang == "uoversatt":
                            await self.websocket_queue.put(skjerm_buffer)
                        skjerm_buffer = ""
                        
            self.audio_queue.task_done()

    def mask_text(self, text, source, target):
        """Pre-maskering: Bytter ut kilde-ord med en unik ID (M99001) for å forhindre NLLB-hallusinasjoner."""
        if not self.oversettelse_ordliste:
            return text, {}
            
        word_map = {}
        if source == "no":
            for no_ord, trans in self.oversettelse_ordliste.items():
                if target in trans:
                    word_map[no_ord.lower()] = trans[target]
        elif target == "no":
            for no_ord, trans in self.oversettelse_ordliste.items():
                if source in trans:
                    word_map[trans[source].lower()] = no_ord
        else:
            for no_ord, trans in self.oversettelse_ordliste.items():
                if source in trans and target in trans:
                    word_map[trans[source].lower()] = trans[target]
                    
        if not word_map:
            return text, {}
            
        mask_map = {}
        mask_counter = 99001
        
        # Sorter fra lengst til kortest for å matche flerordsuttrykk først
        sorted_keys = sorted(word_map.keys(), key=len, reverse=True)
        
        masked_text = text
        for kilde_ord in sorted_keys:
            pattern = re.compile(r'\b' + re.escape(kilde_ord) + r'\b', re.IGNORECASE)
            if pattern.search(masked_text):
                fasit_ord = word_map[kilde_ord]
                placeholder = f"M{mask_counter}"
                mask_map[placeholder] = fasit_ord
                masked_text = pattern.sub(placeholder, masked_text)
                mask_counter += 1
                
        return masked_text, mask_map

    def unmask_text(self, text, mask_map):
        """Post-demaskering: Bytter ut koden (M99001) tilbake til riktig kirkelig ord i målteksten."""
        if not mask_map:
            return text
            
        unmasked_text = text
        for placeholder, fasit_ord in mask_map.items():
            pattern = re.compile(r'\b' + re.escape(placeholder) + r'\b', re.IGNORECASE)
            unmasked_text = pattern.sub(fasit_ord, unmasked_text)
            
        return unmasked_text

    async def translate_worker(self):
        """Tråd 3: NLLB-200 Dynamisk Maskinoversettelse."""
        while True:
            # Svelg unna alt i køen og slå det sammen til én setning hvis vi henger bak
            tekst_å_oversette = await self.translation_queue.get()
            while not self.translation_queue.empty():
                try:
                    neste_bit = self.translation_queue.get_nowait()
                    tekst_å_oversette += " " + neste_bit
                    self.translation_queue.task_done()
                except asyncio.QueueEmpty:
                    break
                    
            if not tekst_å_oversette:
                continue

            target_langs = set()
            active_channels = self.config.get("active_tts_channels", 2)
            for i in range(1, active_channels + 1):
                val = self.config.get(f"tts_ch{i}", "off")
                if val != "off":
                    target_langs.add(val.split("_")[0])
                    
            texting_lang = self.config.get("texting", "no")
            if self.config["source"] != texting_lang:
                target_langs.add(texting_lang)

            resultater = {}
            for base_lang in target_langs:
                # Sjekk om mål- og kildespråk er likt. Hvis ja, hopp over oversettelse!
                if self.config["source"] == base_lang:
                    resultater[base_lang] = tekst_å_oversette
                    continue
                
                if base_lang not in self.nllb_langs:
                    continue
                
                # Masker kildeteksten (Named Entity Masking) for å tvinge NLLB til å kopiere ordliste-koder (f.eks M99001)
                maskert_kilde_tekst, mask_map = self.mask_text(tekst_å_oversette, source=self.config["source"], target=base_lang)
                
                inputs = self.nllb_tokenizer(maskert_kilde_tekst, return_tensors="pt").to("cuda")
                lang_id = self.nllb_tokenizer.convert_tokens_to_ids(self.nllb_langs[base_lang])
                
                tokens = await asyncio.to_thread(
                    self.nllb_model.generate,
                    **inputs, 
                    forced_bos_token_id=lang_id, 
                    max_length=150,
                    num_beams=1,        # Greedy search for hastighet
                    do_sample=False
                )
                oversatt_tekst = self.nllb_tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
                
                # Demasker teksten: Bytter ut M99001 med det eksakte, riktige kirkelige ordet!
                oversatt_tekst = self.unmask_text(oversatt_tekst, mask_map)
                
                resultater[base_lang] = oversatt_tekst
            
            # Send den ferske oversettelsen direkte til skjermen
            texting_lang = self.config.get("texting", "no")
            if self.config["source"] != texting_lang and texting_lang in resultater:
                await self.websocket_queue.put(resultater[texting_lang])
            
            await self.tts_queue.put(resultater)
            self.translation_queue.task_done()

    async def tts_and_stereo_playback(self):
        """Tråd 4: Dynamisk TTS og stereo-splitt med Pipelining."""
        print("[Tråd 4] TTS lytter...")
        PLAYBACK_SR = 48000   # Maskinvarens naturlige kvalitet
        
        # Intern kø for ferdig nedlastet lyd som venter på å bli spilt av
        playback_queue = asyncio.Queue()
        
        async def process_tts_channel(channel, lang, text, speed_rate):
            if not text or lang == "off" or lang not in self.voice_prompts:
                return np.zeros(0, dtype=np.float32)
                
            # Fonetisk omskriving for bedre uttale (kun for lyd, ikke skjerm)
            if lang.startswith("no"):
                for ord_match, fonetisk in self.uttale_ordliste.items():
                    text = re.sub(r'\b' + re.escape(ord_match) + r'\b', fonetisk, text, flags=re.IGNORECASE)
                
            prompt = self.voice_prompts[lang]
                
            def generate_audio():
                base_lang = lang.split("_")[0]
                # OmniVoice (CosyVoice-basert) støtter ofte ikke alle språk offisielt.
                # Vi sjekker spesifisert tts_lang fra sprak_konfig, og fallback-er til 'en' for slike språk for å tvinge modellen til å lese teksten.
                supported_omni = ["en", "zh", "ru", "es", "fr", "de", "it", "ja", "ko", "pt", "tr", "ar", "uk"]
                configured_tts = self.languages.get(base_lang, {}).get("tts_lang")
                if configured_tts and configured_tts in supported_omni:
                    safe_lang = configured_tts
                elif base_lang in supported_omni:
                    safe_lang = base_lang
                else:
                    safe_lang = "en"
                
                try:
                    if isinstance(prompt, str):
                        result = self.omnivoice.generate(text=text, instruct=prompt, speed=speed_rate, language=safe_lang)
                    else:
                        result = self.omnivoice.generate(text=text, voice_clone_prompt=prompt, speed=speed_rate, language=safe_lang)
                except Exception as e:
                    print(f"[TTS Feil] OmniVoice krasjet på {lang}: {e}")
                    return np.zeros(0, dtype=np.float32), 24000
                
                # Robust uthenting av lyd og sample_rate (inspirert av OmniVoice GUI)
                if isinstance(result, list) and len(result) > 0:
                    audio_np = result[0]
                else:
                    audio_np = result

                # Finn base-samplerate (ofte 24000)
                sample_rate = getattr(self.omnivoice.config, "sampling_rate", getattr(self.omnivoice.config, "audio_sample_rate", 24000)) if hasattr(self.omnivoice, 'config') else 24000
                
                if hasattr(audio_np, "numpy"):
                    audio_np = audio_np.numpy()
                elif hasattr(audio_np, "audio"):
                    raw = audio_np.audio
                    # Hvis OmniVoice justerer samplerate dynamisk basert på speed, plukkes det opp her!
                    sample_rate = getattr(audio_np, "sample_rate", sample_rate)
                    audio_np = raw.numpy() if hasattr(raw, "numpy") else np.array(raw)
                else:
                    audio_np = np.array(audio_np)

                if audio_np.ndim > 1:
                    audio_np = audio_np.squeeze()
                
                return audio_np.flatten(), sample_rate
                
            audio_array, dynamic_sr = await asyncio.to_thread(generate_audio)
            
            # Klipp bort digital stillhet, men behold en liten "hale" for å redde siste stavelse!
            if len(audio_array) > 0:
                non_silent = np.where(np.abs(audio_array) > 0.002)[0]
                if len(non_silent) > 0:
                    margin = int(dynamic_sr * 0.08) # 80 millisekunder sikkerhetsmargin (hale)
                    start_idx = max(0, non_silent[0] - margin)
                    end_idx = min(len(audio_array), non_silent[-1] + margin)
                    audio_array = audio_array[start_idx:end_idx]
                    
            # Resample akkurat denne chunken fra dens unike dynamiske samplerate til 48000 Hz med en gang
            if len(audio_array) > 0:
                audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).float()
                resampled_tensor = F_audio.resample(audio_tensor, dynamic_sr, PLAYBACK_SR)
                audio_array = resampled_tensor.squeeze(0).numpy()
                    
            return audio_array

        async def playback_worker():
            """En dedikert arbeider som KUN spiller av lyd, slik at nedlastingen aldri stopper."""
            while True:
                stereo_mix = await playback_queue.get()
                
                # Svelg unna alt som ligger i køen og lim det sammen!
                while not playback_queue.empty():
                    try:
                        next_mix = playback_queue.get_nowait()
                        stereo_mix = np.concatenate((stereo_mix, next_mix))
                        playback_queue.task_done()
                    except asyncio.QueueEmpty:
                        break
                    
                # Chunks er nå allerede resamplet til PLAYBACK_SR (48000) individuelt! 
                # Da unngår vi krasj mellom filer med ulik samplerate!
                final_audio = stereo_mix
                
                # Registrer avspilt TTS i flersporsopptaket synkront på tidslinjen
                if getattr(self, "multitrack_recorder", None) and self.multitrack_recorder.is_recording:
                    self.multitrack_recorder.add_playback_chunk(final_audio)

                # Låser avspillingen til Audient-kortet hvis tilgjengelig
                if self.audio_output_device is not None:
                    try:
                        await asyncio.to_thread(sd.play, final_audio, samplerate=PLAYBACK_SR, blocking=True, device=self.audio_output_device)
                    except Exception as e:
                        print(f"[Playback] Feil ved avspilling: {e}")
                else:
                    # Hvis ingen fysisk lydutgang er valgt, simuler avspillingstiden
                    await asyncio.sleep(len(final_audio) / PLAYBACK_SR)
                playback_queue.task_done()

        # Start avspillings-arbeideren i bakgrunnen
        asyncio.create_task(playback_worker())

        while True:
            data = await self.tts_queue.get()
            
            # Slå sammen opptil 4 ekstra setninger for å ta igjen etterslep, 
            # slik at vi får lange, flytende setninger i stedet for å droppe tekst!
            squashed = 0
            while not self.tts_queue.empty() and squashed < 4:
                try:
                    neste_data = self.tts_queue.get_nowait()
                    for k in set(list(data.keys()) + list(neste_data.keys())):
                        val1 = data.get(k, "")
                        val2 = neste_data.get(k, "")
                        if val1 and val2:
                            data[k] = val1.strip() + " " + val2.strip()
                        elif val2:
                            data[k] = val2.strip()
                    self.tts_queue.task_done()
                    squashed += 1
                except asyncio.QueueEmpty:
                    break
            
            # Beregn totalt etterslep (tekst som venter + lyd som venter på avspilling)
            backlog = self.tts_queue.qsize() + playback_queue.qsize()
            
            # Den dynamiske gasspedalen (Sterkt oppgradert)
            if backlog >= 6:
                speed_rate = 1.50 # Katastrofalt bak, snakk lynraskt!
            elif backlog >= 4:
                speed_rate = 1.35 # Ligger veldig langt bak, snakk fort
            elif backlog >= 2:
                speed_rate = 1.20 # Ligger litt bak, snakk fortere
            else:
                speed_rate = 1.0  # À jour, snakk i normalt, behagelig tempo
            
            # Hent riktig tekst for hver kanal og bygg prosesseringstråder
            active_channels = self.config.get("active_tts_channels", 2)
            tasks = []
            for i in range(1, active_channels + 1):
                lang = self.config.get(f"tts_ch{i}", "off")
                base_lang = lang.split("_")[0] if lang != "off" else "off"
                text = data.get(base_lang, "")
                tasks.append(process_tts_channel(f"ch{i}", lang, text, speed_rate))

            # Prosesser alle aktive kanaler NØYAKTIG samtidig for å spare tid
            audio_channels = await asyncio.gather(*tasks)
            
            # Finn den lengste resulterende lyd-chunken
            max_len = max([len(ch) for ch in audio_channels] + [0])
            
            if max_len > 0:
                padded_channels = []
                for ch in audio_channels:
                    padded_channels.append(np.pad(ch, (0, max_len - len(ch)), mode='constant'))
                
                # Stack vertikalt og transponer for å mikse: shape blir (frames, active_channels)
                multichannel_mix = np.vstack(padded_channels).T
                
                # Legg den ferdige lyden i avspillings-køen umiddelbart, og gå videre til neste setning!
                await playback_queue.put(multichannel_mix)
            
            self.tts_queue.task_done()

    async def broadcast_audio_status(self):
        """Kringkaster lydstatus til alle tilkoblede skjermer."""
        if self.connected_clients:
            msg = json.dumps({
                "type": "audio_status",
                "status": self.config.get("audio_status", "ok"),
                "message": self.config.get("audio_message", "")
            })
            websockets.broadcast(self.connected_clients, msg)

    async def ws_handler(self, websocket):
        """Håndterer nye nettlesere som kobler seg til skjermen."""
        self.connected_clients.add(websocket)
        try:
            # Send systemstatus og lydstatus umiddelbart ved tilkobling
            await websocket.send(json.dumps({
                "type": "system_status",
                "status": self.config.get("system_status", "starting"),
                "error_type": self.config.get("error_type", "none"),
                "title": self.config.get("error_title", ""),
                "detail": self.config.get("error_detail", "")
            }))
            await websocket.send(json.dumps({
                "type": "audio_status",
                "status": self.config.get("audio_status", "ok"),
                "message": self.config.get("audio_message", "")
            }))
            await websocket.wait_closed()
        finally:
            self.connected_clients.remove(websocket)

    async def broadcast_worker(self):
        """Tråd 5: Sender ferdige setninger til nettleseren lynraskt."""
        while True:
            tekst = await self.websocket_queue.get()
            if self.connected_clients:
                websockets.broadcast(self.connected_clients, json.dumps({"text": tekst}))
            self.websocket_queue.task_done()

    _API_HEADERS = {
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
    }

    async def api_status_handler(self, request):
        """REST API: Gir kontrollpanelet fasiten på hvilke språk som er aktive."""
        status_data = dict(self.config)
        for code, info in self.languages.items():
            info["has_voice"] = (code in self.voice_prompts)
            if "enabled" not in info:
                info["enabled"] = True
        status_data["languages"] = self.languages
        status_data["available_voices"] = list(dict.fromkeys([k for k in self.voice_prompts.keys() if self.languages.get(k, {}).get("enabled", True)]))
        status_data["available_languages"] = [k for k, v in self.languages.items() if v.get("enabled", True)]
        status_data["hf_supported_languages"] = list(HF_VOICE_SAMPLES.keys())
        status_data["has_gemini_key"] = bool(self.config.get("gemini_api_key"))
        if getattr(self, "multitrack_recorder", None):
            status_data["recording"] = self.multitrack_recorder.get_status()
        return web.json_response(status_data, headers=self._API_HEADERS)

    async def api_set_handler(self, request):
        """REST API: Oppdaterer inn/ut-språk dynamisk og trigger VRAM-swapping."""
        category = request.match_info.get('category')
        lang = request.match_info.get('lang')
        cors_headers = self._API_HEADERS
        
        gyldige_kategorier = ["source", "texting"] + [f"tts_ch{i}" for i in range(1, 17)]
        if category in gyldige_kategorier:
            # Sikre at hvis et deaktivert eller ukjent språk forespørres, faller vi automatisk tilbake til 'off' (Av)
            if category.startswith("tts_ch") and lang != "off":
                if lang not in self.languages or not self.languages[lang].get("enabled", True):
                    print(f"[API] Språk '{lang}' er ikke tilgjengelig for {category}. Settes til 'off' (Av).")
                    lang = "off"
            elif category == "texting" and lang != "uoversatt":
                if lang not in self.languages or not self.languages[lang].get("enabled", True):
                    active_langs = [k for k, v in self.languages.items() if v.get("enabled", True)]
                    lang = "no" if "no" in active_langs else (active_langs[0] if active_langs else "no")
            elif category == "source":
                if lang not in self.whisper_models or (lang in self.languages and not self.languages[lang].get("enabled", True)):
                    avail = [k for k in self.whisper_models.keys() if k in self.languages and self.languages[k].get("enabled", True)]
                    lang = avail[0] if avail else "no"

            gammel_lang = self.config[category]
            if gammel_lang != lang:
                print(f"[API] Laster inn modell for: {category} -> {lang}. Venter på VRAM...")
                await self.swap_models(category, lang)
                self.config[category] = lang
                self.save_settings()
                print(f"[API] Suksess! {category} lytter nå til {lang}.")
                
            return web.json_response({"status": "success", "category": category, "lang": lang}, headers=cors_headers)
        
        return web.json_response({"error": "Ugyldig kategori"}, status=400, headers=cors_headers)

    async def api_restart_handler(self, request):
        """REST API: Tvinger Python til å avslutte. Systemd vil automatisk starte det på nytt."""
        print("[API] Mottok signal om omstart fra kontrollpanelet. Avslutter systemet...")
        self.save_settings()
        response = web.json_response({"status": "restarting"}, headers=self._API_HEADERS)

        async def kill_process():
            await asyncio.sleep(4)
            os._exit(0)
            
        asyncio.create_task(kill_process())
        return response

    async def api_upload_voice_handler(self, request):
        """REST API (Admin): Laster opp en lydfil og lager prompt.pt"""
        reader = await request.multipart()
        folder_name = "ny_stemme"
        audio_data = None
        
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == "folder_name":
                folder_name = (await part.read()).decode().strip()
            elif part.name == "audio_file":
                audio_data = await part.read(decode=False)
                
        if not audio_data:
            return web.json_response({"error": "Mangler lydfil"}, status=400, headers=self._API_HEADERS)
            
        import os
        os.makedirs(folder_name, exist_ok=True)
        temp_audio_path = os.path.join(folder_name, "temp_audio.wav")
        prompt_path = os.path.join(folder_name, "prompt.pt")
        
        with open(temp_audio_path, "wb") as f:
            f.write(audio_data)
            
        try:
            print(f"[Admin] Genererer klonet stemme fra {temp_audio_path}...")
            # Lås Whisper for å ikke krasje VRAM mens vi prosesserer
            async with self.whisper_lock:
                prompt_obj = await asyncio.to_thread(self.omnivoice.create_voice_clone_prompt, temp_audio_path)
                torch.save(prompt_obj, prompt_path)
            
            # Last inn den nye stemmen dynamisk, uten å skru av programmet
            self.voice_prompts[folder_name] = torch.load(prompt_path, weights_only=False)
            if folder_name in self.languages:
                self.languages[folder_name]["has_voice"] = True
                try:
                    with open("sprak_konfig.json", "w", encoding="utf-8") as f:
                        json.dump(self.languages, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass
            msg = f"Stemmeprofil '{folder_name}' er opprettet."
            print(f"[Admin] {msg}")
            
            if hasattr(self, "connected_clients") and self.connected_clients:
                await self.broadcast_system_status() # Trigger oppdatering

            return web.json_response({"status": "success", "message": msg}, headers=self._API_HEADERS)
        except Exception as e:
            print(f"[Admin] Feil under stemmegenerering: {e}")
            return web.json_response({"error": str(e)}, status=500, headers=self._API_HEADERS)
        finally:
            if os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                    print(f"[Admin] Slettet midlertidig lydfil: {temp_audio_path}")
                except Exception as e:
                    print(f"[Admin] Feil ved sletting av {temp_audio_path}: {e}")

    async def api_set_hw_channels_handler(self, request):
        """REST API (Admin): Setter hvor mange aktive kanaler systemet skal sende lyd ut på."""
        try:
            data = await request.json()
            channels = int(data.get("active_channels", 2))
            hw_max = self.config.get("hw_max_channels", 2)
            if channels < 1 or channels > hw_max:
                card_name = self.config.get("audio_message", "tilkoblet lydkort")
                return web.json_response({
                    "error": f"Tilkoblet lydkort ({card_name}) har kun {hw_max} fysiske utganger. For å sende ut {channels} samtidige tolkespråk må et flerkanals USB-lydkort med minst {channels} utganger kobles til."
                }, status=400, headers=self._API_HEADERS)
                
            self.config["active_tts_channels"] = channels
            self.save_settings()
            print(f"[Admin] Endret aktive kanaler til {channels}. Systemet ruter nå TTS deretter.")
            
            if hasattr(self, "connected_clients") and self.connected_clients:
                await self.broadcast_system_status() # Ber frontends oppdatere GUI
                
            return web.json_response({"status": "success", "active_tts_channels": channels}, headers=self._API_HEADERS)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500, headers=self._API_HEADERS)

    async def api_set_model_handler(self, request):
        """REST API (Admin): Bytter NLLB-200 modellinnstilling (auto, 3.3B, 1.3B, 600M)."""
        try:
            data = await request.json()
            val = data.get("nllb_model_setting", "auto").strip()
            if val not in ["auto", "3.3B", "1.3B", "600M"]:
                return web.json_response({"error": "Ugyldig modellvalg"}, status=400, headers=self._API_HEADERS)
            
            self.config["nllb_model_setting"] = val
            self.resolve_model_profile()
            self.save_settings()
            print(f"[Admin] NLLB modellinnstilling endret til: {val}. Aktiv modell satt til {self.config['active_nllb_model']}")
            return web.json_response({
                "status": "success",
                "nllb_model_setting": val,
                "active_nllb_model": self.config["active_nllb_model"],
                "whisper_compute_type": self.config["whisper_compute_type"]
            }, headers=self._API_HEADERS)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500, headers=self._API_HEADERS)

    async def api_mic_level_handler(self, request):
        """REST API (Admin): Raskt endepunkt for VU-meter før opptak starter."""
        return web.json_response({
            "level": self.config.get("mic_level", 0.0),
            "status": self.config.get("audio_status", "ok"),
            "message": self.config.get("audio_message", "")
        }, headers=self._API_HEADERS)

    async def api_get_dictionary_handler(self, request):
        """REST API (Admin): Henter gjeldende oversettelse_ordliste.json"""
        import json
        try:
            with open("oversettelse_ordliste.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            return web.json_response(data, headers=self._API_HEADERS)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500, headers=self._API_HEADERS)

    async def api_update_dictionary_handler(self, request):
        """REST API (Admin): Oppdaterer oversettelse_ordliste.json og laster den inn på nytt"""
        import json
        try:
            data = await request.json()
            if not isinstance(data, dict):
                return web.json_response({"error": "Ugyldig format. Forventet JSON-objekt (dictionary)."}, status=400, headers=self._API_HEADERS)
            
            with open("oversettelse_ordliste.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.oversettelse_ordliste = data
            print(f"[System] Ordliste oppdatert via Admin ({len(data)} begreper)")
            
            return web.json_response({"status": "ok", "message": "Ordliste oppdatert"}, headers=self._API_HEADERS)
        except Exception as e:
            print(f"[System] Feil ved oppdatering av ordliste: {e}")
            return web.json_response({"error": str(e)}, status=500, headers=self._API_HEADERS)

    async def api_get_pronunciation_handler(self, request):
        """REST API (Admin): Henter gjeldende uttale_ordliste.json"""
        import json
        try:
            if not os.path.exists("uttale_ordliste.json"):
                self.uttale_ordliste = {"Herren": "Hærren"}
                with open("uttale_ordliste.json", "w", encoding="utf-8") as f:
                    json.dump(self.uttale_ordliste, f, indent=2, ensure_ascii=False)
            with open("uttale_ordliste.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            self.uttale_ordliste = data
            return web.json_response(data, headers=self._API_HEADERS)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500, headers=self._API_HEADERS)

    async def api_update_pronunciation_handler(self, request):
        """REST API (Admin): Oppdaterer uttale_ordliste.json og laster den inn på nytt"""
        import json
        try:
            data = await request.json()
            if not isinstance(data, dict):
                return web.json_response({"error": "Ugyldig format. Forventet JSON-objekt."}, status=400, headers=self._API_HEADERS)
            
            with open("uttale_ordliste.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.uttale_ordliste = data
            print(f"[System] Uttale-ordliste oppdatert via Admin ({len(data)} ord)")
            
            return web.json_response({"status": "ok", "message": "Uttale-ordliste oppdatert", "count": len(data)}, headers=self._API_HEADERS)
        except Exception as e:
            print(f"[System] Feil ved oppdatering av uttale-ordliste: {e}")
            return web.json_response({"error": str(e)}, status=500, headers=self._API_HEADERS)

    async def api_start_mixer_recording_handler(self, request):
        """REST API (Admin): Starter opptak av stemmeprøve fra miksebord / mikrofon."""
        if os.path.exists(self.last_recorded_sample_path):
            try:
                os.remove(self.last_recorded_sample_path)
            except Exception:
                pass
        self.sample_recording_buffer = []
        self.is_recording_sample = True
        print("[Admin] Startet opptak av stemmeprøve fra miksebord/mikrofon...")
        return web.json_response({"status": "recording", "message": "Opptak startet"}, headers=self._API_HEADERS)

    async def api_stop_mixer_recording_handler(self, request):
        """REST API (Admin): Stopper opptak og lagrer som 48kHz WAV."""
        self.is_recording_sample = False
        if not self.sample_recording_buffer:
            return web.json_response({
                "error": "Ingen lyd ble registrert fra mikrofonen. Sjekk at lydkort/mikrofon er tilkoblet og aktivt."
            }, status=400, headers=self._API_HEADERS)
        
        full_audio = np.concatenate(self.sample_recording_buffer)
        duration = round(len(full_audio) / 48000.0, 2)
        print(f"[Admin] Opptak fullført: {duration}s ({len(full_audio)} samples).")
        
        if duration < 1.0:
            return web.json_response({
                "error": f"Opptaket var for kort ({duration}s). Vennligst spill inn minst 5–15 sekunder tale for best kvalitet."
            }, status=400, headers=self._API_HEADERS)

        try:
            tensor = torch.from_numpy(full_audio).unsqueeze(0).float()
            tensor = torch.clamp(tensor, -1.0, 1.0)
            torchaudio.save(self.last_recorded_sample_path, tensor, 48000)
            return web.json_response({
                "status": "success",
                "duration": duration,
                "preview_url": "/api/admin/recorded_sample_audio"
            }, headers=self._API_HEADERS)
        except Exception as e:
            print(f"[Admin] Feil ved lagring av WAV fra miksebord: {e}")
            return web.json_response({"error": f"Kunne ikke lagre lydfil: {e}"}, status=500, headers=self._API_HEADERS)

    async def api_recorded_sample_audio_handler(self, request):
        """REST API: Returnerer den innspilte WAV-filen for forhåndslytting i admin-panelet."""
        if not os.path.exists(self.last_recorded_sample_path):
            return web.json_response({"error": "Ingen stemmeprøve er innspilt ennå"}, status=404, headers=self._API_HEADERS)
        return web.FileResponse(self.last_recorded_sample_path, headers={
            "Content-Type": "audio/wav",
            "Cache-Control": "no-cache, no-store, must-revalidate"
        })

    async def api_start_multitrack_recording_handler(self, request):
        """REST API (Admin): Starter flerspors opptak av tale og alle aktive tolkekanaler."""
        if self.multitrack_recorder.is_recording:
            return web.json_response({
                "status": "already_recording",
                "message": "Opptak pågår allerede"
            }, headers=self._API_HEADERS)

        # Bygg sporliste basert på inn- og tolkekanaler
        source_lang = self.config.get("source", "no")
        source_name = self.languages.get(source_lang, {}).get("name", source_lang.upper())
        
        tracks = [
            {
                "id": "source",
                "name": f"Innkommende ({source_name})",
                "lang": source_lang,
                "type": "mic"
            }
        ]
        
        active_tts_channels = self.config.get("active_tts_channels", 2)
        for i in range(1, active_tts_channels + 1):
            lang = self.config.get(f"tts_ch{i}", "off")
            if lang != "off":
                base_lang = lang.split("_")[0]
                lang_name = self.languages.get(base_lang, {}).get("name", base_lang.upper())
                tracks.append({
                    "id": f"tts_ch{i}",
                    "name": f"Tolk {i} ({lang_name})",
                    "lang": base_lang,
                    "type": "tts",
                    "ch_idx": i - 1
                })
                
        res = self.multitrack_recorder.start_recording(tracks)
        return web.json_response(res, headers=self._API_HEADERS)

    async def api_stop_multitrack_recording_handler(self, request):
        """REST API (Admin): Stopper opptak og lagrer flerspors WAV."""
        if not self.multitrack_recorder.is_recording:
            return web.json_response({
                "status": "not_recording",
                "message": "Ingen opptak pågår"
            }, headers=self._API_HEADERS)
            
        res = await asyncio.to_thread(self.multitrack_recorder.stop_recording)
        return web.json_response(res, headers=self._API_HEADERS)

    async def api_recording_status_handler(self, request):
        """REST API: Henter gjeldende opptaksstatus og forhåndsvisning av spor."""
        status = self.multitrack_recorder.get_status()
        source_lang = self.config.get("source", "no")
        source_name = self.languages.get(source_lang, {}).get("name", source_lang.upper())
        preview_tracks = [
            {"id": "source", "name": f"Innkommende ({source_name})", "lang": source_lang, "type": "mic"}
        ]
        active_tts_channels = self.config.get("active_tts_channels", 2)
        for i in range(1, active_tts_channels + 1):
            lang = self.config.get(f"tts_ch{i}", "off")
            if lang != "off":
                base_lang = lang.split("_")[0]
                lang_name = self.languages.get(base_lang, {}).get("name", base_lang.upper())
                preview_tracks.append({
                    "id": f"tts_ch{i}",
                    "name": f"Tolk {i} ({lang_name})",
                    "lang": base_lang,
                    "type": "tts",
                    "ch_idx": i - 1
                })
        status["preview_tracks"] = preview_tracks
        return web.json_response(status, headers=self._API_HEADERS)

    async def api_list_recordings_handler(self, request):
        """REST API: Henter liste over alle tidligere opptak."""
        recordings = await asyncio.to_thread(self.multitrack_recorder.list_recordings)
        return web.json_response({"recordings": recordings}, headers=self._API_HEADERS)

    async def api_download_recording_handler(self, request):
        """REST API: Laster ned en WAV- eller info-fil med attachment header."""
        filename = request.match_info.get('filename')
        safe_filename = os.path.basename(filename)
        filepath = os.path.join(self.multitrack_recorder.recordings_dir, safe_filename)
        if not os.path.exists(filepath):
            return web.json_response({"error": "Fil ikke funnet"}, status=404, headers=self._API_HEADERS)
            
        headers = {
            "Content-Disposition": f'attachment; filename="{safe_filename}"',
            "Content-Type": "audio/wav" if safe_filename.endswith(".wav") else "application/octet-stream",
            "Cache-Control": "no-cache"
        }
        return web.FileResponse(filepath, headers=headers)

    async def api_stream_recording_handler(self, request):
        """REST API: Strømmer lyd for forhåndslytting direkte i nettleseren."""
        filename = request.match_info.get('filename')
        safe_filename = os.path.basename(filename)
        filepath = os.path.join(self.multitrack_recorder.recordings_dir, safe_filename)
        if not os.path.exists(filepath):
            return web.json_response({"error": "Fil ikke funnet"}, status=404, headers=self._API_HEADERS)
            
        headers = {
            "Content-Type": "audio/wav",
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache"
        }
        return web.FileResponse(filepath, headers=headers)

    async def api_delete_recording_handler(self, request):
        """REST API (Admin): Sletter et tidligere opptak og alle dets spor."""
        rec_id = request.match_info.get('id')
        deleted = self.multitrack_recorder.delete_recording(rec_id)
        return web.json_response({
            "status": "success", 
            "deleted": deleted,
            "last_recording": self.multitrack_recorder.last_recording_info
        }, headers=self._API_HEADERS)

    async def api_add_language_handler(self, request):
        """REST API (Admin): Legger til et nytt språk (med valgfri stemmekloning) og lagrer til sprak_konfig.json."""
        reader = await request.multipart()
        lang_code = ""
        lang_name = ""
        nllb_code = ""
        tts_lang = "en"
        voice_source = "keep"
        overwrite_voice = False
        gemini_key = ""
        gemini_gender = "male"
        audio_data = None
        
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == "lang_code":
                lang_code = (await part.read()).decode().strip().lower()
            elif part.name == "lang_name":
                lang_name = (await part.read()).decode().strip()
            elif part.name == "nllb_code":
                nllb_code = (await part.read()).decode().strip()
            elif part.name == "tts_lang":
                tts_lang = (await part.read()).decode().strip()
            elif part.name == "voice_source":
                voice_source = (await part.read()).decode().strip()
            elif part.name == "gemini_api_key":
                gemini_key = (await part.read()).decode().strip()
            elif part.name == "gemini_gender":
                gemini_gender = (await part.read()).decode().strip().lower()
            elif part.name == "overwrite_voice":
                overwrite_val = (await part.read()).decode().strip().lower()
                overwrite_voice = overwrite_val in ["true", "1", "yes"]
            elif part.name == "audio_file":
                audio_data = await part.read(decode=False)
                
        if not lang_code or not nllb_code:
            return web.json_response({"error": "Mangler språkkode eller NLLB-kode"}, status=400, headers=self._API_HEADERS)
            
        if not lang_name:
            lang_name = lang_code.upper()

        DEFAULT_VOICE_FOLDERS = {
            "no": "Norsk_stemme_kvinne",
            "en": "engelsk",
            "sv": "svensk",
            "da": "dansk",
            "uk": "ukrainsk",
            "es": "spansk"
        }
        folder_name = self.languages.get(lang_code, {}).get("voice_folder")
        if not folder_name:
            folder_name = DEFAULT_VOICE_FOLDERS.get(lang_code, lang_code)

        prompt_path = os.path.join(folder_name, "prompt.pt")
        has_existing_prompt = os.path.exists(prompt_path)

        # Sikkerhet: Hvis språket allerede har en prompt.pt og bruker prøver å hente/laste opp/spille inn ny stemme uten bekreftelse:
        if has_existing_prompt and voice_source in ["hf", "custom", "mixer", "gemini"] and not overwrite_voice:
            return web.json_response({
                "status": "conflict",
                "error": f"Språket '{lang_name}' har allerede en lokal stemmeprofil (prompt.pt). Bekreft at du ønsker å overskrive den.",
                "has_existing": True
            }, status=409, headers=self._API_HEADERS)

        # Hvis brukeren valgte opptak fra miksebord / mikrofon:
        if voice_source == "mixer":
            if not os.path.exists(self.last_recorded_sample_path):
                return web.json_response({"error": "Fant ingen innspilt stemmeprøve fra miksebordet. Vennligst spill inn 10-15 sekunder først."}, status=400, headers=self._API_HEADERS)
            try:
                with open(self.last_recorded_sample_path, "rb") as f:
                    audio_data = f.read()
                print(f"[Admin] Benytter innspilt stemmeprøve fra miksebord ({len(audio_data)} bytes).")
            except Exception as e:
                return web.json_response({"error": f"Kunne ikke lese innspilt lydfil: {e}"}, status=500, headers=self._API_HEADERS)

        # Hvis brukeren valgte å generere AI-stemme via Gemini:
        if voice_source == "gemini":
            active_key = gemini_key or self.config.get("gemini_api_key", "").strip()
            if not active_key:
                return web.json_response({
                    "error": "Gemini API-nøkkel mangler. Vennligst følg trinn-for-trinn veiledningen i panelet for å opprette en gratis API-nøkkel på aistudio.google.com."
                }, status=400, headers=self._API_HEADERS)
            
            if gemini_key and gemini_key != self.config.get("gemini_api_key"):
                self.config["gemini_api_key"] = gemini_key
                self.save_settings()

            try:
                print(f"[Admin] Genererer AI-stemmeprøve for {lang_name} ({lang_code}) via Gemini (kjønn: {gemini_gender})...")
                audio_data = await asyncio.to_thread(self.generate_gemini_sample_audio, active_key, lang_name, lang_code, gemini_gender)
                print(f"[Admin] Gemini TTS fullført! Genererte {len(audio_data)} bytes studiokvalitet.")
            except Exception as e:
                print(f"[Admin] Gemini TTS feilet: {e}")
                return web.json_response({"error": f"Gemini TTS feilet: {e}"}, status=500, headers=self._API_HEADERS)

        # Hvis brukeren valgte å hente ekte stemme fra Hugging Face:
        if voice_source == "hf" and (not audio_data or len(audio_data) < 100):
            if lang_code in HF_VOICE_SAMPLES:
                hf_url = HF_VOICE_SAMPLES[lang_code]
                try:
                    print(f"[Admin] Henter ekte menneskestemme for {lang_name} ({lang_code}) fra Hugging Face: {hf_url}...")
                    req = urllib.request.Request(hf_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=20) as resp:
                        audio_data = resp.read()
                    print(f"[Admin] Lastet ned {len(audio_data)} bytes med ekte tale fra Hugging Face.")
                except Exception as e:
                    print(f"[Admin] Feil ved nedlasting fra Hugging Face: {e}")
                    return web.json_response({"error": f"Kunne ikke hente stemme fra Hugging Face: {e}"}, status=500, headers=self._API_HEADERS)
            else:
                return web.json_response({"error": f"Ingen forhåndsdefinert Hugging Face stemme funnet for '{lang_code}'. Vennligst last opp egen fil eller velg kun teksting."}, status=400, headers=self._API_HEADERS)
            
        has_voice = False
        if voice_source == "keep" and has_existing_prompt:
            # Behold eksisterende prompt.pt urørt
            try:
                self.voice_prompts[lang_code] = torch.load(prompt_path, weights_only=False)
                has_voice = True
                print(f"[Admin] Beholder eksisterende prompt.pt for {lang_code}.")
            except Exception as e:
                print(f"[Admin] Feil ved innlasting av eksisterende prompt.pt: {e}")
        elif audio_data and len(audio_data) > 1000 and voice_source != "none":
            # Ny lyd skal klones og lagres til prompt.pt
            os.makedirs(folder_name, exist_ok=True)
            temp_audio_path = os.path.join(folder_name, "temp_audio.wav")
            
            with open(temp_audio_path, "wb") as f:
                f.write(audio_data)
                
            try:
                print(f"[Admin] Genererer klonet stemme for {lang_name} ({lang_code}) fra {temp_audio_path}...")
                async with self.whisper_lock:
                    prompt_obj = await asyncio.to_thread(self.omnivoice.create_voice_clone_prompt, temp_audio_path)
                    torch.save(prompt_obj, prompt_path)
                
                self.voice_prompts[lang_code] = torch.load(prompt_path, weights_only=False)
                has_voice = True
                print(f"[Admin] Ekte stemme klonet og lagret i {prompt_path}")
            except Exception as e:
                print(f"[Admin] Feil under stemmegenerering: {e}")
                return web.json_response({"error": f"Stemmekloning feilet: {e}"}, status=500, headers=self._API_HEADERS)
            finally:
                # Slett midlertidig opptakslyd for å spare harddiskplass
                if os.path.exists(temp_audio_path):
                    try:
                        os.remove(temp_audio_path)
                        print(f"[Admin] Slettet midlertidig fil: {temp_audio_path}")
                    except Exception as e:
                        print(f"[Admin] Feil ved sletting av {temp_audio_path}: {e}")
                if voice_source == "mixer" and os.path.exists(self.last_recorded_sample_path):
                    try:
                        os.remove(self.last_recorded_sample_path)
                        print(f"[Admin] Slettet midlertidig miksebord-opptak: {self.last_recorded_sample_path}")
                    except Exception as e:
                        print(f"[Admin] Feil ved sletting av {self.last_recorded_sample_path}: {e}")
        else:
            # Sjekk om det allerede finnes en prompt.pt for dette språket hvis voice_source != "none"
            if voice_source != "none" and has_existing_prompt:
                try:
                    self.voice_prompts[lang_code] = torch.load(prompt_path, weights_only=False)
                    has_voice = True
                except Exception:
                    pass

        # Oppdater språkkonfigurasjon
        self.languages[lang_code] = {
            "name": lang_name,
            "nllb": nllb_code,
            "tts_lang": tts_lang or "en",
            "voice_folder": folder_name,
            "has_voice": has_voice
        }
        self.nllb_langs[lang_code] = nllb_code
        
        # Lagre til disk
        try:
            with open("sprak_konfig.json", "w", encoding="utf-8") as f:
                json.dump(self.languages, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Admin] Feil ved lagring av sprak_konfig.json: {e}")
            
        print(f"[Admin] Språk '{lang_name}' ({lang_code} -> {nllb_code}) er aktivert i systemet.")
        
        if hasattr(self, "connected_clients") and self.connected_clients:
            await self.broadcast_system_status()
            
        if voice_source == "keep":
            msg = f"Språkkonfigurasjon for '{lang_name}' ble oppdatert. Eksisterende stemme (prompt.pt) ble bevart."
        else:
            msg = f"Språk '{lang_name}' er lagt til!" + (" Ekte stemme klonet og klar for øreplugger." if has_voice else " Klar for teksting på skjerm.")
            
        return web.json_response({"status": "success", "message": msg, "lang_code": lang_code}, headers=self._API_HEADERS)

    async def api_delete_language_handler(self, request):
        """REST API (Admin): Fjerner et språk fra konfigurasjonen."""
        try:
            data = await request.json()
            lang_code = data.get("lang_code", "").strip().lower()
            
            if not lang_code or lang_code not in self.languages:
                return web.json_response({"error": "Språket finnes ikke"}, status=400, headers=self._API_HEADERS)
                
            if len(self.languages) <= 1:
                return web.json_response({"error": "Kan ikke slette det siste gjenværende språket i systemet"}, status=400, headers=self._API_HEADERS)
                
            lang_name = self.languages[lang_code].get("name", lang_code)
            del self.languages[lang_code]
            if lang_code in self.nllb_langs:
                del self.nllb_langs[lang_code]
            if lang_code in self.voice_prompts:
                del self.voice_prompts[lang_code]
                
            # Tilbakestill aktive valg dersom det slettede språket var i bruk
            remaining_langs = [k for k, v in self.languages.items() if v.get("enabled", True)] or list(self.languages.keys())
            fallback_lang = "no" if "no" in remaining_langs else remaining_langs[0]
            
            if self.config.get("texting") == lang_code:
                self.config["texting"] = fallback_lang
                print(f"[Admin] Teksting var satt til slettet språk '{lang_code}', tilbakestilt til '{fallback_lang}'.")
                
            for i in range(1, 17):
                ch_key = f"tts_ch{i}"
                if self.config.get(ch_key) == lang_code:
                    self.config[ch_key] = "off"
                    print(f"[Admin] TTS-kanal {i} var satt til slettet språk '{lang_code}', satt til 'off' (Av).")
                    
            if self.config.get("source") == lang_code:
                avail_whisper = [k for k in self.whisper_models.keys() if k in remaining_langs]
                new_source = avail_whisper[0] if avail_whisper else ("no" if "no" in self.whisper_models else "en")
                if hasattr(self, "swap_models") and self.models_ready:
                    await self.swap_models("source", new_source)
                self.config["source"] = new_source
                print(f"[Admin] Talerspråk var satt til slettet språk '{lang_code}', tilbakestilt til '{new_source}'.")
                
            with open("sprak_konfig.json", "w", encoding="utf-8") as f:
                json.dump(self.languages, f, indent=2, ensure_ascii=False)
                
            self.save_settings()
            print(f"[Admin] Språk '{lang_name}' ({lang_code}) ble fjernet.")
            if hasattr(self, "connected_clients") and self.connected_clients:
                await self.broadcast_system_status()
                
            return web.json_response({"status": "success", "message": f"Språk '{lang_name}' ({lang_code}) ble fjernet."}, headers=self._API_HEADERS)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500, headers=self._API_HEADERS)

    async def api_toggle_language_handler(self, request):
        """REST API (Admin): Slår et språk av eller på for kontrollpanelet uten å slette det."""
        try:
            data = await request.json()
            lang_code = data.get("lang_code", "").strip().lower()
            enabled = bool(data.get("enabled", True))
            
            if not lang_code or lang_code not in self.languages:
                return web.json_response({"error": "Språket finnes ikke"}, status=400, headers=self._API_HEADERS)
                
            # Sikre at vi ikke deaktiverer det siste gjenværende aktive språket
            if not enabled:
                currently_enabled = [k for k, v in self.languages.items() if v.get("enabled", True) and k != lang_code]
                if not currently_enabled:
                    return web.json_response({
                        "error": "Du kan ikke slå av alle språk. Minst ett språk må være aktivt for tolkestyring."
                    }, status=400, headers=self._API_HEADERS)

            self.languages[lang_code]["enabled"] = enabled
            
            # Hvis språket deaktiveres mens det er aktivt i bruk, velg et annet aktivt språk
            if not enabled:
                remaining_active = [k for k, v in self.languages.items() if v.get("enabled", True)]
                fallback_lang = "no" if "no" in remaining_active else (remaining_active[0] if remaining_active else "no")
                
                if self.config.get("texting") == lang_code:
                    self.config["texting"] = fallback_lang
                    print(f"[Admin] Teksting var satt til deaktivert språk '{lang_code}', tilbakestilt til '{fallback_lang}'.")
                    
                for i in range(1, 17):
                    ch_key = f"tts_ch{i}"
                    if self.config.get(ch_key) == lang_code:
                        self.config[ch_key] = "off"
                        print(f"[Admin] TTS-kanal {i} var satt til deaktivert språk '{lang_code}', satt til 'off' (Av).")
                        
                if self.config.get("source") == lang_code:
                    avail_whisper = [k for k in self.whisper_models.keys() if k in remaining_active]
                    new_source = avail_whisper[0] if avail_whisper else ("no" if "no" in self.whisper_models else "en")
                    if hasattr(self, "swap_models") and self.models_ready:
                        await self.swap_models("source", new_source)
                    self.config["source"] = new_source
                    print(f"[Admin] Talerspråk var satt til deaktivert språk '{lang_code}', tilbakestilt til '{new_source}'.")
                    
            with open("sprak_konfig.json", "w", encoding="utf-8") as f:
                json.dump(self.languages, f, indent=2, ensure_ascii=False)
                
            self.save_settings()
            status_str = "aktivert" if enabled else "deaktivert"
            print(f"[Admin] Språk '{lang_code}' ble {status_str} for tolkestyring.")
            if hasattr(self, "connected_clients") and self.connected_clients:
                await self.broadcast_system_status()
                
            return web.json_response({"status": "success", "lang_code": lang_code, "enabled": enabled}, headers=self._API_HEADERS)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500, headers=self._API_HEADERS)

    def generate_gemini_sample_audio(self, api_key: str, lang_name: str, lang_code: str, voice_gender: str = "male") -> bytes:
        """Kaller Google Gemini API for å generere en 10-15s studio-innspilling på målspråket."""
        import base64
        import wave
        import io

        voice_name = "Puck" if voice_gender == "male" else "Kore"
        models_to_try = ["gemini-2.0-flash", "gemini-3.8-flash-tts", "gemini-2.5-flash"]
        last_error = None

        prompt_text = (
            f"Please read the following greeting aloud in clear, natural, and fluent {lang_name} with a calm, warm, reverent tone: "
            f"'May peace, hope, and blessing be with you all. Let love and kindness fill our hearts today and always.'"
        )

        for model_id in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
            payload = {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": prompt_text}]
                }],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": voice_name
                            }
                        }
                    }
                }
            }

            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )

            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    candidates = resp_data.get("candidates", [])
                    if not candidates:
                        raise ValueError(f"Ingen respons-kandidater fra Gemini: {resp_data}")
                    
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for part in parts:
                        inline = part.get("inlineData", {})
                        if "data" in inline:
                            mime = inline.get("mimeType", "").lower()
                            raw_b64 = inline["data"]
                            audio_bytes = base64.b64decode(raw_b64)
                            
                            # Hvis rå PCM uten WAV header (f.eks. audio/x-pcm, audio/pcm, audio/L16):
                            if "pcm" in mime or "l16" in mime or not audio_bytes.startswith(b"RIFF"):
                                wav_buffer = io.BytesIO()
                                with wave.open(wav_buffer, "wb") as wf:
                                    wf.setnchannels(1)
                                    wf.setsampwidth(2)
                                    wf.setframerate(24000)
                                    wf.writeframes(audio_bytes)
                                return wav_buffer.getvalue()
                            else:
                                return audio_bytes
            except urllib.error.HTTPError as he:
                err_body = he.read().decode("utf-8", errors="replace")
                print(f"[Gemini] Modell {model_id} feilet ({he.code}): {err_body}")
                last_error = f"{model_id}: HTTP {he.code} - {err_body}"
                continue
            except Exception as e:
                print(f"[Gemini] Feil med {model_id}: {e}")
                last_error = f"{model_id}: {e}"
                continue

        raise RuntimeError(f"Kunne ikke generere lyd med Gemini API: {last_error}")

    async def api_set_gemini_key_handler(self, request):
        """REST API: Lagrer Google AI Studio Gemini API-nøkkel til system_innstillinger.json."""
        try:
            body = await request.json()
            key = body.get("gemini_api_key", "").strip()
            self.config["gemini_api_key"] = key
            self.save_settings()
            print(f"[Admin] Oppdaterte Gemini API-nøkkel ({'Satt' if key else 'Slettet'}).")
            return web.json_response({
                "status": "success",
                "has_gemini_key": bool(key)
            }, headers=self._API_HEADERS)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500, headers=self._API_HEADERS)

    async def run_api_server(self):
        """Tråd 6: Kjører REST API på port 8080 for ekstern kontroll."""
        app = web.Application()
        app.router.add_get('/api/status', self.api_status_handler)
        app.router.add_get('/api/set/{category}/{lang}', self.api_set_handler)
        app.router.add_get('/api/restart', self.api_restart_handler)
        
        # Admin endpoints
        app.router.add_post('/api/admin/add_language', self.api_add_language_handler)
        app.router.add_post('/api/admin/delete_language', self.api_delete_language_handler)
        app.router.add_post('/api/admin/toggle_language', self.api_toggle_language_handler)
        app.router.add_post('/api/admin/upload_voice', self.api_upload_voice_handler)
        app.router.add_post('/api/admin/set_hw_channels', self.api_set_hw_channels_handler)
        app.router.add_post('/api/admin/set_model', self.api_set_model_handler)
        app.router.add_post('/api/admin/start_mixer_recording', self.api_start_mixer_recording_handler)
        app.router.add_post('/api/admin/stop_mixer_recording', self.api_stop_mixer_recording_handler)
        app.router.add_get('/api/admin/recorded_sample_audio', self.api_recorded_sample_audio_handler)
        app.router.add_get('/api/admin/mic_level', self.api_mic_level_handler)
        app.router.add_get('/api/admin/dictionary', self.api_get_dictionary_handler)
        app.router.add_post('/api/admin/dictionary', self.api_update_dictionary_handler)
        app.router.add_get('/api/admin/pronunciation', self.api_get_pronunciation_handler)
        app.router.add_post('/api/admin/pronunciation', self.api_update_pronunciation_handler)
        app.router.add_post('/api/admin/set_gemini_key', self.api_set_gemini_key_handler)

        # Flerspors Lydopptak (Tale og Tolk)
        app.router.add_post('/api/recording/start', self.api_start_multitrack_recording_handler)
        app.router.add_post('/api/recording/stop', self.api_stop_multitrack_recording_handler)
        app.router.add_get('/api/recording/status', self.api_recording_status_handler)
        app.router.add_get('/api/recording/list', self.api_list_recordings_handler)
        app.router.add_get('/api/recording/download/{filename}', self.api_download_recording_handler)
        app.router.add_get('/api/recording/stream/{filename}', self.api_stream_recording_handler)
        app.router.add_delete('/api/recording/delete/{id}', self.api_delete_recording_handler)
        
        html_no_cache = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"}
        app.router.add_get('/kontrollpanel', lambda r: web.FileResponse('kontrollpanel.html', headers=html_no_cache))
        app.router.add_get('/skjerm', lambda r: web.FileResponse('skjerm.html', headers=html_no_cache))
        app.router.add_get('/admin', lambda r: web.FileResponse('admin_panel.html', headers=html_no_cache))
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', 8080)
        await site.start()
        print("[System] Kontrollpanel: http://localhost:8080/kontrollpanel")
        print("[System] Skjerm: http://localhost:8080/skjerm")
        print("[System] Adminpanel: http://localhost:8080/admin")
        print("[System] REST API lytter på port 8080")
        while True:
            await asyncio.sleep(3600)

    async def run(self):
        """Motoren: Starter alle trådene inkludert nettleser-serveren og REST API."""
        print("[System] Starter KI-Tolken...")
        print("[System] Starter webserver og WebSockets umiddelbart så QR-koder svarer...")
        
        async with websockets.serve(self.ws_handler, "0.0.0.0", 8765):
            tasks = [
                asyncio.create_task(self.run_api_server()),     # Starter webgrensesnitt umiddelbart!
                asyncio.create_task(self.init_models_async()),  # Laster modeller asynkront med feilhåndtering
                asyncio.create_task(self.capture_and_vad()),
                asyncio.create_task(self.transcribe_worker()),
                asyncio.create_task(self.translate_worker()),
                asyncio.create_task(self.tts_and_stereo_playback()),
                asyncio.create_task(self.broadcast_worker())
            ]
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    pipeline = AudioPipeline()
    try:
        asyncio.run(pipeline.run())
    except KeyboardInterrupt:
        print("\n[System] Avsluttet manuelt.")