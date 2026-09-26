# KI-Tolken

KI-Tolken er et 100 % lokalt, asynkront AI-system utviklet for live-tolking og teksting i sanntid. Det lytter til taleren (på feks. norsk eller engelsk) og oversetter automatisk med tekst-til-tale (TTS) ut på flere lydkanaler, og tekst på skjerm. 

Dette prosjektet er spesialtilpasset for kirkelig kontekst (unngår hallusinerte begreper), men kan tilpasses ethvert bruk via ordlister.

## Anbefalt Maskinvare (Hardware)

KI-Tolken kjører **lokalt** (100% offline etter oppsett), og ytelsen avhenger ene og alene av maskinens **skjermkort (GPU)**:

### 1. Skjermkort (GPU) - Det aller viktigste!
Avhengig av hvor mange språk du ønsker å tolke til **samtidig**:
- **For 2 språk (Standard):** Nvidia RTX 3060 (12GB VRAM) eller RTX 4060 Ti (16GB). *Minnet må være minimum 12GB for å unngå VRAM-krasj.*
- **For 3-4 språk:** Nvidia RTX 4070 Ti Super (16GB), RTX 4080 (16GB) eller RTX 4090 (24GB). Fordi Tekst-til-tale genereringen (OmniVoice) er svært krevende, vil lydforsinkelsen øke raskt for hvert nye språk hvis GPUen ikke er rask nok.

### 2. Annet
- **Prosessor (CPU):** Valgfri, men en moderne Intel Core i5/i7 eller AMD Ryzen 5/7 er anbefalt for å håndtere Python-tråder.
- **Minne (RAM):** 32 GB RAM (for å laste modellene inn fra disk før de flyttes til VRAM).
- **Minnepenn (USB):** Minst **8 GB** (til installasjon av operativsystemet).
## Installasjon for Ikke-Tekniske Brukere (Automagisk oppsett på Headless Linux)

For å sikre at tolkemaskinen fungerer som en 100% stabil "appliance" (som en ruter) uten tvungne oppdateringer, mus eller tastatur, anbefales en **headless Linux-installasjon**.

> 💡 **Hva trenger du å laste ned på din vanlige PC/Mac?**
> Du trenger **ikke** laste ned hele kildekoden eller koble opp AI-miljøet på din vanlige PC. Du laster kun ned den lille oppsettspakken **[Oppsett_KI-Tolken.zip](https://github.com/jorbje/KI-tolken/releases/latest)** (eller fra GitHub Releases). Den inneholder kun de få skriptene som trengs for å klargjøre Tolke-PCen.

Etter at du har kjøpt en passende PC (med et RTX 3060 eller bedre skjermkort), gjør du følgende:

1. **Lag Installasjons-minnepenn (Steg 1):** 
   Pakk ut den lille zip-filen på din vanlige Windows-PC eller Mac. Sett inn en tom minnepenn på **minst 8 GB** i maskinen (Ubuntu tar ca. 5.7 GB, så 4 GB blir for lite), og kjør:
   - For **Windows**: Dobbeltklikk på `Lag_Ubuntu_USB.bat`
   - For **Mac**: Dobbeltklikk på `Lag_Ubuntu_USB.command`
   
   *(Du skal **ikke** lete etter eller laste ned Ubuntu selv! Skriptet laster automatisk ned nøyaktig riktig, testet versjon av Ubuntu 24.04 og nødvendig programvare for å legge det på minnepennen).*
2. **Klargjør Tolke-PCen:** 
   Sett minnepennen i den nye Tolke-PCen, koble til en skjerm og et tastatur, og installer **Ubuntu 24.04**. Når du installerer, sørg for å huke av for at du vil skru på **SSH**.
3. **Koble til nettverk og sett Fast IP:** 
   Koble Tolke-PCen til nettverket/internettet med kabel eller WiFi. Gå inn i internett-ruteren deres (spør om hjelp om nødvendig) og lås IP-adressen til Tolke-PCen (sett en statisk IP, for eksempel `192.168.1.50`). Noter ned denne IP-adressen!
   *(Etter dette kan du koble fra skjerm, mus og tastatur og sette maskinen inn i et skap).*
4. **Kjør Installasjonsprogrammet (Steg 2):**
   Gå tilbake til din vanlige Windows-PC eller Mac. Nå skal vi fjernstyre den nye maskinen og installere all AI-teknologien. Kjør:
   - For **Windows**: Dobbeltklikk på `setup_tolken.bat`
   - For **Mac**: Dobbeltklikk på `setup_tolken.command` (eller kjør den fra terminalen)
   
   Programmet vil be om IP-adressen du noterte og passordet til Tolke-PCen.
5. **Len deg tilbake:**
   Skriptet vil logge seg inn på Tolke-PCen (SSH) og gjøre alt i bakgrunnen:
   - Sette opp sikker, passordløs tilkobling (SSH-nøkler).
   - Installere CUDA og KI-biblioteker.
   - Installere Tolkeprogrammet og en systemtjeneste (`systemd`).
   - **Låse fast** Linux-kjernen og Nvidia-driverne slik at operativsystemet *aldri* kan oppdatere dem automatisk og ødelegge for CUDA!
   Når den er ferdig, starter Tolke-PCen lynraskt på nytt, og KI-Tolken starter automatisk!

*(Dersom du absolutt vil bruke Windows i stedet for Linux, finnes det et lokalt installasjonsskript `install_windows.bat` i mappen, men dette anbefales ikke for en 100% headless løsning pga Windows Updates).*

## Systemets Grensesnitt og Fjernbetjening

Når `core_pipeline.py` kjører, oppretter den en lokal webserver. Koble mobiltelefoner og nettbrett til det samme WiFi-nettverket!

- **Skjerm for hørselshemmede (Teksting):** `http://<IP-til-tolkemaskinen>:8080/skjerm`
- **Tolke-ansvarlig / Mobil (QR-Kode panel):** `http://<IP-til-tolkemaskinen>:8080/kontrollpanel`
- **Teknisk Ansvarlig (Admin Panel):** `http://<IP-til-tolkemaskinen>:8080/admin`

### Administrasjon og nye språk
1. Gå til **Admin Panelet** fra en PC.
2. Der kan du spesifisere **hvor mange språk** som skal tolkes samtidig (basert på lydkortets utganger).
3. I dette panelet kan du også **laste opp en lydprøve** (.wav/.mp3) av en person for å klone stemmen. Systemet lager automatisk en ny `.pt` stemmeprofil!
4. Valgene du gjør i Admin-panelet overstyrer umiddelbart hva tolke-ansvarlig ser på mobilen sin i det vanlige Kontrollpanelet.

## Kjente Begrensninger og Løsninger
- **"Gummistrikk-effekt" på lyden:** Hvis tolken velger mange språk (3+) og skjermkortet er for svakt, vil TTS-køen vokse. Systemet er programmert med en "dynamisk gasspedal" som skrur opp lesehastigheten inntil 1.5x, men ved for svak hardware vil det henge bak. Gå inn i Admin-panelet og reduser antall kanaler til GPUen byttes.
- **Feil oversettelse av kirkelige ord:** Rediger filen `oversettelse_ordliste.json` (eller gjør det direkte i Admin-panelet). Systemet bruker "Named Entity Masking" som midlertidig bytter ut ordet ditt med en hemmelig kode (f.eks M9901) for å forhindre at AI'en (NLLB) oversetter det feil.
- **Feil uttale:** Gå til Admin-panelet under **«Fonetisk Uttale»** (eller rediger filen `uttale_ordliste.json`). Her kan du legge inn fonetiske omkodinger (f.eks "Herren": "Hærren") som gjelder KUN for opplesing (skjermen tekstes normalt).
