@echo off
chcp 65001 >nul
echo =======================================================
echo      STEG 1: KLARGJØR MINNEPENN MED UBUNTU 24.04
echo =======================================================
echo.
echo Dette programmet vil nå automatisk laste ned den 
echo ENESTE riktige versjonen av Ubuntu (24.04 LTS) som 
echo garantert fungerer med tolkemotoren.
echo.
echo Vennligst sett inn en tom minnepenn i denne PCen nå.
echo Trykk en tast når minnepennen er satt inn...
pause >nul

echo.
echo [1/3] Laster ned Ubuntu 24.04 LTS (Dette tar tid - ca. 5.7 GB)...
if not exist "ubuntu-24.04.1-desktop-amd64.iso" (
    powershell -Command "Invoke-WebRequest -Uri 'https://releases.ubuntu.com/24.04.1/ubuntu-24.04.1-desktop-amd64.iso' -OutFile 'ubuntu-24.04.1-desktop-amd64.iso'"
) else (
    echo Filen finnes allerede, hopper over nedlasting.
)

echo.
echo [2/3] Laster ned Rufus (Programmet som brenner over til minnepennen)...
if not exist "rufus.exe" (
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/pbatard/rufus/releases/download/v4.5/rufus-4.5p.exe' -OutFile 'rufus.exe'"
)

echo.
echo [3/3] Starter Rufus...
echo =======================================================
echo VIKTIG:
echo 1. I Rufus-vinduet som nå åpner seg, sjekk at din minnepenn er valgt øverst under "Enhet".
echo 2. Alt annet er ferdig utfylt. Bare trykk "START" nederst.
echo 3. Hvis den spør om tillatelser eller manglende filer, trykk "Ja" / "OK".
echo 4. Alt på minnepennen blir slettet!
echo =======================================================
echo.

start "" rufus.exe ubuntu-24.04.1-desktop-amd64.iso

echo Når Rufus er 100%% ferdig, trykk "Lukk".
echo Ta ut minnepennen og sett den i Tolke-PCen for å installere Ubuntu.
echo.
echo (Når Ubuntu er ferdig installert der, går du til STEG 2: setup_tolken.bat)
pause
