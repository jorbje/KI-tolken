@echo off
chcp 65001 >nul
echo =======================================================
echo    AUTOMATISK INSTALLASJON AV KI-TOLKEN (WINDOWS 11)
echo =======================================================
echo.
echo Dette skriptet vil automatisk laste ned og installere:
echo - Python 3.10+
echo - Git
echo - KI-Tolken kildekode
echo - Alle AI-modeller og PyTorch for skjermkortet ditt
echo.
echo Sørg for at du er koblet til internett!
pause

echo.
echo [1/5] Installerer Python og Git via Windows Package Manager...
winget install --id Python.Python.3.10 -e --accept-package-agreements --accept-source-agreements --silent
winget install --id Git.Git -e --accept-package-agreements --accept-source-agreements --silent

echo.
echo [2/5] Laster inn nye systemvariabler...
call "%PROGRAMFILES%\Python310\Scripts\env.bat" 2>nul
set PATH=%PATH%;%LOCALAPPDATA%\Programs\Python\Python310\Scripts;%LOCALAPPDATA%\Programs\Python\Python310\;%PROGRAMFILES%\Git\cmd

echo.
echo [3/5] Laster ned KI-Tolken fra GitHub...
cd %USERPROFILE%
if exist "Tolk" (
    echo Mappen "Tolk" finnes allerede, sletter gammel versjon...
    rmdir /s /q Tolk
)
git clone <din-github-url-her> Tolk
cd Tolk

echo.
echo [4/5] Installerer AI-biblioteker (Dette kan ta litt tid)...
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

echo.
echo [5/5] Setter opp automatisk start i bakgrunnen...
set STARTUP_DIR="%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set SHORTCUT="%STARTUP_DIR%\KI-Tolken.bat"

echo @echo off > %SHORTCUT%
echo cd "%USERPROFILE%\Tolk" >> %SHORTCUT%
echo start /min python core_pipeline.py >> %SHORTCUT%

echo.
echo =======================================================
echo GRATULERER! KI-Tolken er ferdig installert!
echo Hver gang denne PC-en skrus på, vil tolkemotoren starte
echo automatisk i bakgrunnen. 
echo.
echo PC-en din er nå en dedikert Tolke-PC. Finn ut IP-adressen 
echo (ved å skrive ipconfig i CMD), slik at mobiler på nettverket
echo kan koble til kontrollpanelet.
echo =======================================================
pause
