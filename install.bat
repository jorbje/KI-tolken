@echo off
echo ==========================================
echo  Installerer KI-Tolken - Avhengigheter
echo ==========================================
echo.
echo [1/3] Oppdaterer pip...
python -m pip install --upgrade pip
echo.
echo [2/3] Installerer PyTorch for CUDA 12.4 (Kritisk for AI)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
echo.
echo [3/3] Installerer pakker fra requirements.txt...
pip install -r requirements.txt
echo.
echo ==========================================
echo  Installasjon ferdig! Systemet er klart.
echo ==========================================
pause