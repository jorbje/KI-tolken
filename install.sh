#!/bin/bash
echo "=========================================="
echo " Installerer KI-Tolken - Avhengigheter"
echo "=========================================="
echo ""

echo "[1/4] Installerer system-avhengigheter (FFmpeg-biblioteker for Linux)..."
sudo apt update
sudo apt install -y pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev
echo ""

echo "[2/4] Oppdaterer pip..."
python3 -m pip install --upgrade pip
echo ""

echo "[3/4] Installerer PyTorch for CUDA 12.4 (Kritisk for AI)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
echo ""

echo "[4/4] Installerer pakker fra requirements.txt..."
pip install -r requirements.txt
echo ""

echo "=========================================="
echo " Installasjon ferdig! Systemet er klart."
echo "=========================================="