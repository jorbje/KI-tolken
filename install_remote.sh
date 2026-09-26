#!/bin/bash
echo "======================================================="
echo "  LINUX INSTALLASJON AV KI-TOLKEN (AUTOMATISK SCRIPT)"
echo "======================================================="
echo ""
echo "Oppdaterer Linux..."
sudo apt update
sudo apt upgrade -y

echo "Installerer nødvendige verktøy og FFmpeg..."
sudo apt install -y python3-pip python3-venv pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev ubuntu-drivers-common

echo "Installerer offisielle NVIDIA-drivere for skjermkortet..."
sudo ubuntu-drivers autoinstall

echo "Låser (fryser) skjermkort-driverne og kjernen for å hindre automatiske oppdateringer som krasjer CUDA..."
sudo apt-mark hold "^nvidia-.*" "^libnvidia-.*"
sudo apt-mark hold linux-image-generic linux-headers-generic
sudo apt-mark hold "^cuda-.*"

echo "Setter opp Python virtuelt miljø..."
cd ~/Tolk
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

echo "Installerer PyTorch for CUDA (Kritisk for AI)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "Installerer KI-Tolken avhengigheter..."
pip install -r requirements.txt

echo "Lager en tjeneste (systemd) slik at KI-Tolken starter automatisk når PCen skrus på..."
cat <<EOF | sudo tee /etc/systemd/system/kitolken.service
[Unit]
Description=KI-Tolken AI Service
After=network.target

[Service]
User=$USER
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/venv/bin/python core_pipeline.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable kitolken
sudo systemctl start kitolken

echo "Alt er ferdig satt opp!"
echo "PCen starter nå på nytt for å aktivere skjermkort-driverne."
sudo reboot
