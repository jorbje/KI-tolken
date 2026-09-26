#!/bin/bash
echo "======================================================="
echo "    AUTOMATISK INSTALLASJON AV KI-TOLKEN PÅ LINUX-PC"
echo "======================================================="
echo ""
read -p "1. Skriv inn den statiske IP-adressen til Tolke-PCen (f.eks. 192.168.1.50): " IP
read -p "2. Skriv inn brukernavnet på Tolke-PCen: " USERNAME
echo ""
echo "======================================================="
echo "Starter automatisk oppsett via nettverket (SSH)."
echo "Du vil bli bedt om å skrive inn passordet til Tolke-PCen"
echo "for å logge inn, og for admin-rettigheter underveis."
echo "(Teksten vises ikke når du skriver passordet)."
echo ""
echo "======================================================="
echo "Setter opp passordløs innlogging (SSH-nøkkel) for fremtiden..."
echo "======================================================="
if [ ! -f ~/.ssh/id_rsa ]; then
  ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa
fi
ssh-copy-id $USERNAME@$IP

echo ""
echo "======================================================="
echo "Starter selve installasjonen av KI-Tolken på Linux..."
echo "======================================================="
ssh -t $USERNAME@$IP "sudo apt update && sudo apt install -y git && rm -rf ~/Tolk && git clone https://github.com/jorbje/KI-tolken.git ~/Tolk && cd ~/Tolk && chmod +x install_remote.sh && ./install_remote.sh"

echo ""
echo "======================================================="
echo "Installasjonen er fullført! Datamaskinen vil nå starte "
echo "på nytt. Etter omstart kjører KI-Tolken automatisk!"
echo "======================================================="
