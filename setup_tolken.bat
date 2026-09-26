@echo off
chcp 65001 >nul
echo =======================================================
echo     AUTOMATISK INSTALLASJON AV KI-TOLKEN PÅ LINUX-PC
echo =======================================================
echo.
set /p IP="1. Skriv inn den statiske IP-adressen til Tolke-PCen (f.eks. 192.168.1.50): "
set /p USERNAME="2. Skriv inn brukernavnet på Tolke-PCen: "
echo.
echo =======================================================
echo Starter automatisk oppsett via nettverket (SSH). 
echo Du vil bli bedt om å skrive inn passordet til Tolke-PCen 
echo for å logge inn, og for admin-rettigheter underveis.
echo (Teksten vises ikke når du skriver passordet).
echo.
echo =======================================================
echo Setter opp passordløs innlogging (SSH-nøkkel) for fremtiden...
echo =======================================================
if not exist "%USERPROFILE%\.ssh\id_rsa" (
    ssh-keygen -t rsa -b 4096 -N "" -f "%USERPROFILE%\.ssh\id_rsa"
)
type "%USERPROFILE%\.ssh\id_rsa.pub" | ssh %USERNAME%@%IP% "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"

echo.
echo =======================================================
echo Starter selve installasjonen av KI-Tolken på Linux...
echo =======================================================
ssh -t %USERNAME%@%IP% "sudo apt update && sudo apt install -y git && rm -rf ~/Tolk && git clone https://github.com/jorbje/KI-tolken.git ~/Tolk && cd ~/Tolk && chmod +x install_remote.sh && ./install_remote.sh"

echo.
echo =======================================================
echo Installasjonen er fullført! Datamaskinen vil nå starte 
echo på nytt. Etter omstart kjører KI-Tolken automatisk!
echo =======================================================
pause
