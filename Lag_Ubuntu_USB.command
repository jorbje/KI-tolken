#!/bin/bash
echo "======================================================="
echo "      STEG 1: KLARGJØR MINNEPENN MED UBUNTU 24.04"
echo "======================================================="
echo ""
echo "Dette programmet vil nå automatisk laste ned den"
echo "ENESTE riktige versjonen av Ubuntu (24.04 LTS) som"
echo "garantert fungerer med tolkemotoren."
echo ""
echo "[1/2] Laster ned Ubuntu 24.04 LTS (Dette tar tid - ca. 5.7 GB)..."
if [ ! -f "ubuntu-24.04.1-desktop-amd64.iso" ]; then
    curl -L -o ubuntu-24.04.1-desktop-amd64.iso https://releases.ubuntu.com/24.04.1/ubuntu-24.04.1-desktop-amd64.iso
else
    echo "Filen finnes allerede, hopper over nedlasting."
fi

echo ""
echo "[2/2] Åpner nettsiden for BalenaEtcher..."
echo "Siden du bruker Mac, er det tryggest å bruke programmet BalenaEtcher for å lage minnepennen."
echo "Last ned og installer programmet fra nettsiden som nettopp åpnet seg."
open "https://etcher.balena.io/"

echo ""
echo "======================================================="
echo "VIKTIG:"
echo "1. Start BalenaEtcher på Macen din."
echo "2. Velg 'Flash from file' og finn ubuntu-24.04.1-desktop-amd64.iso som ble lastet ned i denne mappen."
echo "3. Sett inn en minnepenn og velg den under 'Select target'."
echo "4. Trykk 'Flash!' (Alt på minnepennen slettes)."
echo "======================================================="
echo ""
echo "Når den er 100% ferdig: Ta ut minnepennen og sett den i Tolke-PCen for å installere Ubuntu."
echo "(Når Ubuntu er ferdig installert der, går du til STEG 2: setup_tolken.command)"
