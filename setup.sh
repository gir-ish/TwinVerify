#!/bin/bash

# Install system dependencies
sudo apt-get update
sudo apt-get install -y portaudio19-dev
sudo apt install alsa-utils
sudo apt-get update
sudo apt install pulseaudio -y

# Install Python dependencies
#sudo pip install -r requirements.txt
sudo pip install -r requirements_audio_vault.txt