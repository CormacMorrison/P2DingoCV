#!/bin/bash

# Exit on error
set -e

echo "Installing pip if not present..."
if ! command -v pip3 &> /dev/null; then
    sudo apt install -y python3-pip
fi

echo "Updating package list..."
sudo apt update

echo "Installing Tesseract OCR engine and dependencies..."
sudo apt install -y tesseract-ocr libtesseract-dev

echo "Verifying installation..."
tesseract --version
python3 -c "import pytesseract; print('pytesseract installed successfully:', pytesseract.get_tesseract_version())"

echo "Installation complete."sudo apt update
sudo apt upgrade
sudo apt install 