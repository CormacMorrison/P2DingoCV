# Temparature Detection Module

# #!/bin/bash

# # Exit on error
# set -e

# echo "Installing pip if not present..."
# if ! command -v pip3 &> /dev/null; then
#     sudo apt install -y python3-pip
# fi

# echo "Updating package list..."
# sudo apt update

# echo "Installing Tesseract OCR engine and dependencies..."
# sudo apt install -y tesseract-ocr libtesseract-dev

# echo "Verifying installation..."
# tesseract --version
# python3 -c "import pytesseract; print('pytesseract installed successfully:', pytesseract.get_tesseract_version())"

# echo "Installation complete."sudo apt update
# sudo apt upgrade
# sudo apt install 

#!/bin/bash
set -e

# ---- Config ----
ENTRY="src/P2DingoCV/cli.py"        # Main entrypoint
OUTDIR="build"            # Output directory
EXE_NAME="p2dingocv"      # Name of the compiled binary

# Clean old build
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

# ---- Compile with Nuitka ----
poetry run python3 -m nuitka \
  --standalone \
  --follow-imports \
  --include-package=HotspotLogic \
  --include-package=Camera \
  --onefile \
  --output-dir="$OUTDIR" \
  --output-filename="$EXE_NAME" \
  "$ENTRY"

echo "✅ Build complete!"
echo "Executable is at $OUTDIR/$EXE_NAME"
