# Temparature Detection Module

# #!/bin/bash

# # Exit on error
# set -e



#!/bin/bash
set -e
sudo apt install python3-dev

# ---- Config ----
ENTRY="src/P2DingoCV/cli.py"        # Main entrypoint
OUTDIR="build"            # Output directory
EXE_NAME="hotspot-cli"      # Name of the compiled binary

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

echo "Build complete!"
echo "Executable is at $OUTDIR/$EXE_NAME"
