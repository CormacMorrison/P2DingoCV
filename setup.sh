#!/bin/bash
set -e  # Exit on any error
sudo apt install pipx -y

# -------- Ensure Poetry is installed --------
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing..."
    pipx install poetry
    # Add Poetry to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"

    # Persist PATH for future sessions (Bash or Zsh)
    SHELL_CONFIG=""
    if [ -n "$BASH_VERSION" ]; then
        SHELL_CONFIG="$HOME/.bashrc"
    elif [ -n "$ZSH_VERSION" ]; then
        SHELL_CONFIG="$HOME/.zshrc"
    fi

    if [ -n "$SHELL_CONFIG" ] && ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$SHELL_CONFIG"; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_CONFIG"
        echo "Updated $SHELL_CONFIG to include Poetry in PATH."
    fi
fi

# -------- Navigate to script directory (repo root) --------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# -------- Install dependencies --------
echo "Installing project dependencies..."
poetry install --no-interaction --no-ansi

# -------- Optional: run commands inside the Poetry environment --------
# Example: run tests
# poetry run pytest

echo "Poetry environment setup complete."

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