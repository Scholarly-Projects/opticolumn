# Install system dependencies
brew install pyenv xz ninja

# Set up pyenv in your shell
SHELL_RC="$HOME/.$(basename "$SHELL")rc"
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$SHELL_RC"
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> "$SHELL_RC"
echo 'eval "$(pyenv init -)"' >> "$SHELL_RC"

# Reload shell
exec "$SHELL"

# Install Python 3.11.9 with explicit _lzma (xz) support
env PYTHON_CONFIGURE_OPTS="--with-liblzma" \
    LDFLAGS="-L$(brew --prefix xz)/lib" \
    CPPFLAGS="-I$(brew --prefix xz)/include" \
    pyenv install 3.11.9

# Set this Python version for the current project
pyenv local 3.11.9

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Upgrade pip and install project dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install Poppler for pdf2image
brew install poppler

# Optional: Set Ninja build flags (not needed for CPU-only PyTorch wheels, but harmless)
export CMAKE_GENERATOR="Ninja"
export USE_NINJA=ON
# Note: TORCH_CUDA_ARCH_LIST is irrelevant on macOS (no CUDA support); safe to omit

# Install PyTorch 2.6.0+ (required due to CVE-2025-32434)
pip install --upgrade torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Run your scripts
python script.py
or avoid idle sleep
caffeinate -i python script.py
or avoid display sleep
caffeinate -di python script.py
and
python review.py
or 
python report.py

# For CDIL Reference:

If the script is "killed" by a PDF file, find the file name in the terminal below and move that file from the "A" folder to the "Review" folder before running:

caffeinate -di python script.py

To continue processing the collection. Otherwise, the **problem file** will be processed and kill the script again. These files in the review folder will be analyzed to troubleshoot the script in future iterations -- but they seem to be around 1 in 300 at the moment.

# If you are getting false positives on blank pages

CONFIDENCE_THRESHOLD           = 0.85
SINGLE_CHAR_CONFIDENCE_THRESHOLD = 0.85
MIN_SEGMENT_HEIGHT             = 20
