# For macOS
brew install pyenv

# Set up pyenv
SHELL_RC="$HOME/.$(basename "$SHELL")rc"
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$SHELL_RC"
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> "$SHELL_RC"
echo 'eval "$(pyenv init -)"' >> "$SHELL_RC"

exec "$SHELL"

# Install specific Python version
pyenv install 3.11.9

# Create and activate virtual environment
pyenv virtualenv 3.11.9 .venv
pyenv activate .venv

python -m venv .venv
.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

brew install poppler

pip install --upgrade torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

python script.py

python review.py