# Setup

Opticolumn runs on **Python 3.11** with **PyTorch 2.6.0 or newer** (required by CVE-2025-32434). Run all commands from inside the `opticolumn` folder.

## macOS

**1. Install pyenv**

```bash
brew install pyenv xz
SHELL_RC="$HOME/.$(basename "$SHELL")rc"
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$SHELL_RC"
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> "$SHELL_RC"
echo 'eval "$(pyenv init -)"' >> "$SHELL_RC"
exec "$SHELL"
```

**2. Install Python 3.11.9** (with the compression support Kraken's models need)

```bash
env PYTHON_CONFIGURE_OPTS="--with-liblzma" \
    LDFLAGS="-L$(brew --prefix xz)/lib" \
    CPPFLAGS="-I$(brew --prefix xz)/include" \
    pyenv install 3.11.9
pyenv local 3.11.9
```

**3. Create the environment and install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install --upgrade torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

**4. Run**

```bash
python script.py                  # OCR: A → B
python review.py                  # review images: B → C
python report.py                  # audit report: A + B → D
```

_To keep your Mac awake during long batches:_

```bash
caffeinate -i python script.py    # prevents idle sleep
caffeinate -di python script.py   # also keeps the display on
```

## Windows

Kraken, the segmentation model Opticolumn uses, runs only on Linux and macOS, so on Windows Opticolumn runs inside **WSL** (Windows Subsystem for Linux).

**1. Install WSL** (in PowerShell, run as administrator)

```powershell
wsl --install
```

Restart when prompted, then open **Ubuntu** from the Start menu and create a username and password. Run every remaining command in the Ubuntu window.

**2. Install pyenv**

```bash
sudo apt update && sudo apt install -y build-essential curl git libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev libncursesw5-dev xz-utils tk-dev \
  libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
exec "$SHELL"
```

**3. Install Python 3.11.9**

```bash
pyenv install 3.11.9
pyenv local 3.11.9
```

**4. Create the environment and install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install --upgrade torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

**5. Run**

```bash
python script.py                  # OCR: A → B
python review.py                  # review images: B → C
python report.py                  # audit report: A + B → D
```

_Your Windows drives are available inside Ubuntu under `/mnt/` (for example, `C:\Users\you\Documents\opticolumn` is `/mnt/c/Users/you/Documents/opticolumn`). Processing is faster if the `opticolumn` folder lives in your Ubuntu home folder (`~`) instead._

_To keep your PC awake during long batches_, note your current settings with `powercfg /query` in PowerShell, then turn sleep off for the run and restore your values afterward:

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
```

## Next time

Reactivate the environment before running any script:

```bash
source .venv/bin/activate
```

## If a file stops the batch

If processing stops with a message ending in `Killed`, a single PDF has used up the computer's memory. The culprit is the file in the last `Processing …` line shown in the terminal before `Killed`.

1. Move that file from the `A` folder to the `review` folder.
2. Run the script again. Files already in `B` are skipped, so processing picks up where it stopped.

```bash
caffeinate -di python script.py   # macOS
python script.py                  # Windows (WSL)
```

If the file is left in `A`, the script will reach it again and stop at the same point. This affects roughly 1 in 300 files; files collected in `review` are used to troubleshoot future versions of the script.