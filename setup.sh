#!/bin/bash
set -e

PROJECT_DIR="$(pwd)"
ENV_DIR="$PROJECT_DIR/envs/easyedit"

if [ ! -d "EasyEdit" ]; then
    git clone https://github.com/zjunlp/EasyEdit.git
fi
cd EasyEdit

if [ ! -d "$ENV_DIR" ]; then
    conda create --prefix "$ENV_DIR" python=3.10 -y
fi

PIP="$ENV_DIR/bin/pip"

export PIP_CACHE_DIR="$PROJECT_DIR/pip-cache"
export HF_HOME="$PROJECT_DIR/huggingface-cache"
export TMPDIR="$PROJECT_DIR/tmp"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TMPDIR"

"$PIP" install -r requirements.txt

cp ../scripts/run_single_edit.py .
mkdir -p data
cp ../data/single_edit.json ./data/single_edit.json

# Persist env vars into ~/.bashrc, but only once — avoid duplicate
# entries if this script is re-run.
MARKER="# >>> kd-evaluate env (auto-added by setup.sh) >>>"
if ! grep -qF "$MARKER" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << EOF
$MARKER
export HF_HOME="$PROJECT_DIR/huggingface-cache"
export PIP_CACHE_DIR="$PROJECT_DIR/pip-cache"
export TMPDIR="$PROJECT_DIR/tmp"
export PATH="$ENV_DIR/bin:\$PATH"
# <<< kd-evaluate env (auto-added by setup.sh) <
EOF
    echo "Added env vars to ~/.bashrc"
else
    echo "~/.bashrc already has these env vars, skipped"
fi

echo ""
echo "Setup complete. Run 'source ~/.bashrc' or open a new terminal to activate."