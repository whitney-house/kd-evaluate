#!/bin/bash
# One-time environment setup.
# Run this from inside your large-storage directory, e.g.:
#   cd ~/temp/kd-evaluate && bash setup.sh
#
# IMPORTANT: on shared servers the home directory quota is often only ~4-6GB,
# which is not enough for a conda env + pip cache + model weights.
# Everything below is deliberately installed under the current directory
# (assumed to be on large/scratch storage), NOT under $HOME.
set -e

PROJECT_DIR="$(pwd)"
ENV_DIR="$PROJECT_DIR/envs/easyedit"

# 1. Clone EasyEdit only if it isn't already there (safe to re-run)
if [ ! -d "EasyEdit" ]; then
    git clone https://github.com/zjunlp/EasyEdit.git
fi
cd EasyEdit

# 2. Create the conda env with --prefix so it lives on this storage,
#    not in ~/.conda/envs (which counts against home quota).
#    Python 3.10+ is required -- EasyEdit's requirements.txt pins
#    torch>=2.9, which has no wheels for Python 3.9.
if [ ! -d "$ENV_DIR" ]; then
    conda create --prefix "$ENV_DIR" python=3.10 -y
fi

# Explicitly source conda's hook so `conda activate` works in a
# non-interactive script (a plain `source activate` will not work here).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# 3. Redirect pip / HuggingFace / temp-file caches off the home quota too.
#    Add these three export lines to ~/.bashrc so they persist across sessions.
export PIP_CACHE_DIR="$PROJECT_DIR/pip-cache"
export HF_HOME="$PROJECT_DIR/huggingface-cache"
export TMPDIR="$PROJECT_DIR/tmp"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TMPDIR"

pip install -r requirements.txt

# 4. Copy this project's own script and data into the EasyEdit repo
#    (EasyEdit uses relative imports, so scripts must be run from its root).
cp ../scripts/run_single_edit.py .
mkdir -p data
cp ../data/single_edit.json ./data/single_edit.json

echo ""
echo "Setup complete."
echo "Next steps:"
echo "  1. huggingface-cli login   (needed to download model weights)"
echo "  2. Check hparams/ROME/<model>.yaml -- set model_name to the HF repo id"
echo "     and device to a free GPU (check with nvidia-smi)"
echo "  3. python run_single_edit.py"
echo ""
echo "To reactivate this env in a new terminal session, run:"
echo "  source \"\$(conda info --base)/etc/profile.d/conda.sh\""
echo "  conda activate $ENV_DIR"
