#!/bin/bash
# Patch a hparams yaml's model_name (to a HuggingFace repo id) and device
# (to a free GPU index), so you don't have to hand-write sed commands
# every time you add a new model.
#
# Usage:
#   bash patch_hparams.sh <method> <model_yaml_stem> <hf_repo_id> <device_index>
#
# Example:
#   bash patch_hparams.sh ROME gpt2-xl gpt2-xl 2
#   bash patch_hparams.sh ROME qwen2.5-7b Qwen/Qwen2.5-7B 3
set -e

METHOD="$1"
MODEL_STEM="$2"
HF_REPO_ID="$3"
DEVICE="$4"

if [ -z "$DEVICE" ]; then
    echo "Usage: bash patch_hparams.sh <method> <model_yaml_stem> <hf_repo_id> <device_index>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EASYEDIT_DIR="${EASYEDIT_DIR:-$SCRIPT_DIR/../EasyEdit}"
HPARAMS_FILE="$EASYEDIT_DIR/hparams/$METHOD/$MODEL_STEM.yaml"

if [ ! -f "$HPARAMS_FILE" ]; then
    echo "No such hparams file: $HPARAMS_FILE"
    exit 1
fi

# Replace whatever model_name currently is (local-path placeholder or
# another repo id) with the given HuggingFace repo id.
sed -i -E "s|model_name: .*|model_name: \"$HF_REPO_ID\"|" "$HPARAMS_FILE"

# Replace whatever device is currently set with the given index.
sed -i -E "s|device: [0-9]+|device: $DEVICE|" "$HPARAMS_FILE"

echo "Patched $HPARAMS_FILE:"
grep -E "model_name|device" "$HPARAMS_FILE"