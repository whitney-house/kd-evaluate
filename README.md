# Knowledge Editing — Minimal Runnable Project

Goal: use EasyEdit + ROME to edit a single fact into Qwen2.5-7B (Paris -> Lyon
as the capital of France), and check the three basic editing metrics:
Reliability, Generalization, and Locality. This is the minimal proof of
concept before extending to sequential editing / multi-model comparison.

## Repo layout

```
knowledge-editing-project/
├── setup.sh                     # environment setup (clone EasyEdit + install deps)
├── data/single_edit.json        # one hand-written CounterFact-style edit case
├── scripts/run_single_edit.py   # edit + evaluate script
└── README.md
```

## Server-specific gotchas (read this first)

This was run on a shared university server. A few things were not obvious
and cost real time to figure out — documenting them here so future-me
doesn't repeat the debugging:

- **Home directory quota was only 4GB (hard limit 6GB).** Not enough for a
  conda env + pip cache + model weights. Everything (conda env, pip cache,
  HuggingFace cache, the cloned repo itself) must live under a
  large-storage directory instead of `$HOME` — on this server that's
  `~/temp` (a symlink to `/mount/studenten-temp1/users/<username>`).
- **`conda create -n <name>` installs into `~/.conda/envs` by default**,
  which still counts against the home quota even if your code lives
  elsewhere. Use `conda create --prefix <path-under-temp>` instead.
- **`source activate <env>` fails inside a non-interactive script** (e.g.
  `bash setup.sh`) even if `conda activate` works fine in an interactive
  shell — the script doesn't load `~/.bashrc`'s conda init. Fix: explicitly
  `source "$(conda info --base)/etc/profile.d/conda.sh"` before
  `conda activate`.
- **`pip install -r requirements.txt` failed with "Could not find a version
  that satisfies torch==2.9.1"** on a Python 3.9 env — torch >= 2.9
  requires Python >= 3.10. Recreate the env with `python=3.10`.
- **Every new terminal session starts un-activated.** `conda env list`
  showing your env without a `*` next to it means you're back on system
  Python — always re-run `conda activate <path>` (or `which python` to
  sanity check) before installing anything or running the script.
- **GPUs are shared across users.** Run `nvidia-smi` before each run and
  pick a GPU with ~0% utilization and near-zero memory used; don't reuse a
  hardcoded device index, since availability changes.
- **The hparams yaml ships with a placeholder local path**
  (`model_name: "./hugging_cache/Qwen2.5-7B"`), not a HuggingFace repo id.
  Change it to `Qwen/Qwen2.5-7B` so `transformers` downloads it automatically.

## Setup

```bash
cd ~/temp/kd-evaluate   # or wherever your large-storage directory is
bash setup.sh
```

This clones EasyEdit, creates a conda env at `./envs/easyedit` (Python 3.10),
redirects pip/HuggingFace/tmp caches into the project directory, and installs
`requirements.txt`.

To reactivate the env in a fresh terminal:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/temp/kd-evaluate/envs/easyedit
```

## Model access: Qwen2.5-7B

Chose Qwen2.5-7B specifically because it's a fully open model on
HuggingFace — no gated-access request needed, unlike Llama-3-8B or
Mistral-7B.

```bash
huggingface-cli login   # paste a token generated at huggingface.co/settings/tokens
```

## hparams changes

`EasyEdit/hparams/ROME/qwen2.5-7b.yaml` ships with two fields that need
changing:

```yaml
# Was: "./hugging_cache/Qwen2.5-7B" (a local-path placeholder).
# Changed to the HuggingFace repo id so it downloads automatically:
model_name: "Qwen/Qwen2.5-7B"

# Was: 5 (whatever GPU the EasyEdit authors used).
# Changed to a GPU index confirmed free via `nvidia-smi`:
device: 1
```

## Run

```bash
cd EasyEdit
python run_single_edit.py
```

Metrics are saved to `./results/single_edit_result.json`. Key fields:

- `post.rewrite_acc` — Reliability (did the edit take effect)
- `post.rephrase_acc` — Generalization (does it hold under a reworded prompt)
- `post.locality.neighborhood_acc` — Locality (did unrelated facts stay
  intact; close to 1.0 is good)

**First successful run** (single CounterFact-style case, ROME):
`rewrite_acc = 1.0`, `rephrase_acc = 0.0`, `locality.neighborhood_acc = 0.0`.
The edit itself worked, but generalization and locality were poor — a
known behavior of single-fact ROME edits, and a useful thing to track once
sequential editing is added.

## Next steps (not done yet)

- Set `sequential_edit=True` and extend `data/single_edit.json` to multiple
  cases for sequential editing.
- Swap `ROMEHyperParams` for `MEMITHyperParams`
  (`hparams/MEMIT/qwen2.5-7b.yaml`) to compare editing methods.
- Add a general-capability check (e.g. an MMLU subset) to see whether
  overall model quality degrades as the number of edits grows.
