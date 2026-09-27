# Knowledge Editing Experiments via Methods Fine-Tuning (FT) and Rank one Model Editing (ROME)

This project is to explore 2 different knwoledge editing methods on foundation models with limited RAM.The experiment is carried on NVIDIA RTX A6000 through the IMS server service.The knowledge editing methods are implemented and evaluted by using [EasyEdit](https://github.com/zjunlp/EasyEdit) repo.  

## Models 

| Model    | Params | Layers  |
| -------- | ------- | ------ |
| Qwen2.5-7b | 7B | 48        |
| GPT2-xl | 1.5B  | 28        |


## How to Run This Project

### 1. Run the `setup.sh`
```bash
bash setup.sh
```
This clones EasyEdit, creates a conda env at ./envs/easyedit (Python 3.10 — required, torch>=2.9 has no wheels for 3.9), points pip/HuggingFace/tmp caches into this directory instead of $HOME, and installs requirements.txt. Safe to re-run — it skips steps that are already done.

### Activate the env in every new terminal/session:

```bash
export PATH="$(pwd)/envs/easyedit/bin:$PATH"
which python   # sanity check: should print .../envs/easyedit/bin/python
```


### 2. HuggingFace LogIn:

```bash
huggingface-cli login
```

### 3. Prepare Data

```bash
curl -L -o data/counterfact.json https://rome.baulab.info/data/dsets/counterfact.json

python scripts/prepare_counterfact_sample.py --n 150 --seed 42
```

### 4. Configuration Setup

Every hparams yaml ships with a placeholder model_name and an arbitrary GPU device; patch both before running:

```bash
nvidia-smi   # find a genuinely idle GPU (0% util, ~0 MiB used) — don't hardcode from a previous session

bash scripts/patch_hparams.sh ROME qwen2.5-7b Qwen/Qwen2.5-7B <gpu_index>
# The same is applied to gpt2-xl

```
For FT only, also check batch_size: 1 is set (needed for one-edit-at-a-time sequential editing):

```bash
grep batch_size EasyEdit/hparams/FT/qwen2.5-7b.yaml
grep batch_size EasyEdit/hparams/FT/gpt2-xl.yaml
# if not 1:
sed -i 's/batch_size: [0-9]*/batch_size: 1/' EasyEdit/hparams/FT/<model>.yaml
```

### 5.Run the four experiments

Each run takes tens of minutes to ~1.5 hours depending on GPU load — use tmux/screen so an SSH disconnect doesn't kill it:

```bash
tmux new -s edit
python scripts/run_sequential_edit.py --method ROME --model qwen2.5-7b
python scripts/run_sequential_edit.py --method ROME --model gpt2-xl
python scripts/run_sequential_edit.py --method FT   --model qwen2.5-7b
python scripts/run_sequential_edit.py --method FT   --model g
```
Each writes `results/sequential_edit_<method>_<model>.json`.

All results are saved in `/results`.