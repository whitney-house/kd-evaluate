"""
Sequential editing: apply N edits one after another to the SAME model
(not reset between edits), and save the metrics returned at each step.
This is what lets us observe how Reliability / Generalization / Locality
degrade as the number of accumulated edits grows.

Prerequisites: same as run_single_edit.py (run from inside the EasyEdit
repo root, HF login done, hparams model_name/device already set).

Run:
    python run_sequential_edit.py
"""

import json
import os

from easyeditor import BaseEditor, ROMEHyperParams

DATA_PATH = "./data/sequential_edits.json"
HPARAMS_PATH = "./hparams/ROME/qwen2.5-7b.yaml"
RESULTS_PATH = "./results/sequential_edit_result.json"


def load_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    cases = load_cases(DATA_PATH)

    hparams = ROMEHyperParams.from_hparams(HPARAMS_PATH)
    editor = BaseEditor.from_hparams(hparams)

    prompts = [c["prompt"] for c in cases]
    subject = [c["subject"] for c in cases]
    ground_truth = [c["ground_truth"] for c in cases]
    target_new = [c["target_new"] for c in cases]
    rephrase_prompts = [c["rephrase_prompt"] for c in cases]
    locality_inputs = {
        "neighborhood": {
            "prompt": [c["locality_prompt"] for c in cases],
            "ground_truth": [c["locality_ground_truth"] for c in cases],
        }
    }

    # sequential_edit=True: edits are applied one after another on the same
    # model, instead of each case being edited on a fresh copy.
    # keep_original_weight=False: don't roll back weights after each edit —
    # we want them to accumulate, which is the whole point of "sequential".
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        subject=subject,
        ground_truth=ground_truth,
        target_new=target_new,
        rephrase_prompts=rephrase_prompts,
        locality_inputs=locality_inputs,
        sequential_edit=True,
        keep_original_weight=False,
    )

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Sequential edit complete ({len(cases)} edits). Saved to:", RESULTS_PATH)

    # Quick summary so you can eyeball the trend without opening the json.
    print("\nstep  rewrite_acc  rephrase_acc  locality_acc")
    for i, m in enumerate(metrics):
        post = m["post"]
        rewrite = post["rewrite_acc"][0]
        rephrase = post["rephrase_acc"][0]
        locality = post["locality"]["neighborhood_acc"][0]
        print(f"{i:>4}  {rewrite:>11.2f}  {rephrase:>12.2f}  {locality:>13.2f}")


if __name__ == "__main__":
    main()