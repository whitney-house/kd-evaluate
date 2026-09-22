"""
Sequential editing: apply N edits one after another to the SAME model
(not reset between edits), and save the metrics returned at each step.

Runs directly from this repo's scripts/ folder — no need to copy this file
into EasyEdit/ first (see easyedit_path.py for how that's made to work).

Run from the project root:
    python scripts/run_sequential_edit.py --method ROME --model qwen2.5-7b
    python scripts/run_sequential_edit.py --method ROME --model gpt2-xl
"""

import argparse
import json
import os

import easyedit_path  # noqa: F401  (adds EasyEdit/ to sys.path as a side effect)
from easyedit_path import EASYEDIT_DIR
from easyeditor import BaseEditor, ROMEHyperParams, MEMITHyperParams, FTHyperParams

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "counterfact_sample.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

HPARAMS_CLASSES = {
    "ROME": ROMEHyperParams,
    "MEMIT": MEMITHyperParams,
    "FT": FTHyperParams,
}


def load_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ROME", "MEMIT", "FT"], default="ROME")
    parser.add_argument("--model", default="qwen2.5-7b", help="matches a filename under EasyEdit/hparams/<method>/")
    args = parser.parse_args()

    # hparams/data/stats live inside the EasyEdit checkout itself (that's
    # where EasyEdit expects to find and write them); our own sampled
    # edit data and results stay in this repo.
    hparams_path = os.path.join(EASYEDIT_DIR, "hparams", args.method, f"{args.model}.yaml")
    results_path = os.path.join(RESULTS_DIR, f"sequential_edit_{args.method.lower()}_{args.model}.json")

    cases = load_cases(DATA_PATH)

    hparams_cls = HPARAMS_CLASSES[args.method]
    hparams = hparams_cls.from_hparams(hparams_path)
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

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[{args.method} / {args.model}] Sequential edit complete ({len(cases)} edits). Saved to:", results_path)

    print("\nstep  rewrite_acc  rephrase_acc  locality_acc")
    for i, m in enumerate(metrics):
        post = m["post"]
        rewrite = post["rewrite_acc"][0]
        rephrase = post["rephrase_acc"][0]
        locality = post["locality"]["neighborhood_acc"][0]
        print(f"{i:>4}  {rewrite:>11.2f}  {rephrase:>12.2f}  {locality:>13.2f}")


if __name__ == "__main__":
    main()