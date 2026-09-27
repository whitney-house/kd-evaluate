import json
import os

from easyeditor import BaseEditor, ROMEHyperParams

DATA_PATH = "./data/single_edit.json"
HPARAMS_PATH = "./hparams/ROME/qwen2.5-7b.yaml"
RESULTS_PATH = "./results/single_edit_result.json"


def load_edit_case(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[0]  


def main():
    case = load_edit_case(DATA_PATH)

    hparams = ROMEHyperParams.from_hparams(HPARAMS_PATH)

    editor = BaseEditor.from_hparams(hparams)

    prompts = [case["prompt"]]
    subject = [case["subject"]]
    ground_truth = [case["ground_truth"]]
    target_new = [case["target_new"]]
    rephrase_prompts = [case["rephrase_prompt"]]
    locality_inputs = {
        "neighborhood": {
            "prompt": [case["locality_prompt"]],
            "ground_truth": [case["locality_ground_truth"]],
        }
    }

    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        subject=subject,
        ground_truth=ground_truth,
        target_new=target_new,
        rephrase_prompts=rephrase_prompts,
        locality_inputs=locality_inputs,
        sequential_edit=False,  
        keep_original_weight=True, 
    )

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
