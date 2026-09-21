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

    # Step 1: 加载编辑方法的超参数(模型结构、要改哪几层、学习率等都在这个yaml里)
    hparams = ROMEHyperParams.from_hparams(HPARAMS_PATH)

    # Step 2: 构造 Editor(内部会根据 hparams.model_name 自动去 HuggingFace / 本地路径
    # 加载模型和 tokenizer,不需要自己写 from_pretrained)
    editor = BaseEditor.from_hparams(hparams)

    # Step 3: 组织编辑数据。
    # prompts / ground_truth / target_new 是必需的三元组:
    #   在 prompts 上,模型原本应该输出 ground_truth,编辑后应该输出 target_new。
    # rephrase_prompts 用于测 Generalization(换一种问法,编辑是否依然生效)。
    # locality_inputs 用于测 Locality(无关知识有没有被破坏)。
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

    # Step 4: 执行编辑。metrics 里直接包含 Reliability / Generalization / Locality 分数,
    # 不需要自己再写评测逻辑。
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        subject=subject,
        ground_truth=ground_truth,
        target_new=target_new,
        rephrase_prompts=rephrase_prompts,
        locality_inputs=locality_inputs,
        sequential_edit=False,  # 单知识点编辑,不需要连续编辑模式
        keep_original_weight=True,  # 保留原始权重的备份,方便后续对比/回滚
    )

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("编辑完成,指标已保存到:", RESULTS_PATH)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
