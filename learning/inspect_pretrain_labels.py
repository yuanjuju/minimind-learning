"""用一条短文本观察 MiniMind 预训练的输入、标签和下一 token 对齐。"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402

from dataset.lm_dataset import PretrainDataset  # noqa: E402


def shown(tokenizer, token_id):
    return repr(tokenizer.decode([int(token_id)], skip_special_tokens=False))


def main():
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    dataset = PretrainDataset(str(ROOT / "learning" / "sample_pretrain.jsonl"), tokenizer, max_length=64)
    input_ids, labels = dataset[0]

    print("示例：预训练文本『你好！』")
    used = int((input_ids != tokenizer.pad_token_id).sum())
    print("模型实际看到的文本：")
    print(tokenizer.decode(input_ids[:used], skip_special_tokens=False))
    print("\n位置  当前 token          要预测的下一 token    下一位置的 label  计入 loss")
    print("-" * 77)
    for i in range(min(used, len(input_ids) - 1)):
        target_label = int(labels[i + 1])
        print(
            f"{i:>4}  {shown(tokenizer, input_ids[i]):<20} "
            f"{shown(tokenizer, input_ids[i + 1]):<20} "
            f"{target_label:>14}  {'是' if target_label != -100 else '否'}"
        )

    scored = int((labels[1:] != -100).sum())
    print(f"\n共 {len(input_ids)} 个输入 token；其中 {scored} 次下一 token 预测计入预训练损失。")
    print("注意：labels[0] 是 BOS 的 ID，但模型计算 loss 时从 labels[1] 开始，因此它不计分。")
    assert scored > 0 and int(labels[0]) == tokenizer.bos_token_id


if __name__ == "__main__":
    main()
