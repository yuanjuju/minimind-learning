"""对比上游 eval_llm.py 为预训练和 SFT 构造的输入，不加载权重。"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

from transformers import AutoTokenizer  # noqa: E402


def main():
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    question = "请用一句话介绍你自己"
    pretrain_prompt = tokenizer.bos_token + question
    sft_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=False,
    )

    print("同一个问题，不同阶段的输入格式：")
    print("\n[预训练权重的文本续写前缀]")
    print(repr(pretrain_prompt))
    print("\n[SFT 权重的对话前缀]")
    print(repr(sft_prompt))
    print(f"\n长度：预训练 {len(tokenizer(pretrain_prompt).input_ids)} tokens；SFT {len(tokenizer(sft_prompt).input_ids)} tokens")
    print("这里只比较提示词，不加载权重，也不判断两种模型的回答质量。")

    assert tokenizer.bos_token in pretrain_prompt
    assert "assistant" in sft_prompt and tokenizer.eos_token in sft_prompt


if __name__ == "__main__":
    main()
