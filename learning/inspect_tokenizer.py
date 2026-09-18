"""观察 MiniMind tokenizer：文本、token、ID、特殊标记和聊天模板。"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

from transformers import AutoTokenizer  # noqa: E402


def main():
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    text = "你好，MiniMind！"
    ids = tokenizer(text, add_special_tokens=False).input_ids
    pieces = tokenizer.convert_ids_to_tokens(ids)

    print(f"原文：{text}")
    print("token 片段：", pieces)
    print("token ID：", ids)
    print("解码结果：", tokenizer.decode(ids))
    print(f"字符数：{len(text)}；token 数：{len(ids)}。两者不必相等。")
    assert tokenizer.decode(ids) == text

    print("\n特殊标记（名称 → 字符串 → ID）：")
    for name in ("bos", "eos", "pad"):
        print(f"{name:>3}: {getattr(tokenizer, name + '_token')!r} → {getattr(tokenizer, name + '_token_id')}")

    messages = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    print("\n同一 tokenizer 渲染聊天模板后的文本：")
    print(rendered)
    print("模板中包含角色、结束标记；不能把它当成普通的『你好你好！』文本。")
    assert tokenizer.bos_token in rendered and tokenizer.eos_token in rendered


if __name__ == "__main__":
    main()
