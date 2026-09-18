"""只在内存里训练一个很小的 MiniMind，观察梯度和参数更新。"""

import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from dataset.lm_dataset import SFTDataset  # noqa: E402
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402


def main():
    torch.manual_seed(42)
    random.seed(0)

    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    dataset = SFTDataset(str(ROOT / "learning" / "sample_sft.jsonl"), tokenizer, max_length=32)
    input_ids, labels = dataset[0]
    input_ids, labels = input_ids.unsqueeze(0), labels.unsqueeze(0)  # batch=1

    # 与正式训练使用同一个模型类，但缩小到 2 层、hidden_size=64；不加载任何 .pth。
    config = MiniMindConfig(hidden_size=64, num_hidden_layers=2, flash_attn=False)
    model = MiniMindForCausalLM(config).cpu().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    before = model.lm_head.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)

    result = model(input_ids, labels=labels)  # 1. 前向传播：得到 loss
    loss = result.loss + result.aux_loss
    loss.backward()  # 2. 反向传播：得到每个参数的梯度
    grad_norm = model.lm_head.weight.grad.norm().item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()  # 3. 优化器按梯度更新参数

    change = (model.lm_head.weight.detach() - before).abs().max().item()
    print(f"临时模型参数量：{sum(p.numel() for p in model.parameters()):,}")
    print(f"本次前向计算的 loss：{loss.item():.4f}")
    print(f"lm_head.weight 的梯度范数：{grad_norm:.6f}")
    print(f"optimizer.step() 后该权重的最大变化：{change:.8f}")
    print("没有加载或保存任何训练权重文件；进程结束后这个临时模型就消失。")
    assert grad_norm > 0 and change > 0


if __name__ == "__main__":
    main()
