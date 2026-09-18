"""用随机初始化的微型 MiniMind 观察结构与张量形状，不做正式推理。"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402


def main():
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    config = MiniMindConfig(hidden_size=64, num_hidden_layers=2, flash_attn=False, max_position_embeddings=128)
    model = MiniMindForCausalLM(config).cpu().eval()
    ids = tokenizer("你好，模型！", return_tensors="pt", add_special_tokens=False).input_ids

    with torch.inference_mode():
        embeddings = model.model.embed_tokens(ids)
        first_attention = model.model.layers[0].self_attn
        q = first_attention.q_proj(embeddings)
        k = first_attention.k_proj(embeddings)
        v = first_attention.v_proj(embeddings)
        result = model(ids)

    print(f"输入 ID [batch, seq]：{tuple(ids.shape)}")
    print(f"embedding [batch, seq, hidden]：{tuple(embeddings.shape)}")
    print(f"第一层 Q/K/V 投影：{tuple(q.shape)} / {tuple(k.shape)} / {tuple(v.shape)}")
    print(f"注意力头：Q={config.num_attention_heads}，K/V={config.num_key_value_heads}，head_dim={config.head_dim}")
    print(f"最终 logits [batch, seq, vocab]：{tuple(result.logits.shape)}")
    print(f"临时模型的独立参数量：{sum(p.numel() for p in model.parameters()):,}")
    print("注意：Q/K/V 是同一输入的三种投影；这里只看投影形状，完整注意力仍在模型前向计算中。")
    print("模型未加载预训练权重，logits 是随机初始化模型的输出，不要拿它测试回答质量。")

    assert embeddings.shape == (*ids.shape, config.hidden_size)
    assert q.shape[-1] == config.num_attention_heads * config.head_dim
    assert k.shape[-1] == v.shape[-1] == config.num_key_value_heads * config.head_dim
    assert result.logits.shape == (*ids.shape, config.vocab_size)


if __name__ == "__main__":
    main()
