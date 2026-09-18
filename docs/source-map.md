# 从一条样本到一个回答：源码导航

本文对应本仓库固定的上游版本，先读主线，再读可选的 MoE、LoRA、DPO、RL。这里的个人复现实验使用 **Dense 64M**，没有启用 MoE。

若是第一次系统学习，可先沿着 [分阶段导学](../guide/00-start.md)做 CPU 实验，再把本页当作查源码的索引。

## 主线文件

| 问题 | 源码 | 阅读时看什么 |
| --- | --- | --- |
| 字符串怎么变成 token？ | [`model/tokenizer_config.json`](../model/tokenizer_config.json)、[`model/tokenizer.json`](../model/tokenizer.json) | `<|im_start|>`、`<|im_end|>`、`<|endoftext|>` 与聊天模板 |
| 预训练样本怎么做？ | [`dataset/lm_dataset.py`](../dataset/lm_dataset.py) 的 `PretrainDataset` | 文本前后加起止标记；padding 的 label 变成 `-100` |
| SFT 样本怎么做？ | 同文件的 `SFTDataset` | 对话模板、助手片段定位、`generate_labels` 屏蔽用户位置 |
| 模型结构在哪？ | [`model/model_minimind.py`](../model/model_minimind.py) | `MiniMindConfig → Attention → MiniMindBlock → MiniMindModel → MiniMindForCausalLM` |
| loss 在哪算？ | `MiniMindForCausalLM.forward` | `logits[..., :-1, :]` 对齐 `labels[..., 1:]`；交叉熵忽略 `-100` |
| 训练时谁更新参数？ | [`trainer/train_pretrain.py`](../trainer/train_pretrain.py)、[`trainer/train_full_sft.py`](../trainer/train_full_sft.py) | 前向、`backward`、梯度裁剪、`optimizer.step`、`zero_grad` |
| 权重从哪里来？ | [`trainer/trainer_utils.py`](../trainer/trainer_utils.py) 的 `init_model` | 预训练默认从头开始；Full SFT 默认读取 `pretrain_768.pth` |
| 怎么做命令行推理？ | [`eval_llm.py`](../eval_llm.py) | 预训练使用文本前缀；SFT 使用聊天模板；输出长度受 `max_new_tokens` 限制 |

## 数据流

```text
预训练 JSONL 的 text
  → PretrainDataset: input_ids + labels（padding=-100）
  → MiniMindForCausalLM: 预测下一个 token、计算 loss
  → train_pretrain.py: 反向传播与参数更新
  → out/pretrain_768.pth

SFT JSONL 的 conversations + 预训练权重
  → SFTDataset: 整段对话作为 input_ids，仅助手目标位置计分
  → 同一个模型类和训练机制
  → out/full_sft_768.pth
  → eval_llm.py 按聊天模板推理
```

## Dense 模型结构，按执行顺序读

1. `MiniMindConfig` 默认 `hidden_size=768`、`num_hidden_layers=8`、`vocab_size=6400`、`use_moe=False`。这与复现实验中的约 63.91M 参数一致。
2. `MiniMindModel` 先做 token embedding，再经过 8 个 `MiniMindBlock`，最后做 RMSNorm。
3. 每个 block 先做带因果遮罩的自注意力，再做前馈网络，并保留残差连接。默认注意力有 8 个 query 头、4 个 key/value 头；Q/K 使用 RoPE 位置信息。
4. `lm_head` 把隐藏状态投影成词表各 token 的分数（logits）；训练时通过交叉熵得到 loss。

这些是**这份源码的实现事实**，不是所有大模型的统一配置。想看每个术语的直白解释，先读 [`core-concepts.md`](core-concepts.md)，再逐类打开源码。原作者更完整的介绍仍在 [`README.upstream.md`](../README.upstream.md)。
