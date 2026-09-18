# 第二章：从 token ID 到 logits

**目标**：不要求背公式，先能沿着真实代码追踪一次前向传播。主源码是 [`model/model_minimind.py`](../model/model_minimind.py)；可同时打开[上游提供的结构图](../images/LLM-structure.jpg)定位模块。本人的已完成训练使用 `use_moe=False` 的 Dense 路线；这里讲的是该固定版本的具体实现，不代表所有 Transformer 都同样配置。

## 1. 看形状，再看计算

运行 `python learning/inspect_model_shapes.py`。脚本用同一个 `MiniMindForCausalLM` 类，但配置成 `hidden_size=64`、2 层且随机初始化，仅用于 CPU 观察。预期主线是：

```text
input_ids [B, T]
  → token embedding [B, T, H]
  → block × 层数 [B, T, H]
  → final norm [B, T, H]
  → lm_head / logits [B, T, V]
```

`B` 是批量、`T` 是 token 序列长度、`H` 是隐藏维度、`V` 是词表大小。最后一维是 `V`，因为模型在每个位置要给**每个候选下一 token**一个分数；logits 还不是概率，也不是直接可读的词。

本机运行的微型配置输出是 `input_ids=(1,4)`、`embedding=(1,4,64)`、`Q=(1,4,64)`、`K/V=(1,4,32)`、`logits=(1,4,6400)`，独立参数约 53 万。这些数字来自教学脚本的随机小模型，不是正式 64M 权重的结构记录。

## 2. 一个 block 里发生什么

`MiniMindBlock` 先把输入经 RMSNorm 送入 `Attention`，加回残差；再经 RMSNorm、`FeedForward`，再次加回残差。`Attention` 将同一隐藏状态分别投影为 Q、K、V。默认配置是 8 个查询头、4 个 K/V 头，后者会通过 `repeat_kv` 对齐查询头的数量；本章实验中分别能看到 Q 最后一维 64、K/V 最后一维 32。RoPE 作用在 Q/K 上以引入位置信息；因果注意力使当前位置不能利用未来 token。

前馈网络由 `gate_proj`、`up_proj`、激活函数和 `down_proj` 组成；它不同于“把 logits 选成下一个 token”的生成步骤。把 `Attention.forward` 与 `FeedForward.forward` 的输入输出形状对照起来，能避免把模型内部表示和最终词表分数混为一谈。

## 3. 为什么 CPU 玩具模型回答不了问题？

这个实验只随机初始化模型，没有加载 `out/*.pth`。它验证**代码连通和张量形状**，不验证语言能力。正式权重来自训练；[一步训练实验](../learning/one_step_training.py)能展示权重被更新，但一次更新也不会产生可靠的聊天助手。

## 自检与待实践

1. 在 `MiniMindForCausalLM.forward` 找到 `lm_head`，解释为何最终 shape 是 `[B,T,V]`。
2. 在 `Attention.forward` 找到因果遮罩与 `repeat_kv`，解释“不能偷看未来”和“K/V 头更少”的代码依据。
3. **待实践**：将实验中的 `num_hidden_layers` 从 2 改为 3，记录参数量如何变化；再将 `hidden_size` 改为 128，比较 Q/K/V 形状。不要把这当成训练加速或模型质量结论。

下一章：[训练、推理与评估](03-training-and-inference.md)。
