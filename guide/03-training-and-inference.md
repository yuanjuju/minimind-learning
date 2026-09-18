# 第三章：loss、权重更新、推理与评估

**目标**：把“模型会输出”拆成两个不同阶段：训练时根据目标改参数；推理时固定参数、一步步选 token。对应源码为 [`model/model_minimind.py`](../model/model_minimind.py)、[`trainer/train_pretrain.py`](../trainer/train_pretrain.py)、[`trainer/train_full_sft.py`](../trainer/train_full_sft.py) 和 [`eval_llm.py`](../eval_llm.py)。

## 1. loss 从哪里来

`MiniMindForCausalLM.forward` 的 `logits[..., :-1, :]` 与 `labels[..., 1:]` 错位对齐，让当前位置预测下一个 token。交叉熵用目标 ID 衡量预测偏差，并跳过 `-100`。预训练与 SFT 仍用同一模型类，但数据、起始权重和哪些 label 计分不同。先完成 [第一章的标签实验](01-tokenizer-and-data.md)，否则只盯训练日志的数字容易误解目标。

运行 `python learning/one_step_training.py`：看 loss、非零梯度与 `optimizer.step()` 后的权重差异。这个实验只训练一个随机的小模型，证明**更新发生了**；它不证明 loss 会一直下降，也不代表正式权重在本地被再次训练。

## 2. 从头预训练与 Full SFT

固定版本的预训练脚本默认 `from_weight=none`；Full SFT 默认 `from_weight=pretrain`。个人云端实验是先完成 Dense 约 64M 的预训练，再用其权重进行 Full SFT。已核实的参数、未留下的 SFT 精确命令、文件哈希和局限都在 [复现记录](../docs/reproduction.md)。

训练循环的顺序可按 `model(...) → loss/accumulation_steps → backward() → clip_grad_norm_() → optimizer.step() → zero_grad()` 阅读。梯度累积让一次优化器更新跨多个数据批次；日志中的 step 是数据批次编号，通常不等于权重更新次数。`out/*.pth` 保存模型权重，而续训检查点还保存优化器等状态；二者用途不同。

## 3. 固定权重后如何生成

运行 `python learning/inspect_prompts.py`。上游 `eval_llm.py` 对预训练权重使用 `BOS + 原始问题`，对 SFT 权重使用 `apply_chat_template(..., add_generation_prompt=True)`。同一句问题可能变成长度不同、角色不同的 token 序列。因此比较两种权重时，应该使用各自正确的输入格式。

固定脚本中的同一句“请用一句话介绍你自己”，在本仓库 tokenizer 下分别成为 **8 个**与 **24 个** token。这个观察只说明输入格式不同，不能单靠 token 数推出哪个权重更好。

模型每一步给出词表上的 logits；上游 `generate` 通过温度、`top_k`、`top_p` 等规则选择下一个 token，再把它接回输入，直到遇到结束 token 或达到 `max_new_tokens`。**回答到一半突然结束**可能是长度上限，也可能模型采样到了结束 token；只看屏幕末尾不能区分。用固定提示词与生成参数重试，再记录结束原因，才有证据判断。

## 4. 怎么证明“更好”

训练 loss 不是通用能力分数，单次聊天也容易受采样影响。待做的评估至少记录：权重版本、源码提交、问题集、提示词模板、随机种子、`max_new_tokens`、温度与 `top_p`、原始输出及简单评分标准。先用小而固定的非实时问题集，对比“按要求回答、事实错误、完整结束”这些可观察项；不要只挑好看的例子。更详细的下一步在 [学习路线](../docs/learning-path.md)。

## 自检

1. 画出一个批次从 `input_ids` 到 `optimizer.step()` 的路径，并解释 `-100` 在哪里生效。
2. 解释模型推理时为何不应该再调用 `backward()`。
3. **待实践**：编写一份固定问题集与结果记录，分别评估预训练和 SFT 权重；在完成前不宣称已有系统评测结论。

下一章：[扩展路线](04-next-topics.md)。
