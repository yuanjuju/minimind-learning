# 从实验里学到的核心概念

这一页是速查笔记；要按“tokenizer → 模型 → 训练/推理”从头走一遍，可从 [导学页](../guide/00-start.md)开始。

## 1. `input_ids`、`labels` 与 `-100`

`input_ids` 是模型实际读取的 token 序列；`labels` 是相应位置要用来检查预测是否正确的答案序列。语言模型在位置 `i` 预测位置 `i+1`，所以源码用 `logits[..., :-1, :]` 与 `labels[..., 1:]` 对齐。`-100` 是 PyTorch 交叉熵忽略的标签值，并不是词表中的“特殊词”。

- 预训练：文本中的绝大多数非 padding 内容都成为下一 token 预测目标。`labels[0]` 虽保留起始标记，但因错位计算而不参与 loss。
- SFT：整段对话仍作为上下文输入；用户段与助手头部的 label 是 `-100`，助手回答及结束标记成为计分目标。本版聊天模板可能包含空 `<think>` 标签，数据处理会随机移除一部分，所以原始文本与实际训练字符串不一定完全相同。
- “用户段不直接计分”不等于用户内容没用。模型仍读取用户内容，以便预测后续助手回答。

亲自运行 [`learning/inspect_pretrain_labels.py`](../learning/inspect_pretrain_labels.py) 和 [`learning/inspect_sft_labels.py`](../learning/inspect_sft_labels.py)，比只看抽象定义更直观。

## 2. loss、梯度与参数更新

`loss` 衡量当前批次、当前目标下模型预测与目标的差距。`backward()` 根据 loss 计算参数梯度；`optimizer.step()` 根据梯度和优化器状态更新权重；`zero_grad()` 清理本轮梯度，准备下一轮。本仓库的 [`one_step_training.py`](../learning/one_step_training.py) 使用临时小模型验证了“梯度非零且参数确实改变”，不会覆盖正式权重。

训练 loss 下降只能说明模型在当前训练目标上拟合得更好；不等于回答一定真实、可靠，也不能直接拿预训练 loss 与 SFT loss 比高低，因为它们的数据和计分位置不同。

## 3. batch、梯度累积、epoch

- `batch_size`：一次前向/反向计算送入多少条样本。
- `accumulation_steps`：连续多少个批次累计梯度后才做一次 `optimizer.step()`。单卡、无其它调整时，粗略有效 batch 为两者的乘积；例如 `32 × 8 = 256`。
- `epoch`：训练数据集被遍历一轮。一个 epoch 含很多批次，也通常含很多次权重更新。
- 本版脚本日志的 `step` 随数据加载批次增加；当 `accumulation_steps > 1`，**一个日志 step 不等于一次权重更新**。

## 4. 权重文件与续训检查点

`out/pretrain_768.pth` 和 `out/full_sft_768.pth` 是模型权重，适合用来加载/推理；`checkpoints/*_resume.pth` 还包含优化器等训练状态，更适合从中断位置续训。不能把“有模型权重”误认为“可以无损接着原训练进度跑”。本仓库只记录正式权重的 SHA-256，不把二进制文件提交到 Git。

## 5. 为什么 SFT 更像聊天助手

预训练主要学习文本续写；SFT 在预训练参数基础上，用带角色模板的问答数据继续优化助手回答。两者的任务形式、数据与输入模板都不一样。个人主观对话中 SFT 表现更贴近指令，但要得出更可靠的质量结论，还需固定测试问题、生成参数和评价标准；见 [`learning-path.md`](learning-path.md)。
