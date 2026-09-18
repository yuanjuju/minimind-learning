# CPU 实验台：把大模型拆开观察

这里的 `.py` 和小型 `.jsonl` 样本是本仓库新增的学习材料，不属于上游 MiniMind 源码；所有示例都在项目根目录运行。建议先读 [导学页](../guide/00-start.md)，每次运行前预测输出，再回到相应上游实现核对。

| 顺序 | 命令 | 要观察什么 |
| --- | --- | --- |
| 1 | `python learning/inspect_tokenizer.py` | 文本如何切成 token，聊天模板加入了什么；[讲义](../guide/01-tokenizer-and-data.md) |
| 2 | `python learning/inspect_pretrain_labels.py` | 预训练中非填充 token 如何成为预测目标；[讲义](../guide/01-tokenizer-and-data.md) |
| 3 | `python learning/inspect_sft_labels.py` | SFT 中用户段为何是 `-100`，助手段为何计分；[讲义](../guide/01-tokenizer-and-data.md) |
| 4 | `python learning/inspect_model_shapes.py` | embedding、Q/K/V、logits 的形状；[讲义](../guide/02-transformer.md) |
| 5 | `python learning/one_step_training.py` | `loss → backward → optimizer.step` 如何让权重变化；[讲义](../guide/03-training-and-inference.md) |
| 6 | `python learning/inspect_prompts.py` | 同一句问题在预训练与 SFT 推理中怎样被格式化；[讲义](../guide/03-training-and-inference.md) |

部分脚本直接调用上游的 `PretrainDataset`、`SFTDataset` 和 `MiniMindForCausalLM`，但仅处理这里的玩具样本；结构和一步训练脚本创建的是内存中的 **约 53 万参数随机初始化模型**，不加载、保存或覆盖 `out/` 里的正式权重。它们的 logits/loss 不应与 64M 正式模型的训练日志或回答质量比较。

推荐在独立 Python 3.12 环境中安装运行这些实验所需的最小依赖：`torch==2.6.0`、`transformers==4.57.6`、`numpy==1.26.4`、`datasets==3.6.0`。完整上游依赖见根目录 [`requirements.txt`](../requirements.txt)；是否能直接安装 PyTorch 2.6.0 取决于操作系统和 Python 架构。

两个 `.jsonl` 是人为编写的极小示例，不是上游大规模训练数据，也不是性能评测集。
