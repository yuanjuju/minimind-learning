# 可重复的小实验

这里的 `.py` 和小型 `.jsonl` 样本是个人学习材料，不属于上游 MiniMind 源码；所有示例都在项目根目录运行。

| 顺序 | 命令 | 要观察什么 |
| --- | --- | --- |
| 1 | `python learning/inspect_pretrain_labels.py` | 预训练中非填充 token 如何成为预测目标 |
| 2 | `python learning/inspect_sft_labels.py` | SFT 中用户段为何是 `-100`，助手段为何计分 |
| 3 | `python learning/one_step_training.py` | `loss → backward → optimizer.step` 如何让权重变化 |

脚本直接调用本仓库的 `PretrainDataset`、`SFTDataset` 和 `MiniMindForCausalLM`，但仅处理这里的玩具样本；第三个脚本创建的是内存中的 **53 万参数临时模型**，不加载、保存或覆盖 `out/` 里的正式权重。它的 loss 数值不应与 64M 正式模型的训练日志比较。

推荐在独立 Python 3.12 环境中安装运行这三个实验所需的最小依赖：`torch==2.6.0`、`transformers==4.57.6`、`numpy==1.26.4`、`datasets==3.6.0`。完整上游依赖见根目录 [`requirements.txt`](../requirements.txt)；是否能直接安装 PyTorch 2.6.0 取决于操作系统和 Python 架构。

两个 `.jsonl` 是人为编写的极小示例，不是上游大规模训练数据，也不是性能评测集。
