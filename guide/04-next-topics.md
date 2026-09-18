# 第四章：从主线走向扩展

本章是**源码阅读地图与未来实验设计**，不是“这些训练我都已经做过”。上游固定快照包含更多训练器与模型实现；可以直接借它们学习算法，但每个方向都要用自己的实验补足理解与证据。先读完前三章，尤其是数据标签、模型前向与更新步骤。

| 方向 | 想解决什么问题 | 从上游哪里读起 | 建议先问自己 |
| --- | --- | --- | --- |
| LoRA | 不更新所有基础参数时，能否做低成本适配？ | [`model/model_lora.py`](../model/model_lora.py)、[`trainer/train_lora.py`](../trainer/train_lora.py) | 哪些参数有梯度？保存的是完整权重还是增量？ |
| DPO | 有“偏好/不偏好”回答对时，怎样优化相对偏好？ | [`dataset/lm_dataset.py`](../dataset/lm_dataset.py) 的 `DPODataset`、[`trainer/train_dpo.py`](../trainer/train_dpo.py) | chosen 与 rejected 的输入、mask 是否一致？ |
| MoE | 用多个专家层，如何按 token 选择计算路径？ | [`model/model_minimind.py`](../model/model_minimind.py) 的 `MOEFeedForward` | 门控如何选专家？辅助 loss 如何进入总目标？ |
| 蒸馏 | 如何用教师模型信号帮助小模型？ | [`trainer/train_distillation.py`](../trainer/train_distillation.py) | 教师和学生各提供什么分布/目标？ |
| RL / Agent | 有交互或奖励反馈时，训练目标怎样改变？ | [`trainer/train_ppo.py`](../trainer/train_ppo.py)、[`trainer/train_grpo.py`](../trainer/train_grpo.py)、[`trainer/train_agent.py`](../trainer/train_agent.py) | 奖励来自哪里？如何避免把训练 reward 当作真实质量？ |

## 建议的下一项真正实验

先做 [第三章](03-training-and-inference.md)提出的固定问题集评估，再选择**一个**方向。例如 LoRA：先在小样本上确认只有目标参数可训练、增量权重可保存并重新加载；然后才考虑租 GPU 跑更完整的任务。实验日志至少写清：源码提交、数据来源和许可、命令、运行环境、预期、实际输出、费用/时间与失败案例。

其他方向也可以只做源码研读，不必为了“项目显得完整”而运行所有训练。个人已完成范围以 [复现记录](../docs/reproduction.md)为准；上游官方提供的功能清单以 [原始 README](../README.upstream.md) 为准。
