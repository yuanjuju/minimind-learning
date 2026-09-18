# MiniMind LLM Lab：从 token 到对话模型

这是我基于 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 建立的大模型学习项目。我用它沿着 **文本 → token → 训练样本 → Transformer → loss/梯度 → 预训练 → SFT → 生成与评估** 这条链路学习，而不只是保存一次训练的命令。

我已经在 RTX 4090 上完成模型的预训练与 Full SFT，并在 Mac 上验证权重、运行推理。仓库里的教学脚本和学习笔记把这段实践拆成可以在 CPU 上重复观察的小问题。

> **来源与边界**：MiniMind 模型、tokenizer、训练器等基础实现来自原作者。本仓库是独立的、非官方的学习工程；我的新增内容是学习路线、讲解、CPU 小实验及个人复现记录。上游源码固定在提交 [`7a9137d`](UPSTREAM_SOURCE.md)，保留原 README 与 Apache-2.0 许可。不能把引用的上游实现当成本人原创。

## 学习地图

| 阶段 | 要回答的问题 | 动手入口 | 当前状态 |
| --- | --- | --- | --- |
| 1. Tokenizer | 一句话怎样变成 token ID？聊天角色标记在哪里？ | [分词实验](learning/inspect_tokenizer.py) · [讲义](guide/01-tokenizer-and-data.md) | CPU 实验已加入 |
| 2. 数据与目标 | 预训练和 SFT 分别让哪些位置参与 loss？ | [预训练标签](learning/inspect_pretrain_labels.py) · [SFT 标签](learning/inspect_sft_labels.py) | CPU 实验已运行 |
| 3. 模型结构 | embedding、注意力、MLP 如何改变张量形状？ | [结构实验](learning/inspect_model_shapes.py) · [讲义](guide/02-transformer.md) | CPU 实验已加入 |
| 4. 优化 | loss 怎样经反向传播改变权重？ | [一步训练](learning/one_step_training.py) · [讲义](guide/03-training-and-inference.md) | CPU 实验已运行 |
| 5. 两阶段训练 | 从头预训练与基于其权重做 SFT 有何区别？ | [个人复现记录](docs/reproduction.md) · [源码路线](docs/source-map.md) | 云端训练已完成 |
| 6. 推理与评估 | 为什么会截断、胡说或在不同采样设置下输出不同？ | [提示词实验](learning/inspect_prompts.py) · [评估计划](docs/learning-path.md) | 提示词 CPU 实验已加入；系统评估待做 |
| 7. 扩展 | LoRA、DPO、MoE 等要解决什么问题？ | [扩展阅读](guide/04-next-topics.md) | 仅源码阅读路线，未宣称训练 |

从头学习可先读 [导学页](guide/00-start.md)，按表中顺序运行实验，再对照上游源码。已经有基础的读者可以直接看 [源码地图](docs/source-map.md) 或 [完整原作者 README](README.upstream.md)。

## 快速开始：不租 GPU 也能学

以下脚本只用本仓库内的小样本或随机初始化的小模型；不下载 GB 级数据、不加载正式权重，也不会改动 `out/`。在 macOS/Linux 的仓库根目录运行：

```bash
conda create -y --prefix ./.venv python=3.12
./.venv/bin/python -m pip install -r requirements-learning.txt

./.venv/bin/python learning/inspect_tokenizer.py
./.venv/bin/python learning/inspect_pretrain_labels.py
./.venv/bin/python learning/inspect_sft_labels.py
./.venv/bin/python learning/inspect_model_shapes.py
./.venv/bin/python learning/one_step_training.py
./.venv/bin/python learning/inspect_prompts.py
```

没有 Conda 时，可用 `python3.12 -m venv .venv` 创建环境。更多说明与每个实验的观察点见 [learning/README.md](learning/README.md)。这套轻量依赖只为学习实验准备；上游完整训练和其它功能的依赖见 [`requirements.txt`](requirements.txt)。

## 这个仓库里哪些是我的工作？

- **上游快照**：根目录的 `model/`、`dataset/`、`trainer/`、`scripts/`、`images/`、`eval_llm.py` 等来自指定的 MiniMind 提交；原始中文说明保存在 [README.upstream.md](README.upstream.md)，英文说明保存在 [README_en.md](README_en.md)。
- **我的学习工程**：`learning/` 的可运行小实验、`guide/` 的分阶段讲义、`docs/` 的源码导航与个人训练记录、这个首页。教学脚本会导入上游类，不声称重新发明了模型。
- **我的训练产物**：正式权重仍在我自己的电脑，不在公开仓库里。Git 中只放 [SHA-256 校验值](docs/artifacts.sha256) 和[已核实的复现事实](docs/reproduction.md)。云端凭据、训练大数据、虚拟环境和权重均不上传。

这个项目的目标是能**解释、运行、修改并检验**一个小语言模型。我的两阶段训练是一份已完成的案例，不是所有章节都已做完的证明。`guide/` 中的“试一试”和“待实践”是留给下一步的任务；当实验真正完成时再补运行条件、输出和结论。

## 结果、局限与下一步

个人对话试验中，SFT 权重比预训练权重更容易按聊天格式作答；但少量主观例子和训练 loss 不能证明模型可靠，也不能替代固定数据集、固定生成参数和多次评估。下一步优先做 [预训练/SFT 的可比评测](docs/learning-path.md)，之后再选 LoRA 或 DPO 等方向深挖。

想核对本地权重时，将 `pretrain_768.pth`、`full_sft_768.pth` 放到未追踪的 `out/`，或用 `--weights-dir` 指向原位置，然后运行 `learning/verify_weights.py`。不要把 `.pth`、训练 JSONL 或密钥提交到 GitHub。

## 致谢与许可

感谢 [MiniMind 原作者及贡献者](https://github.com/jingyaogong/minimind)。本仓库保留上游的 [Apache-2.0 LICENSE](LICENSE) 与[来源说明](UPSTREAM_SOURCE.md)；数据、权重和第三方服务可能另有使用条件，本仓库不替它们作授权声明。
