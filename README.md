# MiniMind 学习与复现记录

> 非官方学习仓库。上游项目是 [jingyaogong/minimind](https://github.com/jingyaogong/minimind)；这里保存一份与本人训练权重对应的**固定版本源码**、可重复的小实验和学习笔记。原作者的工作与本人的学习记录在仓库中明确区分。

**English note:** This is an independent learning and reproduction record, not an official MiniMind release. The upstream source is preserved at a pinned revision with its Apache-2.0 license and original README.

## 为什么有这个仓库

我用单张 RTX 4090 完成了 MiniMind Dense 64M 的预训练和 Full SFT，随后把两个最终权重下载到 Mac，核对 SHA-256，并观察到 SFT 版本更能按对话格式回答。只把训练命令和模型文件留在电脑里，很难复盘“数据如何变成标签、loss 如何影响权重、遇到的问题如何解决”；这个仓库把这些证据和理解集中保存。

这里**不**发布训练数据、模型权重、云平台凭据，也不把个人对话测试当作正式评测。GitHub 上的源码快照固定在提交 [`7a9137d2e90294df80ce9178b89e82657e19f5a7`](UPSTREAM_SOURCE.md)，以便对应本人训练的权重，而不是追随不断变化的上游 `main`。

## 内容导航

| 想看什么 | 从这里开始 |
| --- | --- |
| 原作者的完整介绍、训练步骤与图示 | [`README.upstream.md`](README.upstream.md)；英文版 [`README_en.md`](README_en.md) |
| 代码的阅读顺序和数据流 | [`docs/source-map.md`](docs/source-map.md) |
| token、label、loss、梯度、batch 的直白解释 | [`docs/core-concepts.md`](docs/core-concepts.md) |
| 预训练和 SFT 的小样本实验 | [`learning/README.md`](learning/README.md) |
| 本人的云端训练过程、参数与校验值 | [`docs/reproduction.md`](docs/reproduction.md) |
| 本次遇到的报错和误区 | [`docs/troubleshooting.md`](docs/troubleshooting.md) |
| 后续学习计划与评测方向 | [`docs/learning-path.md`](docs/learning-path.md) |
| 上游来源、版本与许可 | [`UPSTREAM_SOURCE.md`](UPSTREAM_SOURCE.md)、[`LICENSE`](LICENSE) |

仓库根目录的 `model/`、`dataset/`、`trainer/`、`scripts/` 和 `images/` 主要是上游源码快照；`learning/` 下的小实验和 `docs/` 下的个人记录是后来新增的学习材料。原始 `README.md` 只改名为 `README.upstream.md`，因此原有相对图片链接仍可使用。

## 在 Mac 或 Linux 上运行三个小实验

这些实验使用**玩具数据和 CPU**，不用下载 GB 级数据，也不用租 GPU；第三个实验创建的临时小模型不会读写正式权重。准备 Python 3.12 环境后，在仓库根目录执行：

```bash
conda create -y --prefix ./.venv python=3.12
./.venv/bin/python -m pip install -r requirements-learning.txt

./.venv/bin/python learning/inspect_pretrain_labels.py
./.venv/bin/python learning/inspect_sft_labels.py
./.venv/bin/python learning/one_step_training.py
```

没有 Conda 时，也可以用已安装的 `python3.12 -m venv .venv` 建环境，然后执行后续命令。完整的训练环境与上游全部可选功能需要 [`requirements.txt`](requirements.txt) 中更多依赖；上面较短的依赖清单只保证这些学习实验。

三个实验依次回答：

1. 预训练文本中哪些下一 token 预测计入 loss？
2. SFT 为什么读取整个对话，却主要给助手回答位置计分？
3. `backward()` 与 `optimizer.step()` 分别做什么？

## 权重如何保存与校验

正式权重**不在 Git 中**。已核对的文件名是 `pretrain_768.pth` 与 `full_sft_768.pth`，期望哈希值见 [`docs/artifacts.sha256`](docs/artifacts.sha256)。若你自行取得这两个文件并放进本仓库的 `out/`，可以运行：

```bash
./.venv/bin/python learning/verify_weights.py
```

校验程序只读取文件并计算 SHA-256，不加载模型对象。文件不在本仓库时可用 `--weights-dir` 指向自己的目录；**不要把 `.pth`、训练 JSONL、`.venv`、缓存或密钥提交到 GitHub**。正式推理要用对应版本的源码、tokenizer 和权重，且先了解上游的 [`eval_llm.py`](eval_llm.py) 用法。

## 复现结论与局限

- 本次确实完成了 Dense 64M 的预训练和 SFT 两阶段，并在 Mac 上做了人工推理检查；详细证据与未记录的参数分开写在 [`docs/reproduction.md`](docs/reproduction.md)。
- SFT 在这些对话中比预训练更像聊天助手，但这不是“模型可靠”或“达到通用大模型水平”的证明。训练 loss 与少量主观例子不能替代固定测试集、固定生成参数和多次评估。
- 本仓库不是对上游项目的替代维护。要查最新功能、下载方式、模型或数据许可证，请到[上游仓库](https://github.com/jingyaogong/minimind)核对。

## 版权与致谢

感谢 MiniMind 作者及贡献者开放源码。本仓库保留上游的 Apache-2.0 [`LICENSE`](LICENSE) 与两个原始 README；有关固定提交、文件来源和本人新增内容的界限见 [`UPSTREAM_SOURCE.md`](UPSTREAM_SOURCE.md)。复制或使用上游源码时请继续保留这些信息。训练数据、模型权重及第三方服务可能有各自的使用条件，此仓库不替它们作授权声明。
