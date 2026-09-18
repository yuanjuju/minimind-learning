# 上游源码来源与许可

本仓库包含 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 的一份固定版本源码快照，用于复现和学习，而非宣称这些上游代码由本仓库作者原创。

- 上游提交：[`7a9137d2e90294df80ce9178b89e82657e19f5a7`](https://github.com/jingyaogong/minimind/tree/7a9137d2e90294df80ce9178b89e82657e19f5a7)
- 快照日期：2026-09-18
- 上游许可：Apache License 2.0；本仓库保留上游的 [`LICENSE`](LICENSE)
- 原始中文 README：[`README.upstream.md`](README.upstream.md)（仅改名，以便根目录 README 记录个人学习内容）
- 原始英文 README：[`README_en.md`](README_en.md)

快照中的 `model/`、`dataset/`、`trainer/`、`scripts/`、`images/`、`eval_llm.py` 等文件来自上述提交；`learning/`、`guide/` 和 `docs/` 下标明为学习记录的文件为后续新增。教学脚本会调用上游类，但这些脚本本身是本仓库的实验材料。若以后修改上游文件，会在相关提交中说明。

训练数据、模型权重、云平台凭据、实例地址、虚拟环境和缓存均**不**随源码快照上传。它们可能有独立的分发条件，且不适合普通 Git 仓库存储。
