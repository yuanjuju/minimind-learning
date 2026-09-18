# 2026-09-17 个人复现实验记录

本页根据当时保留的终端输出整理，是**个人复现记录**，不是官方基准测试。涉及云平台实例的地址、端口、密钥、账户信息和账单未收录。

## 1. 版本和环境

- 源码固定在 [`7a9137d2e90294df80ce9178b89e82657e19f5a7`](../UPSTREAM_SOURCE.md)，避免后续上游变化影响权重加载。
- 云端训练硬件：单张 NVIDIA GeForce RTX 4090，约 24 GB 显存；训练前 `nvidia-smi` 与 `torch.cuda.is_available()` 均检查通过。
- 云端镜像：NGC PyTorch 2.6.0 系列，Python 3.12；具体依赖以云端当时的安装状态为准。
- 本机：MacBook Air，使用项目内独立的 Python 3.12 环境完成 CPU 推理及教学实验。云端训练不依赖 Mac 保持亮屏，但云端实例与训练进程必须持续运行。

## 2. 数据与训练顺序

1. 下载 `pretrain_t2t_mini.jsonl`，终端显示约 1.2 GB；下载 `sft_t2t_mini.jsonl`，显示约 1.7 GB。大数据文件**未复制进本仓库**。上游数据获取方式见 [`README.upstream.md`](../README.upstream.md)。
2. 运行 Dense 64M 预训练 1 个 epoch。终端输出 `Model Params: 63.91M`、`39695/39695`，最终 loss 约 2.0066。loss 是该批训练目标的数值，不是独立测试集成绩。
3. 基于预训练权重执行 Full SFT 1 个 epoch。终端输出总计 `56608` 个数据加载步；运行时使用的 batch size 为 16。**完整 SFT 命令当时未保存**，不能把默认参数当作已核实的实际命令。
4. 使用 `eval_llm.py --weight full_sft` 做过人工对话检查。SFT 版本比预训练版本更像聊天助手，但仍出现实时日期类问题回答不可靠的情况；这只是主观观察，不是系统评测。

当时记录的正式预训练命令（在云端仓库的 `trainer/` 目录下运行）：

```bash
python -u train_pretrain.py \
  --epochs 1 --batch_size 32 --accumulation_steps 8 \
  --num_workers 4 --max_seq_len 340 \
  --save_interval 2000 --log_interval 100 --from_resume 1
```

`--from_resume 1` 表示允许自动检测已有检查点；仅凭这条命令无法证明本次一定从检查点恢复。日志中的 `step` 是批次步，梯度累积为 8 时通常每 8 个批次更新一次参数。训练脚本的 `epoch_time` 字段实际按剩余步数估计，不能当作已用训练时长。

## 3. 最终文件与校验

云端 `out/pretrain_768.pth` 和 `out/full_sft_768.pth` 各约 132 MB，下载到 Mac 后逐个核对 SHA-256，结果见 [`artifacts.sha256`](artifacts.sha256)。这里**只保存校验值，不公开权重文件**。

若已经把自己的两个权重放进本仓库的 `out/`，运行：

```bash
python learning/verify_weights.py
```

若权重仍在另一个目录，运行：

```bash
python learning/verify_weights.py --weights-dir /path/to/minimind/out
```

`verify_weights.py` 只做逐块哈希，不调用 `torch.load`，因此不执行来自权重文件的反序列化逻辑。

## 4. 范围和未记录事项

- 未保留云端完整依赖锁定文件、SFT 精确启动命令、验证集分数或多个随机种子结果；因此本页不宣称“严格可复现”的质量指标。
- 当时存在约 619 MB 的 SFT 续训检查点，但**未列入本机已校验的交付权重**；正式推理只需要相应 `out/*.pth` 和同版源码、tokenizer。
- 本仓库不分发训练数据、云平台镜像、模型权重或商业平台的使用凭据。
