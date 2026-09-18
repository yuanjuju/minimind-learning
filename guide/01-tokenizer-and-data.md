# 第一章：Tokenizer 与训练样本

**目标**：分清“人看到的文字”“模型看到的 token ID”和“训练时要预测的 label”。本章依托上游 [`model/tokenizer_config.json`](../model/tokenizer_config.json) 与 [`dataset/lm_dataset.py`](../dataset/lm_dataset.py)，不是自己重新训练 tokenizer。

## 1. 从字符串到 ID

运行 `python learning/inspect_tokenizer.py`。它加载仓库自带的 tokenizer，展示一句话的 token 片段、ID 与解码结果。汉字、标点、英文不保证一字符对应一 token；模型只接收整数 ID。输出的 `bos/eos/pad` 在这份 tokenizer 中分别用于消息开始、结束和填充，**不能**把 `pad` 的数值与 `-100` 混为一谈：前者是合法 token ID，后者是损失函数忽略标记。

在本仓库固定的 tokenizer 上，脚本用“你好，MiniMind！”测得 **12 个字符、8 个 token**；这只是一个具体例子，换一段文本数字就会变。特殊标记 ID 分别是 `bos=1`、`eos=2`、`pad=0`，可在实验输出与配置文件中互相核对。

脚本也会展示 `apply_chat_template` 产生的角色头、结束标记以及可能出现的 `<think>` 块。聊天模板是训练和推理输入格式的一部分，不是 UI 上的装饰。观察时对比一句裸文本与同一句放进 `user/assistant` 消息后的变化。

## 2. 预训练：几乎每个非填充位置都是目标

运行 `python learning/inspect_pretrain_labels.py`，再看上游的 `PretrainDataset.__getitem__`。它把样本文本截断，加上开始和结束 token，并补齐到固定长度；填充位置的 label 改为 `-100`。模型在位置 `i` 预测位置 `i+1`，因此真正参与损失的是 `labels[1:]` 中非 `-100` 的位置。

本章的玩具 JSONL 只有一条手写文本，不代表正式训练数据。修改 `learning/sample_pretrain.jsonl` 后重新运行，可以看到 token 数与有效目标位置随文字改变；改动前可先复制一份样本留底。

## 3. SFT：读完整对话，主要给助手回答计分

运行 `python learning/inspect_sft_labels.py`，再看 `SFTDataset.create_chat_prompt`、`generate_labels`。完整对话都进入 `input_ids`；标签生成器把非助手目标位置设为 `-100`，主要让助手回答及结束标记进入交叉熵。这不是“不读取用户提问”：提问仍是预测回答的上下文。

源码中的 `pre_processing_chat` 和 `post_processing_chat` 含有随机处理；教学脚本固定随机种子让输出可比。正式训练时不要以为样本 JSONL 就与最终送给模型的字符串完全一致。

## 自检

1. 指着实验输出说清某个 token ID 对应哪段文本，以及能否从 ID 解码回来。
2. 找到一个预训练 padding 位置，说明它为什么不计入 loss。
3. 找到一个 SFT 用户位置和助手回答位置，说明两者虽都被模型读取，但计分方式不同。
4. **待实践**：手工加一轮用户/助手消息，再观察长对话的模板和 `max_length` 截断如何改变标签；记录输入、预期和输出后再写结论。

下一章：[模型结构](02-transformer.md)。
