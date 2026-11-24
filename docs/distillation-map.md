# 蒸馏路径地图：业界全景 × 本项目点亮情况

> 这张表回答一个问题：大模型蒸馏小模型，业界走通的路有哪些，本项目（GPT_teacher 找路 + re-pilot 落地）实证到了哪一步。
> 找路过程详记见 [GPT_teacher/docs/distill-roadmap.md](https://github.com/helloworldtang/GPT_teacher-3.37M-cn)（三路径决策树）；落地实验详记见 [experiment.md](experiment.md)。

## 全景与点亮矩阵

| # | 业界路径 | 代表案例 | 本项目状态 | 证据 |
|---|---|---|---|---|
| 1 | **数据级蒸馏**：teacher 合成语料 → student SFT | DeepSeek-R1（80 万 CoT 样本训 6 个小模型，蒸馏版实测打赢 GPT-4o）；Alpaca | ✅✅ 产品级点亮 | GPT_teacher 路径 A（域内 90%、OOD 零塌缩）+ re-pilot 全程（底座 22% → 蒸馏 91%，CLI 可用） |
| 2 | **伪标签蒸馏**：teacher 给海量无标注数据打标 | Distil-Whisper（22k 小时伪标签，8 倍速） | ◐ 半点亮 | re-pilot 程序化用例 = "合成标签 + 自动验证"，同思想；"teacher 给真实数据打标"待做 |
| 3 | **CoT 推理链蒸馏** | R1 / Distilling Step-by-Step / Masked CoT 内化 | ✅ 点亮适用边界 | 4M student 上负结果（域内 27→8/30）：容量门槛，student ≥1.5B 才装得下"知识+推理"两份 |
| 4 | **Logit 白盒 KD** | Hinton 原教旨；Minitron（剪枝+同词表重训，+16% MMLU） | ✅ 点亮为负结果+边界 | 跨词表后训练两轮失败（词表投影+位置对齐装置已入库 vocab_bridge）；与 DeepSeek 有意绕开互相印证；同词表场景可复用 |
| 5 | **On-policy 蒸馏**：student 采样、teacher 批改 | Qwen3 规模验证；HF H4（小模型上优于 RL） | ✅ 机制点亮 | 自举自解率 93%（1079/1158，`corpus/selftrain.py`）：本机验证器免费打分，GKD 成本大头消失。效果注脚：未撬动 OOD（见三重证伪） |
| 6 | **教科书级合成数据** | Phi 系列（质量 > 数量） | ✅ 思想同构 | "正确性由构造保证、自然性由 teacher 保证"是数据质量优先的极致版 |
| 配套 | 量化部署 | 所有端侧标配 | ✅ | fuse+convert 4bit：1GB、0.3s/题、精度零损失 |
| 配套 | 客观 benchmark | 各家有官方榜 | ✅ 自建 | 执行验证 100% 客观 + 蒸馏前后同题对比协议 |

## 本项目的自创增量

**语料架构反转**（业界公开配方未见同款）：teacher 直接出题通过率仅 23%（答案与用例自相矛盾），
反转为——regex 与正反用例由程序构造（100% 自洽，12 种特性覆盖），teacher 只把 regex 批量翻译成
自然中文需求。适用于一切"输出可机器验证"的任务（SQL / 命令行 / 格式转换同理）。

## 边界的诚实刻画

- 域内 91%（held-out 需求、执行验证）
- **训练外新场景 15%**（手造 20 题压测，`data/ood_real.jsonl`）：失败画像 = 场景误绑 + 前缀细节丢失
- 救援三连全部证伪：扩场景族（v2）、on-policy 自举（v3）、推理时检索（PoC）均未撬动 15%
- **机制已定位（2026-11-23）**：四格对照实验实锤灾难性遗忘——检索示例让底座 35%→50%，
  却让蒸馏版 15%→10%（连照抄都不会）；LoRA 领域 SFT 把泛化与 ICL 能力一并洗掉。
  蒸馏的真实交易：域内 22%→91%（+69pp）是用 OOD 35%→15%（-20pp）换的
- 三条对策：**双模型路由 ✅已验证（2026-11-24）**——OOD 15%→40%、域内 92%→94%，验证器做路由器零训练零误伤；
  SFT 混通用数据防遗忘（治本，待验证）→ 降蒸馏强度找帕累托点（待验证）；
  方法论教训：蒸馏必须同时挂域内+域外双基准

## 参考文献

- [DeepSeek-R1 Distillation: Synthetic vs. Human CoT Data](https://fireworks.ai/research/deepseek-r1-distillation)
- [Distilling Step-by-Step (Google Research)](https://research.google/blog/distilling-step-by-step-outperforming-larger-language-models-with-less-training-data-and-smaller-model-sizes)
- [Minitron: Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/abs/2407.14679)
- [Unlocking On-Policy Distillation (HuggingFace H4)](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation)
- [awesome-on-policy-distillation](https://github.com/chrisliu298/awesome-on-policy-distillation)
