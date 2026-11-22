# re-pilot

中文需求 → 正则表达式，跑在你电脑上的离线小模型（Qwen3.5-2B LoRA 蒸馏自本机 Qwen3-8B）。

## 快速开始

```bash
uv sync

# 语料（已随仓库提供 data/train.jsonl，可重新生成）
uv run python -m corpus.build --total 2200 --eval-holdout 200

# LoRA 训练（MPS 约 2.5 小时；adapters 已含在仓库）
uv run python train_sft.py --iters 2000 --batch-size 2 --grad-accumulate 2

# 评测
uv run python eval.py --model Qwen/Qwen3.5-2B --tag base
uv run python eval.py --model Qwen/Qwen3.5-2B --adapter adapters --tag distilled

# 使用
uv run python cli.py "匹配中国大陆手机号" --test "13812345678"
```

## 效果（held-out 200 题，本机 `re` 执行验证）

| 模型 | 通过率 | 均时 |
|---|---|---|
| 底座裸跑 | 22% | 0.9s |
| **蒸馏版** | **91%** | 1.0s |

蒸馏增益 +69pp，完全离线。实验全记录见 [docs/experiment.md](docs/experiment.md)，
业界蒸馏路径全景与本项目的点亮情况见 [docs/distillation-map.md](docs/distillation-map.md)。

## 结构

```
corpus/   语料管线：程序化出题引擎（100% 自洽）+ teacher 批量需求翻译
data/     训练/评测语料（验证通过的样本）
train_sft.py  LoRA 训练入口（mlx-lm）
eval.py   执行验证评测：底座 vs 蒸馏版
cli.py    端侧入口：需求 → regex + 样例即时验证
```

## License

MIT
