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

| 模型 | 域内 200 题 | 训练外 20 题 | 均时 |
|---|---|---|---|
| 底座裸跑 | 22% | 35% | 0.9s |
| 蒸馏版（LoRA） | 91% | 15% | 1.0s |
| **双模型路由（v0.2.0 默认）** | **94%** | **40%** | ~1s |

蒸馏增益 +69pp；代价是灾难性遗忘（训练外 35%→15%，详见
[docs/experiment.md](docs/experiment.md) 的四格对照实验）。v0.2.0 的双模型路由用
验证器做路由器（蒸馏版先答，失败转底座+检索），把两端同时抬升、零训练零误伤。
业界蒸馏路径全景与点亮情况见 [docs/distillation-map.md](docs/distillation-map.md)。

## 实验台账

所有训练尝试（成功与失败）登记在 [EXPERIMENTS.md](EXPERIMENTS.md)：顶部永远是当前最优版本，下方按时间倒序记录每次尝试的方案、结果、思考与后续建议。

## 结构

```
corpus/   语料管线：程序化出题引擎（100% 自洽）+ teacher 批量需求翻译
data/     训练/评测语料（验证通过的样本）
EXPERIMENTS.md  实验台账（顶部=当前最优，下方时间倒序）
train_sft.py  LoRA 训练入口（mlx-lm）
eval.py   执行验证评测：底座 vs 蒸馏版
cli.py    端侧入口：需求 → regex + 样例即时验证
```

## License

MIT
