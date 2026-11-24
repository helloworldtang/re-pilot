# 实验台账

> 域内 = `data/eval.jsonl` 200 题（训练见过的场景族，新题）；OOD = `data/ood_real.jsonl` 20 题（训练未见过的场景族）。判分均为本机 `re` 执行验证。

## 当前最优

**v0.2.0 · 双模型路由**（tag `v0.2.0`）：**域内 94% ｜ OOD 40%**

蒸馏版（v2 LoRA）先答 → 验证器判可编译+样例命中 → 失败转底座（Qwen3.5-2B 裸模型）+ 检索 few-shot 重答。
`uv run re-pilot "需求" --test "样例"`，默认即路由模式。

| 组成件 | 域内 | OOD |
|---|---|---|
| 蒸馏版 v2（`adapters/`，`data/train.jsonl` 3797 条训出） | 92% | 15% |
| 底座 + 检索 few-shot | — | 50% |
| 底座裸跑 | 22% | 35% |

## 全版本

| 版本 | 方案 | 域内 | OOD | 结论 |
|---|---|---|---|---|
| **v2 + 路由** | 蒸馏版 + 验证器路由到基座 | **94%** | **40%** | **当前最优** |
| v2 | 37 族 / 3800 条语料 LoRA | 92% | 15% | keeper 组成件 |
| v4 | v2 + 通用指令 1:1 混训 | 94% | 15% | ❌ 防遗忘失败 |
| v3 | v2 + on-policy 自举（自解率 93%） | 90% | 15% | ❌ 换分布无效 |
| v1 | 22 族 / 2000 条语料 LoRA | 91% | 15% | 被 v2 取代 |
| 底座 | Qwen3.5-2B 不训练 | 22% | 35% | 基线 |

## 一条结论

四次正交干预（扩数据 / 混通用 / on-policy / 检索）OOD 全部 15% 不动——**LoRA 领域 SFT 对泛化能力的清洗不可逆也不可预防**，是范式属性。域内 +69pp 是用 OOD -20pp 换的。当前量级下路由是终点；保泛化需范式级改变（全参 + 数十万条数据）。完整实验记录见 [docs/experiment.md](docs/experiment.md)。

## 复现 keeper

```bash
uv run python -m corpus.build --total 4000 --eval-holdout 200   # 或直接用仓内 data/train.jsonl
uv run python train_sft.py --train-file data/train.jsonl --iters 3500 --batch-size 2 --grad-accumulate 2
uv run python eval.py --adapter adapters --tag v2
```
