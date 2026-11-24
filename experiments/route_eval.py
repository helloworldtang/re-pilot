"""双模型路由评测：验证器做路由器，域内走蒸馏版、路由失败走底座+检索。

路由信号（只用生产环境可得的信息，不用任何 ground truth 作弊）：
  1. 蒸馏版输出不可编译 → 路由
  2. 用户样例（正例）有未命中的 → 路由
路由目标：底座 + 轻量检索 few-shot（字符二元组 Jaccard，从训练语料取 top-3）

评测两组基准：
  - OOD 20 题：目标 15% → 显著提升
  - 域内 200 题：目标不掉（路由误伤率可控）

用法：uv run python experiments/route_eval.py
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

from corpus.validate import parse_flags
from experiments.fewshot_poc import build_prompt, parse_answer


def bigrams(text: str) -> set[str]:
    """字符二元组（中英文通吃的轻量相似度基础）。"""
    return {text[i : i + 2] for i in range(len(text) - 1)} if len(text) > 1 else {text}


def retrieve_shots(requirement: str, corpus: list[dict[str, str]], k: int = 3) -> list[dict[str, object]]:
    """字符 Jaccard 检索最相似的 k 条训练语料做 few-shot。"""
    rq = bigrams(requirement)
    scored = []
    for row in corpus:
        sim = len(rq & bigrams(row["requirement"])) / max(1, len(rq | bigrams(row["requirement"])))
        scored.append((sim, row))
    scored.sort(key=lambda x: -x[0])
    return [
        {"requirement": row["requirement"], "answer": {"regex": row["regex"], "flags": row.get("flags", "")}}
        for _, row in scored[:k]
    ]


def gen(model: Any, tokenizer: Any, requirement: str, shots: list[dict[str, object]] | None = None) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    text = build_prompt(tokenizer, requirement, shots or [])
    out: str = generate(model, tokenizer, prompt=text, max_tokens=256, sampler=make_sampler(temp=0.0), verbose=False)
    return out


def route_check(regex: str, flags: str, samples: list[str]) -> bool:
    """路由判定：可编译且用户样例全命中 → 接受蒸馏版。"""
    try:
        pattern = re.compile(regex, parse_flags(flags))
    except (re.error, ValueError):
        return False
    return all(pattern.search(s) for s in samples)


def final_check(regex: str, flags: str, positive: list[str], negative: list[str]) -> bool:
    """评测判分（用完整正反例，只在 eval 时用）。"""
    try:
        pattern = re.compile(regex, parse_flags(flags))
    except (re.error, ValueError):
        return False
    return all(pattern.search(c) for c in positive) and not any(pattern.search(c) for c in negative)


def main() -> None:
    """跑两组基准，出路由版成绩。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="adapters")
    ap.add_argument("--base", default="Qwen/Qwen3.5-2B")
    args = ap.parse_args()

    from mlx_lm import load

    d_loaded = load(args.base, adapter_path=args.adapter)
    d_model, d_tok = d_loaded[0], d_loaded[1]
    b_loaded = load(args.base)
    b_model, b_tok = b_loaded[0], b_loaded[1]

    with open("data/train.jsonl", encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f]

    for name, path, limit in [("OOD", "data/ood_real.jsonl", 20), ("域内", "data/eval.jsonl", 200)]:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f][:limit]
        passed = routed = 0
        for row in rows:
            # 用户样例 = 正例（生产可得：用户说"要匹配这些"）
            samples = row["positive"][:3]
            ans_d = parse_answer(gen(d_model, d_tok, row["requirement"]))
            accept = False
            if ans_d is not None and route_check(ans_d["regex"], ans_d.get("flags", ""), samples):
                accept = final_check(ans_d["regex"], ans_d.get("flags", ""), row["positive"], row["negative"])
            else:
                routed += 1
                shots = retrieve_shots(row["requirement"], corpus)
                ans_b = parse_answer(gen(b_model, b_tok, row["requirement"], shots))
                if ans_b is not None:
                    accept = final_check(ans_b["regex"], ans_b.get("flags", ""), row["positive"], row["negative"])
            passed += 1 if accept else 0
        print(f"[路由版-{name}] 通过 {passed}/{len(rows)} ({passed / len(rows):.0%}) | 路由触发 {routed}/{len(rows)}")


if __name__ == "__main__":
    main()
