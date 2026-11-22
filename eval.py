"""评测：held-out 需求上跑模型 → 解析 regex → 本机执行验证通过率。

对比三种模型：底座裸跑（蒸馏前）/ LoRA 蒸馏版 / teacher 参照（构造保证 100%）。

用法：
    uv run python eval.py --model Qwen/Qwen3.5-2B --tag base
    uv run python eval.py --model Qwen/Qwen3.5-2B --adapter adapters --tag distilled
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter

from corpus.validate import parse_flags, parse_teacher_json
from train_sft import SYSTEM_PROMPT


def run_eval(model_path: str, adapter: str | None, eval_path: str, limit: int) -> dict[str, object]:
    """在 eval 集上评测一个模型，返回指标 dict。"""
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    loaded = load(model_path, adapter_path=adapter) if adapter else load(model_path)
    model, tokenizer = loaded[0], loaded[1]

    rows = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    rows = rows[:limit]

    total = compile_ok = passed = 0
    fail_reasons: list[str] = []
    durations = 0.0
    for row in rows:
        text = tokenizer.apply_chat_template(  # type: ignore[no-untyped-call]
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["requirement"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        t0 = time.time()
        out = generate(
            model,
            tokenizer,
            prompt=text,
            max_tokens=256,
            sampler=make_sampler(temp=0.0),
            verbose=False,
        )
        durations += time.time() - t0
        total += 1

        sample = parse_teacher_json(out) if "{" in out else None
        regex = sample.get("regex") if sample else None
        if not isinstance(regex, str) or not regex:
            fail_reasons.append("格式失败")
            continue
        try:
            flags = parse_flags(sample.get("flags", "") if sample else "")
            pattern = re.compile(regex, flags)
        except (re.error, ValueError):
            fail_reasons.append("编译失败")
            continue
        compile_ok += 1
        if all(pattern.search(c) for c in row["positive"]) and not any(pattern.search(c) for c in row["negative"]):
            passed += 1
        else:
            fail_reasons.append("用例未过")

    return {
        "total": total,
        "compile_ok": compile_ok,
        "passed": passed,
        "pass_rate": passed / max(1, total),
        "avg_seconds": durations / max(1, total),
        "fail_reasons": fail_reasons,
        "fail_counts": dict(Counter(fail_reasons)),
    }


def main() -> None:
    """命令行入口：跑评测并打印对比行。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--eval", default="data/eval.jsonl")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--tag", default="model")
    args = ap.parse_args()

    result = run_eval(args.model, args.adapter, args.eval, args.limit)
    print(
        f"[{args.tag}] 通过 {result['passed']}/{result['total']} ({result['pass_rate']:.0%}) | "
        f"可编译 {result['compile_ok']} | 平均 {result['avg_seconds']:.1f}s/题"
    )
    print("  失败归因:", result["fail_counts"])
    with open(f"docs/eval_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
