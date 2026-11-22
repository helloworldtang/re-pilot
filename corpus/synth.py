"""teacher 语料合成：mlx-lm 本机生成 + 验证器过滤 + 落盘。

用法：
    uv run python corpus/synth.py --total 2600 --eval-holdout 200
    uv run python corpus/synth.py --total 5 --inspect   # 冒烟 + 打印样本
"""

from __future__ import annotations

import argparse
import json
import random

from corpus.features import build_prompt, combo_plan
from corpus.validate import parse_teacher_json, validate_sample

SYSTEM_PROMPT = "你是正则表达式出题与答题专家。严格按用户要求的 JSON 格式输出，不要任何解释。"


def synth(
    total: int,
    eval_holdout: int,
    teacher_model: str,
    max_tokens: int,
    temperature: float,
) -> None:
    """生成 total 条候选 → 验证过滤 → train/eval 两份 jsonl 落盘。"""
    from mlx_lm import generate, load

    # load 返回类型是 2/3 元组的联合，索引取值避免 mypy 联合解包报错
    loaded = load(teacher_model)
    model, tokenizer = loaded[0], loaded[1]
    print(f"Teacher: {teacher_model}")

    plan = combo_plan(total, seed=42)
    accepted: list[dict[str, object]] = []
    rejected = 0
    seen_requirements: set[str] = set()

    for i, (feature, scenario, difficulty, variant) in enumerate(plan):
        prompt = build_prompt(feature, scenario, difficulty, variant)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = generate(
            model,
            tokenizer,
            prompt=text,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
        )
        sample = parse_teacher_json(response)
        if sample is None:
            rejected += 1
        else:
            ok, reason = validate_sample(sample)
            requirement = sample.get("requirement")
            if ok and isinstance(requirement, str) and requirement not in seen_requirements:
                seen_requirements.add(requirement)
                accepted.append(sample)
            else:
                rejected += 1
                if i < 20:
                    print(f"  ✗ {reason}: {scenario}/{feature[0]}")
        if (i + 1) % 50 == 0:
            print(f"  进度 {i + 1}/{len(plan)} | 通过 {len(accepted)} | 拒绝 {rejected}")

    rng = random.Random(42)
    rng.shuffle(accepted)
    eval_rows = accepted[:eval_holdout]
    train_rows = accepted[eval_holdout:]

    with open("data/train.jsonl", "w", encoding="utf-8") as f:
        for item in train_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open("data/eval.jsonl", "w", encoding="utf-8") as f:
        for item in eval_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    rate = len(accepted) / max(1, len(plan))
    print(f"\n完成: 候选 {len(plan)} | 通过 {len(accepted)} ({rate:.0%}) | 拒绝 {rejected}")
    print(f"data/train.jsonl {len(train_rows)} 条 | data/eval.jsonl {len(eval_rows)} 条")


def main() -> None:
    """命令行入口。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=2600, help="候选总数（按 1.3 倍冗余备料）")
    ap.add_argument("--eval-holdout", type=int, default=200)
    ap.add_argument("--teacher", default="mlx-community/Qwen3-8B-8bit")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--inspect", action="store_true", help="冒烟模式：只出 5 条并打印")
    args = ap.parse_args()

    if args.inspect:
        args.total = 5
        args.eval_holdout = 0

    synth(args.total, args.eval_holdout, args.teacher, args.max_tokens, args.temperature)

    if args.inspect:
        with open("data/train.jsonl", encoding="utf-8") as f:
            for line in f:
                print(json.dumps(json.loads(line), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
