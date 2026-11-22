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
    from mlx_lm.sample_utils import make_sampler

    # load 返回类型是 2/3 元组的联合，索引取值避免 mypy 联合解包报错
    loaded = load(teacher_model)
    model, tokenizer = loaded[0], loaded[1]
    print(f"Teacher: {teacher_model}")

    plan = combo_plan(total, seed=42)
    accepted: list[dict[str, object]] = []
    rejected = 0
    seen_requirements: set[str] = set()

    def chat(messages: list[dict[str, str]]) -> str:
        # Qwen3 是 thinking 模型，默认输出 <think> 段会破坏 JSON 解析
        text = tokenizer.apply_chat_template(  # type: ignore[no-untyped-call]
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        response: str = generate(
            model,
            tokenizer,
            prompt=text,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature, top_p=0.95),
            verbose=False,
        )
        return response

    for i, (feature, scenario, difficulty, variant) in enumerate(plan):
        prompt = build_prompt(feature, scenario, difficulty, variant)
        sample = parse_teacher_json(
            chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            )
        )
        ok, reason = (False, "JSON 解析失败") if sample is None else validate_sample(sample)

        # 回炉一次：带着验证器的具体错误让 teacher 自修（通过率 20%→显著提升）
        if not ok and sample is not None:
            repair = parse_teacher_json(
                chat(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": json.dumps(sample, ensure_ascii=False)},
                        {
                            "role": "user",
                            "content": (
                                f"你的答案未通过自动验证：{reason}。请修正（保持需求不变），只输出修正后的 JSON 对象。"
                            ),
                        },
                    ]
                )
            )
            if repair is not None:
                sample = repair
                ok, reason = validate_sample(sample)

        if sample is None or not ok:
            rejected += 1
            if i < 20:
                print(f"  ✗ {reason}: {scenario}/{feature[0]}")
        else:
            requirement = sample.get("requirement")
            if isinstance(requirement, str) and requirement not in seen_requirements:
                seen_requirements.add(requirement)
                accepted.append(sample)
            else:
                rejected += 1
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
