"""语料编排：程序化原子（正确性由构造保证）+ teacher 批量需求翻译。

用法：
    uv run python -m corpus.build --total 2200 --eval-holdout 200
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable

from corpus.programmatic import generate_atoms
from corpus.validate import validate_sample

SYSTEM_PROMPT = "你是正则表达式工程师。严格按用户要求的 JSON 格式输出，不要任何解释。"

STYLES = [
    "直接描述",
    "口语化提问式",
    "工程场景式（如从日志/表格/文本中提取）",
    "约束条件式（强调不匹配什么）",
]


def translate_batch(
    batch: list[dict[str, object]],
    chat: Callable[[list[dict[str, str]]], str],
) -> dict[int, str]:
    """让 teacher 为一批原子写自然中文需求，返回 {序号: 需求句}。"""
    items = []
    for i, atom in enumerate(batch):
        pos = atom["positive"]
        neg = atom["negative"]
        assert isinstance(pos, list) and isinstance(neg, list)
        items.append(
            {
                "id": i,
                "用途": atom["scenario"],
                "regex": atom["regex"],
                "应匹配示例": pos[:3],
                "不应匹配示例": neg[:2],
            }
        )
    prompt = (
        "为下列每个正则表达式写一句自然的中文需求描述（用户视角，会说的话），"
        f"风格多样（{'; '.join(STYLES)}），每句不超过 30 字，"
        "不要出现正则语法本身，不要解释匹配原理。\n"
        '只输出 JSON 数组，元素形如 {"id": 序号, "requirement": "..."}。\n\n'
        + json.dumps(items, ensure_ascii=False, indent=1)
    )
    raw = chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        return {}
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}
    out: dict[int, str] = {}
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and isinstance(item.get("id"), int) and isinstance(item.get("requirement"), str):
                out[item["id"]] = item["requirement"].strip()
    return out


def build(total: int, eval_holdout: int, teacher_model: str, batch_size: int) -> None:
    """程序化生成 + 批量翻译 + holdout 切分 + 落盘。"""
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    atoms = generate_atoms(total, seed=42)

    loaded = load(teacher_model)
    model, tokenizer = loaded[0], loaded[1]
    print(f"Teacher: {teacher_model}（仅做需求翻译）")

    def chat(messages: list[dict[str, str]]) -> str:
        text = tokenizer.apply_chat_template(  # type: ignore[no-untyped-call]
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        response: str = generate(
            model,
            tokenizer,
            prompt=text,
            max_tokens=2048,
            sampler=make_sampler(temp=0.7, top_p=0.95),
            verbose=False,
        )
        return response

    translated = 0
    for start in range(0, len(atoms), batch_size):
        batch = atoms[start : start + batch_size]
        reqs = translate_batch(batch, chat)
        if len(reqs) < len(batch):  # 整批解析失败重试一次
            reqs = translate_batch(batch, chat)
        for i, atom in enumerate(batch):
            req = reqs.get(i)
            if req and 6 <= len(req) <= 60:
                atom["requirement"] = req
                translated += 1
            else:
                atom["requirement"] = ""  # 标记丢弃
        done = min(start + batch_size, len(atoms))
        print(f"  翻译进度 {done}/{len(atoms)} | 有效 {translated}")

    kept = [a for a in atoms if a["requirement"]]
    final: list[dict[str, object]] = []
    for atom in kept:
        sample = {
            "requirement": atom["requirement"],
            "regex": atom["regex"],
            "flags": atom["flags"],
            "positive": atom["positive"],
            "negative": atom["negative"],
        }
        ok, _ = validate_sample(sample)
        if ok:
            final.append(sample)
    print(f"终版样本: {len(final)}/{len(atoms)}（翻译有效 {translated}，验证通过 {len(final)}）")

    rng = random.Random(42)
    rng.shuffle(final)
    eval_rows = final[:eval_holdout]
    train_rows = final[eval_holdout:]
    with open("data/train.jsonl", "w", encoding="utf-8") as f:
        for item in train_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open("data/eval.jsonl", "w", encoding="utf-8") as f:
        for item in eval_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"data/train.jsonl {len(train_rows)} 条 | data/eval.jsonl {len(eval_rows)} 条")


def main() -> None:
    """命令行入口。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=2200)
    ap.add_argument("--eval-holdout", type=int, default=200)
    ap.add_argument("--teacher", default="mlx-community/Qwen3-8B-8bit")
    ap.add_argument("--batch-size", type=int, default=20)
    args = ap.parse_args()
    build(args.total, args.eval_holdout, args.teacher, args.batch_size)


if __name__ == "__main__":
    main()
