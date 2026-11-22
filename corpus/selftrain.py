"""On-policy 自举（rejection sampling / STaR 式）：学生自己出答案，验证器当老师。

流程：
  A 出题：程序化原子（新 seed，与训练语料不相交）+ teacher 翻译需求
  B 学生作答：student 每题采样 k 个 regex，本机验证器对题目的正反用例打分，
    通过的第一个收为该题答案（on-policy：用的是学生自己的分布）
  C 回炉：全 k 皆败的题回退 teacher 的 regex（难例保底），落 selftrain.jsonl

之后与原语料混合重训。验证器免费是这条路径在本项目成本极低的原因
（业界 GKD 的成本大头就是 teacher 逐 token 打分）。

用法：
    uv run python -m corpus.selftrain --pool 1200 --k 8 --adapter adapters
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable

from corpus.build import translate_batch
from corpus.programmatic import generate_atoms
from corpus.validate import parse_flags, validate_sample


def student_answers(
    requirement: str,
    chat_sampler: Callable[[list[dict[str, str]]], str],
    k: int,
) -> list[str]:
    """student 采样 k 个候选 regex（temp>0 保证多样性）。"""
    from train_sft import SYSTEM_PROMPT

    out: list[str] = []
    for _ in range(k):
        raw = chat_sampler(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": requirement},
            ]
        )
        s, e = raw.find("{"), raw.rfind("}")
        if s == -1 or e <= s:
            continue
        try:
            obj = json.loads(raw[s : e + 1])
            if isinstance(obj.get("regex"), str) and obj["regex"]:
                out.append(obj["regex"])
        except json.JSONDecodeError:
            continue
    return out


def check_candidate(regex: str, flags: str, positive: list[str], negative: list[str]) -> bool:
    """验证器打分：可编译 + 正例全中 + 反例全不中。"""
    try:
        pattern = re.compile(regex, parse_flags(flags))
    except (re.error, ValueError):
        return False
    return all(pattern.search(c) for c in positive) and not any(pattern.search(c) for c in negative)


def main() -> None:
    """命令行入口：出题 → 学生作答 → 验证 → 落盘。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=1200, help="自举出题数")
    ap.add_argument("--k", type=int, default=8, help="每题采样数")
    ap.add_argument("--teacher", default="mlx-community/Qwen3-8B-8bit")
    ap.add_argument("--student", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--adapter", default="adapters")
    args = ap.parse_args()

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    # ── A 出题：新 seed 原子 + teacher 翻译 ──
    atoms = generate_atoms(args.pool, seed=4242)
    print(f"出题池: {len(atoms)}（seed=4242，与训练语料不相交）")

    t_loaded = load(args.teacher)
    t_model, t_tok = t_loaded[0], t_loaded[1]

    def teacher_chat(messages: list[dict[str, str]]) -> str:
        text = t_tok.apply_chat_template(  # type: ignore[no-untyped-call]
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        out: str = generate(
            t_model, t_tok, prompt=text, max_tokens=2048, sampler=make_sampler(temp=0.7, top_p=0.95), verbose=False
        )
        return out

    for i in range(0, len(atoms), 20):
        batch = atoms[i : i + 20]
        reqs = translate_batch(batch, teacher_chat)
        for j, atom in enumerate(batch):
            req = reqs.get(j)
            atom["requirement"] = req if req and 6 <= len(req) <= 60 else ""
        print(f"  翻译 {min(i + 20, len(atoms))}/{len(atoms)}")

    pool = [a for a in atoms if a["requirement"]]
    print(f"有效题池: {len(pool)}")

    # ── B 学生作答 + 验证（换 student 模型；teacher 模型交还内存）──
    t_model = None
    s_loaded = load(args.student, adapter_path=args.adapter or None)
    s_model, s_tok = s_loaded[0], s_loaded[1]

    def student_chat(messages: list[dict[str, str]]) -> str:
        text = s_tok.apply_chat_template(  # type: ignore[no-untyped-call]
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        out: str = generate(
            s_model, s_tok, prompt=text, max_tokens=256, sampler=make_sampler(temp=0.8, top_p=0.95), verbose=False
        )
        return out

    rows: list[dict[str, object]] = []
    self_solved = teacher_fallback = 0
    for i, item in enumerate(pool):
        assert isinstance(item["requirement"], str)
        assert isinstance(item["positive"], list) and isinstance(item["negative"], list)
        assert isinstance(item["regex"], str) and isinstance(item["flags"], str)
        candidates = student_answers(item["requirement"], student_chat, args.k)
        picked = None
        for cand in candidates:
            if check_candidate(cand, item["flags"], item["positive"], item["negative"]):
                picked = cand
                break
        if picked is not None:
            self_solved += 1
            rows.append(
                {
                    "requirement": item["requirement"],
                    "regex": picked,
                    "flags": item["flags"],
                    "positive": item["positive"],
                    "negative": item["negative"],
                    "source": "self",
                }
            )
        else:
            # 难例保底：teacher 的标准答案进训练（学生对这些题还没学会）
            teacher_fallback += 1
            sample = {
                "requirement": item["requirement"],
                "regex": item["regex"],
                "flags": item["flags"],
                "positive": item["positive"],
                "negative": item["negative"],
            }
            if validate_sample(sample)[0]:
                rows.append({**sample, "source": "teacher"})
        if (i + 1) % 50 == 0:
            print(f"  自举进度 {i + 1}/{len(pool)} | 自解 {self_solved} | 保底 {teacher_fallback}")

    with open("data/selftrain.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rate = self_solved / max(1, len(pool))
    print(f"\n自举完成: 自解率 {rate:.0%}（{self_solved}/{len(pool)}）| teacher 保底 {teacher_fallback}")
    print("data/selftrain.jsonl")


if __name__ == "__main__":
    main()
