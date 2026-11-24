"""端侧 CLI：中文需求 → 正则（本地 2B 蒸馏模型 + 双模型路由，完全离线）。

双模型路由（v0.2.0）：蒸馏版（专才）先答，验证器判可编译 + 样例命中；
失败则转底座（通才）+ 检索 few-shot 重答——OOD 40% / 域内 94% 的路由版成绩。

用法：
    uv run re-pilot "匹配中国大陆手机号" --test "13812345678"
    uv run re-pilot "提取日志里的错误等级" --test "[ERROR] boom"
    uv run re-pilot "匹配微信号" --no-route   # 关路由，只看蒸馏版
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

from experiments.fewshot_poc import build_prompt, parse_answer
from experiments.route_eval import retrieve_shots

DEFAULT_MODEL = "models/re-pilot-4bit"  # 融合+4bit 量化版：0.3s/题，精度无损
DEFAULT_ADAPTER = ""
DEFAULT_BASE = "Qwen/Qwen3.5-2B"  # 路由备用：底座（通才，检索 few-shot）

FLAG_BITS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}


def compile_flags(flags: str) -> int:
    bits = 0
    for ch in flags or "":
        bits |= FLAG_BITS.get(ch, 0)
    return bits


def main() -> None:
    """命令行入口：蒸馏版生成 → 验证器路由 → 必要时底座+检索重答。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("requirement", help="中文需求描述")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--base", default=DEFAULT_BASE, help="路由备用的底座模型")
    ap.add_argument("--no-route", action="store_true", help="关闭双模型路由（只用蒸馏版）")
    ap.add_argument("--corpus", default="data/train.jsonl", help="检索 few-shot 的语料库")
    ap.add_argument("--test", action="append", default=None, help="样例文本（可多次），路由信号 + 即时验证")
    args = ap.parse_args()

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    def run(model: Any, tokenizer: Any, requirement: str, shots: list[dict[str, object]] | None = None) -> str:
        text = build_prompt(tokenizer, requirement, shots or [])
        return generate(model, tokenizer, prompt=text, max_tokens=256, sampler=make_sampler(temp=0.0), verbose=False)

    loaded = load(args.model, adapter_path=args.adapter if args.adapter else None)
    model, tokenizer = loaded[0], loaded[1]

    base_model = base_tok = None
    corpus: list[dict[str, str]] | None = None
    if not args.no_route:
        b_loaded = load(args.base)
        base_model, base_tok = b_loaded[0], b_loaded[1]
        with open(args.corpus, encoding="utf-8") as f:
            corpus = [json.loads(line) for line in f]

    def accepts(regex: str, flags: str, samples: list[str]) -> bool:
        """路由判定：可编译且样例全命中。"""
        try:
            pattern = re.compile(regex, compile_flags(flags))
        except re.error:
            return False
        return all(pattern.search(s) for s in samples)

    # ── 第一跳：蒸馏版（专才）──
    answer = parse_answer(run(model, tokenizer, args.requirement))
    source = "蒸馏版"
    routed = False
    first_hop_bad = answer is None or not accepts(answer["regex"], answer.get("flags", ""), args.test or [])
    if first_hop_bad and base_model is not None and corpus is not None and base_tok is not None:
        # ── 第二跳：底座（通才）+ 检索 few-shot ──
        routed = True
        shots = retrieve_shots(args.requirement, corpus)
        answer_b = parse_answer(run(base_model, base_tok, args.requirement, shots))
        if answer_b is not None:
            answer = answer_b
            source = "路由→底座+检索"

    if answer is None:
        print("两个模型均无法解析出 JSON 输出", file=sys.stderr)
        raise SystemExit(1)

    regex, flags = answer["regex"], answer.get("flags", "")
    print(f"regex: {regex}")
    print(f"flags: {flags or '(无)'}")
    print(f"来源: {source}" + ("（建议提供 --test 样例以启用完整路由信号）" if routed and not args.test else ""))
    try:
        pattern = re.compile(regex, compile_flags(flags))
    except re.error as e:
        print(f"⚠ 生成结果无法编译: {e}", file=sys.stderr)
        raise SystemExit(2) from e

    if args.test:
        for sample_text in args.test:
            m = pattern.search(sample_text)
            verdict = f"命中: {m.group(0)!r}" if m else "未命中"
            print(f"  {sample_text!r} → {verdict}")


if __name__ == "__main__":
    main()
