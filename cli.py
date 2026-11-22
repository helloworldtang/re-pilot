"""端侧 CLI：中文需求 → 正则（本地 2B 蒸馏模型，完全离线）。

用法：
    uv run python cli.py "匹配中国大陆手机号"
    uv run python cli.py "提取日志里的错误等级" --test "[ERROR] boom"
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from train_sft import SYSTEM_PROMPT

DEFAULT_MODEL = "models/re-pilot-4bit"  # 融合+4bit 量化版：0.3s/题，精度无损
DEFAULT_ADAPTER = ""


def main() -> None:
    """命令行入口：生成 regex 并可选即时验证样例。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("requirement", help="中文需求描述")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--test", action="append", default=None, help="样例文本（可多次），本地即时验证")
    args = ap.parse_args()

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    loaded = load(args.model, adapter_path=args.adapter if args.adapter else None)
    model, tokenizer = loaded[0], loaded[1]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": args.requirement},
    ]
    try:
        text = tokenizer.apply_chat_template(  # type: ignore[no-untyped-call]
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore[no-untyped-call]

    out = generate(
        model,
        tokenizer,
        prompt=text,
        max_tokens=256,
        sampler=make_sampler(temp=0.0),
        verbose=False,
    )
    start, end = out.find("{"), out.rfind("}")
    if start == -1 or end <= start:
        print(f"模型输出无法解析：{out[:120]}", file=sys.stderr)
        raise SystemExit(1)
    try:
        parsed = json.loads(out[start : end + 1])
        regex, flags = parsed["regex"], parsed.get("flags", "")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"模型输出 JSON 不完整：{out[:120]}", file=sys.stderr)
        raise SystemExit(1) from e

    print(f"regex: {regex}")
    print(f"flags: {flags or '(无)'}")
    flag_bits = 0
    for ch in flags or "":
        flag_bits |= {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}.get(ch, 0)
    try:
        pattern = re.compile(regex, flag_bits)
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
