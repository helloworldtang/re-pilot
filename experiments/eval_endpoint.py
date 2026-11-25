"""四格对照评测（走 OpenAI 兼容端点）：底座/任意模型 × 零样本/+神谕示例 × 域内/OOD。

与 experiments/fewshot_poc.py 同一套提示词与神谕表，只是推理从本机 MLX 换成 HTTP 端点，
使 9B/125B 这类本机放不下的模型也能跑同一套基准。

用法：
    OMLK_API_KEY=... uv run python experiments/eval_endpoint.py --model Qwen3.5-9B-OptiQ-4bit --tag base_9b
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypedDict

from experiments.fewshot_poc import ORACLE_MAP, build_messages, check, parse_answer

DEFAULT_URL = "http://10.0.2.13:8001/v1/chat/completions"


class EvalRow(TypedDict):
    """评测集的一行。"""

    requirement: str
    regex: str
    flags: str
    positive: list[str]
    negative: list[str]


class CellResult(TypedDict):
    """一格的成绩。"""

    passed: int
    total: int
    rate: float
    seconds: float


def load_rows(path: str, limit: int) -> list[EvalRow]:
    with open(path, encoding="utf-8") as f:
        rows: list[EvalRow] = [json.loads(line) for line in f]
    return rows[:limit]


def chat(base_url: str, api_key: str, model: str, messages: list[dict[str, str]], max_tokens: int = 256) -> str:
    """OpenAI 兼容调用，贪婪解码，关思考链。"""
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        base_url, data=payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body: Any = json.loads(resp.read(), strict=False)
    return str(body["choices"][0]["message"].get("content") or "")


def oracle_for(requirement: str) -> dict[str, str] | None:
    for key, spec in ORACLE_MAP.items():
        if key in requirement:
            return spec
    return None


def main() -> None:
    """跑四格（或指定格），结果落 docs/eval_<tag>.json。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit-indist", type=int, default=200)
    ap.add_argument("--limit-ood", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default=None, help="结果 json 路径（默认 docs/eval_<tag>.json）")
    args = ap.parse_args()

    api_key = os.environ["OMLK_API_KEY"]
    indist = load_rows("data/eval.jsonl", args.limit_indist)
    ood = load_rows("data/ood_real.jsonl", args.limit_ood)

    def run_cell(rows: list[EvalRow], shots_mode: str) -> CellResult:
        def one(row: EvalRow) -> bool:
            requirement = row["requirement"]
            positive, negative = row["positive"], row["negative"]
            shots: list[dict[str, object]] = []
            if shots_mode == "oracle":
                o = oracle_for(requirement)
                if o is not None:
                    shots = [
                        {"requirement": o["requirement"], "answer": {"regex": o["regex"], "flags": ""}, "positive": []}
                    ]
            out = chat(args.url, api_key, args.model, build_messages(requirement, shots))
            ans = parse_answer(out)
            return bool(ans and check(ans["regex"], ans.get("flags", ""), positive, negative))

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            results = list(ex.map(one, rows))
        passed = sum(results)
        return {
            "passed": passed,
            "total": len(rows),
            "rate": passed / max(1, len(rows)),
            "seconds": round(time.time() - t0, 1),
        }

    report: dict[str, object] = {"model": args.model, "cells": {}}
    cells: dict[str, CellResult] = {}
    report["cells"] = cells
    for bench, rows in (("indist", indist), ("ood", ood)):
        for mode in ("zero", "oracle"):
            key = f"{bench}_{mode}"
            cells[key] = run_cell(rows, mode)
            c = cells[key]
            print(f"[{args.tag}] {key:14s} {c['passed']}/{c['total']} ({c['rate']:.0%})  {c['seconds']}s", flush=True)

    out = args.out or f"docs/eval_{args.tag}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
