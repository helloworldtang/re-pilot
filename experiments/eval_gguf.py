"""GGUF 版等价性评测：走 llama-server 的 OpenAI 兼容接口。

部署前门禁：MLX(91%) → GGUF Q4_K_M 后同套考题重跑，掉点超过 2pp 不上线。

前置：llama-server -m models/re-pilot-q4km.gguf --port 8012 已在跑
用法：uv run python experiments/eval_gguf.py [--eval data/eval.jsonl] [--limit 200] [--tag gguf_q4]
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from typing import Any

from corpus.validate import parse_flags
from train_sft import SYSTEM_PROMPT


def chat(base_url: str, requirement: str) -> str:
    """OpenAI 兼容调用，贪婪解码。"""
    payload = json.dumps(
        {
            "model": "re-pilot",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": requirement},
            ],
            "temperature": 0.0,
            "max_tokens": 256,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body: Any = json.loads(resp.read())
    return str(body["choices"][0]["message"]["content"])


def main() -> None:
    """跑评测并打印通过率。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8012")
    ap.add_argument("--eval", default="data/eval.jsonl")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--tag", default="gguf_q4")
    args = ap.parse_args()

    with open(args.eval, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f][: args.limit]

    passed = compile_ok = 0
    fails: list[str] = []
    t0 = time.time()
    for i, row in enumerate(rows):
        out = chat(args.base_url, row["requirement"])
        s, e = out.find("{"), out.rfind("}")
        ok = False
        if s != -1 and e > s:
            try:
                ans = json.loads(out[s : e + 1])
                regex = ans.get("regex")
                if isinstance(regex, str) and regex:
                    compile_ok += 1
                    pattern = re.compile(regex, parse_flags(ans.get("flags", "")))
                    if all(pattern.search(c) for c in row["positive"]) and not any(
                        pattern.search(c) for c in row["negative"]
                    ):
                        passed += 1
                        ok = True
            except (json.JSONDecodeError, re.error, ValueError):
                pass
        if not ok:
            fails.append(row["requirement"][:20])
        if (i + 1) % 50 == 0:
            print(f"  进度 {i + 1}/{len(rows)} | 通过 {passed}")

    rate = passed / max(1, len(rows))
    print(
        f"[{args.tag}] 通过 {passed}/{len(rows)} ({rate:.0%}) | 可编译 {compile_ok} | "
        f"平均 {(time.time() - t0) / len(rows):.1f}s/题"
    )
    with open(f"docs/eval_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump({"passed": passed, "total": len(rows), "rate": rate, "compile_ok": compile_ok}, f, indent=2)


if __name__ == "__main__":
    main()
