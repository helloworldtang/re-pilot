"""LoRA SFT：把 data/train.jsonl 转成 mlx-lm chat 格式并启动训练。

student 学的映射：user=中文需求 → assistant={"regex":..., "flags":...}。
正反用例只用于评测，不进训练目标（输出更干净、收敛更快）。

用法：
    uv run python train_sft.py --iters 2000
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys

SYSTEM_PROMPT = '你是正则助手。根据用户的中文需求输出 JSON：{"regex": "...", "flags": "i/m/s 或空"}。不要输出其他内容。'


def to_chat_rows(path: str) -> list[dict[str, list[dict[str, str]]]]:
    """把语料转成 mlx-lm 的 messages 格式。"""
    rows: list[dict[str, list[dict[str, str]]]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            answer = json.dumps({"regex": item["regex"], "flags": item["flags"]}, ensure_ascii=False)
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": item["requirement"]},
                        {"role": "assistant", "content": answer},
                    ]
                }
            )
    return rows


def main() -> None:
    """切分 train/valid 并调用 mlx_lm.lora 训练。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--valid", type=int, default=100, help="从训练集划出的验证条数")
    args = ap.parse_args()

    rows = to_chat_rows("data/train.jsonl")
    rng = random.Random(42)
    rng.shuffle(rows)
    valid_rows = rows[: args.valid]
    train_rows = rows[args.valid :]
    import os

    os.makedirs("data_lora", exist_ok=True)
    with open("data_lora/train.jsonl", "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open("data_lora/valid.jsonl", "w", encoding="utf-8") as f:
        for r in valid_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"data_lora: train {len(train_rows)} | valid {len(valid_rows)}")

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        args.model,
        "--train",
        "--data",
        "data_lora",
        "--iters",
        str(args.iters),
        "--batch-size",
        str(args.batch_size),
        "--num-layers",
        "16",
        "--learning-rate",
        str(args.lr),
    ]
    print(" ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
