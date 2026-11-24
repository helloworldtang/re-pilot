"""通用指令语料生成：teacher 多任务生成，防遗忘实验（对策一）的数据源。

五类通用任务轮询生成，每条 user 指令 + assistant 回答，直接产出 mlx-lm
chat 格式（不进领域语料的验证器——通用任务无免费执行裁判，靠 prompt 约束）。

用法：uv run python -m corpus.generic --total 2000
"""

from __future__ import annotations

import argparse
import json
import random

SYSTEM_PROMPT = "你是一个乐于助人的中文助手，回答简洁准确。"

# 五类通用任务 × 每类的题材池（防同质化）
TASKS = [
    ("改写", ["把这句话改得更口语化", "把这段话改写成正式书面语", "用更简洁的话复述"], ["生活琐事", "工作邮件", "产品介绍", "新闻标题"]),
    ("摘要", ["用一句话概括这段文字", "提取下面内容的三 个要点", "给这段文字写个标题"], ["科技新闻", "历史事件", "科普知识", "使用说明"]),
    ("常识问答", ["简答", "为什么", "是什么"], ["天文地理", "生物化学", "历史文化", "日常物理"]),
    ("翻译", ["翻译成英文", "翻译成中文"], ["日常用语", "技术术语", "谚语俗语", "短句"]),
    ("代码问答", ["这段代码做什么", "怎么写", "报错原因"], ["Python 基础", "Shell 命令", "SQL 查询", "正则入门"]),
]

GENERIC_SYSTEM = "你在为助手模型生成训练数据。给一条自然的用户指令和一条高质量的简短回答。只输出 JSON：{\"instruction\": \"...\", \"answer\": \"...\"}。回答不超过 60 字。"


def chat(tokenizer, model, generate_fn, prompt: str) -> str:
    from mlx_lm.sample_utils import make_sampler

    messages = [
        {"role": "system", "content": GENERIC_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    out: str = generate_fn(model, tokenizer, prompt=text, max_tokens=512, sampler=make_sampler(temp=0.8, top_p=0.95), verbose=False)
    return out


def main() -> None:
    """生成 total 条通用指令样本，落 data/generic.jsonl（mlx-lom chat 格式）。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=2000)
    ap.add_argument("--teacher", default="mlx-community/Qwen3-8B-8bit")
    args = ap.parse_args()

    from mlx_lm import generate, load

    loaded = load(args.teacher)
    model, tokenizer = loaded[0], loaded[1]

    rng = random.Random(42)
    rows: list[dict[str, list[dict[str, str]]]] = []
    for i in range(args.total):
        task, asks, topics = TASKS[i % len(TASKS)]
        ask, topic = rng.choice(asks), rng.choice(topics)
        prompt = f"任务类型：{task}（{ask}）。题材：{topic}。变体 {i // len(TASKS) + 1}，内容别重复。"
        raw = chat(tokenizer, model, generate, prompt)
        s, e = raw.find("{"), raw.rfind("}")
        if s == -1 or e <= s:
            continue
        try:
            obj = json.loads(raw[s : e + 1])
            instruction, answer = str(obj.get("instruction", "")), str(obj.get("answer", ""))
        except json.JSONDecodeError:
            continue
        if not (6 <= len(instruction) <= 120 and 4 <= len(answer) <= 200):
            continue
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": answer},
                ]
            }
        )
        if (i + 1) % 100 == 0:
            print(f"  进度 {i + 1}/{args.total} | 有效 {len(rows)}")

    with open("data/generic.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"完成: {len(rows)}/{args.total} → data/generic.jsonl")


if __name__ == "__main__":
    main()
