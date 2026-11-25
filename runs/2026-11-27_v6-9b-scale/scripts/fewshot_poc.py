"""检索 PoC：few-shot 场景示例能否撬动 OOD 15%。

三条件对照（同一 ood_real 20 题）：
  A 零样本（基线，已有数据 15%）
  B 固定 few-shot：3 个通用域内示例（与考题无关，测"随便给例子有没有用"）
  C 神谕 few-shot：每题人工配语义相邻的域内示例（测"检索做对了的上限"）

判读：
  C >> B ≈ A → 检索有效且相似度是关键 → 值得建真检索（embedding 近邻）
  C ≈ A     → 2B 连照着抄都不会，边界是能力问题 → 检索也救不了
  B ≈ C >> A → 随便给例子就行，不需要精细检索

用法：uv run python experiments/fewshot_poc.py
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

from corpus.validate import parse_flags

# B 条件：3 个固定通用示例（覆盖数字类/符号类/文本类，与考题刻意不相关）
FIXED_SHOTS = [
    {
        "requirement": "匹配中国大陆手机号",
        "answer": {"regex": r"1[3-9]\d{9}", "flags": ""},
        "positive": ["13812345678"],
    },
    {
        "requirement": "提取十六进制颜色码",
        "answer": {"regex": r"#[0-9a-fA-F]{6}\b", "flags": ""},
        "positive": ["#FF5733"],
    },
    {
        "requirement": "匹配日志里的等级标记",
        "answer": {"regex": r"\[(INFO|WARN|ERROR|DEBUG)\]", "flags": ""},
        "positive": ["[ERROR] boom"],
    },
]

# C 条件：每道 OOD 考题 → 神谕配对的域内相似场景示例（人工指定）
ORACLE_MAP: dict[str, dict[str, str]] = {
    "微信号": {
        "requirement": "匹配中国大陆手机号",
        "regex": r"^[a-zA-Z][a-zA-Z0-9_]{5,19}$",
        "hint": "字符类+量词约束的账号格式",
    },
    "QQ号": {"requirement": "匹配中国大陆手机号", "regex": r"^[1-9]\d{4,10}$", "hint": "纯数字+首字符约束"},
    "邮政编码": {"requirement": "匹配中国大陆手机号", "regex": r"^\d{6}$", "hint": "固定位数纯数字"},
    "驼峰": {"requirement": "匹配 SQL 标识符", "regex": r"\b[a-z_][a-z0-9_]*\b", "hint": "小写开头的标识符"},
    "Markdown 超链接": {"requirement": "匹配 Markdown 行内代码", "regex": r"`[^`]+`", "hint": "成对符号包裹"},
    "HTML 标签": {"requirement": "匹配 HTML 标签", "regex": r"<[a-zA-Z][^>]*>", "hint": "尖括号标签"},
    "bash 变量": {"requirement": "匹配命令行长选项", "regex": r"--[a-z][a-z0-9-]{2,}", "hint": "符号前缀+字符类"},
    "UUID": {"requirement": "匹配 MAC 地址", "regex": r"([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}", "hint": "十六进制分段"},
    "Windows 文件路径": {"requirement": "匹配相对路径", "regex": r"\.{1,2}/[\w./-]+", "hint": "路径分隔符结构"},
    "emoji": {"requirement": "匹配中文词语", "regex": r"[一-龥]{2,4}", "hint": "Unicode 字符范围"},
    "重复出现": {"requirement": "匹配英文单词边界", "regex": r"\berror\b", "hint": "词边界"},
    "带 +86 前缀": {"requirement": "匹配中国大陆手机号", "regex": r"1[3-9]\d{9}", "hint": "可选前缀"},
    "Markdown 标题": {"requirement": "匹配 Markdown 粗体", "regex": r"\*\*[^*]+\*\*", "hint": "行首标记+锚点"},
    "SQL 行注释": {"requirement": "匹配 SQL 字符串字面量", "regex": r"'[^']*'", "hint": "到行尾的注释"},
    "CSV 带引号字段": {"requirement": "匹配带引号字符串", "regex": r'"[^"]+"', "hint": "成对引号"},
    "IPv6": {
        "requirement": "匹配 IPv4 地址",
        "regex": r"((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)",
        "hint": "多组数字分段",
    },
    "百分数": {"requirement": "匹配金额", "regex": r"[¥$]\d{1,3}(,\d{3})*(?:\.\d{2})?", "hint": "数字+单位后缀"},
    "温度": {"requirement": "匹配时间", "regex": r"\d{1,2}:\d{2}(:\d{2})?", "hint": "数字+单位符号"},
    "时区": {"requirement": "匹配 HTTP 状态行", "regex": r"HTTP/1\.[01] \d{3}", "hint": "固定字母前缀+数字"},
    "中文姓名": {"requirement": "匹配中文词语", "regex": r"^[一-龥]{2,4}$", "hint": "中文字符+长度"},
}


def build_prompt(
    tokenizer: Any,
    requirement: str,
    shots: list[dict[str, object]],
) -> str:
    """system + few-shot 多轮 + 真题。"""
    from train_sft import SYSTEM_PROMPT

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot in shots:
        messages.append({"role": "user", "content": str(shot["requirement"])})
        messages.append({"role": "assistant", "content": json.dumps(shot["answer"], ensure_ascii=False)})
    messages.append({"role": "user", "content": requirement})
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return str(text)


def parse_answer(out: str) -> dict[str, str] | None:
    s, e = out.find("{"), out.rfind("}")
    if s == -1 or e <= s:
        return None
    try:
        obj: dict[str, str] = json.loads(out[s : e + 1])
        if isinstance(obj.get("regex"), str):
            return obj
    except json.JSONDecodeError:
        pass
    return None


def check(regex: str, flags: str, positive: list[str], negative: list[str]) -> bool:
    try:
        pattern = re.compile(regex, parse_flags(flags))
    except (re.error, ValueError):
        return False
    return all(pattern.search(c) for c in positive) and not any(pattern.search(c) for c in negative)


def main() -> None:
    """跑 B/C 两条件（A 基线已有），出对照表。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--adapter", default="adapters")
    ap.add_argument("--eval", default="data/ood_real.jsonl")
    args = ap.parse_args()

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    loaded = load(args.model, adapter_path=args.adapter)
    model, tokenizer = loaded[0], loaded[1]

    with open(args.eval, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    results: dict[str, int] = {"B_fixed": 0, "C_oracle": 0}
    total = 0
    for row in rows:
        total += 1
        oracle: dict[str, str] | None = None
        for key, spec in ORACLE_MAP.items():
            if key in row["requirement"]:
                oracle = spec
                break
        if oracle is None:
            print(f"  ⚠ 神谕未匹配: {row['requirement'][:20]}")

        conditions = {
            "B_fixed": FIXED_SHOTS,
            "C_oracle": [
                {
                    "requirement": oracle["requirement"] if oracle else FIXED_SHOTS[0]["requirement"],
                    "answer": {"regex": oracle["regex"] if oracle else FIXED_SHOTS[0]["answer"]["regex"], "flags": ""},  # type: ignore[index]
                    "positive": [],
                }
            ],
        }
        for name, shots in conditions.items():
            text = build_prompt(tokenizer, row["requirement"], shots)  # type: ignore[arg-type]
            out = generate(model, tokenizer, prompt=text, max_tokens=256, sampler=make_sampler(temp=0.0), verbose=False)
            ans = parse_answer(out)
            if ans and check(ans["regex"], ans.get("flags", ""), row["positive"], row["negative"]):
                results[name] += 1

    print(f"\n条件对照（{total} 题，A 基线 = 15%/3 题）：")
    print(f"  B 固定 few-shot:  {results['B_fixed']}/{total} ({results['B_fixed'] / total:.0%})")
    print(f"  C 神谕 few-shot:  {results['C_oracle']}/{total} ({results['C_oracle'] / total:.0%})")
    with open("docs/fewshot_poc.json", "w", encoding="utf-8") as f:
        json.dump({"total": total, "A_baseline": 3, **results}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
