"""语料验证器：本机 re 执行是唯一裁判。

teacher 生成的样本必须通过：可编译 + 正例全命中 + 反例全不命中 + 反平凡。
验证器只保证"regex 与用例自洽"，语义正确性靠抽检与考题交叉验证兜底。
"""

from __future__ import annotations

import json
import re
from typing import Any

FLAG_CHARS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}

TRIVIAL_PATTERNS = {"", ".*", ".+", "(.*)"}


def parse_flags(flag_str: Any) -> int:
    """把 flags 字符串（如 "im"）解析为 re 标志位，非法字符报 ValueError。"""
    if not flag_str:
        return 0
    if not isinstance(flag_str, str):
        raise ValueError(f"flags 非字符串: {flag_str!r}")
    flags = 0
    for ch in flag_str:
        if ch not in FLAG_CHARS:
            raise ValueError(f"非法 flag 字符: {ch!r}")
        flags |= FLAG_CHARS[ch]
    return flags


def validate_sample(sample: dict[str, Any]) -> tuple[bool, str]:
    """校验一条样本的自洽性，返回 (是否通过, 原因)。

    Args:
        sample: 含 requirement/regex/flags/positive/negative 的字典。

    Returns:
        (True, "ok") 或 (False, 具体原因)。
    """
    regex = sample.get("regex")
    if not isinstance(regex, str) or regex.strip() in TRIVIAL_PATTERNS:
        return False, "regex 缺失或平凡"
    try:
        flags = parse_flags(sample.get("flags", ""))
    except ValueError as e:
        return False, str(e)
    try:
        pattern = re.compile(regex, flags)
    except re.error as e:
        return False, f"编译失败: {e}"

    positive = sample.get("positive")
    negative = sample.get("negative")
    if not isinstance(positive, list) or not isinstance(negative, list):
        return False, "正反例缺失"
    if len(positive) < 3 or len(negative) < 2:
        return False, f"用例数不足 (pos={len(positive)}, neg={len(negative)})"

    for case in positive:
        if not isinstance(case, str) or not pattern.search(case):
            return False, f"正例未命中: {case!r}"
    for case in negative:
        if not isinstance(case, str):
            return False, "反例非字符串"
        if pattern.search(case):
            return False, f"反例误命中: {case!r}"

    requirement = sample.get("requirement")
    if not isinstance(requirement, str) or len(requirement) < 6:
        return False, "需求描述缺失或过短"

    return True, "ok"


def parse_teacher_json(raw: str) -> dict[str, Any] | None:
    """从 teacher 输出解析 JSON 对象（容忍围栏与前后噪声）。"""
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed: dict[str, Any] = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None
