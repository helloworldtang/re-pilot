"""多样性轴：正则特性 × 应用场景，驱动 teacher 合成不重样。

teacher 直接"出题"会翻来覆去手机号邮箱；显式轮询特性×场景组合，
每个组合要求不同难度/不同具体物件，语料覆盖面才够 LoRA 学出泛化。
"""

from __future__ import annotations

import random

# 正则语言特性（教学覆盖面）
FEATURES = [
    ("char_class", "字符类 [..] 与预定义类 \\d \\w \\s"),
    ("quantifier", "量词 {n,m} / * / + / ?"),
    ("lazy", "懒惰匹配 .*? 或 .+?"),
    ("alternation", "交替分支 (a|b)"),
    ("group", "捕获分组与反向引用"),
    ("named_group", "命名分组 (?P<name>...)"),
    ("non_capture", "非捕获分组 (?:...)"),
    ("lookahead", "正向先行断言 (?=...)"),
    ("neg_lookahead", "负向先行断言 (?!...)"),
    ("lookbehind", "后行断言 (?<=...) / (?<!...)"),
    ("anchor", "锚点 ^ $ \\b"),
    ("escape", "特殊字符转义与 Unicode 类 \\p 或中文字符范围"),
    ("flags", "flags 组合（忽略大小写/多行/dotall）"),
]

# 应用场景（真实世界物件）
SCENARIOS = [
    "中国大陆手机号",
    "电子邮箱地址",
    "日期时间（多种格式）",
    "IPv4 地址",
    "IPv6 地址",
    "URL 与域名",
    "JSON 字段值抽取",
    "日志行（等级/时间戳/组件）",
    "中文字符/中文词语",
    "金额与货币",
    "文件路径与扩展名",
    "密码强度判定",
    "身份证号",
    "车牌号",
    "十六进制颜色码",
    "Markdown 语法元素",
    "CSV 字段切分",
    "版本号",
    "SQL 注入可疑模式",
    "空白与缩进清洗",
    "HTML 标签",
    "时间间隔（如 1h30m）",
    "科学计数法数字",
    "电话分机与区号",
    "用户名/昵称规则",
    "MAC 地址",
    "经纬度坐标",
    "ISBN / 条形码",
    "表格对齐行",
    "重复行检测",
]

DIFFICULTIES = ["简单直接", "中等（含边界情况）", "较难（需断言或组合特性）"]


def combo_plan(total: int, seed: int = 42) -> list[tuple[tuple[str, str], str, str, int]]:
    """生成 total 个 (特性, 场景, 难度, 变体号) 组合计划。

    特性×场景全组合轮询 + 难度与变体号打散，保证覆盖均匀。
    """
    rng = random.Random(seed)
    combos = [(f, s) for f in FEATURES for s in SCENARIOS]
    rng.shuffle(combos)
    plan: list[tuple[tuple[str, str], str, str, int]] = []
    variant = 0
    while len(plan) < total:
        for f, s in combos:
            diff = rng.choice(DIFFICULTIES)
            plan.append((f, s, diff, variant))
            if len(plan) >= total:
                break
        variant += 1
    return plan


def build_prompt(feature: tuple[str, str], scenario: str, difficulty: str, variant: int) -> str:
    """把组合计划渲染成 teacher 出题 prompt。"""
    _, feature_desc = feature
    return (
        f"请出一道正则表达式的练习样本并给出标准答案。要求：\n"
        f"1. 应用场景：{scenario}（变体 {variant + 1}，换具体物件/格式，别与其他变体重复）\n"
        f"2. 必须用到特性：{feature_desc}\n"
        f"3. 难度：{difficulty}\n"
        f"4. 需求用一句中文描述（不透露正则写法）；给出 regex、可选 flags"
        f"（只能是 i/m/s 的组合字符串，无则空串）、至少 3 个应匹配的正例"
        f"和至少 3 个不应匹配的反例。正反例必须是完整待检文本片段。\n"
        f"5. **语法必须是 Python re 模块（3.11）支持的语法**：不支持 \\p{{...}}、"
        f"(?R)、递归、possessive 量词等 PCRE/JS 特性。提交前在心里自测每个用例。\n"
        f"只输出 JSON 对象，字段：requirement, regex, flags, positive（字符串数组）、"
        f"negative（字符串数组）。不要解释。"
    )
