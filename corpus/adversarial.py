"""对抗样本生成：在易混淆场景对之间构造判别性样本。

场景误绑的机制是"11 位数字 → 手机号正则"。原语料的多样性是场景级的（37 族各说各的），
缺少同一句式、相近形态、答案却不同的判别性样本。这里显式构造混淆对：每个对里
两条样本需求句式几乎相同，字符形态相近，但答案指向不同正则——强迫映射精细化。

约束：与 data/ood_real.jsonl 的 held-out 场景保持纯净——只训练"判别"，不训练 OOD
题的标准答案。对里若涉及 OOD 场景（如 QQ 号），其正则用形态不同的近邻替代
（如 10 位工号），而非 OOD 基准里的答案本身。

用法：uv run python -m corpus.adversarial --total 1200
"""

from __future__ import annotations

import argparse
import json
import random
from typing import NotRequired, TypedDict

from corpus.validate import validate_sample


class Side(TypedDict):
    """混淆对的一侧：一种物件及其正则与用例。"""

    obj: str
    regex: str
    pos: list[str]
    neg: list[str]
    flags: NotRequired[str]


class Pair(TypedDict):
    """易混淆的一对：句式相同、形态相近、答案不同。"""

    style: str
    a: Side
    b: Side


# 混淆对：(需求A, 正则A, 正例A, 反例A, 需求B, 正则B, 正例B, 反例B, 模板句式)
# 句式相同只换关键词——模型必须从"语义细节"而非"句式"决定答案。
CONFUSABLE_PAIRS: list[Pair] = [
    {
        "style": "匹配{obj}",
        "a": {
            "obj": "11 位手机号",
            "regex": "1[3-9]\\d{9}",
            "pos": ["13812345678", "15912345678", "18600001111", "19999999999"],
            "neg": ["1234567890", "1381234567"],
        },
        "b": {
            "obj": "10 位纯数字工号",
            "regex": "^\\d{10}$",
            "pos": ["1234567890", "0087654321", "9999999999"],
            "neg": ["13812345678", "123456789"],
        },
    },
    {
        "style": "匹配{obj}",
        "a": {
            "obj": "6 位邮政编码",
            "regex": "^\\d{6}$",
            "pos": ["200000", "518000", "100010"],
            "neg": ["20000", "2000000"],
        },
        "b": {
            "obj": "5 位数字编号",
            "regex": "^\\d{5}$",
            "pos": ["12345", "00001", "99999"],
            "neg": ["123456", "1234"],
        },
    },
    {
        "style": "匹配{obj}",
        "a": {
            "obj": "17 位数字串",
            "regex": "^\\d{17}$",
            "pos": ["11010119900101123", "00000000000000000", "99999999999999999"],
            "neg": ["1101011990010112", "110101199001011234"],
        },
        "b": {
            "obj": "18 位字符的身份编号",
            "regex": "^\\d{17}[\\dXx]$",
            "pos": ["11010119900101123X", "110101199001011234", "00000000000000000x"],
            "neg": ["11010119900101123", "X11010119900101123"],
        },
    },
    {
        "style": "匹配{obj}",
        "a": {
            "obj": "带 +86 前缀的中国手机号",
            "regex": "\\+861[3-9]\\d{9}",
            "pos": ["+8613812345678", "+8615912345678", "+8618600001111"],
            "neg": ["8613812345678", "13812345678"],
        },
        "b": {
            "obj": "不带前缀的中国手机号",
            "regex": "(?<!\\d)1[3-9]\\d{9}(?!\\d)",
            "pos": ["手机 13812345678", "13812345678", "联系 15912345678 即可"],
            "neg": ["+8613812345678", "8613812345678"],
        },
    },
    {
        "style": "匹配{obj}",
        "a": {
            "obj": "冒号分隔的 MAC 地址",
            "regex": "([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}",
            "pos": ["00:1A:2B:3C:4D:5E", "ff:ff:ff:ff:ff:ff", "AA:BB:CC:DD:EE:01"],
            "neg": ["00-1A-2B-3C-4D-5E", "00:1A:2B"],
        },
        "b": {
            "obj": "横线分隔的 MAC 地址",
            "regex": "([0-9A-Fa-f]{2}-){5}[0-9A-Fa-f]{2}",
            "pos": ["00-1A-2B-3C-4D-5E", "FF-FF-FF-FF-FF-FF", "aa-bb-cc-dd-ee-01"],
            "neg": ["00:1A:2B:3C:4D:5E"],
        },
    },
    {
        "style": "匹配{obj}",
        "a": {
            "obj": "四段点分的 IPv4 地址",
            "regex": "((25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]?\\d)\\.){3}(25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]?\\d)",
            "pos": ["192.168.1.1", "10.0.0.1", "255.255.255.255"],
            "neg": ["::1", "2001:db8::1"],
        },
        "b": {
            "obj": "冒号分隔的 IPv6 地址",
            "regex": "(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}",
            "pos": ["2001:0db8:85a3:0000:0000:8a2e:0370:7334", "1:2:3:4:5:6:7:8", "fe80:0:0:0:0:0:0:1"],
            "neg": ["192.168.1.1", "1.2.3.4"],
        },
    },
    {
        "style": "匹配{obj}",
        "a": {
            "obj": "用连字符分隔的日期",
            "regex": "\\d{4}-\\d{2}-\\d{2}",
            "pos": ["2024-03-14", "1999-12-31", "2026-01-01"],
            "neg": ["2024/03/14", "20240314"],
        },
        "b": {
            "obj": "用斜杠分隔的日期",
            "regex": "\\d{4}/\\d{2}/\\d{2}",
            "pos": ["2024/03/14", "1999/12/31", "2026/01/01"],
            "neg": ["2024-03-14", "2024.03.14"],
        },
    },
    {
        "style": "识别{obj}",
        "a": {
            "obj": "全小写下划线风格的标识符",
            "regex": "^\\b[a-z]+(?:_[a-z]+)+\\b$",
            "pos": ["user_name", "get_user_by_id", "max_retry_count"],
            "neg": ["userName", "getUserId"],
        },
        "b": {
            "obj": "全大写下划线风格的常量名",
            "regex": "^\\b[A-Z]+(?:_[A-Z]+)+\\b$",
            "pos": ["MAX_SIZE", "USER_NAME", "DEFAULT_TIMEOUT"],
            "neg": ["user_name", "MaxSize"],
        },
    },
    {
        "style": "识别{obj}",
        "a": {
            "obj": "以单引号包裹的字符串",
            "regex": "'[^']*'",
            "pos": ["'hello'", "name = '张三'", "'a b c'"],
            "neg": ['"hello"', "hello"],
        },
        "b": {
            "obj": "以双引号包裹的字符串",
            "regex": '"[^"]*"',
            "pos": ['"hello"', 'name = "张三"', '"a b c"'],
            "neg": ["'hello'", "hello"],
        },
    },
    {
        "style": "匹配{obj}",
        "a": {
            "obj": "带两位小数的金额",
            "regex": "\\d+\\.\\d{2}\\b",
            "pos": ["199.00", "价格 12.50 元", "0.99"],
            "neg": ["199", "12.5"],
        },
        "b": {
            "obj": "不带小数的整数金额",
            "regex": "(?<!\\.)\\b\\d+\\b(?!\\.)",
            "pos": ["199", "共 200 件", "数量：5"],
            "neg": ["199.00", "12.50"],
        },
    },
    {
        "style": "匹配{obj}",
        "a": {
            "obj": "摄氏温度值",
            "regex": "-?\\d+(?:\\.\\d+)?\\s*°C",
            "pos": ["36.5°C", "-5 °C", "100°C"],
            "neg": ["98°F", "36.5"],
        },
        "b": {
            "obj": "华氏温度值",
            "regex": "-?\\d+(?:\\.\\d+)?\\s*°F",
            "pos": ["98°F", "32 °F", "212°F"],
            "neg": ["36.5°C", "98"],
        },
    },
    {
        "style": "匹配{obj}",
        "a": {
            "obj": "以百分号结尾的比例",
            "regex": "\\d+(?:\\.\\d+)?%",
            "pos": ["85%", "12.5%", "100%"],
            "neg": ["85", "0.85"],
        },
        "b": {
            "obj": "0 到 1 之间的小数比例",
            "regex": "\\b0\\.\\d+\\b",
            "pos": ["0.85", "0.125", "比例 0.5 结束"],
            "neg": ["85%", "50%", "1.5"],
        },
    },
    {
        "style": "匹配{obj}",
        "a": {
            "obj": "以点斜杠开头的相对路径",
            "regex": "(?<!\\.)\\./[\\w./-]+",
            "pos": ["./src/main.py", "./config.yml", "./a/b/c.txt"],
            "neg": ["../lib/x.py", "/usr/bin"],
        },
        "b": {
            "obj": "以两点斜杠开头的上级路径",
            "regex": "\\.\\./[\\w./-]+",
            "pos": ["../lib/x.py", "../../config", "../a.txt"],
            "neg": ["./src/main.py", "/usr/bin"],
        },
    },
    {
        "style": "匹配{obj}",
        "a": {
            "obj": "单横线的命令行短选项",
            "regex": "(?<!-)-[a-zA-Z]\\b",
            "pos": ["ls -l", "grep -i x", "tar -z"],
            "neg": ["--all", "word", "no-flag"],
        },
        "b": {
            "obj": "双横线的命令行长选项",
            "regex": "--[a-z][a-z0-9-]+",
            "pos": ["uv sync --extra dev", "--verbose", "git log --oneline"],
            "neg": ["-l", "a--b"],
        },
    },
    {
        "style": "匹配{obj}",
        "a": {
            "obj": "井号开头的一级 Markdown 标题",
            "regex": "^# [^\\n]+$",
            "flags": "m",
            "pos": ["# 标题", "# 另一个标题", "# Hello world"],
            "neg": ["## 二级", "#无空格"],
        },
        "b": {
            "obj": "井号开头的二级 Markdown 标题",
            "regex": "^## [^\\n]+$",
            "flags": "m",
            "pos": ["## 二级", "## 另一个二级", "## Sub"],
            "neg": ["# 标题", "### 三级"],
        },
    },
    {
        "style": "匹配{obj}",
        "a": {
            "obj": "单星号包裹的斜体文本",
            "regex": "(?<!\\*)\\*[^*\\n]+\\*(?!\\*)",
            "pos": ["*强调*", "*斜体*", "这是*重点*内容"],
            "neg": ["**粗体**", "普通"],
        },
        "b": {
            "obj": "双星号包裹的粗体文本",
            "regex": "\\*\\*[^*\\n]+\\*\\*",
            "pos": ["**粗体**", "**重点**", "这是**加粗**内容"],
            "neg": ["*强调*", "普通"],
        },
    },
    {
        "style": "匹配{obj}",
        "a": {
            "obj": "反引号包裹的行内代码",
            "regex": "`[^`\\n]+`",
            "pos": ["`uv sync`", "看 `cli.py` 文件", "`x`"],
            "neg": ["```python", "code"],
        },
        "b": {
            "obj": "三反引号包裹的代码块围栏",
            "regex": "^```[a-zA-Z]*$",
            "flags": "m",
            "pos": ["```python", "```", "```bash"],
            "neg": ["`x`", "code"],
        },
    },
    {
        "style": "匹配{obj}",
        "a": {
            "obj": "方括号包裹的日志等级",
            "regex": "\\[(INFO|WARN|ERROR|DEBUG)\\]",
            "pos": ["[ERROR] boom", "[INFO] ok", "server [WARN] x"],
            "neg": ["{ERROR}", "ERROR"],
        },
        "b": {
            "obj": "花括号包裹的日志等级",
            "regex": "\\{(INFO|WARN|ERROR|DEBUG)\\}",
            "pos": ["{ERROR} boom", "{INFO} ok", "server {WARN} x"],
            "neg": ["[ERROR]", "ERROR"],
        },
    },
]


# 句式变体：同一混淆对用不同问法，防止模型记住句式本身
STYLES = ["{obj}", "匹配{obj}", "识别{obj}", "提取{obj}", "找出文本中的{obj}", "写个正则匹配{obj}", "{obj}的正则"]


def _vary(style: str, obj: str, rng: random.Random) -> str:
    """给需求句加变体。"""
    verb = rng.choice(["匹配", "识别", "提取", "找出", "抓取"])
    s = style.replace("匹配{obj}", f"{verb}{{obj}}").replace("识别{obj}", f"{verb}{{obj}}")
    s = s.format(obj=obj)
    if rng.random() < 0.3:
        s = s + rng.choice(["的文本", "的字符串", "片段"])
    return s


def build_pair_samples(pair: Pair, rng: random.Random, strict: bool = False) -> list[dict[str, object]]:
    """由一个混淆对生成两条判别样本；每条样本的正例含自己、反例含对方（互为反例）。"""
    out: list[dict[str, object]] = []
    for me, ot in ((pair["a"], pair["b"]), (pair["b"], pair["a"])):
        requirement = _vary(pair["style"], me["obj"], rng)
        # 反例 = 自带反例 + 对方的正例（互为反例是判别信号的核心）
        negative = list(me["neg"]) + list(ot["pos"])[:1]
        sample: dict[str, object] = {
            "requirement": requirement,
            "regex": me["regex"],
            "flags": me.get("flags", ""),
            "positive": list(me["pos"]),
            "negative": negative,
        }
        ok, reason = validate_sample(sample)
        if not ok:
            if strict:
                print(f"  ✗ 对抗样本未过验证: {me['obj'][:14]} — {reason}")
            continue
        out.append(sample)
    return out


def main() -> None:
    """生成 total 条对抗样本，落 runs/2026-11-26_v5-adversarial/data/adversarial.jsonl。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=1200)
    args = ap.parse_args()

    rng = random.Random(42)
    rows: list[dict[str, object]] = []
    # 先跑一轮全量验证，统计可用对数
    base = []
    for pair in CONFUSABLE_PAIRS:
        base.extend(build_pair_samples(pair, rng, strict=True))
    print(f"混淆对 {len(CONFUSABLE_PAIRS)} 个 → 一次轮询产出 {len(base)} 条有效对抗样本")

    if not base:
        raise SystemExit("混淆对全部未过验证，检查用例数（验证器要求 ≥3 正例、≥2 反例）")
    # 循环上界：每轮至少产出 len(base) 条，超过 total 轮仍凑不够即判定异常退出（防死循环）
    i = 0
    while len(rows) < args.total and i < args.total * len(CONFUSABLE_PAIRS):
        pair = CONFUSABLE_PAIRS[i % len(CONFUSABLE_PAIRS)]
        i += 1
        rows.extend(build_pair_samples(pair, rng))
    if len(rows) < args.total:
        print(f"⚠ 仅产出 {len(rows)}/{args.total}，按实际数量继续")

    rows = rows[: args.total]
    rng.shuffle(rows)
    with open("runs/2026-11-26_v5-adversarial/data/adversarial.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    uniq_req = len({r["requirement"] for r in rows})
    uniq_re = len({r["regex"] for r in rows})
    print(f"完成: {len(rows)} 条 | 唯一需求 {uniq_req} | 唯一正则 {uniq_re}")


if __name__ == "__main__":
    main()
