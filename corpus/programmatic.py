"""程序化出题引擎：原子场景生成器 × 结构包装器，构造 100% 自洽样本。

teacher 直接出题实测通过率仅 23%（regex 与用例自相矛盾），这里反转架构：
正确性由构造保证——原子生成器每场景随机化产出 (regex, 正例, 反例, 特性)，
包装器做语义安全的结构变换，全部产物过 validate_sample 兜底。

反例设计的坑（search 语义）：反例中不能含有"能被 regex 局部匹配"的子串
（如金额反例 "¥1,23,456" 含合法前缀 "¥1"、惰性 <.+?> 在 "<<>>" 上匹配 "<<>"）；反斜杠 w 默认 Unicode 会匹配中文。
teacher 的角色退为"需求翻译"（见 synth.py）。
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from typing import Any

from corpus.extra_scenarios import EXTRA_GENERATORS
from corpus.validate import validate_sample

Atom = tuple[str, list[str], list[str], set[str]]  # (regex, positive, negative, features)


def make_rng(seed: int) -> random.Random:
    return random.Random(seed)


def _digits(r: random.Random, n: int, alphabet: str = "0123456789") -> str:
    return "".join(r.choice(alphabet) for _ in range(n))


def _hex(r: random.Random, n: int) -> str:
    return _digits(r, n, "0123456789abcdefABCDEF")


def _cn(r: random.Random, n: int) -> str:
    return "".join(chr(r.randint(0x4E00, 0x9FA5)) for _ in range(n))


def atom_phone(r: random.Random) -> Atom:
    regex = r"1[3-9]\d{9}"
    pos = [r.choice(["13", "15", "17", "18", "19"]) + _digits(r, 9) for _ in range(3)]
    neg = [_digits(r, 10), _digits(r, 12), "2" + _digits(r, 10), "13" + _digits(r, 8)]
    return regex, pos, neg, {"char_class", "quantifier"}


def atom_email(r: random.Random) -> Atom:
    regex = r"[\w.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,4}"
    names = ["alice", "bob.wu", "team_dev", "user01", "contact"]
    domains = ["example.com", "mail.cn", "test.org", "api.io"]
    pos = [f"{r.choice(names)}@{r.choice(domains)}" for _ in range(3)]
    neg = ["plainaddress", "a@@b.com", "no.tld@", "@nodomain.com"]
    return regex, pos, neg, {"char_class", "quantifier"}


def atom_date(r: random.Random) -> Atom:
    y = r.choice(["2024", "2025", "2026"])
    sep = r.choice(["-", "/", "."])
    regex = rf"\d{{4}}{re.escape(sep)}\d{{2}}{re.escape(sep)}\d{{2}}"
    pos = [f"{y}{sep}{m:02d}{sep}{d:02d}" for m, d in [(3, 14), (11, 2), (12, 31)]]
    neg = [f"{y}{sep}3{sep}14", "20240314", f"14{sep}03{sep}{y}"]
    return regex, pos, neg, {"quantifier", "escape"}


def atom_ipv4(r: random.Random) -> Atom:
    regex = r"((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
    pos = [f"{r.randint(1, 223)}.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(1, 254)}" for _ in range(3)]
    neg = ["256.1.1.1", "1.2.3", "1..2.3"]
    return regex, pos, neg, {"alternation", "group", "quantifier"}


def atom_url(r: random.Random) -> Atom:
    regex = r"https?://[\w.-]+(?:/[a-zA-Z0-9./?%&=-]*)?"
    hosts = ["example.com", "api.service.cn", "docs.test.org"]
    pos = [f"https://{h}{p}" for h, p in zip(hosts, ["/a/b?x=1", "", "/docs"], strict=True)]
    neg = ["example.com", "ftp://a.com", "https://", "http:/a.com"]
    return regex, pos, neg, {"quantifier", "non_capture"}


def atom_hex_color(r: random.Random) -> Atom:
    regex = r"#[0-9a-fA-F]{6}\b"
    pos = [f"#{_hex(r, 6)}" for _ in range(3)]
    neg = [f"#{_hex(r, 5)}", _hex(r, 6), "#GGGGGG", f"#{_hex(r, 8)}"]
    return regex, pos, neg, {"char_class", "quantifier", "anchor"}


def atom_time(r: random.Random) -> Atom:
    regex = r"\d{1,2}:\d{2}(:\d{2})?"
    pos = [f"{h}:{m:02d}" for h, m in [(9, 5), (23, 59), (0, 0)]]
    neg = ["9:5", "9-30", "095", "9:"]
    return regex, pos, neg, {"quantifier", "group"}


def atom_amount(r: random.Random) -> Atom:
    regex = r"[¥$]\d{1,3}(,\d{3})*(?:\.\d{2})?"
    pos = [f"{r.choice(['¥', '$'])}{v}" for v in ["199", "1,299", "12,345.67"]]
    neg = ["199", "¥", "cost: $", "￥199"]
    return regex, pos, neg, {"char_class", "quantifier", "non_capture"}


def atom_version(r: random.Random) -> Atom:
    regex = r"v?\d+\.\d+(\.\d+)?(-[a-zA-Z0-9]+)?"
    pos = ["1.2", "v2.0.1", "3.4.5-beta"]
    neg = ["1", "v1", "1..2", "a.b.c"]
    return regex, pos, neg, {"quantifier", "group"}


def atom_mac(r: random.Random) -> Atom:
    sep = r.choice([":", "-"])
    regex = rf"([0-9A-Fa-f]{{2}}{re.escape(sep)}){{5}}[0-9A-Fa-f]{{2}}"
    pos = [sep.join(_hex(r, 2) for _ in range(6)) for _ in range(3)]
    bad = [_hex(r, 2) for _ in range(6)]
    bad[0] = "g7"  # 非十六进制字符
    neg = [sep.join(_hex(r, 2) for _ in range(5)), sep.join(bad)]
    return regex, pos, neg, {"group", "quantifier", "char_class"}


def atom_chinese(r: random.Random) -> Atom:
    n = r.randint(2, 4)
    regex = rf"[一-龥]{{{n},{n + 2}}}"
    pos = [_cn(r, n + r.randint(0, 2)) for _ in range(3)]
    neg = [_cn(r, 1), "hello123", "中x"]
    return regex, pos, neg, {"escape", "quantifier"}


def atom_log_level(r: random.Random) -> Atom:
    regex = r"\[(INFO|WARN|ERROR|DEBUG)\]"
    lv = r.choice(["INFO", "WARN", "ERROR", "DEBUG"])
    pos = [f"[{lv}] request handled", f"2024-01-01 [{lv}] done", f"server [{lv}] ok"]
    neg = ["[info] x", "[TRACE] y", "INFO z", "[INFOWARN]"]
    return regex, pos, neg, {"group", "alternation", "anchor"}


def atom_filename(r: random.Random) -> Atom:
    exts = ["png", "jpg", "jpeg", "gif"]
    ext_alt = "|".join(r.sample(exts, 3))
    regex = rf"[\w.-]+\.({ext_alt})"
    pos = [f"{r.choice(['logo', 'shot-2024', 'img_v2', 'photo01'])}.{r.choice(exts)}" for _ in range(3)]
    neg = ["readme.md", "archive.zip", "noext", ".png"]
    return regex, pos, neg, {"char_class", "alternation", "group"}


def atom_password(r: random.Random) -> Atom:
    regex = r"(?=.*\d)(?=.*[a-z])(?=.*[A-Z])\w{8,16}"
    pos = ["Abcdef12", "Passw0rd123", "xY9zzzzz"]
    neg = ["abcdefgh", "ABCDEFGH", "Abc12"]
    return regex, pos, neg, {"lookahead", "quantifier"}


def atom_id_card(r: random.Random) -> Atom:
    regex = r"\d{17}[\dXx]"
    pos = [_digits(r, 17) + r.choice("0123456789X") for _ in range(3)]
    neg = [_digits(r, 16), _digits(r, 18) + "Y", _digits(r, 17)]
    return regex, pos, neg, {"quantifier", "char_class"}


def atom_plate(r: random.Random) -> Atom:
    regex = r"[京津沪粤苏浙][A-Z][A-Z0-9]{5}"
    pos = [r.choice("京津沪粤苏浙") + r.choice("ABCDEFG") + _digits(r, 5, "0123456789ABC") for _ in range(3)]
    neg = ["A12345", "京12345", "京A1234", "沪A-12345"]
    return regex, pos, neg, {"char_class", "quantifier"}


def atom_duration(r: random.Random) -> Atom:
    regex = r"(\d+h)?(\d+m)(\d+s)?"
    pos = ["90m", "1h30m", "45m20s"]
    neg = ["1.5h", "m30", "30", "hm"]
    return regex, pos, neg, {"group", "quantifier"}


def atom_scientific(r: random.Random) -> Atom:
    regex = r"-?\d+(\.\d+)?[eE][+-]?\d+"
    pos = ["3.14e10", "-2E-5", "1e3"]
    neg = ["3.14", "e10", "1.5e", "-e3"]
    return regex, pos, neg, {"quantifier", "group"}


def atom_quoted(r: random.Random) -> Atom:
    regex = r'"[^"]+"'
    pos = ['"hello"', '"a b c"', '"json_key"']
    neg = ["hello", '"unterminated', '""']
    return regex, pos, neg, {"char_class", "quantifier"}


def atom_lookbehind(r: random.Random) -> Atom:
    regex = r"(?<=总价: )\d{1,6}(\.\d{2})?"
    pos = ["总价: 12800", "总价: 99.50", "账单 总价: 3000"]
    neg = ["12800", "价格: 12800", "总价 12800"]
    return regex, pos, neg, {"lookbehind", "quantifier"}


def atom_lazy(r: random.Random) -> Atom:
    regex = r"<.+?>"
    pos = ["<b>bold</b>", "<i>it</i>", "<u>x</u>"]
    neg = ["plain text", "a < b", "no tags"]
    return regex, pos, neg, {"lazy", "quantifier"}


def atom_word_ci(r: random.Random) -> Atom:
    word = r.choice(["error", "warning", "success", "failed"])
    regex = rf"\b{word}\b"
    pos = [word, word.upper(), f"start {word} end"]
    neg = ["myerror", "errors", word + "x"]
    return regex, pos, neg, {"anchor", "flags"}


ATOM_GENERATORS: list[tuple[str, Callable[[random.Random], Atom]]] = [
    ("中国大陆手机号", atom_phone),
    ("电子邮箱地址", atom_email),
    ("日期", atom_date),
    ("IPv4 地址", atom_ipv4),
    ("URL", atom_url),
    ("十六进制颜色码", atom_hex_color),
    ("时间", atom_time),
    ("金额", atom_amount),
    ("版本号", atom_version),
    ("MAC 地址", atom_mac),
    ("中文词语", atom_chinese),
    ("日志等级", atom_log_level),
    ("文件名", atom_filename),
    ("密码强度", atom_password),
    ("身份证号", atom_id_card),
    ("车牌号", atom_plate),
    ("时长", atom_duration),
    ("科学计数法", atom_scientific),
    ("带引号字符串", atom_quoted),
    ("后行断言取数", atom_lookbehind),
    ("惰性匹配标签", atom_lazy),
    ("英文单词边界", atom_word_ci),
]


def wrap_named_group(
    r: random.Random, regex: str, pos: list[str], neg: list[str]
) -> tuple[str, list[str], list[str], str]:
    name = r.choice(["value", "num", "key", "data", "item"])
    return f"(?P<{name}>{regex})", pos, neg, "named_group"


def wrap_anchor(r: random.Random, regex: str, pos: list[str], neg: list[str]) -> tuple[str, list[str], list[str], str]:
    return f"^{regex}$", pos, neg, "anchor"


def wrap_capture(r: random.Random, regex: str, pos: list[str], neg: list[str]) -> tuple[str, list[str], list[str], str]:
    return f"({regex})", pos, neg, "group"


def wrap_non_capture(
    r: random.Random, regex: str, pos: list[str], neg: list[str]
) -> tuple[str, list[str], list[str], str]:
    return f"(?:{regex})", pos, neg, "non_capture"


def wrap_alternation(
    r: random.Random, regex: str, pos: list[str], neg: list[str]
) -> tuple[str, list[str], list[str], str]:
    other_regex, other_pos, _, _ = r.choice(ATOM_GENERATORS)[1](r)
    return f"{regex}|{other_regex}", pos + other_pos, neg, "alternation"


WRAPPERS: list[Callable[..., tuple[str, list[str], list[str], str]]] = [
    wrap_named_group,
    wrap_anchor,
    wrap_capture,
    wrap_non_capture,
    wrap_alternation,
]


def generate_atoms(count: int, seed: int = 42) -> list[dict[str, Any]]:
    """生成 count 个自洽原子样本（1/3 套结构包装器），全部过验证器。"""
    r = make_rng(seed)
    out: list[dict[str, Any]] = []
    feature_coverage: set[str] = set()
    i = 0
    while len(out) < count and i < count * 3:
        i += 1
        scenario, gen = r.choice(ATOM_GENERATORS)
        regex, pos, neg, features = gen(r)
        flags = "i" if scenario == "英文单词边界" else r.choice(["", "", "", "i", "m"])
        if r.random() < 0.34:
            wrapper = r.choice(WRAPPERS)
            regex, pos, neg, extra = wrapper(r, regex, pos, neg)
            features = features | {extra}
        sample = {
            # 占位需求：teacher 翻译阶段覆盖；此处仅为通过验证器结构校验
            "requirement": f"匹配{scenario}的文本片段",
            "regex": regex,
            "flags": flags,
            "positive": pos,
            "negative": neg,
            "scenario": scenario,
            "features": sorted(features),
        }
        ok, _ = validate_sample(sample)
        if ok:
            feature_coverage |= features
            out.append(sample)
    print(f"程序化原子: {len(out)}/{count} | 特性覆盖 {sorted(feature_coverage)}")
    return out


ATOM_GENERATORS = ATOM_GENERATORS + EXTRA_GENERATORS
