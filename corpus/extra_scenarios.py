"""第二批场景生成器：SQL/命令行/工程向 15 族，扩多样性治 OOD 误绑。

与手造 OOD 考题重叠的族（驼峰/UTC/百分数/温度/bash变量等）刻意不收，
保持 data/ood_real.jsonl 的 held-out 纯净。

OOD 压力测试（手造 20 题训练外需求）实测 15%：失败画像 = 场景误绑
（QQ 号→手机号、驼峰→邮箱）+ 前缀细节丢失。药方是扩大场景族多样性，
让"需求→regex"映射本身泛化。手造 OOD 集保持 held-out 不进训练。
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import TypeAlias

# 与 programmatic.Atom 同构（避免循环导入，本地声明）
Atom: TypeAlias = tuple[str, list[str], list[str], set[str]]


def atom_sql_ident(r: random.Random) -> Atom:
    regex = r"\b[a-z_][a-z0-9_]*\b"
    pos = ["SELECT user_id FROM t", "字段 name 有效", "取 order_no 列"]
    neg = ["1abc", "user-id", "UPPER CASE"]
    return regex, pos, neg, {"anchor", "char_class"}


def atom_sql_string(r: random.Random) -> Atom:
    regex = r"'[^']*'"
    pos = ["WHERE name='张三' 结束", "值 'abc123' 存入", "SET x='a b'"]
    neg = ["name=张三", '"双引号"', "'未闭合"]
    return regex, pos, neg, {"char_class", "quantifier"}


def atom_shell_short_flag(r: random.Random) -> Atom:
    regex = r"\s-[a-zA-Z]\b"
    pos = ["ls -l 显示", "rm -f 强制", "grep -i 搜索"]
    neg = ["ls --all", "a-b 连字符", "无flag"]
    return regex, pos, neg, {"anchor", "char_class"}


def atom_shell_long_flag(r: random.Random) -> Atom:
    regex = r"--[a-z][a-z0-9-]{2,}"
    pos = ["uv sync --extra dev", "--verbose 输出", "git log --oneline 简洁"]
    neg = ["单横线 -x", "-- 无字", "a--b"]
    return regex, pos, neg, {"char_class", "quantifier"}


def atom_cron(r: random.Random) -> Atom:
    regex = r"(\d{1,2} |\* ){4}(\d{1,2}|\*)"
    pos = ["0 8 * * * 任务", "30 4 1 * * 备份", "15 2 3 4 5 触发"]
    neg = ["* * 不好", "0 8 *", "cron 表达式"]
    return regex, pos, neg, {"group", "quantifier"}


def atom_json_number(r: random.Random) -> Atom:
    regex = r":\s*-?\d+(?:\.\d+)?\b"
    pos = ['{"count": 42}', '{"price": -3.5 有', '{"id":100}']
    neg = ['{"name": "abc"}', "冒号: abc", "no json"]
    return regex, pos, neg, {"quantifier", "non_capture"}


def atom_markdown_bold(r: random.Random) -> Atom:
    regex = r"\*\*[^*]+\*\*"
    pos = ["这是**重点**内容", "**标题**开头", "看**加粗字**这里"]
    neg = ["*单星*", "普通文本", "**空**"]
    return regex, pos, neg, {"char_class", "quantifier"}


def atom_markdown_code(r: random.Random) -> Atom:
    regex = r"`[^`]+`"
    pos = ["运行 `uv sync` 即可", "看 `cli.py` 文件", "命令 `ls -l` 输出"]
    neg = ["普通命令", "``", "无代码"]
    return regex, pos, neg, {"char_class", "quantifier"}


def atom_http_status(r: random.Random) -> Atom:
    regex = r"HTTP/1\.[01] \d{3}"
    pos = ["HTTP/1.1 200 OK", "响应 HTTP/1.0 404 未找到", "HTTP/1.1 500 错误"]
    neg = ["200 OK", "HTTP 200", "HTTP/2 200"]
    return regex, pos, neg, {"escape", "quantifier"}


def atom_git_sha(r: random.Random) -> Atom:
    regex = r"\b[0-9a-f]{7,40}\b"
    pos = ["commit 8988603 已合", "sha e61564f9c2 提交", "hash abc1234"]
    neg = ["123456", "xyzabcd", "g123456"]
    return regex, pos, neg, {"char_class", "quantifier"}


def atom_port(r: random.Random) -> Atom:
    regex = r":\d{2,5}\b"
    pos = ["服务 localhost:8080 运行", "连接 db:5432 成功", "端口 :3000 开放"]
    neg = ["句末:", "a:b", "时间 12:30 秒"]
    return regex, pos, neg, {"quantifier", "anchor"}


def atom_latitude(r: random.Random) -> Atom:
    regex = r"-?\d{1,2}\.\d{4,},\s*-?\d{1,3}\.\d{4,}"
    pos = ["31.2304, 121.4737 上海", "坐标 39.9042,116.4074", "-33.8688, 151.2093"]
    neg = ["31.23, 121.47", "经纬 31 度", "31.2304 单值"]
    return regex, pos, neg, {"quantifier", "escape"}


def atom_base64(r: random.Random) -> Atom:
    regex = r"\b[A-Za-z0-9+/]{20,}={0,2}\b"
    pos = ["token SGVsbG9Xb3JsZDEyMzQ1 全文", "串 aGVsbG8gd29ybGQh 对照", "YWFhYmJiY2NjZGRkZWVmZg=="]
    neg = ["短串 abc", "!!!不是", "普通词 word"]
    return regex, pos, neg, {"char_class", "quantifier"}


def atom_utc_offset(r: random.Random) -> Atom:
    regex = r"UTC[+-]\d{2}:\d{2}"
    pos = ["时间 UTC+08:00 开盘", "UTC-05:00 纽约", "log UTC+00:00"]
    neg = ["GMT+8", "UTC+8", "UTC08:00"]
    return regex, pos, neg, {"quantifier", "escape"}


def atom_percent(r: random.Random) -> Atom:
    regex = r"\d+(?:\.\d+)?%"
    pos = ["通过率 85%", "误差 12.5%以内", "100%"]
    neg = ["百分之五十", "85", "%"]
    return regex, pos, neg, {"quantifier", "non_capture"}


def atom_temperature(r: random.Random) -> Atom:
    regex = r"-?\d+(?:\.\d+)?\s*°[CF]"
    pos = ["体温 36.5°C", "今天 98°F 热", "降到了 -5°C"]
    neg = ["36.5 度", "°C", "36.5K"]
    return regex, pos, neg, {"quantifier", "escape"}


def atom_camel(r: random.Random) -> Atom:
    regex = r"\b[a-z]+[A-Z][a-zA-Z]*\b"
    pos = ["getUserById 和 fetchData", "myVarName", "调用 parseJson 完毕"]
    neg = ["getuser", "GetUser", "USER_ID"]
    return regex, pos, neg, {"char_class", "anchor"}


def atom_relative_path(r: random.Random) -> Atom:
    regex = r"\.{1,2}/[\w./-]+"
    pos = ["引入 ./lib/utils.py", "配置 ../config.yml 读取", "走 ./a/b/c 路径"]
    neg = ["/usr/bin", "C:\\x", "相对 plain"]
    return regex, pos, neg, {"quantifier", "char_class"}


def atom_fullwidth_digit(r: random.Random) -> Atom:
    regex = "[０-９]{2,}"
    pos = ["编号１２３４", "房间 ０５ 室", "电话０８６"]
    neg = ["12345", "一百分", "abc"]
    return regex, pos, neg, {"char_class", "quantifier"}


EXTRA_GENERATORS: list[tuple[str, Callable[[random.Random], Atom]]] = [
    ("SQL 标识符", atom_sql_ident),
    ("SQL 字符串字面量", atom_sql_string),
    ("命令行短选项", atom_shell_short_flag),
    ("命令行长选项", atom_shell_long_flag),
    ("cron 表达式", atom_cron),
    ("JSON 数字值", atom_json_number),
    ("Markdown 粗体", atom_markdown_bold),
    ("Markdown 行内代码", atom_markdown_code),
    ("HTTP 状态行", atom_http_status),
    ("git 短哈希", atom_git_sha),
    ("端口号", atom_port),
    ("经纬度坐标", atom_latitude),
    ("base64 串", atom_base64),
    ("相对路径", atom_relative_path),
    ("全角数字", atom_fullwidth_digit),
]
