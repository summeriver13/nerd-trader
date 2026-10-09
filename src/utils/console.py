"""
src/utils/console.py
控制台输出环境适配
"""

import io
import logging
import sys
import unicodedata


def force_utf8_stdout() -> None:
    """Windows 下 stdout/stderr 默认 GBK，输出 emoji 重定向到文件时会 UnicodeEncodeError，统一转 utf-8"""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            encoding = (stream.encoding or "").lower()
            if "utf" not in encoding:
                stream.reconfigure(encoding="utf-8")


def setup_logging(level: int = logging.INFO) -> None:
    """统一日志出口：没有它 logger.info 永不打印、warning/error 只会裸打 stderr"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def display_width(text: str) -> int:
    """东亚全角字符占 2 列，len() 会算成 1，中文表头按字符数补空格必然错位"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text: str, width: int, align: str = "left") -> str:
    """按显示宽度补空格（中文表格对齐用）"""
    fill = " " * max(0, width - display_width(text))
    return text + fill if align == "left" else fill + text
