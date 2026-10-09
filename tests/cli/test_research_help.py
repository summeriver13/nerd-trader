"""CLI 冒烟：入口脚本必须能渲染帮助并正常退出。

argparse 会对 help 文本做 % 插值，字面 % 没写成 %% 时 --help 直接抛 ValueError，
这条测试专门拦这类"参数表崩了但主流程没人发现"的问题。
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_research_help_exits_zero() -> None:
    proc = subprocess.run(
        [sys.executable, "research.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--bars" in proc.stdout
