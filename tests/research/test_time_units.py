"""时间单位：--bars 是根数，抓数深度才换算成日历天；窗口裁剪必须精确。

历史教训：抓数按日历天、评估按根，两个单位混用会让 1h 的「730」只回测 30 天；
缓存不裁剪则同一个命令的结果随本地缓存变深而漂移。
"""

from typing import Any

import pandas as pd

from src.research.data_window import bars_to_days, load_window


class _FakeLoader:
    """假加载器：只回放一段固定长度的历史，不碰网络。"""

    def __init__(self, rows: int) -> None:
        self.frame = pd.DataFrame(
            {"close": range(rows)},
            index=pd.date_range("2020-01-01", periods=rows, freq="1D"),
        )
        self.calls: list[dict[str, Any]] = []

    def prepare_data(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        self.calls.append(kwargs)
        return self.frame


def test_bars_to_days_conversion() -> None:
    """根数 → 日历天按周期换算，至少 1 天。"""
    assert bars_to_days(730, "1d") == 730
    assert bars_to_days(730, "4h") == 122
    assert bars_to_days(730, "1h") == 31
    assert bars_to_days(1, "1m") == 1


def test_load_window_trims_to_bars_plus_warmup() -> None:
    """返回的根数恰好是 评估 + 预热，多出来的旧数据裁掉。"""
    loader = _FakeLoader(rows=5000)
    out = load_window(loader, "BTC/USDT", "1d", bars=500, warmup=618)
    assert out is not None
    assert len(out) == 1118
    # 抓数深度按「评估 + 预热」换算，不再额外加缓冲（required_buffer 已删除）
    assert loader.calls[0]["days_back"] == 1118
    assert "required_buffer" not in loader.calls[0]


def test_load_window_keeps_short_history_and_warns(
    capsys: Any,
) -> None:
    """历史不足时原样返回并提示，不静默缩短评估区间。"""
    loader = _FakeLoader(rows=100)
    out = load_window(loader, "BTC/USDT", "1d", bars=500, warmup=618)
    assert out is not None
    assert len(out) == 100
    assert "不足 1118 根" in capsys.readouterr().out
