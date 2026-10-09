"""
src/analytics/reporting.py
回测结果落盘：净值/基准/回撤序列与成交明细
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.types import TradeData


def save_equity_series(result: dict[str, Any], path: Path) -> None:
    """把净值、基准与回撤三条序列落成 CSV。

    终端内字符曲线与离线分析都读这份文件；PNG 只供人看，程序读不了。
    """
    frame = pd.DataFrame(
        {
            "timestamp": result["equity_curve"].index,
            "equity": result["equity_curve"].to_numpy(),
            "benchmark": result["benchmark_curve"].to_numpy(),
            "drawdown": result["drawdown_curve"].to_numpy(),
        }
    )
    frame.to_csv(path, index=False)


def save_trades(trades: Sequence[TradeData], path: Path) -> None:
    """把成交明细落成 parquet。

    方向与开平仓写枚举英文名（LONG/SHORT + OPEN/CLOSE），消费方按它判断开平仓；
    datetime 保持时间类型，读取端不必再解析字符串。
    """
    frame = pd.DataFrame(
        [
            {
                "symbol": trade.symbol,
                "direction": trade.direction.name,
                "offset": trade.offset.name,
                "price": trade.price,
                "volume": trade.volume,
                "datetime": trade.datetime,
            }
            for trade in trades
        ]
    )
    frame.to_parquet(path, index=False, compression="snappy")
