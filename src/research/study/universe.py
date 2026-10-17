"""
src/research/study/universe.py
标的池装载：目录级读日线缓存 + 流动性排序（三条 TSMOM 研究线共用）。

为什么抽出来：现货/永续/容量三个脚本各自复制过一遍同样的 glob→读 parquet→根数门槛，
第三处出现时按 Rule of Three 提为共享实现。这里只做「读进来」与「按成交额排序」，
不做任何筛选决策——池子的定义属于调用方（各研究线口径不同，混在一起会互相污染）。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_bars_dir(root: Path, min_bars: int) -> dict[str, pd.DataFrame]:
    """读目录下全部 parquet 日线，只留不少于 ``min_bars`` 根的标的；文件名即标的代码。"""
    out: dict[str, pd.DataFrame] = {}
    for path in sorted(root.glob("*.parquet")):
        bars = pd.read_parquet(path)
        if len(bars) >= min_bars:
            out[path.stem] = bars
    return out


def liquidity_order(bars_map: dict[str, pd.DataFrame], top: int) -> list[str]:
    """按全历史中位日名义成交额降序取前 ``top`` 只。

    这是**池子的定义**（事后选池，本身带前视），不是交易规则：用它回答「只做头部标的
    会怎样」，但绝不能用它决定某天能不能下单——时变的额度判断见
    ``src/research/backtest/leverage_cap.py``（滞后成交额，无前视）。
    """
    median: dict[str, float] = {}
    for stem, bars in bars_map.items():
        dollar = bars["close"] * bars["volume"]
        median[stem] = float(dollar.median())
    return sorted(median, key=lambda stem: median[stem], reverse=True)[:top]
