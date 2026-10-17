"""carry 的基差维度：delta 中性组合的真实损益 = 资金费 − Δbasis。

`carry.py` 只算了资金费，把基差收敛当成 0。真实情况是：做多现货 + 做空等额永续的
净值变化 = 日资金费 − Δbasis，其中 `basis = (永续 − 现货) / 现货`。

推导：建仓时买现货（−spot₀）、卖永续（+perp₀），平仓时卖现货（+spot₁）、买永续
平仓（−perp₁），总现金流 = (spot₁ − spot₀) − (perp₁ − perp₀) ≈ −Δbasis。
所以**永续贴水**（通常伴随负费率）若在建仓后收敛，基差项是亏损而不是 0；反之升水
标的除费率外还能赚一笔收敛。累计上 telescope 求和即 `basis_first − basis_last`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd

from src.config.paths import (
    CRYPTO_ROOT,
    MARKET_SPOT,
    MARKET_SWAP,
    crypto_dataset_dir,
)
from src.research.funding.carry_ranking import load_rates
from src.utils.data_sources.funding import daily_rates

TIMEFRAME = "1d"
EXCHANGE = "okx"
FUNDING_EXCHANGE = "binance"
# |basis| 超过此值视为异常日（脱钩或数据错误，实测 MAGIC 曾出现 83%），剔除并计数
MAX_ABS_BASIS = 0.20


@dataclass(frozen=True)
class BasisRow:
    """一只标的的基差统计与调整后收益（全部为区间累计值，不是年化）。"""

    stem: str
    days: int
    basis_mean: float
    basis_std: float
    basis_first: float
    basis_last: float
    funding_sum: float
    basis_sum: float
    adjusted_sum: float
    dropped: int


def load_close(market: str, stem: str) -> pd.Series:
    """读一个市场的日线收盘价；缺文件返回空序列。"""
    path = (
        crypto_dataset_dir(EXCHANGE, market, TIMEFRAME, root=CRYPTO_ROOT)
        / f"{stem}.parquet"
    )
    if not path.exists():
        return pd.Series(dtype=float)
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return cast(pd.Series, frame.sort_index()["close"])


def load_basis(stem: str) -> tuple[pd.Series, int]:
    """永续与现货的同期基差序列，返回 (序列, 剔除的异常日数)。

    最后一根通常是当日未收盘 K 线，两边快照时点不同，直接去掉。
    """
    perp = load_close(MARKET_SWAP, stem)
    spot = load_close(MARKET_SPOT, stem)
    if perp.empty or spot.empty:
        return pd.Series(dtype=float), 0
    shared = perp.index.intersection(spot.index)
    if len(shared) < 2:
        return pd.Series(dtype=float), 0
    basis = (perp.loc[shared] - spot.loc[shared]) / spot.loc[shared]
    basis = cast(pd.Series, basis.iloc[:-1])
    bad = basis.abs() > MAX_ABS_BASIS
    return cast(pd.Series, basis[~bad]), int(bad.sum())


def basis_row(stem: str, funding_days: int) -> BasisRow | None:
    basis, dropped = load_basis(stem)
    if len(basis) < 2:
        return None
    funding = daily_rates(load_rates(FUNDING_EXCHANGE, stem))
    funding = cast(pd.Series, funding.iloc[-funding_days:]).dropna()
    # 基差序列对齐到资金费的同一批日子：Δbasis 只在两边都有时才实现
    shared = funding.index.intersection(basis.index)
    if len(shared) < 2:
        return None
    funding_aligned = cast(pd.Series, funding.loc[shared])
    basis_aligned = cast(pd.Series, basis.loc[shared])
    delta = basis_aligned.diff()
    adjusted = funding_aligned - delta
    adjusted = cast(pd.Series, adjusted.dropna())
    return BasisRow(
        stem=stem,
        days=len(funding_aligned),
        basis_mean=float(basis_aligned.mean()),
        basis_std=float(basis_aligned.std()),
        basis_first=float(basis_aligned.iloc[0]),
        basis_last=float(basis_aligned.iloc[-1]),
        funding_sum=float(funding_aligned.sum()),
        basis_sum=float(-delta.sum()),
        adjusted_sum=float(adjusted.sum()),
        dropped=dropped,
    )


def basis_rows(stems: list[str], funding_days: int) -> list[BasisRow]:
    """批量算基差贡献；缺现货或数据不足的标的直接跳过。"""
    rows: list[BasisRow] = []
    for stem in stems:
        row = basis_row(stem, funding_days)
        if row is not None:
            rows.append(row)
    return rows
