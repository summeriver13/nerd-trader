"""
src/research/paper.py
前向纸面账本：把每日目标权重与已实现净值追加成带哈希链的 JSONL。

为什么要有这一层：趋势线的全部读数都来自同一段历史的回看（`docs/trend/README.md`
把「零前向验证」列在仍未解决的限制里），而统计门槛（DSR/PBO/配对差）都已用尽。
前向记录是唯一还能继续增加真样本的动作，所以它必须**只增不改**：每行带
`hash = sha256(prev_hash + 行内容)`，事后改动任何一行都会断链（`verify_ledger`），
使「这条记录是当时写下的」可被第三方复核。

三条口径约束：

- **不下单**：输入是自持日线缓存的收盘价，输出是「今天收盘后该持有什么」加当日净值。
  下单属执行层，与这一层解耦。
- **与回测同源**：信号用 `tsmom_signal`（多速度突破 × 逐标的波动率缩放），权重除以
  当日有数据的标的数（等权 1/N_t）、持仓滞后一根、成本取现货 taker + 滑点——与
  `scripts/tsmom_study.py` 的 `tsmom_ms` 臂同一套原语，故 `--check` 能让纸面读数与
  研究读数直接对照。
- **起点不回填**：首次运行时起点取最新一根 K 线（`PaperSpec.start=None`），账本里
  第一行就是「从今天开始持仓」——回填历史等于又做一次回看。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd

from src.config.paths import RESULTS_ROOT
from src.core.costs import SPOT_FEES
from src.research.tsmom import (
    BARS_PER_YEAR,
    DEFAULT_ENTRY_PERIODS,
    DEFAULT_VOL_LOOKBACK,
    DEFAULT_VOL_TARGET,
    TsmomOptions,
    tsmom_signal,
)

# 账本落研究产物区（可重建），与只增不改的数据湖 raw 分开
LEDGER_DIR: Final[Path] = RESULTS_ROOT / "live"
LEDGER_NAME: Final[str] = "tsmom_paper.jsonl"
# 哈希链的起点标记，行内 prev_hash 取它表示这是第一行
GENESIS: Final[str] = "genesis"
# 现货基准：账本的对照标的（研究侧同口径）
BTC_STEM: Final[str] = "BTC-USDT"
# 单边成本 = 现货 taker 费率 + 市价单滑点，与 SPOT_FEES 同源
COST_RATE: Final[float] = SPOT_FEES.taker + SPOT_FEES.slippage
# 账本行的日期字段（追加按它去重，链按它排序）
DATE_FIELD: Final[str] = "date"


@dataclass(frozen=True)
class PaperSpec:
    """前向账本的固定口径：``start=None`` 表示从最新一根 K 线起（真前向，不回填）。"""

    start: pd.Timestamp | None = None
    vol_target: float | None = DEFAULT_VOL_TARGET
    vol_lookback: int = DEFAULT_VOL_LOOKBACK
    entry_periods: tuple[int, ...] = DEFAULT_ENTRY_PERIODS

    def options(self) -> TsmomOptions:
        """转成信号层选项，保证与 `tsmom_study.py` 的主口径臂同参。"""
        return TsmomOptions(
            entry_periods=self.entry_periods,
            vol_target=self.vol_target,
            vol_lookback=self.vol_lookback,
        )


def signal_panel(bars_map: dict[str, pd.DataFrame], spec: PaperSpec) -> pd.DataFrame:
    """逐标的原始信号（多速度突破 × 波动率缩放，未除 N）。"""
    options = spec.options()
    frames: dict[str, pd.Series] = {
        stem: tsmom_signal(bars, options) for stem, bars in bars_map.items()
    }
    return pd.DataFrame(frames)


def scale_signals(signals: pd.DataFrame, counts: pd.Series) -> pd.DataFrame:
    """信号 ÷ N_t：**信号当日**的口径，即收盘后要调的下一根目标（实盘视角）。"""
    scaled = signals.div(counts.replace(0.0, float("nan")), axis=0)
    return scaled.fillna(0.0)


def positions(signals: pd.DataFrame, counts: pd.Series) -> pd.DataFrame:
    """持仓：信号滞后一根、再除**持仓当日**的 N_t（与引擎的 `mean(axis=1)` 同口径）。

    两个时点都不能错：除法必须在 `shift` 之后（先除后移等于拿前一日的标的数当分母，
    2026-10-10 只有 8 只标的出数，拿 205 当分母会把当日收益少算 0.89%）；无当根数据的
    格子必须留 NaN 而不是 0（置 0 会被当成清仓，197 只滞后标的凭空产生 4.8 的换手、
    白白扣掉 72bp）。
    """
    lagged = signals.shift(1)
    scaled = lagged.div(counts.replace(0.0, float("nan")), axis=0)
    return scaled.where(signals.notna())


def coverage_ok(returns: pd.DataFrame, *, min_ratio: float = 0.8, window: int = 30) -> bool:
    """最新一根的标的数是否接近近期常态（数据滞后的日子不能记账）。

    只查最新一根：N_t 就是等权权重本身，若当日只有零星几只标的出数（本地缓存没
    刷完、交易所接口未回），持仓会被放大到远超常态。
    """
    counts = returns.notna().sum(axis=1).astype(float)
    if len(counts) < 2:
        return False
    reference = float(counts.shift(1).rolling(window, min_periods=1).median().iloc[-1])
    return reference > 0 and float(counts.iloc[-1]) >= min_ratio * reference


def _align_tz(index: pd.Index, boundary: Any) -> pd.Timestamp:
    """把边界日期对齐到索引的时区（账本里存的是无时区的日期字符串）。

    K 线索引是 tz-aware（UTC），而账本行的 ``date`` 是 ``YYYY-MM-DD``；直接比较会抛
    ``Invalid comparison between dtype=datetime64[ns, UTC] and Timestamp``。
    """
    stamp = pd.Timestamp(boundary)
    tz = getattr(index, "tz", None)
    if tz is not None and stamp.tzinfo is None:
        return cast(pd.Timestamp, stamp.tz_localize(tz))
    if tz is None and stamp.tzinfo is not None:
        return cast(pd.Timestamp, stamp.tz_localize(None))
    return cast(pd.Timestamp, stamp)


def nav_panel(
    held: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    btc_returns: pd.Series | None = None,
    start: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """按持仓算纸面净值、换手与暴露（逐根收益 = Σ pos×ret − 换手×费率）。

    ``start`` 之后的第一行**重新起算净值**（首行 = 1）：起点之前没有仓位，所以首行
    不计收益、只计建仓成本（否则等于把上一根的涨跌回填进前向记录）。成本的复利写法
    与引擎「按初始资本扣」略有差异（<1bp/次），不影响对照。
    """
    frame = pd.DataFrame(
        {
            "gross_ret": (held * returns).sum(axis=1).astype(float),
            "turnover": held.diff().abs().sum(axis=1).astype(float),
            "exposure": held.abs().sum(axis=1).astype(float),
            "n_symbols": returns.notna().sum(axis=1).astype(int),
        }
    )
    if btc_returns is not None:
        frame["btc_ret"] = btc_returns.reindex(frame.index).fillna(0.0)
    if start is not None:
        frame = frame.loc[frame.index >= _align_tz(frame.index, start)].copy()
        if len(frame) > 0:
            first = frame.index[0]
            frame.loc[first, "turnover"] = float(held.loc[first].abs().sum())
            frame.loc[first, "gross_ret"] = 0.0
            if btc_returns is not None:
                frame.loc[first, "btc_ret"] = 0.0
    elif len(frame) > 0:
        frame.loc[frame.index[0], "turnover"] = 0.0
    frame["cost_ret"] = frame["turnover"] * COST_RATE
    frame["net_ret"] = frame["gross_ret"] - frame["cost_ret"]
    frame["nav"] = (1.0 + frame["net_ret"]).cumprod()
    frame["btc_nav"] = (
        (1.0 + frame["btc_ret"]).cumprod()
        if btc_returns is not None
        else float("nan")
    )
    return frame


def _nonzero(series: pd.Series, digits: int = 6) -> dict[str, float]:
    """只留非零项并定精度：既避开几百个 0，也避开 NaN（NaN ≠ NaN 会被当成数据修订）。"""
    out: dict[str, float] = {}
    for column, value in series.items():
        number = float(value)
        if number != 0.0 and not math.isnan(number):
            out[str(column)] = round(number, digits)
    return out


def ledger_rows(
    panel: pd.DataFrame,
    held: pd.DataFrame,
    target: pd.DataFrame,
) -> list[dict[str, Any]]:
    """把净值面板与仓位面板转成账本行：当日读数 + 当前持仓 + 收盘后的目标权重。

    持仓明细全量写入（只留非零项，约几十条），不做截断——截断会让不同阈值的两次
    运行互报「数据修订」，把真信号淹在假告警里。
    """
    rows: list[dict[str, Any]] = []
    for stamp in panel.index:
        row = panel.loc[stamp]
        moment = cast(pd.Timestamp, pd.Timestamp(stamp))
        want = target.loc[stamp].sort_values(ascending=False)
        current = held.loc[stamp].sort_values(ascending=False)
        rows.append(
            {
                DATE_FIELD: moment.strftime("%Y-%m-%d"),
                "nav": round(float(row["nav"]), 6),
                "btc_nav": round(float(row["btc_nav"]), 6),
                "net_ret": round(float(row["net_ret"]), 6),
                "gross_ret": round(float(row["gross_ret"]), 6),
                "cost_ret": round(float(row["cost_ret"]), 6),
                "turnover": round(float(row["turnover"]), 6),
                "exposure": round(float(row["exposure"]), 6),
                "n_symbols": int(row["n_symbols"]),
                "hold": _nonzero(current),
                "target": _nonzero(want),
                "n_hold": int(current.ne(0.0).sum()),
                "n_target": int(want.ne(0.0).sum()),
            }
        )
    return rows


def _digest(payload: dict[str, Any]) -> str:
    """行摘要：字段排序后序列化再 sha256，保证同内容必得同哈希。"""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def chain_rows(rows: list[dict[str, Any]], prev_hash: str = GENESIS) -> list[dict[str, Any]]:
    """给行序列串上哈希链：每行写入 prev_hash 与自身 hash。"""
    out: list[dict[str, Any]] = []
    for row in rows:
        payload = {**row, "prev_hash": prev_hash}
        payload["hash"] = _digest(payload)
        out.append(payload)
        prev_hash = payload["hash"]
    return out


def read_ledger(path: Path) -> list[dict[str, Any]]:
    """读账本（行序即追加序）；文件不存在时返回空表。"""
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_ledger(rows: list[dict[str, Any]]) -> str | None:
    """校验哈希链；返回首个断链的日期，全链完好返回 None。"""
    prev = GENESIS
    for row in rows:
        payload = {key: value for key, value in row.items() if key != "hash"}
        if payload.get("prev_hash") != prev or row.get("hash") != _digest(payload):
            return str(row.get(DATE_FIELD))
        prev = str(row["hash"])
    return None


def append_ledger(
    path: Path, rows: list[dict[str, Any]]
) -> tuple[int, list[str]]:
    """追加新日期行，返回（新增行数，内容不一致的日期）。

    已存在的日期绝不覆盖：同一日期重算结果不同意味着数据被修订过，这种情况必须
    摆到明面上（冲突日期交调用方报警），而不是悄悄改掉当时的记录。
    """
    existing = read_ledger(path)
    stored = {str(row[DATE_FIELD]): row for row in existing}
    fresh = [row for row in rows if str(row[DATE_FIELD]) not in stored]
    conflicts = [
        str(row[DATE_FIELD])
        for row in rows
        if str(row[DATE_FIELD]) in stored
        and _canonical(row) != _canonical(stored[str(row[DATE_FIELD])])
    ]
    if not fresh:
        return 0, conflicts
    prev = str(existing[-1]["hash"]) if existing else GENESIS
    chained = chain_rows(fresh, prev)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in chained:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(chained), conflicts


def _comparable(row: dict[str, Any]) -> dict[str, Any]:
    """去掉链字段后的可比较内容（哈希与 prev_hash 由 append 时写入，不参与比较）。"""
    return {key: value for key, value in row.items() if key not in {"hash", "prev_hash"}}


def _canonical(row: dict[str, Any]) -> str:
    """比较用的规范形式：字段排序序列化，使 NaN 这类自身不等的值也能逐字对比。"""
    return json.dumps(_comparable(row), sort_keys=True, ensure_ascii=False)


def ledger_panel(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """把账本行还原成面板（读侧）：净值、BTC 对照、暴露与换手。"""
    if not rows:
        return pd.DataFrame(
            {
                "nav": pd.Series(dtype=float),
                "btc_nav": pd.Series(dtype=float),
                "exposure": pd.Series(dtype=float),
                "turnover": pd.Series(dtype=float),
            }
        )
    index = pd.to_datetime([str(row[DATE_FIELD]) for row in rows])
    return pd.DataFrame(
        {
            "nav": [float(row["nav"]) for row in rows],
            "btc_nav": [float(row["btc_nav"]) for row in rows],
            "exposure": [float(row["exposure"]) for row in rows],
            "turnover": [float(row["turnover"]) for row in rows],
        },
        index=index,
    )


def forward_stats(panel: pd.DataFrame) -> dict[str, float]:
    """前向记录的即时读数：期数、年化、夏普、回撤，以及对 BTC 的同期对照。"""
    if len(panel) < 2:
        return {"bars": 0.0}
    nav = panel["nav"].astype(float)
    btc = panel["btc_nav"].astype(float)
    net = nav.pct_change().dropna()
    bars = float(len(net))
    growth = float(nav.iloc[-1]) ** (BARS_PER_YEAR / bars) - 1.0
    btc_growth = float(btc.iloc[-1]) ** (BARS_PER_YEAR / bars) - 1.0
    std = float(net.std())
    return {
        "bars": bars,
        "nav": float(nav.iloc[-1]),
        "total_ret": float(nav.iloc[-1]) - 1.0,
        "annual_ret": growth,
        "sharpe": float(net.mean()) / std * (BARS_PER_YEAR**0.5) if std > 0 else 0.0,
        "max_dd": float((nav / nav.cummax() - 1.0).min()),
        "btc_total_ret": float(btc.iloc[-1]) - 1.0,
        "btc_annual_ret": btc_growth,
        "exposure": float(panel["exposure"].astype(float).mean()),
    }
