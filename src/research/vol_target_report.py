"""
src/research/vol_target_report.py
波动率目标研究的报告层：全网格、等暴露对照、分段稳定性、机制检验。

四张表各回答一个问题：

- **grid**：150 个参数组合各自的读数，以及每个组合减去「等平均暴露对照」后的
  增量。核心列是 `sharpe_gain`——它是波动率目标的真实贡献，不是「仓位变小」的
  贡献（对照已经把后者扣掉了）。
- **segments**：把全样本四等分，看 `sharpe_gain` 是否每段都为正。单段为正可能
  是运气，四段全正才说明是机制。
- **mechanism**：波动率目标成立要两个前提——波动率可预测（自相关）且高波动期
  未来收益更低。两个都不成立的话，前面两张表再好看也只是曲线拟合。
- **cost**：换手成本敏感性。波动率目标每根都在动仓位，这是它相对满仓多出来的
  成本项。

判读顺序：先看 mechanism（前提成不成立），再看 grid 的 `sharpe_gain` 分布，
最后看 segments 的一致性。前提不成立就直接否决，不必往下看。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from src.analytics.metrics import ANNUAL_DAYS
from src.config.paths import RESULTS_ROOT
from src.research.factors.portfolio import leg_stats
from src.research.vol_target import (
    VolTargetRun,
    VolTargetSpec,
    matched_control,
    run_baseline,
    run_spec,
)

OUTPUT_DIR: Final[Path] = RESULTS_ROOT / "vol_target"
DEFAULT_SEGMENTS: Final[int] = 4
DEFAULT_COST_LEVELS: Final[tuple[float, ...]] = (0.0, 0.0005, 0.001, 0.002)
DEFAULT_COST_SPEC: Final[VolTargetSpec] = VolTargetSpec(0.40, 30, 7, 1.0)
VOL_LOOKBACK: Final[int] = 30
VOL_LAGS: Final[tuple[int, ...]] = (1, 2, 3, 5, 10)
VOL_HORIZONS: Final[tuple[int, ...]] = (1, 7, 30)
VOL_QUANTILES: Final[int] = 5

# 配对块自举：块长 30 天覆盖已实现波动率的记忆长度，策略与对照用同一组下标重采样
# （配对），差值的方差因此远小于各自单独估计。
BOOTSTRAP_ROUNDS: Final[int] = 2000
BOOTSTRAP_BLOCK: Final[int] = 30
BOOTSTRAP_SEED: Final[int] = 20261011
DEFAULT_BOOTSTRAP_SPECS: Final[tuple[VolTargetSpec, ...]] = (
    VolTargetSpec(0.40, 10, 7, 1.0),
    VolTargetSpec(0.40, 30, 7, 1.0),
    VolTargetSpec(0.40, 90, 7, 1.0),
)


@dataclass(frozen=True)
class VolTargetStudy:
    """一次研究的全部读数。"""

    universe: pd.DataFrame
    grid: pd.DataFrame
    segments: pd.DataFrame
    mechanism: pd.DataFrame
    cost: pd.DataFrame
    bootstrap: pd.DataFrame


def _run_frame(run: VolTargetRun, prefix: str) -> dict[str, Any]:
    """把一次读数拍平成带前缀的列。"""
    return {
        f"{prefix}annual": run.stats.annual_compounded,
        f"{prefix}sharpe": run.stats.sharpe,
        f"{prefix}vol": run.stats.volatility,
        f"{prefix}dd": run.stats.max_drawdown,
    }


def _spec_frame(bars: pd.DataFrame, spec: VolTargetSpec) -> dict[str, Any]:
    """一个参数组合：自身读数 + 等平均暴露对照 + 增量列。"""
    run = run_spec(bars, spec)
    control = matched_control(bars, run)
    row: dict[str, Any] = {
        "label": run.label,
        "target_vol": spec.target_vol,
        "lookback": spec.lookback,
        "rebalance_every": spec.rebalance_every,
        "max_exposure": spec.max_exposure,
        "avg_exposure": run.avg_exposure,
        "turnover": run.turnover,
        "cost": run.cost,
        "ctrl_exposure": control.avg_exposure,
    }
    row |= _run_frame(run, "")
    row |= _run_frame(control, "ctrl_")
    # 增量列：正数代表波动率目标比「同平均暴露的固定仓位」更好
    row["sharpe_gain"] = row["sharpe"] - row["ctrl_sharpe"]
    row["annual_gain"] = row["annual"] - row["ctrl_annual"]
    row["dd_gain"] = row["dd"] - row["ctrl_dd"]
    row["vol_gain"] = row["ctrl_vol"] - row["vol"]
    return row


def _segment_edges(n_bars: int, n_segments: int) -> list[tuple[int, int]]:
    """把样本按根数等分，返回每段的 [起, 止) 下标。"""
    edges = np.linspace(0, n_bars, n_segments + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_segments)]


def _segment_rows(
    bars: pd.DataFrame, spec: VolTargetSpec, n_segments: int
) -> list[dict[str, Any]]:
    """一个参数组合在每段上的 sharpe_gain（对照用全样本平均暴露，保持单一策略）。"""
    run = run_spec(bars, spec)
    control = matched_control(bars, run)
    rows: list[dict[str, Any]] = []
    for i, (start, stop) in enumerate(_segment_edges(bars.index.size, n_segments)):
        target = leg_stats(run.net_ret.iloc[start:stop], ANNUAL_DAYS)
        base = leg_stats(control.net_ret.iloc[start:stop], ANNUAL_DAYS)
        rows.append(
            {
                "label": run.label,
                "rebalance_every": spec.rebalance_every,
                "segment": i,
                "start": bars.index[start],
                "stop": bars.index[stop - 1],
                "sharpe": target.sharpe,
                "ctrl_sharpe": base.sharpe,
                "sharpe_gain": target.sharpe - base.sharpe,
            }
        )
    return rows


def _mechanism_rows(close: pd.Series) -> list[dict[str, Any]]:
    """两个前提：波动率可预测、且高波动期未来收益更低。"""
    realized = close.pct_change().rolling(VOL_LOOKBACK).std() * np.sqrt(ANNUAL_DAYS)
    rows: list[dict[str, Any]] = [
        {"check": f"autocorr_lag{lag}", "value": float(realized.autocorr(lag))}
        for lag in VOL_LAGS
    ]
    bucket = pd.qcut(realized, VOL_QUANTILES, labels=False, duplicates="drop")
    for horizon in VOL_HORIZONS:
        forward = close.pct_change(horizon).shift(-horizon)
        grouped = forward.groupby(bucket)
        # 转成 ndarray 再取值：pandas stub 把 Series[label] 标成含 DataFrame 的联合类型
        means = cast(pd.Series, grouped.mean()).to_numpy(dtype=float)
        counts = cast(pd.Series, grouped.count()).to_numpy(dtype=int)
        rows.extend(
            {
                "check": f"fwd{horizon}_q{level + 1}",
                "value": float(means[level]),
                "observations": int(counts[level]),
            }
            for level in range(means.size)
        )
    return rows


def _cost_rows(bars: pd.DataFrame, spec: VolTargetSpec) -> list[dict[str, Any]]:
    """换手成本敏感性：同一参数组合在几档单边费率下的读数。"""
    rows: list[dict[str, Any]] = []
    for fee in DEFAULT_COST_LEVELS:
        run = run_spec(bars, spec, fee_rate=fee, slippage=0.0)
        control = matched_control(bars, run, fee_rate=fee, slippage=0.0)
        rows.append(
            {
                "fee_one_way": fee,
                "annual": run.stats.annual_compounded,
                "sharpe": run.stats.sharpe,
                "ctrl_annual": control.stats.annual_compounded,
                "ctrl_sharpe": control.stats.sharpe,
                "sharpe_gain": run.stats.sharpe - control.stats.sharpe,
                "cost": run.cost,
            }
        )
    return rows


def _block_sample(n_bars: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """环形块自举下标：连续 block 根一起抽，保留收益与波动的自相关结构。"""
    n_blocks = int(np.ceil(n_bars / block))
    starts = rng.integers(0, n_bars, size=n_blocks)
    offsets = np.arange(block)
    return ((starts[:, None] + offsets) % n_bars).ravel()[:n_bars]


def _bootstrap_row(
    bars: pd.DataFrame, spec: VolTargetSpec, *, block: int, seed: int
) -> dict[str, Any]:
    """一个参数组合的夏普增量置信区间（配对块自举）。

    夏普的解析标准误在年化波动 60% 的标的上大到无法判断 0.05 的差异；配对自举把
    策略与对照放在同一批重采样日子里，把「市场本身」的波动消掉，剩下的才是波动率
    目标的贡献。
    """
    run = run_spec(bars, spec)
    control = matched_control(bars, run)
    target = run.net_ret.to_numpy(dtype=float)
    base = control.net_ret.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    gains = np.empty(BOOTSTRAP_ROUNDS, dtype=float)
    for i in range(BOOTSTRAP_ROUNDS):
        index = _block_sample(target.size, block, rng)
        gains[i] = (
            leg_stats(pd.Series(target[index]), ANNUAL_DAYS).sharpe
            - leg_stats(pd.Series(base[index]), ANNUAL_DAYS).sharpe
        )
    observed = run.stats.sharpe - control.stats.sharpe
    return {
        "label": run.label,
        "observed": observed,
        "boot_mean": float(gains.mean()),
        "boot_std": float(gains.std(ddof=1)),
        "ci_low": float(np.percentile(gains, 2.5)),
        "ci_high": float(np.percentile(gains, 97.5)),
        "p_positive": float((gains > 0).mean()),
    }


def build_study(
    bars: pd.DataFrame,
    specs: list[VolTargetSpec],
    *,
    n_segments: int = DEFAULT_SEGMENTS,
    bootstrap_specs: tuple[VolTargetSpec, ...] = DEFAULT_BOOTSTRAP_SPECS,
) -> VolTargetStudy:
    """跑完全网格与六张表。"""
    close = cast(pd.Series, bars["close"]).astype(float)
    baseline = run_baseline(bars)
    universe = pd.DataFrame(
        [
            {
                "symbol": "BTC/USDT",
                "bars": bars.index.size,
                "start": bars.index[0],
                "stop": bars.index[-1],
                "years": bars.index.size / ANNUAL_DAYS,
                "specs": len(specs),
                **_run_frame(baseline, "base_"),
            }
        ]
    )
    segment_rows: list[dict[str, Any]] = []
    for spec in specs:
        segment_rows.extend(_segment_rows(bars, spec, n_segments))
    return VolTargetStudy(
        universe=universe,
        grid=pd.DataFrame([_spec_frame(bars, spec) for spec in specs]),
        segments=pd.DataFrame(segment_rows),
        mechanism=pd.DataFrame(_mechanism_rows(close)),
        cost=pd.DataFrame(_cost_rows(bars, DEFAULT_COST_SPEC)),
        bootstrap=pd.DataFrame(
            [
                _bootstrap_row(
                    bars, spec, block=BOOTSTRAP_BLOCK, seed=BOOTSTRAP_SEED + i
                )
                for i, spec in enumerate(bootstrap_specs)
            ]
        ),
    )


def save_outputs(study: VolTargetStudy) -> list[Path]:
    """落盘五份 parquet，文件名与研究主题一致。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, frame in (
        ("vol_target_universe", study.universe),
        ("vol_target_grid", study.grid),
        ("vol_target_segments", study.segments),
        ("vol_target_mechanism", study.mechanism),
        ("vol_target_cost", study.cost),
        ("vol_target_bootstrap", study.bootstrap),
    ):
        path = OUTPUT_DIR / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        written.append(path)
    return written
