"""
src/research/factors/variants.py
在 library.py 的基因子上派生的候选：平滑变体（Alpha101 算子）与量价交互因子。

两件东西，各自回答一个具体问题：
- `@decay{w}` / `@tsrank{w}`：对任意基因子做线性衰减 / 自身历史分位，回答
  「平滑能不能把平衡成本推过 taker 线」——本库的死因报告里「信号真、成本不允许」
  那一族（rev_3 永续 h=3 平衡成本 10.0bp）等的就是这类变换；
- `corr_pv_{w}`：收盘价与成交量的滚动相关，是本库原先完全缺席的「量价交互」家族
  第一条。方向取负号沿用 Alpha101 的 corr 族惯例（量价齐升越强越看空），是先验
  定向、不是按样本挑的。

派生因子不进 FACTOR_LIBRARY：主库的 8 个因子被 factor_suite/factor_sweep 的报告
与文档读数钉住，而 trials 账本按「库里有多少因子」计试验次数，往里加东西会静默
抬高全库的 Bonferroni/DSR 门槛。要用变体就显式点名（见 scripts/factor_smoothing.py）。
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from src.research.factors.library import (
    DIRECTION_SHORT,
    FactorSpec,
    resolve_factors,
)
from src.research.factors.operators import (
    CORRELATION_WINDOWS,
    DECAY_WINDOWS,
    TS_RANK_WINDOWS,
    correlation,
    decay_linear,
    ts_rank,
)

OPERATOR_DECAY: str = "decay_linear"
OPERATOR_TS_RANK: str = "ts_rank"

# CLI 写法 → 算子。窗口必须显式写在键里（decay10），因为 5 与 20 是两件事
OPERATOR_PREFIXES: tuple[tuple[str, str], ...] = (
    ("decay", OPERATOR_DECAY),
    ("tsrank", OPERATOR_TS_RANK),
)
SMOOTH_OPERATORS: tuple[str, ...] = tuple(
    f"decay{window}" for window in DECAY_WINDOWS
) + tuple(f"tsrank{window}" for window in TS_RANK_WINDOWS)

TRANSFORMS: dict[str, Callable[[pd.DataFrame, int], pd.DataFrame]] = {
    OPERATOR_DECAY: decay_linear,
    OPERATOR_TS_RANK: ts_rank,
}

FAMILY_INTERACTION: str = "量价交互"


def parse_operator(name: str) -> tuple[str, int]:
    """把 decay10/tsrank20 解析成（算子名，窗口）；写法错当场报错，不静默跳过。"""
    for prefix, kind in OPERATOR_PREFIXES:
        if not name.startswith(prefix):
            continue
        raw = name[len(prefix) :]
        if not raw.isdigit() or int(raw) < 2:
            raise ValueError(f"算子窗口必须是 ≥2 的整数：{name}")
        return kind, int(raw)
    raise KeyError(f"未知算子 {name}；可用：{', '.join(SMOOTH_OPERATORS)}")


def smoothing_spec(base: FactorSpec, operator: str) -> FactorSpec:
    """基因子做一次平滑变换后的派生 spec：方向继承，预热 = 基因子 + 窗口 − 1。

    算子对基因子取值的**符号**是可交换的（decay_linear 线性、ts_rank 单调），所以
    先变换原值、再由 `FactorSpec.values()` 统一乘方向，与先定向再变换等价。
    """
    kind, window = parse_operator(operator)
    transform = TRANSFORMS[kind]
    return FactorSpec(
        key=f"{base.key}@{operator}",
        label=f"{base.label} → {kind}({window})",
        family=base.family,
        direction=base.direction,
        warmup=base.warmup + window - 1,
        compute=lambda panel: transform(base.compute(panel), window),
    )


def resolve_smoothing(
    base_keys: str | None = None, operators: str | None = None
) -> list[FactorSpec]:
    """基因子 × 算子 的全部平滑变体；省略任一侧取全量（8 因子 × 5 算子 = 40 条）。"""
    bases = resolve_factors(base_keys)
    names = SMOOTH_OPERATORS if not operators else _split(operators)
    return [smoothing_spec(base, name) for base in bases for name in names]


def interaction_spec(window: int) -> FactorSpec:
    """量价交互：收盘价与成交量的滚动相关（值越大 = 量价齐升越强）。"""
    return FactorSpec(
        key=f"corr_pv_{window}",
        label=f"收盘价与成交量 {window} 根滚动相关",
        family=FAMILY_INTERACTION,
        direction=DIRECTION_SHORT,
        # 两个输入序列本身无前置 NaN，满 min_periods 落在下标 window−1
        warmup=window - 1,
        compute=lambda panel: correlation(panel.close, panel.volume, window),
    )


def resolve_interaction() -> list[FactorSpec]:
    """量价交互族（当前 2 条：窗口 10/20）。"""
    return [interaction_spec(window) for window in CORRELATION_WINDOWS]


def _split(value: str) -> list[str]:
    """逗号分隔的 CLI 字符串 → 去空项列表（保持顺序，不去重）。"""
    return [item.strip() for item in value.split(",") if item.strip()]
