"""
三重障碍标签（de Prado Ch 3）。

固定期限标签（「h 天后的收益符号」）有两个问题：一是它假设你**只**在 h 天后平仓，
而真实交易会在止盈/止损/超时三条路径上先到先出；二是它的标签长度恒为 h，
使唯一性权重退化成 1/h（与抽稀数学等价），Ch 4 因此无法独立实现。

三重障碍把标签长度变成**可变**的：每条路径在自己的触及时刻结束。这带来两个后果——
标签更接近可实现的交易，以及唯一性权重终于有内容（见 overlap.py）。

三条口径（改代码前先读）：

1. **障碍宽度按波动率标度，不是固定百分比。** 同一个 1% 对 BTC 和 PEPE 是完全不同的
   事件；固定百分比会让高波动标的几乎总是先触上障碍。
2. **首次触及决定标签，标签是触及时刻的收益符号**（不是「哪个障碍被触及」）。
   这是作者 getBins 的口径：bin = sign(ret)，所以碰到垂直障碍时可能是 0。
   同一根上两条障碍都被穿越时取上障碍——与作者 min(axis=1) 的任意取舍一致
   （同一时刻两个时间戳相等，取谁都对）。
3. **t1 返回整数位置而非时间戳。** 唯一性权重与 purge 都按根序号算，
   返回时间戳还要再折回来一次。

参考：López de Prado (2018) Advances in Financial Machine Learning, Ch 3。

---

同一章还给了另一套标签：**趋势扫描**（`trend_scan`）。它不预设持有期，而是在前向窗口
内枚举所有端点、取「价格对时间」回归斜率 |t| 最大的那一个，标签就是那个斜率的符号。
与三重障碍的分工：三重障碍回答「这一注先碰到哪条障碍」，趋势扫描回答「未来哪一段
时间里趋势最显著」——前者给退出规则，后者给方向。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_VOL_SPAN = 100  # 作者 getDailyVol 的默认 span
MIN_VOL_OBSERVATIONS = 20  # 波动率可用前的最少根数

BARRIER_COLUMNS = ("label", "t1", "ret", "upper", "lower", "vol")

TREND_COLUMNS = ("label", "t1", "t_value", "span", "ret")

MIN_TREND_WINDOW = 3
"""趋势扫描的回归窗口下限：斜率 t 值的自由度是 j−2，j=3 才有一个自由度。"""

TREND_THRESHOLD_MIN_PERIODS = 20
"""动态阈值可用前的最少历史观测数（不足时不分档，照给标签）。"""


@dataclass(frozen=True)
class TripleBarrierOptions:
    """三重障碍参数。side 在调用处给，见 triple_barrier。"""

    profit_take: float = 1.0  # 上障碍 = +profit_take × 波动率
    stop_loss: float = 1.0  # 下障碍 = −stop_loss × 波动率
    vertical: int = 20  # 垂直障碍：最多持有多少根
    vol_span: float = DEFAULT_VOL_SPAN

    def validate(self) -> None:
        if self.profit_take <= 0 or self.stop_loss <= 0:
            raise ValueError("障碍宽度必须为正")
        if self.vertical < 1:
            raise ValueError("垂直障碍至少 1 根")
        if self.vol_span <= 1:
            raise ValueError("波动率 span 至少 2")


def ewma_volatility(close: pd.Series, span: float = DEFAULT_VOL_SPAN) -> pd.Series:
    """EWMA 波动率（作者 getDailyVol：收益率的标准差，非年化）。"""
    return close.pct_change().ewm(span=span).std()


def triple_barrier(
    close: pd.Series,
    options: TripleBarrierOptions,
    side: int = 1,
) -> pd.DataFrame:
    """逐根出发的三重障碍标签，返回索引与 close 相同的表。

    side=1 看多、side=-1 看空：路径收益统一乘 side 后再比障碍，
    所以做空时「上障碍」就是价格下跌那一侧（作者的 sides 口径）。

    列：label（触及时刻收益的符号）、t1（触及位置，整数根序号）、
    ret（触及时刻相对起点的收益，已按 side 定向）、upper/lower（障碍价格）、
    vol（起点处的波动率）。

    末尾 vertical 根不出事件——它们凑不满垂直窗口。
    """
    options.validate()
    if side not in (1, -1):
        raise ValueError("side 只能是 1 或 -1")
    size = len(close)
    empty = pd.DataFrame(index=close.index, columns=pd.Index(BARRIER_COLUMNS))
    if size <= options.vertical + MIN_VOL_OBSERVATIONS:
        return empty

    values = close.to_numpy(dtype=float)
    vol = ewma_volatility(close, options.vol_span).to_numpy(dtype=float)
    starts = np.arange(size - options.vertical)
    widths = vol[starts]
    usable = np.isfinite(widths) & (widths > 0)
    starts, widths = starts[usable], widths[usable]
    if starts.size == 0:
        return empty

    # 路径矩阵：第 i 行是第 i 个事件的 vertical 根前向收益（已按 side 定向）
    steps = np.arange(1, options.vertical + 1)
    window = starts[:, None] + steps[None, :]
    path = (values[window] / values[starts][:, None] - 1.0) * side
    # 首次触及：argmax 在全 False 时返回 0，必须用哨兵换掉
    hit_up = path > (options.profit_take * widths)[:, None]
    hit_down = path < (-options.stop_loss * widths)[:, None]
    last = options.vertical - 1
    first_up = np.where(hit_up.any(axis=1), hit_up.argmax(axis=1), last)
    first_down = np.where(hit_down.any(axis=1), hit_down.argmax(axis=1), last)
    column = np.minimum(first_up, first_down)
    rows = np.arange(starts.size)

    upper = values[starts] * (1.0 + options.profit_take * widths)
    lower = values[starts] * (1.0 - options.stop_loss * widths)
    if side == -1:  # 做空时价格方向反过来
        upper, lower = lower, upper

    return pd.DataFrame(
        {
            "label": np.sign(path[rows, column]).astype(int),
            "t1": starts + column + 1,
            "ret": path[rows, column],
            "upper": upper,
            "lower": lower,
            "vol": widths,
        },
        index=close.index.take(starts),
    )


@dataclass(frozen=True)
class TrendScanOptions:
    """趋势扫描参数（de Prado Ch 3.6）。"""

    span: int = 20  # 前向扫描的最大窗口（根）
    min_window: int = MIN_TREND_WINDOW  # 最小的回归窗口（根）
    threshold: float = 0.0  # 动态阈值分位：0 = 不设阈值（原文 Snippet 3.4 口径）

    def validate(self) -> None:
        if self.min_window < MIN_TREND_WINDOW:
            raise ValueError(
                f"回归窗口至少 {MIN_TREND_WINDOW} 根（斜率 t 值的自由度 j−2）"
            )
        if self.span < self.min_window:
            raise ValueError("扫描窗口不能小于最小回归窗口")
        if not 0.0 <= self.threshold < 1.0:
            raise ValueError("动态阈值必须是 [0, 1) 里的分位")


def slope_t_values(windows: np.ndarray) -> np.ndarray:
    """一批等长窗口里「价格对时间」回归的斜率 t 值。

    简单线性回归的斜率 t 值等于相关系数 r 的单调变换 `r·sqrt((j−2)/(1−r²))`，
    所以整批窗口可以一次算完——面板上 196 只 × 2475 天不必逐窗口调 OLS。

    方差为 0 的窗口（价格没动）返回 NaN：那种窗口没有趋势可言，不能当成「|t| = 0
    的平趋势」参与取最大，否则价格静止的标的会一直拿到标签。
    """
    length = windows.shape[1]
    ramp = np.arange(length, dtype=float)
    centered_x = ramp - ramp.mean()
    centered_y = windows - windows.mean(axis=1, keepdims=True)
    variance_x = float(centered_x @ centered_x)
    variance_y = np.einsum("ij,ij->i", centered_y, centered_y)
    covariance = centered_y @ centered_x
    denominator = np.sqrt(variance_x * variance_y)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.where(denominator > 0.0, covariance / denominator, np.nan)
        # r = ±1（完美线性）时 t 发散，夹到开区间内侧：t 大而有限
        clipped = np.clip(correlation, -1.0 + 1e-12, 1.0 - 1e-12)
        return clipped * np.sqrt((length - 2.0) / (1.0 - clipped**2))


def _threshold_weak(
    index: pd.Index, magnitude: np.ndarray, threshold: float
) -> np.ndarray:
    """动态阈值：|t| 低于**过去** |t| 该分位的观测（True = 记 0 不下注）。

    阈值取扩展分位而不是同一事件日窗口内的分位：阈值不许看未来，否则标签上会
    多出第二条前向通道。`TREND_THRESHOLD_MIN_PERIODS` 之前分位是 NaN，不分档。
    """
    history = (
        pd.Series(magnitude, index=index)
        .expanding(min_periods=TREND_THRESHOLD_MIN_PERIODS)
        .quantile(threshold)
        .to_numpy(dtype=float)
    )
    return np.isfinite(history) & (magnitude < history)


def _trend_frame(
    close: pd.Series,
    rows: np.ndarray,
    spans: np.ndarray,
    t_values: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """把「每个起点选中的那个端点」铺回与 close 同索引的表。"""
    size = len(close)
    values = close.to_numpy(dtype=float)
    endpoints = (rows + spans - 1).astype(float)
    columns: dict[str, np.ndarray] = {}
    for name, filled in (
        ("label", labels),
        ("t1", endpoints),
        ("t_value", t_values),
        ("span", spans.astype(float)),
        ("ret", values[endpoints.astype(int)] / values[rows] - 1.0),
    ):
        column = np.full(size, np.nan)
        column[rows] = filled
        columns[name] = column
    return pd.DataFrame(columns, index=close.index)


def trend_scan(close: pd.Series, options: TrendScanOptions) -> pd.DataFrame:
    """逐根出发的趋势扫描标签（de Prado Ch 3.6，原文 Snippet 3.4/3.5）。

    对每个起点 t0，在前向窗口 (t0, t0+span] 内枚举所有端点 t1，对 [t0, t1] 上的价格
    做「价格对时间」回归，取 |t| 最大的那个端点：标签是那个斜率的符号，t1 是端点位置。

    四条口径（改代码前先读）：

    1. **回归的是价格序列本身**（原文口径），不是对数价格。斜率 t 值对 y 的整体
       缩放不变，所以标的之间的价格量级差异不影响读数。
    2. **t1 是整数根序号**，与三重障碍的 t1 同一坐标——都要喂给唯一性权重与 purge。
    3. **动态阈值只用过去**（见 `_threshold_weak`）。
    4. **尾部 min_window−1 根没有标签**，返回 NaN 而不是 0——0 在下游是「有观点但
       不下注」（被阈值判为弱趋势），两者不能混。

    列：label（{−1, 0, +1}）、t1（端点位置）、t_value（有符号 t 值）、
    span（该端点的窗口长度）、ret（起点到端点的收益）。
    """
    options.validate()
    size = len(close)
    empty = pd.DataFrame(index=close.index, columns=pd.Index(TREND_COLUMNS))
    if size <= options.span:
        return empty

    values = close.to_numpy(dtype=float)
    window_lengths = list(range(options.min_window, options.span + 1))
    t_values = np.full((size, len(window_lengths)), np.nan)
    for column, length in enumerate(window_lengths):
        # 切片索引而不是 sliding_window_view：numpy 桩对后者的 window_shape
        # 只认 Python int，循环变量的 int 类型会被反复误报（本仓禁用 pyright ignore）
        starts = np.arange(size - length + 1)
        windows = values[starts[:, None] + np.arange(length)]
        t_values[: starts.size, column] = slope_t_values(windows)

    magnitude = np.abs(t_values)
    usable = np.isfinite(magnitude).any(axis=1)
    rows = np.flatnonzero(usable)
    if rows.size == 0:
        return empty
    best = np.nanargmax(magnitude[rows], axis=1)
    chosen_t = t_values[rows, best]
    chosen_span = np.asarray(window_lengths, dtype=int)[best]
    labels = np.sign(chosen_t)
    if options.threshold > 0.0:
        weak = _threshold_weak(
            close.index.take(rows), np.abs(chosen_t), options.threshold
        )
        labels = np.where(weak, 0.0, labels)
    return _trend_frame(close, rows, chosen_span, chosen_t, labels)
