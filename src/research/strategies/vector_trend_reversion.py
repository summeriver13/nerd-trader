"""
src/research/strategies/vector_trend_reversion.py
趋势+反转缝合怪策略 (Trend + Reversion) [最终架构适配版]
核心改进:
1. 适配 BaseStrategy 的 Crypto 市场规则
2. 杠杆逻辑升级：取 "策略设定上限" 和 "市场物理上限" 的较小值
3. 熊市做空逻辑受 self.can_short 控制
"""

from typing import Any, Literal, overload, override

import numpy as np
import pandas as pd

from src.research.strategies.base import EWM_CONVERGENCE_MULT, BaseStrategy


class TrendReversionStrategy(BaseStrategy):
    """
    牛市趋势 + 熊市反转 + 非对称凯利资金管理
    """

    def __init__(
        self,
        # === 1. 市场状态判断 ===
        trend_period: int = 200,
        # === 2. 牛市趋势参数 ===
        bull_fast: int = 14,
        bull_slow: int = 25,
        bull_adx_thresh: int = 30,
        bull_atr_mult: float = 3.0,
        # === 3. 熊市反转参数 ===
        bear_bb_period: int = 20,
        bear_bb_std: float = 2.0,
        bear_rsi_period: int = 14,
        bear_rsi_thresh: int = 30,
        bear_stop_loss: float = 0.05,
        # === 4. 资金管理 (凯利公式) ===
        kelly_window: int = 20,  # 回溯最近多少笔交易来计算
        max_leverage: float = 2.0,  # 策略期望的最大杠杆
        min_leverage: float = 0.5,  # 杠杆下限
        # === 通用参数 ===
        atr_period: int = 14,
        adx_period: int = 14,
    ):
        # 1. 初始化父类 (获取 self.can_short, self.max_leverage 等市场规则)
        super().__init__()

        self.trend_period = trend_period

        self.bull_fast = bull_fast
        self.bull_slow = bull_slow
        self.bull_adx_thresh = bull_adx_thresh
        self.bull_atr_mult = bull_atr_mult

        self.bear_bb_period = bear_bb_period
        self.bear_bb_std = bear_bb_std
        self.bear_rsi_period = bear_rsi_period
        self.bear_rsi_thresh = bear_rsi_thresh
        self.bear_stop_loss = bear_stop_loss

        self.kelly_window = kelly_window

        # === [关键修改] 杠杆双重锁 ===
        # 策略设定的上限 vs 交易所允许的物理上限，取较小值
        self.effective_max_lev = min(max_leverage, self.max_leverage)

        # 最小杠杆也不能超过最大杠杆
        self.effective_min_lev = min(min_leverage, self.effective_max_lev)

        self.atr_period = atr_period
        self.adx_period = adx_period

    @property
    @override
    def warmup_bars(self) -> int:
        """最长滚动窗口 + ADX 的 EWM 收敛余量"""
        longest = max(
            self.trend_period,
            self.bull_slow,
            self.bear_bb_period,
            self.bear_rsi_period,
            self.kelly_window,
            self.atr_period,
        )
        return longest + EWM_CONVERGENCE_MULT * self.adx_period

    @override
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.sort_index()

        # 1. 长期趋势线
        df["MA_Trend"] = df["close"].rolling(self.trend_period).mean()

        # 2. 牛市指标
        df["MA_Fast"] = df["close"].rolling(self.bull_fast).mean()
        df["MA_Slow"] = df["close"].rolling(self.bull_slow).mean()

        # ATR
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["ATR"] = tr.rolling(self.atr_period).mean()

        # ADX
        up = df["high"] - df["high"].shift(1)
        down = df["low"].shift(1) - df["low"]
        plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
        minus_dm = pd.Series(
            np.where((down > up) & (down > 0), down, 0.0), index=df.index
        )

        alpha = 1 / self.adx_period
        tr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / tr_smooth)
        minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / tr_smooth)

        sum_di = pd.Series(plus_di + minus_di, index=df.index).replace(0, np.nan)
        dx = pd.Series(100 * (abs(plus_di - minus_di) / sum_di), index=df.index)
        df["ADX"] = pd.Series(dx).fillna(0).ewm(alpha=alpha, adjust=False).mean()

        # 3. 熊市指标
        bb_mid = df["close"].rolling(self.bear_bb_period).mean()
        bb_std = df["close"].rolling(self.bear_bb_period).std()
        df["BB_Upper"] = bb_mid + (bb_std * self.bear_bb_std)
        df["BB_Mid"] = bb_mid
        df["BB_Lower"] = bb_mid - (bb_std * self.bear_bb_std)

        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.bear_rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.bear_rsi_period).mean()
        rs = gain / loss
        rsi_raw = pd.Series(100 - (100 / (1 + rs)), index=df.index)
        df["RSI"] = pd.Series(rsi_raw).fillna(50)

        return df

    @override
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if "MA_Trend" not in df.columns:
            df = self.calculate_indicators(df)

        signals, _ = self._run_simulation_logic(
            df["open"].to_numpy(),
            df["close"].to_numpy(),
            df["high"].to_numpy(),
            df["low"].to_numpy(),
            df["MA_Trend"].to_numpy(),
            df["MA_Fast"].to_numpy(),
            df["MA_Slow"].to_numpy(),
            df["ATR"].to_numpy(),
            df["ADX"].to_numpy(),
            df["BB_Lower"].to_numpy(),
            df["BB_Mid"].to_numpy(),
            df["RSI"].to_numpy(),
            # 策略参数
            self.bull_adx_thresh,
            self.bull_atr_mult,
            self.bear_rsi_thresh,
            self.bear_stop_loss,
            # 凯利参数
            self.kelly_window,
            # [关键] 传入计算后的有效杠杆
            self.effective_max_lev,
            self.effective_min_lev,
            # 时间戳
            timestamps=df.index,
        )
        df["signal"] = signals
        return df

    def get_trade_history(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if "MA_Trend" not in df.columns:
            df = self.calculate_indicators(df)

        return self._run_simulation_logic(
            df["open"].to_numpy(),
            df["close"].to_numpy(),
            df["high"].to_numpy(),
            df["low"].to_numpy(),
            df["MA_Trend"].to_numpy(),
            df["MA_Fast"].to_numpy(),
            df["MA_Slow"].to_numpy(),
            df["ATR"].to_numpy(),
            df["ADX"].to_numpy(),
            df["BB_Lower"].to_numpy(),
            df["BB_Mid"].to_numpy(),
            df["RSI"].to_numpy(),
            self.bull_adx_thresh,
            self.bull_atr_mult,
            self.bear_rsi_thresh,
            self.bear_stop_loss,
            self.kelly_window,
            self.effective_max_lev,
            self.effective_min_lev,
            record_trades=True,
            timestamps=df.index,
        )

    def _calculate_kelly_leverage(
        self,
        trade_history: list[dict[str, Any]],
        window: int,
        max_lev: float,
        min_lev: float,
    ) -> float:
        """
        计算半凯利杠杆
        """
        if len(trade_history) < 5:
            return 1.0  # 初始默认满仓/1倍

        recent_trades = trade_history[-window:]
        pnls = [t["pnl"] for t in recent_trades]

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        if not wins:
            return min_lev
        if not losses:
            return max_lev

        win_rate = len(wins) / len(pnls)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        if avg_loss == 0:
            return max_lev
        b_ratio = avg_win / avg_loss

        # 半凯利公式：负值直接压到下限，否则夹在 [min_lev, max_lev]
        kelly_f = (win_rate * b_ratio - (1 - win_rate)) / b_ratio
        half_kelly = kelly_f / 2
        final_lev = (
            min_lev if half_kelly < 0 else float(np.clip(half_kelly, min_lev, max_lev))
        )

        return final_lev

    @overload
    def _run_simulation_logic(
        self,
        open_arr: np.ndarray,
        close_arr: np.ndarray,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        trend_ma: np.ndarray,
        ma_fast: np.ndarray,
        ma_slow: np.ndarray,
        atr: np.ndarray,
        adx: np.ndarray,
        bb_lower: np.ndarray,
        bb_mid: np.ndarray,
        rsi: np.ndarray,
        bull_adx_thresh: float,
        bull_atr_mult: float,
        bear_rsi_thresh: float,
        bear_stop_loss_pct: float,
        kelly_window: int,
        max_lev: float,
        min_lev: float,
        record_trades: Literal[True],
        timestamps: pd.Index | None = None,
    ) -> list[dict[str, Any]]: ...

    @overload
    def _run_simulation_logic(
        self,
        open_arr: np.ndarray,
        close_arr: np.ndarray,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        trend_ma: np.ndarray,
        ma_fast: np.ndarray,
        ma_slow: np.ndarray,
        atr: np.ndarray,
        adx: np.ndarray,
        bb_lower: np.ndarray,
        bb_mid: np.ndarray,
        rsi: np.ndarray,
        bull_adx_thresh: float,
        bull_atr_mult: float,
        bear_rsi_thresh: float,
        bear_stop_loss_pct: float,
        kelly_window: int,
        max_lev: float,
        min_lev: float,
        record_trades: Literal[False] = False,
        timestamps: pd.Index | None = None,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def _run_simulation_logic(
        self,
        open_arr: np.ndarray,
        close_arr: np.ndarray,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        trend_ma: np.ndarray,
        ma_fast: np.ndarray,
        ma_slow: np.ndarray,
        atr: np.ndarray,
        adx: np.ndarray,
        bb_lower: np.ndarray,
        bb_mid: np.ndarray,
        rsi: np.ndarray,
        bull_adx_thresh: float,
        bull_atr_mult: float,
        bear_rsi_thresh: float,
        bear_stop_loss_pct: float,
        kelly_window: int,
        max_lev: float,
        min_lev: float,
        record_trades: bool = False,
        timestamps: pd.Index | None = None,
    ) -> list[dict[str, Any]] | tuple[np.ndarray, np.ndarray]:
        length = len(close_arr)
        signals = np.zeros(length)
        returns = np.zeros(length)
        trade_logs = []

        position = 0.0
        stop_price = 0.0
        entry_price = 0.0
        entry_idx = 0
        extreme_price = 0.0
        strategy_mode = ""

        current_leverage = 1.0

        for i in range(1, length):
            if np.isnan(trend_ma[i]) or np.isnan(bb_mid[i]):
                continue

            prev_close = close_arr[i - 1]
            bar_return = 0.0

            is_bull_regime = close_arr[i] > trend_ma[i]
            exit_reason = ""
            exit_price = 0.0

            # === 2. 持仓管理 ===
            if position > 0:
                bar_return = ((close_arr[i] - prev_close) / prev_close) * position

                if strategy_mode == "Bull_Trend":
                    # 牛市止损：跌破 ATR 移动止损线
                    if low_arr[i] < stop_price:
                        position = 0.0
                        exit_reason = "Bull_StopLoss"
                        exit_price = (
                            min(open_arr[i], stop_price)
                            if open_arr[i] < stop_price
                            else stop_price
                        )
                        bar_return = (
                            (exit_price - prev_close) / prev_close
                        ) * current_leverage
                    # 牛市止盈：均线死叉
                    elif ma_fast[i] < ma_slow[i]:
                        position = 0.0
                        exit_reason = "Bull_DeadCross"
                        exit_price = close_arr[i]
                    else:
                        # 移动止损上移
                        if high_arr[i] > extreme_price:
                            extreme_price = high_arr[i]
                            new_stop = extreme_price - (bull_atr_mult * atr[i])
                            stop_price = max(stop_price, new_stop)

                elif strategy_mode == "Bear_Reversion":
                    # 熊市止损：固定百分比
                    if low_arr[i] < stop_price:
                        position = 0.0
                        exit_reason = "Bear_StopLoss"
                        exit_price = (
                            min(open_arr[i], stop_price)
                            if open_arr[i] < stop_price
                            else stop_price
                        )
                        bar_return = (
                            (exit_price - prev_close) / prev_close
                        ) * current_leverage
                    # 熊市止盈：回归布林带中轨
                    elif high_arr[i] > bb_mid[i]:
                        position = 0.0
                        exit_reason = "Bear_TakeProfit"
                        exit_price = bb_mid[i]
                        bar_return = (
                            (exit_price - prev_close) / prev_close
                        ) * current_leverage
                    # 熊市时间止损：持仓太久不涨就跑
                    elif (i - entry_idx) > 18:
                        position = 0.0
                        exit_reason = "Bear_TimeStop"
                        exit_price = close_arr[i]

            # === 3. 开仓逻辑 ===
            else:  # position == 0
                suggested_lev = self._calculate_kelly_leverage(
                    trade_logs, kelly_window, max_lev, min_lev
                )

                # --- A. 牛市 ---
                if is_bull_regime:
                    if (ma_fast[i] > ma_slow[i]) and (adx[i] > bull_adx_thresh):
                        # 尝试放大仓位，但不能超过 effective_max_lev
                        aggressive_lev = max(suggested_lev * 1.5, 1.0)
                        current_leverage = min(aggressive_lev, max_lev)

                        position = current_leverage

                        strategy_mode = "Bull_Trend"
                        entry_price = close_arr[i]
                        entry_idx = i
                        extreme_price = high_arr[i]
                        stop_price = extreme_price - (bull_atr_mult * atr[i])

                # --- B. 熊市 ---
                else:
                    # 抄底信号：跌破下轨且 RSI 超卖
                    is_oversold = (low_arr[i] < bb_lower[i]) and (rsi[i] < 30)
                    is_confirmed = close_arr[i] > bb_lower[i]  # 收盘收回下轨内

                    if is_oversold and is_confirmed:
                        # [关键] 熊市只做多抄底，不做空

                        # 熊市仓位保守：建议值和 1.0 取小
                        bear_lev = min(suggested_lev, 1.0)
                        current_leverage = min(bear_lev, max_lev)

                        position = current_leverage

                        strategy_mode = "Bear_Reversion"
                        entry_price = close_arr[i]
                        entry_idx = i
                        stop_price = entry_price * (1 - bear_stop_loss_pct)

            signals[i] = position
            returns[i] = bar_return

            if position == 0 and exit_reason and timestamps is not None:
                raw_pnl = (exit_price - entry_price) / entry_price
                # 移除了 Short 相关逻辑，因为此策略只做多

                trade_logs.append(
                    {
                        "entry_time": timestamps[entry_idx],
                        "exit_time": timestamps[i],
                        "reason": exit_reason,
                        "pnl": raw_pnl,
                        "leverage_used": current_leverage,
                    }
                )

        if record_trades:
            return trade_logs
        return signals, returns
