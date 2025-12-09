"""
src/research/strategies/vector_trend_reversion.py
趋势+反转缝合怪策略 (Trend + Reversion) [非对称凯利版]
核心改进:
1. 非对称凯利 (Asymmetric Kelly): 牛市放大仓位(至少满仓)，熊市缩小仓位(最多1倍)
2. 动态调整开仓杠杆
"""

import numpy as np
import pandas as pd

from src.research.strategies.base import BaseStrategy


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
        max_leverage: float = 2.0,  # 全局杠杆上限 (风控锁)
        min_leverage: float = 0.5,  # 杠杆下限
        # === 通用参数 ===
        atr_period: int = 14,
        adx_period: int = 14,
    ):
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
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage

        self.atr_period = atr_period
        self.adx_period = adx_period

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
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
        df["ADX"] = dx.fillna(0).ewm(alpha=alpha, adjust=False).mean()  # type: ignore

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
        df["RSI"] = rsi_raw.fillna(50)  # type: ignore

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if "MA_Trend" not in df.columns:
            df = self.calculate_indicators(df)

        signals, _ = self._run_simulation_logic(
            df["open"].values,
            df["close"].values,
            df["high"].values,
            df["low"].values,
            df["MA_Trend"].values,
            df["MA_Fast"].values,
            df["MA_Slow"].values,
            df["ATR"].values,
            df["ADX"].values,
            df["BB_Lower"].values,
            df["BB_Mid"].values,
            df["RSI"].values,
            # 策略参数
            self.bull_adx_thresh,
            self.bull_atr_mult,
            self.bear_rsi_thresh,
            self.bear_stop_loss,
            # 凯利参数
            self.kelly_window,
            self.max_leverage,
            self.min_leverage,
            # 时间戳
            timestamps=df.index,
        )
        df["signal"] = signals
        return df

    def get_trade_history(self, df: pd.DataFrame) -> list:
        if "MA_Trend" not in df.columns:
            df = self.calculate_indicators(df)

        return self._run_simulation_logic(
            df["open"].values,
            df["close"].values,
            df["high"].values,
            df["low"].values,
            df["MA_Trend"].values,
            df["MA_Fast"].values,
            df["MA_Slow"].values,
            df["ATR"].values,
            df["ADX"].values,
            df["BB_Lower"].values,
            df["BB_Mid"].values,
            df["RSI"].values,
            self.bull_adx_thresh,
            self.bull_atr_mult,
            self.bear_rsi_thresh,
            self.bear_stop_loss,
            self.kelly_window,
            self.max_leverage,
            self.min_leverage,
            record_trades=True,
            timestamps=df.index,
        )  # type: ignore

    def _calculate_kelly_leverage(self, trade_history, window, max_lev, min_lev):
        """
        计算半凯利杠杆 (Base Half-Kelly)
        """
        if len(trade_history) < 5:
            return 1.0

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

        kelly_f = (win_rate * b_ratio - (1 - win_rate)) / b_ratio
        half_kelly = kelly_f / 2

        if half_kelly < 0:
            final_lev = min_lev
        else:
            # 注意：这里我们返回 raw value，具体的 limit 逻辑放到 loop 里根据牛熊去定
            # 但为了防止计算出极端值，这里先做一个宽松的 clip
            final_lev = np.clip(half_kelly, min_lev, max_lev * 2.0)

        return final_lev

    def _run_simulation_logic(
        self,
        open_arr,
        close_arr,
        high_arr,
        low_arr,
        trend_ma,
        ma_fast,
        ma_slow,
        atr,
        adx,
        bb_lower,
        bb_mid,
        rsi,
        bull_adx_thresh,
        bull_atr_mult,
        bear_rsi_thresh,
        bear_stop_loss_pct,
        kelly_window,
        max_lev,
        min_lev,
        record_trades=False,
        timestamps=None,
    ):
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
                    elif ma_fast[i] < ma_slow[i]:
                        position = 0.0
                        exit_reason = "Bull_DeadCross"
                        exit_price = close_arr[i]
                    else:
                        if high_arr[i] > extreme_price:
                            extreme_price = high_arr[i]
                            new_stop = extreme_price - (bull_atr_mult * atr[i])
                            stop_price = max(stop_price, new_stop)

                elif strategy_mode == "Bear_Reversion":
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
                    elif high_arr[i] > bb_mid[i]:
                        position = 0.0
                        exit_reason = "Bear_TakeProfit"
                        exit_price = bb_mid[i]
                        bar_return = (
                            (exit_price - prev_close) / prev_close
                        ) * current_leverage
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
                        # === [非对称凯利核心逻辑] ===
                        # 1. 牛市放大：在凯利建议基础上 * 1.5
                        # 2. 满仓保底：至少开 1.0 倍 (只要出信号，绝不轻仓)
                        # 3. 物理封顶：不能超过 max_lev (比如 2.0)
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
                    is_oversold = (low_arr[i] < bb_lower[i]) and (rsi[i] < 30)
                    is_confirmed = close_arr[i] > bb_lower[i]

                    if is_oversold and is_confirmed:
                        # === [非对称凯利核心逻辑] ===
                        # 熊市保守：强制限制在 1.0 倍以下
                        bear_lev = min(suggested_lev, 1.0)

                        current_leverage = bear_lev
                        position = current_leverage

                        strategy_mode = "Bear_Reversion"
                        entry_price = close_arr[i]
                        entry_idx = i
                        stop_price = entry_price * (1 - bear_stop_loss_pct)

            signals[i] = position
            returns[i] = bar_return

            if position == 0 and exit_reason:
                raw_pnl = (exit_price - entry_price) / entry_price
                if "Short" in exit_reason:
                    raw_pnl *= -1

                trade_logs.append(
                    {
                        "entry_time": timestamps[entry_idx],  # type: ignore
                        "exit_time": timestamps[i],  # type: ignore
                        "reason": exit_reason,
                        "pnl": raw_pnl,
                        "leverage_used": current_leverage,
                    }
                )

        if record_trades:
            return trade_logs
        return signals, returns

    def optimize(self, df, **kwargs):
        print("⚠️ 混合策略参数过于复杂，建议使用默认经验参数。")
        return {}
