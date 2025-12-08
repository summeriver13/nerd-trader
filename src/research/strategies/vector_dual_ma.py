"""
src/research/strategies/vector_dual_ma.py
双均线策略 - 向量化回测版 (Vectorized)
改进点：加入 ADX 趋势滤网，过滤震荡行情，降低回撤
"""

import itertools

import numpy as np
import pandas as pd

from src.research.strategies.base import BaseStrategy


class DualMAStrategy(BaseStrategy):
    """
    双均线趋势策略 (Dual MA) + ATR 移动止损 + ADX 趋势过滤
    """

    def __init__(
        self,
        fast_period: int = 9,
        slow_period: int = 30,
        atr_period: int = 14,
        atr_multiplier: float = 3.0,
        adx_period: int = 14,
        adx_threshold: int = 25,  # 新增：ADX 阈值，低于此值不开仓
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        df = df.sort_index()

        # 1. 均线
        df["MA_Fast"] = df["close"].rolling(self.fast_period).mean()
        df["MA_Slow"] = df["close"].rolling(self.slow_period).mean()

        # 2. ATR & ADX 计算 (纯 Pandas 实现，无需 TA-Lib)
        # 计算 TR
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(self.atr_period).mean()

        # 计算 ADX
        up = df["high"] - df["high"].shift(1)
        down = df["low"].shift(1) - df["low"]

        # 显式转为 Series
        plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
        minus_dm = pd.Series(
            np.where((down > up) & (down > 0), down, 0.0), index=df.index
        )

        # 使用 Wilder 的平滑方法 (alpha=1/period)
        alpha = 1 / self.adx_period
        tr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()

        plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / tr_smooth)
        minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / tr_smooth)

        # === 核心修复点 ===
        # Linter 认为 plus_di + minus_di 可能是 int，
        # 我们这里强制将其包装为 Series，彻底解决 "Cannot access attribute replace for class int"
        sum_di = pd.Series(plus_di + minus_di, index=df.index)

        # 现在 sum_di 确认为 Series，可以使用 replace
        sum_di = sum_di.replace(0, np.nan)

        dx = 100 * (abs(plus_di - minus_di) / sum_di)

        # 同样，强制 dx 为 Series 后再调用 ewm
        df["ADX"] = (
            pd.Series(dx, index=df.index)
            .fillna(0)
            .ewm(alpha=alpha, adjust=False)
            .mean()
        )

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if "MA_Fast" not in df.columns or "ADX" not in df.columns:
            df = self.calculate_indicators(df)

        signals, _ = self._run_simulation_logic(
            df["open"].values,
            df["close"].values,
            df["high"].values,
            df["low"].values,
            df["MA_Fast"].values,
            df["MA_Slow"].values,
            df["ATR"].values,
            df["ADX"].values,  # 传入 ADX
            self.atr_multiplier,
            self.adx_threshold,  # 传入阈值
        )
        df["signal"] = signals
        return df

    def get_trade_history(self, df: pd.DataFrame) -> list:
        if "MA_Fast" not in df.columns:
            df = self.calculate_indicators(df)

        return self._run_simulation_logic(
            df["open"].values,
            df["close"].values,
            df["high"].values,
            df["low"].values,
            df["MA_Fast"].values,
            df["MA_Slow"].values,
            df["ATR"].values,
            df["ADX"].values,
            self.atr_multiplier,
            self.adx_threshold,
            record_trades=True,
            timestamps=df.index,
        )  # type: ignore

    def _run_simulation_logic(
        self,
        open_arr,
        close_arr,
        high_arr,
        low_arr,
        ma_fast_arr,
        ma_slow_arr,
        atr_arr,
        adx_arr,  # 新增参数
        atr_mult,
        adx_thresh,  # 新增参数
        record_trades=False,
        timestamps=None,
    ):
        """
        核心逻辑: 双均线交叉 + ADX 过滤 + ATR 追踪止损
        """
        length = len(close_arr)
        signals = np.zeros(length)
        returns = np.zeros(length)
        trade_logs = []

        position = 0
        stop_price = 0.0
        highest_price = 0.0

        entry_price = 0.0
        entry_idx = 0

        for i in range(1, length):
            if (
                np.isnan(ma_fast_arr[i])
                or np.isnan(ma_slow_arr[i])
                or np.isnan(atr_arr[i])
                or np.isnan(adx_arr[i])
            ):
                continue

            prev_close = close_arr[i - 1]
            bar_return = (close_arr[i] - prev_close) / prev_close

            exit_reason = ""
            exit_price = 0.0

            # === 持仓逻辑 ===
            if position == 1:
                # 1. 检查止损 (ATR 追踪)
                if low_arr[i] < stop_price:
                    position = 0
                    exit_reason = "StopLoss"
                    exit_price = stop_price
                    # 处理跳空低开
                    if open_arr[i] < stop_price:
                        exit_price = open_arr[i]

                    # 修正当根K线的收益计算
                    bar_return = (exit_price - prev_close) / prev_close

                # 2. 检查死叉 (趋势结束)
                elif ma_fast_arr[i] < ma_slow_arr[i]:
                    position = 0
                    exit_reason = "DeathCross"
                    exit_price = close_arr[i]

                # 3. 持有更新 (上移止损)
                else:
                    if high_arr[i] > highest_price:
                        highest_price = high_arr[i]
                        new_stop = highest_price - (atr_mult * atr_arr[i])
                        stop_price = max(stop_price, new_stop)

                # 记录平仓交易
                if position == 0 and record_trades:
                    trade_pnl = (exit_price - entry_price) / entry_price
                    trade_logs.append(
                        {
                            "entry_time": timestamps[entry_idx],  # type: ignore
                            "entry_price": entry_price,
                            "exit_time": timestamps[i],  # type: ignore
                            "exit_price": exit_price,
                            "reason": exit_reason,
                            "pnl": trade_pnl,
                            "bars_held": i - entry_idx,
                            "max_favorable": (highest_price - entry_price)
                            / entry_price,
                        }
                    )

            # === 空仓逻辑 (开仓信号) ===
            else:
                # 必须满足两个条件:
                # 1. 均线金叉 (快线上穿慢线)
                # 2. ADX 大于阈值 (趋势够强)
                if (ma_fast_arr[i] > ma_slow_arr[i]) and (adx_arr[i] > adx_thresh):
                    position = 1
                    highest_price = high_arr[i]
                    stop_price = highest_price - (atr_mult * atr_arr[i])

                    entry_price = close_arr[i]
                    entry_idx = i
                    bar_return = 0.0

            signals[i] = position
            # 如果昨天收盘持有，今天的涨跌幅计入收益
            if signals[i - 1] == 1:
                returns[i] = bar_return
            else:
                returns[i] = 0.0

        if record_trades:
            return trade_logs
        return signals, returns

    def optimize(
        self,
        df: pd.DataFrame,
        fast_range=range(5, 25, 5),
        slow_range=range(20, 60, 10),
        atr_mult_range=[3.0, 5.0],
        adx_thresh_range=[20, 25, 30],
    ) -> dict:
        print("\n🚀 [策略优化] 启动网格搜索 (含 ADX 滤网)...")
        results = []

        # === 核心改动：全员使用 np.array() 包裹 ===
        # 这种写法是 "防弹" 的：无论输入是 Series 还是 array，np.array() 都能处理，
        # 且不会触发 Linter 关于 .values 或 .to_numpy() 的属性报错。

        close_arr = np.array(df["close"])
        open_arr = np.array(df["open"])
        high_arr = np.array(df["high"])
        low_arr = np.array(df["low"])

        # 预计算 ADX
        temp_strat = DualMAStrategy(
            adx_period=self.adx_period, atr_period=self.atr_period
        )
        df_temp = temp_strat.calculate_indicators(df.copy())

        atr_arr = np.array(df_temp["ATR"])
        adx_arr = np.array(df_temp["ADX"])

        combinations = list(
            itertools.product(fast_range, slow_range, atr_mult_range, adx_thresh_range)
        )

        # 为了 rolling 计算方便，构造一个明确的 Series
        close_series = pd.Series(close_arr)

        for fast, slow, atr_mult, adx_thresh in combinations:
            if fast >= slow:
                continue

            # === 这里也是重点 ===
            # 不再调用 .to_numpy() 或 .values
            # 直接把 rolling 的结果喂给 np.array()
            ma_fast = np.array(close_series.rolling(fast).mean())
            ma_slow = np.array(close_series.rolling(slow).mean())

            signals, strategy_ret_arr = self._run_simulation_logic(
                open_arr,
                close_arr,
                high_arr,
                low_arr,
                ma_fast,
                ma_slow,
                atr_arr,
                adx_arr,
                atr_mult,
                adx_thresh,
            )

            trades = np.sum(np.abs(np.diff(signals)))

            # 简单的夏普估算
            std_dev = np.std(strategy_ret_arr)
            if std_dev == 0:
                sharpe = 0
            else:
                sharpe = (np.mean(strategy_ret_arr) / std_dev) * np.sqrt(365 * 6)

            net_ret = np.sum(strategy_ret_arr) - (trades * 0.0005)

            results.append(
                {
                    "fast": fast,
                    "slow": slow,
                    "atr_mult": atr_mult,
                    "adx_thresh": adx_thresh,
                    "net_ret": net_ret,
                    "sharpe": sharpe,
                    "trades": trades,
                }
            )

        if not results:
            return {"fast": 9, "slow": 30, "atr_mult": 5.0, "adx_thresh": 25}

        res_df = pd.DataFrame(results)
        best = res_df.sort_values(by="sharpe", ascending=False).iloc[0]

        self.fast_period = int(best["fast"])
        self.slow_period = int(best["slow"])
        self.atr_multiplier = float(best["atr_mult"])
        self.adx_threshold = int(best["adx_thresh"])

        return {
            "fast": self.fast_period,
            "slow": self.slow_period,
            "atr_mult": self.atr_multiplier,
            "adx_thresh": self.adx_threshold,
            "sharpe": best["sharpe"],
        }
