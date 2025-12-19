"""
src/research/research.py
科研控制台：集成动态预热缓冲与中文绘图修复版
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

import ccxt
import matplotlib.pyplot as plt

# 1. 绘图环境配置：解决中文乱码与负号显示
plt.rcParams["font.sans-serif"] = ["SimHei"]  # Windows 环境使用黑体
plt.rcParams["axes.unicode_minus"] = False

# 2. 路径补丁
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# noqa: E402
from config.settings import GLOBAL_CONFIG  # noqa: E402
from src.research.domains.crypto import CryptoDomain  # noqa: E402
from src.research.domains.stock_cn import StockCNDomain  # noqa: E402
from src.research.engine import BacktestEngine  # noqa: E402
from src.research.strategies.rsrs_trend_follower import RSRSTrendStrategy  # noqa: E402
from src.research.strategies.vector_dual_ma import DualMAStrategy  # noqa: E402
from src.research.strategies.vector_trend_reversion import (
    TrendReversionStrategy,  # noqa: E402
)
from src.research.strategies.vector_value import ValueStrategy  # noqa: E402
from src.utils.data_loader import MarketDataHandler  # noqa: E402

# === 快捷预设菜单 ===
SYMBOL_PRESETS: dict[str, dict[str, tuple[str, str]]] = {
    "Crypto": {
        "c1": ("BTC/USDT", "比特币 (Bitcoin)"),
        "c2": ("ETH/USDT", "以太坊 (Ethereum)"),
    },
    "A股-ETF (免税)": {
        "e1": ("513100", "纳指 ETF (美股牛市)"),
        "e2": ("510300", "沪深300 ETF (大盘)"),
        "e3": ("159915", "创业板 ETF (成长)"),
        "e4": ("512880", "证券 ETF (波动/牛市旗手)"),
        "e5": ("510880", "红利 ETF (高股息)"),
    },
    "A股-个股": {
        "s1": ("600519", "贵州茅台 (消费/核心资产)"),
        "s2": ("600036", "招商银行 (低估值/价值)"),
        "s3": ("601088", "中国神华 (煤炭/高分红)"),
        "s4": ("601127", "赛力斯 (高波动/华为合作)"),
    },
}

# === 策略映射表 ===
STRATEGY_MAP: dict[str, dict[str, Any]] = {
    "1": {
        "name": "自适应双均线 (Dual MA)",
        "class": DualMAStrategy,
        "default_params": {"trend_period": 120, "bull_fast": 10, "bull_slow": 30},
    },
    "2": {
        "name": "趋势+反转缝合怪 (Trend+Reversion)",
        "class": TrendReversionStrategy,
        "default_params": {"trend_period": 200, "bull_fast": 10},
    },
    "3": {
        "name": "低估值择时 (PE/PB Valuation)",
        "class": ValueStrategy,
        "default_params": {"factor_name": "pe", "buy_threshold": 15.0},
    },
    "4": {
        "name": "RSRS 阻力支撑相对强度 (RSRS Trend)",
        "class": RSRSTrendStrategy,
        "default_params": {
            "rsrs_n": 18,
            "rsrs_m": 600,
            "buy_threshold": 0.7,
            "sell_threshold": -0.7,
            "max_leverage": 1.0,
            "stop_loss_mult": 2.5,
        },
    },
}


def customize_params(default_params: dict[str, Any]) -> dict[str, Any]:
    """交互式修改策略参数。"""
    print("\n⚙️ 参数设置 (直接回车保持默认):")
    new_params = default_params.copy()
    for key, default_val in default_params.items():
        while True:
            try:
                user_input = input(f"   - {key} [默认 {default_val}]: ").strip()
                if not user_input:
                    break
                target_type = type(default_val)
                new_params[key] = target_type(user_input)
                break
            except ValueError:
                print(f"   ❌ 格式错误，请输入有效的 {type(default_val).__name__}")
    return new_params


def select_symbol_ui() -> str:
    """标的选择界面。"""
    print("\n🎯 请选择回测标的:")
    quick_map = {}
    for category, items in SYMBOL_PRESETS.items():
        print(f"\n--- {category} ---")
        for key, (code, desc) in items.items():
            print(f"  [{key}] {desc} ({code})")
            quick_map[key] = code

    user_input = input("\n请输入 [序号] 或 [代码] (默认 e2): ").strip().lower()
    if not user_input:
        return quick_map["e2"]
    return quick_map.get(user_input, user_input.upper())


def run_backtest(
    symbol: str,
    timeframe: str,
    strategy_conf: dict[str, Any],
    days: int,
    leverage: float = 1.0,
) -> None:
    """执行单次回测流程，支持动态数据预热。"""
    StrategyClass = strategy_conf["class"]
    params = strategy_conf["default_params"].copy()
    strategy_name = strategy_conf["name"]

    print(
        f"\n{'=' * 60}\n🧪 启动回测 | 策略: {strategy_name} | 标的: {symbol}\n{'=' * 60}"
    )

    # 1. 动态计算预热缓冲区
    required_buffer = 700 if "RSRS" in strategy_name else 200

    loader = MarketDataHandler(
        ccxt.okx(GLOBAL_CONFIG.to_ccxt_config()) if "/" in symbol else None
    )

    try:
        # 调用改进后的 prepare_data，传入 required_buffer
        df = loader.prepare_data(
            symbol, timeframe, days_back=days, required_buffer=required_buffer
        )
        if df is None or df.empty:
            print("❌ 数据为空，请检查网络、数据源或 Tushare 积分。")
            return
        print(f"📊 数据准备完毕: 总行数 {len(df)}, 起始时间 {df.index[0]}")
    except Exception as e:
        print(f"❌ 数据获取失败: {e}")
        return

    # 2. 策略计算
    market_type = "crypto" if "/" in symbol else "stock_cn"
    params["market_type"] = market_type
    strategy = StrategyClass(**params)
    df = strategy.calculate_indicators(df)
    df = strategy.generate_signals(df)

    # 3. 引擎适配
    if market_type == "crypto":
        domain = CryptoDomain(fee_rate=0.0005, slippage=0.0005)
        capital = 10000.0
    else:
        is_etf = any(symbol.startswith(pre) for pre in ["159", "51", "588"])
        domain = StockCNDomain(
            commission_rate=0.0002 if is_etf else 0.00025,
            stamp_duty=0.0 if is_etf else 0.0005,
            slippage=0.001,
        )
        capital = 200000.0

    engine = BacktestEngine(domain=domain, initial_capital=capital, leverage=leverage)
    result = engine.run(df)

    if not result:
        print("⚠️ 回测未产生有效交易。")
        return

    # 4. 结果输出摘要
    print(f"\n📊 结果摘要 | {symbol}")
    print(
        f"💰 总收益: {result['total_ret']:.2%} (基准: {result.get('bench_ret', 0):.2%})"
    )
    print(
        f"📉 最大回撤: {result['max_drawdown']:.2%} | 夏普: {result['sharpe_ratio']:.2f}"
    )
    print(f"🔄 交易次数: {result['trade_count']}")

    if input("\n是否生成图表? (y/n) [默认 y]: ").strip().lower() != "n":
        save_path = (
            PROJECT_ROOT
            / "data"
            / "results"
            / f"{symbol.replace('/', '_')}_{strategy_name}.png"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        engine.plot(
            result, title=f"{symbol} - {strategy_name}", save_path=str(save_path)
        )


def main() -> None:
    print("🚀 Unified Quant Research Lab v2.0")
    print("-" * 40)

    while True:
        try:
            symbol = select_symbol_ui()
            timeframe = input("\nK线周期 (1h, 4h, 1d) [默认 1d]: ").strip() or "1d"
            days_input = input("回测天数 [默认 730]: ").strip()
            days = int(days_input) if days_input else 730

            print("\n♟️ 选择策略:")
            for k, v in STRATEGY_MAP.items():
                print(f"  [{k}] {v['name']}")
            strat_choice = input("请输入序号 [默认 4]: ").strip() or "4"
            strategy_conf = STRATEGY_MAP.get(strat_choice)

            if not strategy_conf:
                print("❌ 无效的策略序号")
                continue

            mode = input("\n🔧 [1]回测 [2]优化 [默认 1]: ").strip() or "1"

            if mode == "1":
                conf = strategy_conf.copy()
                if input("修改参数? (y/n) [默认 n]: ").strip().lower() == "y":
                    conf["default_params"] = customize_params(conf["default_params"])

                leverage = 1.0
                if "/" in symbol:
                    lev_input = input("杠杆倍数 [默认 1]: ").strip()
                    leverage = float(lev_input) if lev_input else 1.0

                run_backtest(symbol, timeframe, conf, days, leverage)

            elif mode == "2":
                print("⚠️ 优化模式目前仅支持均线策略。")

            if input("\n继续其他实验? (y/n) [默认 y]: ").strip().lower() == "n":
                print("👋 实验结束，再见！")
                break

        except KeyboardInterrupt:
            print("\n👋 用户中断实验")
            break
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    main()
