"""
research.py
科研控制台：问答式交互回测与参数优化
[UX Optimized] 体验优化版
1. 新增：标的快捷预设菜单 (Presets)
2. 优化：参数输入流程
"""

import sys
from pathlib import Path

import ccxt

# 1. 路径补丁
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# 2. 导入模块
from config.settings import GLOBAL_CONFIG  # noqa: E402
from src.research.domains.crypto import CryptoDomain  # noqa: E402
from src.research.domains.stock_cn import StockCNDomain  # noqa: E402
from src.research.engine import BacktestEngine  # noqa: E402

# 导入策略
from src.research.strategies.vector_dual_ma import DualMAStrategy  # noqa: E402
from src.research.strategies.vector_trend_reversion import (  # noqa: E402
    TrendReversionStrategy,  # noqa: E402
)
from src.research.strategies.vector_value import ValueStrategy  # noqa: E402
from src.utils.data_loader import MarketDataHandler  # noqa: E402

# === [新增] 标的预设菜单 ===
SYMBOL_PRESETS = {
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
        "s3": ("601088", "中国神华 (煤炭/高分红/PB策略)"),
        "s4": ("601127", "赛力斯 (新能源/高波动)"),
    },
}

# 3. 策略映射表
STRATEGY_MAP = {
    "1": {
        "name": "自适应双均线 (Adaptive Dual MA)",
        "class": DualMAStrategy,
        "default_params": {
            "trend_period": 120,
            "bull_fast": 10,
            "bull_slow": 30,
            "bull_adx_thresh": 25,
            "bull_atr_mult": 3.0,
            "bear_fast": 20,
            "bear_slow": 60,
            "bear_adx_thresh": 100,
            "bear_atr_mult": 3.0,
            "atr_period": 14,
            "adx_period": 14,
        },
    },
    "2": {
        "name": "趋势+反转缝合怪 (Trend + Reversion)",
        "class": TrendReversionStrategy,
        "default_params": {
            "trend_period": 200,
            "bull_fast": 10,
            "bull_slow": 30,
            "bull_adx_thresh": 25,
            "bull_atr_mult": 3.0,
            "bear_bb_period": 20,
            "bear_bb_std": 2.0,
            "bear_rsi_thresh": 30,
            "bear_stop_loss": 0.05,
            "atr_period": 14,
            "adx_period": 14,
        },
    },
    "3": {
        "name": "低估值择时 (PE/PB Valuation)",
        "class": ValueStrategy,
        "default_params": {
            "factor_name": "pe",
            "buy_threshold": 15.0,
            "sell_threshold": 25.0,
            "stop_loss_pct": 0.15,
        },
    },
}


def customize_params(default_params: dict) -> dict:
    print("\n⚙️  参数设置 (直接回车保持默认):")
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


def get_data_loader(symbol: str) -> MarketDataHandler:
    if "/" in symbol:
        exchange = ccxt.okx(GLOBAL_CONFIG.to_ccxt_config())  # type: ignore
        return MarketDataHandler(exchange)
    else:
        return MarketDataHandler(None)


def run_backtest(symbol, timeframe, strategy_conf, days, leverage=1.0):
    StrategyClass = strategy_conf["class"]
    params = strategy_conf["default_params"]
    strategy_name = strategy_conf["name"]

    print(f"\n{'=' * 60}")
    print(f"🧪 启动回测 | 策略: {strategy_name} | 标的: {symbol} | 周期: {timeframe}")
    print(f"⚙️  当前参数: {params}")
    print(f"⚖️  使用杠杆: {leverage}x")
    print(f"📅 回测窗口: 过去 {days} 天")
    print(f"{'=' * 60}")

    print("1. 正在获取数据...")
    loader = get_data_loader(symbol)

    try:
        df = loader.prepare_data(symbol, timeframe, days_back=days)
        if df is None or df.empty:
            print("   ❌ 数据为空！(请检查代码拼写/网络/Tushare积分)")
            return
        print(f"   🔍 实际回测范围: {df.index[0]} ~ {df.index[-1]}")
    except Exception as e:
        print(f"❌ 数据获取失败: {e}")
        return

    is_crypto = "/" in symbol
    market_type = "crypto" if is_crypto else "stock_cn"

    print("2. 计算策略信号...")
    params["market_type"] = market_type
    strategy = StrategyClass(**params)
    df = strategy.calculate_indicators(df)
    df = strategy.generate_signals(df)

    print("3. 执行回测...")

    if is_crypto:
        print("   🌐 识别为 Crypto 市场 (万5手续费)")
        domain = CryptoDomain(fee_rate=0.0005, slippage=0.0005)
        initial_capital = 10000.0
    else:
        is_etf = (
            symbol.startswith("159")
            or symbol.startswith("51")
            or symbol.startswith("588")
        )
        if is_etf:
            print(f"   📈 识别为 A股 ETF ({symbol}) -> [免印花税!]")
            domain = StockCNDomain(
                commission_rate=0.0002,
                min_commission=0.0,
                stamp_duty=0.0,
                slippage=0.001,
            )
        else:
            print(f"   🇨🇳 识别为 A股 股票 ({symbol}) -> [含印花税]")
            domain = StockCNDomain(
                commission_rate=0.00025,
                min_commission=5.0,
                stamp_duty=0.0005,
                slippage=0.001,
            )
        initial_capital = 200000.0

    engine = BacktestEngine(
        domain=domain, initial_capital=initial_capital, leverage=leverage
    )
    result = engine.run(df)

    if not result:
        print("⚠️ 回测未生成结果")
        return

    print("\n" + "=" * 40)
    print("📊 回测结果摘要")
    print("=" * 40)
    bench_val = result.get("bench_ret", 0.0)
    print(f"💰 总收益率: {result['total_ret']:.2%} (基准: {bench_val:.2%})")
    print(f"📅 年化收益: {result['annual_ret']:.2%}")
    print(f"📉 最大回撤: {result['max_drawdown']:.2%}")
    print(f"⚡ 夏普比率: {result['sharpe_ratio']:.2f}")
    print(f"🔄 交易次数: {result['trade_count']:.0f}")

    if "cost" in result["df"].columns:
        total_cost = result["df"]["cost"].sum() * engine.initial_capital
        print(f"💸 交易成本: {total_cost:.2f} 元 (佣金+印花税)")

    print("=" * 40 + "\n")

    plot_choice = input("是否生成图表? (y/n) [默认 y]: ").strip().lower()
    if plot_choice != "n":
        param_vals = [str(v) for v in params.values()]
        param_str = "_".join(param_vals)
        save_name = f"{symbol.replace('/', '_')}_{timeframe}_{days}d_{param_str}_lev{leverage}.png"
        save_path = PROJECT_ROOT / "data" / "results" / save_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"🖼️  正在绘图 -> {save_path}")
        engine.plot(
            result,
            title=f"{symbol} {timeframe} ({days} days)",
            save_path=str(save_path),
        )


def run_optimization(symbol, timeframe, strategy_conf, days):
    """暴力网格搜索模式"""
    StrategyClass = strategy_conf["class"]
    strategy_name = strategy_conf["name"]

    print(f"\n{'=' * 60}")
    print(f"🔥 启动暴力优化 | 策略: {strategy_name} | 标的: {symbol}")
    print(f"{'=' * 60}")

    print("1. 正在获取数据...")
    loader = get_data_loader(symbol)
    df = loader.prepare_data(symbol, timeframe, days_back=days)

    if df is None or df.empty:
        print("❌ 数据为空")
        return

    market_type = "crypto" if "/" in symbol else "stock_cn"

    print("\n2. 定义搜索参数范围 (回车使用默认推荐值)...")
    try:
        # 这里只给出一个通用的模板，具体策略可能会覆盖 optimize 方法自定义范围
        # 对于 PE 策略，我们可能需要不同的范围，但这里作为 Demo 保持原样
        f_start = input("   - 快线/买入阈值 [默认 5]: ").strip() or 5
        f_end = input("   - 慢线/卖出阈值 [默认 30]: ").strip() or 30
        fast_range = range(int(f_start), int(f_end), 5)
        # ... 这里简化处理，直接调用策略自带的 optimize 默认逻辑
        # 实际情况中，应该根据 strategy_name 动态决定让用户输什么参数
        slow_range = range(20, 80, 10)
        atr_mult_range = [3.0]
        adx_thresh_range = [0, 10, 25]  # 默认加入0适应ETF
    except ValueError:
        fast_range = range(5, 30, 5)
        slow_range = range(20, 80, 10)
        atr_mult_range = [3.0]
        adx_thresh_range = [0, 10, 25]

    print("\n⚡ 开始优化...")

    strategy = StrategyClass()
    if not hasattr(strategy, "optimize"):
        print("❌ 该策略不支持 optimize 方法")
        return

    best_params = strategy.optimize(
        df,
        fast_range=fast_range,
        slow_range=slow_range,
        atr_mult_range=atr_mult_range,
        adx_thresh_range=adx_thresh_range,
        market_type=market_type,
    )

    print("\n" + "=" * 40)
    print("🏆 优化完成！历史最优参数如下：")
    print("=" * 40)
    print(f"✅ 夏普比率 (Sharpe): {best_params.get('sharpe', 0):.2f}")
    print("\n📝 推荐配置:")
    for key, val in best_params.items():
        if key not in ["sharpe", "net_ret"]:
            print(f"  - {key}: {val}")


def select_symbol_ui():
    """
    [新增] 优化的标的选择界面
    """
    print("\n🎯 请选择回测标的:")

    # 建立一个扁平的映射表用于查找: {'e1': '513100', ...}
    quick_map = {}

    for category, items in SYMBOL_PRESETS.items():
        print(f"\n--- {category} ---")
        for key, (code, desc) in items.items():
            print(f"  [{key}] {desc}")
            quick_map[key] = code

    user_input = input("\n请输入 [序号] 或直接输入 [代码] (默认 e1): ").strip()

    if not user_input:
        return quick_map["e1"]  # 默认纳指

    # 如果用户输入的是快捷键 (如 's1')
    if user_input in quick_map:
        selected_code = quick_map[user_input]
        print(f"✅ 已选择: {selected_code}")
        return selected_code

    # 否则假设用户输入的是自定义代码
    return user_input.upper()


def main():
    print("🚀 欢迎来到量化实验室 (Unified Lab)")
    print("-" * 40)

    while True:
        try:
            # 1. 标的选择 (新 UI)
            symbol = select_symbol_ui()

            timeframe = (
                input("\n请输入K线周期 (e.g. 1h, 4h, 1d) [默认 1d]: ").strip().lower()
            ) or "1d"

            days_str = input("回测天数 [默认 730 (2年)]: ").strip()
            days = int(days_str) if days_str else 730

            # 2. 策略选择
            print("\n♟️ 请选择策略:")
            for key, val in STRATEGY_MAP.items():
                print(f"  [{key}] {val['name']}")
            strat_choice = (input("请输入序号 [默认 1]: ").strip()) or "1"
            strategy_conf = STRATEGY_MAP.get(strat_choice)

            if not strategy_conf:
                print("❌ 无效的策略序号")
                continue

            # 3. 模式选择
            print("\n🔧 请选择模式:")
            print("  [1] 单次回测 (Backtest)")
            print("  [2] 暴力优化 (Optimization)")
            mode = (input("请输入序号 [默认 1]: ").strip()) or "1"

            if mode == "1":
                current_conf = strategy_conf.copy()
                current_conf["default_params"] = strategy_conf["default_params"].copy()

                # 询问是否修改参数
                if input("是否修改策略参数? (y/n) [默认 n]: ").strip().lower() == "y":
                    current_conf["default_params"] = customize_params(
                        current_conf["default_params"]
                    )

                # 杠杆设置 (仅 Crypto)
                leverage = 1.0
                if "/" in symbol:
                    lev_str = input("请输入杠杆倍数 [默认 1]: ").strip()
                    leverage = float(lev_str) if lev_str else 1.0

                run_backtest(symbol, timeframe, current_conf, days, leverage=leverage)

            elif mode == "2":
                run_optimization(symbol, timeframe, strategy_conf, days)

            if input("\n还要再来一次吗? (y/n) [默认 y]: ").strip().lower() == "n":
                print("👋 Bye!")
                break

        except KeyboardInterrupt:
            print("\n👋 用户强制退出")
            break
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"\n❌ 发生错误: {e}")


if __name__ == "__main__":
    main()
