"""
research.py
科研控制台：问答式交互回测与参数优化
[修复版] 强制按天数切片数据
"""

import sys
from pathlib import Path

import ccxt
import pandas as pd

# 1. 路径补丁
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# 2. 导入模块
from config.settings import GLOBAL_CONFIG  # noqa: E402
from src.research.engine import BacktestEngine  # noqa: E402

# 导入策略
from src.research.strategies.vector_dual_ma import DualMAStrategy  # noqa: E402
from src.utils.data_loader import MarketDataHandler  # noqa: E402

# 3. 策略映射表
STRATEGY_MAP = {
    "1": {
        "name": "自适应双均线 (Adaptive Dual MA)",
        "class": DualMAStrategy,
        "default_params": {
            # === 趋势判断 ===
            "trend_period": 120,  # 牛熊分界线(半年线)
            # === 牛市参数 (进攻) ===
            "bull_fast": 14,
            "bull_slow": 25,
            "bull_adx_thresh": 30,
            "bull_atr_mult": 3.0,
            # === 熊市参数 (防守) ===
            "bear_fast": 20,
            "bear_slow": 75,
            "bear_adx_thresh": 100,
            "bear_atr_mult": 3.0,
            # === 通用参数 ===
            "atr_period": 14,
            "adx_period": 14,
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


def _calculate_required_bars(timeframe, days):
    """辅助函数：计算需要的 K 线根数"""
    bars_per_day = 1
    if "m" in timeframe:
        minutes = int(timeframe.replace("m", ""))
        bars_per_day = 1440 // minutes
    elif "h" in timeframe:
        hours = int(timeframe.replace("h", ""))
        bars_per_day = 24 // hours
    elif "d" in timeframe:
        bars_per_day = 1
    return days * bars_per_day


def run_backtest(symbol, timeframe, strategy_conf, days, leverage=1.0):
    """单次回测模式"""
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
    exchange = ccxt.okx(GLOBAL_CONFIG.to_ccxt_config())  # type: ignore
    loader = MarketDataHandler(exchange)

    try:
        df = loader.prepare_data(symbol, timeframe, days_back=days)
        if df is None or df.empty:
            print("   ❌ 数据为空！")
            return

        # === [核心修复] 强制切片逻辑 ===
        # 无论 Loader 返回多少数据，我们只取用户要求的最后 N 天
        required_bars = _calculate_required_bars(timeframe, days)

        # 预留一点 Buffer (比如多取 100 根) 用于计算 MA/ADX 等指标的预热，防止刚开始的信号是 NaN
        buffer_bars = 100
        total_needed = required_bars + buffer_bars

        if len(df) > total_needed:
            print(
                f"   ✂️ [自动切片] 原始数据 ({len(df)}条) -> 截取最后 {required_bars} 根K线 (+预热缓冲)"
            )
            df = df.tail(total_needed)
        else:
            print(
                f"   ⚠️ [提示] 数据量 ({len(df)}) 少于理论需求 ({total_needed})，使用全部数据。"
            )

        # 打印时间范围供确认
        print(f"   🔍 实际回测范围: {df.index[0]} ~ {df.index[-1]}")

    except Exception as e:
        print(f"❌ 数据获取失败: {e}")
        return

    print("2. 计算策略信号...")
    strategy = StrategyClass(**params)
    df = strategy.calculate_indicators(df)
    df = strategy.generate_signals(df)

    print("3. 执行回测...")
    engine = BacktestEngine(initial_capital=10000, leverage=leverage)
    result = engine.run(df)

    if not result:
        print("⚠️ 回测未生成结果 (可能是数据不足)")
        return

    print("\n" + "=" * 40)
    print("📊 回测结果摘要")
    print("=" * 40)
    bench_val = result.get("bench_ret", 0.0)
    print(f"💰 总收益率: {result['total_ret']:.2%} (基准: {bench_val:.2%})")
    print(f"📅 年化收益: {result['annual_ret']:.2%}")
    print(f"📉 最大回撤: {result['max_drawdown']:.2%}")
    print(f"⚡ 夏普比率: {result['sharpe_ratio']:.2f}")
    print(f"🔄 交易次数: {result['trade_count']}")
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
    """
    [新增] 暴力网格搜索模式
    """
    StrategyClass = strategy_conf["class"]
    strategy_name = strategy_conf["name"]

    print(f"\n{'=' * 60}")
    print(f"🔥 启动暴力优化 | 策略: {strategy_name} | 标的: {symbol}")
    print(f"📅 优化窗口: 过去 {days} 天")
    print(f"{'=' * 60}")

    # 1. 获取数据
    print("1. 正在获取数据...")
    exchange = ccxt.okx(GLOBAL_CONFIG.to_ccxt_config())  # type: ignore
    loader = MarketDataHandler(exchange)

    df = loader.prepare_data(symbol, timeframe, days_back=days)
    if df is None or df.empty:
        print("❌ 数据为空")
        return

    # === [核心修复] 优化模式也要切片 ===
    required_bars = _calculate_required_bars(timeframe, days)
    # 优化需要更多数据来确保统计显著性，但也得遵守用户的时间窗口
    if len(df) > required_bars:
        print(f"   ✂️ [自动切片] 截取最后 {days} 天用于优化...")
        df = df.tail(required_bars)

    # 2. 定义搜索空间
    print("\n2. 定义搜索参数范围 (回车使用默认推荐值)...")

    try:
        f_start = input("   - 快线起始值 [默认 5]: ").strip() or 5
        f_end = input("   - 快线结束值 [默认 30]: ").strip() or 30
        fast_range = range(int(f_start), int(f_end), 3)

        s_start = input("   - 慢线起始值 [默认 20]: ").strip() or 20
        s_end = input("   - 慢线结束值 [默认 80]: ").strip() or 80
        slow_range = range(int(s_start), int(s_end), 5)

        atr_mult_range = [3.0, 4.0, 5.0, 6.0]
        adx_thresh_range = [20, 25, 30]

    except ValueError:
        print("❌ 输入格式错误，使用默认范围...")
        fast_range = range(5, 30, 3)
        slow_range = range(20, 80, 5)
        atr_mult_range = [3.0, 5.0]
        adx_thresh_range = [20, 30]

    total_combs = (
        len(fast_range) * len(slow_range) * len(atr_mult_range) * len(adx_thresh_range)
    )
    print(f"\n⚡ 准备测试 {total_combs} 种参数组合，这可能需要几十秒...")

    # 3. 运行优化
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
    )

    # 4. 输出结果
    print("\n" + "=" * 40)
    print("🏆 优化完成！历史最优参数如下：")
    print("=" * 40)
    print(f"✅ 夏普比率 (Sharpe): {best_params.get('sharpe', 0):.2f}")

    print("\n📝 推荐配置:")
    for key, val in best_params.items():
        if key not in ["sharpe", "net_ret"]:
            print(f"  - {key}: {val}")

    print("\n💡 建议：请记下这组参数，然后使用 [1] 单次回测 模式，")
    print("   输入这组参数，看看具体的资金曲线长什么样。")


def main():
    print("🚀 欢迎来到量化实验室 (Simple CLI)")
    print("-" * 40)

    while True:
        try:
            # 1. 基础设置
            symbol = (
                input("\n请输入交易对 (例如 BTC/USDT) [默认 BTC/USDT]: ")
                .strip()
                .upper()
            ) or "BTC/USDT"
            timeframe = (
                input("请输入K线周期 (例如 1h, 4h, 1d) [默认 4h]: ").strip().lower()
            ) or "4h"
            days_str = input("回测天数 [默认 365]: ").strip()
            days = int(days_str) if days_str else 365

            # 2. 选择策略
            print("\n请选择策略:")
            for key, val in STRATEGY_MAP.items():
                print(f"  [{key}] {val['name']}")
            strat_choice = (input("请输入序号 [默认 1]: ").strip()) or "1"
            strategy_conf = STRATEGY_MAP.get(strat_choice)

            if not strategy_conf:
                print("❌ 无效的策略序号")
                continue

            # 3. 选择模式 (回测 vs 优化)
            print("\n请选择模式:")
            print("  [1] 单次回测 (Backtest) - 看详情")
            print("  [2] 暴力优化 (Optimization) - 找参数")

            mode = (input("请输入序号 [默认 1]: ").strip()) or "1"

            if mode == "1":
                # === 单次回测流程 ===
                current_conf = strategy_conf.copy()
                current_conf["default_params"] = strategy_conf["default_params"].copy()

                if input("是否修改策略参数? (y/n) [默认 n]: ").strip().lower() == "y":
                    current_conf["default_params"] = customize_params(
                        current_conf["default_params"]
                    )

                lev_str = input("请输入杠杆倍数 (例如 1, 2, 3) [默认 1]: ").strip()
                leverage = float(lev_str) if lev_str else 1.0
                if leverage <= 0:
                    leverage = 1.0

                run_backtest(symbol, timeframe, current_conf, days, leverage=leverage)

            elif mode == "2":
                # === 暴力优化流程 ===
                run_optimization(symbol, timeframe, strategy_conf, days)

            # 继续吗？
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
