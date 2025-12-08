"""
research.py
科研控制台：问答式交互回测
使用方法：直接运行 python research.py，然后根据提示输入
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

# 3. 策略映射表 (已更新 ADX 参数)
STRATEGY_MAP = {
    "1": {
        "name": "双均线策略 (Dual MA + ADX)",
        "class": DualMAStrategy,
        "default_params": {
            "fast_period": 9,
            "slow_period": 30,
            "atr_multiplier": 5.0,
            "adx_threshold": 25,  # 新增：默认 ADX 阈值
            "adx_period": 14,  # 新增：默认 ADX 周期
        },
    },
}


def customize_params(default_params: dict) -> dict:
    """
    交互式修改参数
    自动根据默认值的类型 (int/float) 转换输入
    """
    print("\n⚙️  参数设置 (直接回车保持默认):")
    new_params = default_params.copy()

    for key, default_val in default_params.items():
        while True:
            try:
                user_input = input(f"   - {key} [默认 {default_val}]: ").strip()

                # 如果用户直接回车，使用默认值
                if not user_input:
                    break

                # 智能类型转换
                target_type = type(default_val)
                new_params[key] = target_type(user_input)
                break  # 输入有效，跳出循环

            except ValueError:
                print(f"   ❌ 格式错误，请输入有效的 {type(default_val).__name__}")

    return new_params


def run_backtest(symbol, timeframe, strategy_conf, days):
    StrategyClass = strategy_conf["class"]
    params = strategy_conf["default_params"]
    strategy_name = strategy_conf["name"]

    print(f"\n{'=' * 60}")
    print(f"🧪 启动回测 | 策略: {strategy_name} | 标的: {symbol} | 周期: {timeframe}")
    print(f"⚙️  当前参数: {params}")
    print(f"{'=' * 60}")

    # --- 数据准备 ---
    print("1. 正在获取数据...")
    exchange = ccxt.okx(GLOBAL_CONFIG.to_ccxt_config())  # type: ignore

    loader = MarketDataHandler(exchange)
    try:
        df = loader.prepare_data(symbol, timeframe, days_back=days)

        if df is not None and not df.empty:
            # === [新增] 1. 计算理论需要的 K 线数量 ===
            # 解析周期字符串 (e.g. '4h', '15m', '1d')
            if "m" in timeframe:
                minutes = int(timeframe.replace("m", ""))
                bars_per_day = 1440 // minutes
            elif "h" in timeframe:
                hours = int(timeframe.replace("h", ""))
                bars_per_day = 24 // hours
            elif "d" in timeframe:
                bars_per_day = 1
            else:
                bars_per_day = 1  # 默认兜底

            required_bars = days * bars_per_day

            # === [新增] 2. 执行切片逻辑 ===
            # 如果当前数据量比需要的量多出 10% 以上，说明读取的是旧的长缓存
            # 我们强制截取最后 N 条，保证回测窗口准确
            if len(df) > required_bars * 1.1:
                print(
                    f"   ✂️ [自动切片] 缓存数据 ({len(df)}) 多于请求 ({required_bars})，正在截取最后 {days} 天..."
                )
                df = df.tail(required_bars)

            # === [修改] 3. Debug 信息输出 (此时 df 已经是切片后的了) ===
            start_ts = pd.Timestamp(str(df.index[0]))
            end_ts = pd.Timestamp(str(df.index[-1]))

            start_str = str(start_ts)[:10]
            end_str = str(end_ts)[:10]

            print(
                f"   🔍 [数据检查] 范围: {start_str} ~ {end_str} | 总K线数: {len(df)}"
            )

            # 再次检查数据量是否过少 (防止数据本身就不够)
            if len(df) < required_bars * 0.9:
                print(
                    f"   ⚠️ [警告] 数据量 ({len(df)}) 少于预期 ({required_bars})，可能是历史数据不足。"
                )

        else:
            print("   ❌ 数据为空！")
            return

    except Exception as e:
        print(f"❌ 数据获取失败: {e}")
        import traceback

        traceback.print_exc()
        return

    # --- 策略计算 ---
    print("2. 计算策略信号...")
    strategy = StrategyClass(**params)
    df = strategy.calculate_indicators(df)
    df = strategy.generate_signals(df)

    # --- 引擎运行 ---
    print("3. 执行回测...")
    engine = BacktestEngine(initial_capital=10000)
    result = engine.run(df)

    if not result:
        print("⚠️ 回测未生成结果 (可能是数据不足)")
        return

    # --- 结果输出 ---
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

    # --- 绘图 ---
    plot_choice = input("是否生成图表? (y/n) [默认 y]: ").strip().lower()
    if plot_choice != "n":
        # 文件名带上参数特征
        param_vals = [str(v) for v in params.values()]
        param_str = "_".join(param_vals)
        save_name = f"{symbol.replace('/', '_')}_{timeframe}_{param_str}.png"

        save_path = PROJECT_ROOT / "data" / "results" / save_name
        save_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"🖼️  正在绘图 -> {save_path}")
        engine.plot(result, title=f"{symbol} {timeframe}", save_path=str(save_path))


def main():
    print("🚀 欢迎来到量化实验室 (Simple CLI)")
    print("-" * 40)

    while True:
        try:
            # 1. 选择币种
            symbol = (
                input("\n请输入交易对 (例如 BTC/USDT) [默认 BTC/USDT]: ")
                .strip()
                .upper()
            )
            if not symbol:
                symbol = "BTC/USDT"

            # 2. 选择周期
            timeframe = (
                input("请输入K线周期 (例如 1h, 4h, 1d) [默认 4h]: ").strip().lower()
            )
            if not timeframe:
                timeframe = "4h"

            # 3. 选择天数
            days_str = input("回测天数 [默认 365]: ").strip()
            days = int(days_str) if days_str else 365

            # 4. 选择策略
            print("\n请选择策略:")
            for key, val in STRATEGY_MAP.items():
                print(f"  [{key}] {val['name']}")

            strat_choice = input("请输入序号 [默认 1]: ").strip()
            if not strat_choice:
                strat_choice = "1"

            strategy_conf = STRATEGY_MAP.get(strat_choice)
            if not strategy_conf:
                print("❌ 无效的策略序号")
                continue

            # 参数自定义流程
            current_conf = strategy_conf.copy()
            # 必须深拷贝 params 字典，否则会修改全局默认值
            current_conf["default_params"] = strategy_conf["default_params"].copy()

            need_custom = input("是否修改策略参数? (y/n) [默认 n]: ").strip().lower()
            if need_custom == "y":
                new_params = customize_params(current_conf["default_params"])
                current_conf["default_params"] = new_params

            # 运行回测
            run_backtest(symbol, timeframe, current_conf, days)

            # 继续吗？
            cont = input("\n还要再测一次吗? (y/n) [默认 y]: ").strip().lower()
            if cont == "n":
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
