"""
src/research/optimize.py
参数优化实验室：通过网格搜索 (Grid Search) 寻找最优参数组合
"""

import itertools
import sys
import time
from pathlib import Path

import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. 路径补丁
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# 2. 导入模块
from config.settings import GLOBAL_CONFIG  # noqa: E402
from src.research.engine import BacktestEngine  # noqa: E402
from src.research.strategies.vector_dual_ma import DualMAStrategy  # noqa: E402
from src.utils.data_loader import MarketDataHandler  # noqa: E402


def run_grid_search(symbol="BTC/USDT", timeframe="4h", days=365):
    print(f"\n🚀 启动参数优化 | 标的: {symbol} | 周期: {timeframe} | 范围: {days}天")

    # --- A. 准备数据 (只读一次) ---
    print("1. 加载数据...")
    exchange = ccxt.okx(GLOBAL_CONFIG.to_ccxt_config())  # type: ignore
    loader = MarketDataHandler(exchange)
    try:
        df_raw = loader.prepare_data(symbol, timeframe, days_back=days)
    except Exception as e:
        print(f"❌ 数据获取失败: {e}")
        return

    # --- B. 定义参数网格 ---
    # 快线: 5 到 20, 步长 2 (5, 7, 9 ... 19)
    fast_range = range(5, 25, 2)
    # 慢线: 20 到 90, 步长 5 (20, 25, 30 ... 85)
    slow_range = range(20, 100, 5)

    # 生成所有组合 (Cartesian Product)
    combinations = list(itertools.product(fast_range, slow_range))
    # 过滤掉非法组合 (快线 >= 慢线)
    combinations = [(f, s) for f, s in combinations if f < s]

    print(f"2. 开始扫描 {len(combinations)} 组参数组合...")
    print("-" * 40)

    results = []
    start_time = time.time()

    # --- C. 暴力循环 ---
    # 初始化回测引擎 (资金不重要，主要看比率)
    engine = BacktestEngine(initial_capital=10000)

    for i, (fast, slow) in enumerate(combinations):
        # 实例化策略
        # 注意：这里我们只优化周期，ATR倍数暂时固定为 5.0
        strategy = DualMAStrategy(
            fast_period=fast, slow_period=slow, atr_multiplier=5.0
        )

        # 计算
        # 注意：每次都要基于 df_raw 的副本计算，防止污染
        df_test = strategy.calculate_indicators(df_raw.copy())
        df_test = strategy.generate_signals(df_test)

        # 回测
        res = engine.run(df_test)

        # 记录结果
        results.append(
            {
                "fast": fast,
                "slow": slow,
                "total_ret": res["total_ret"],
                "sharpe": res["sharpe_ratio"],
                "max_dd": res["max_drawdown"],
                "trades": res["trade_count"],
            }
        )

        # 简易进度条
        if (i + 1) % 10 == 0:
            sys.stdout.write(f"\r⚡ 已处理: {i + 1}/{len(combinations)}")
            sys.stdout.flush()

    elapsed = time.time() - start_time
    print(f"\n✅ 扫描完成! 耗时: {elapsed:.2f}秒")

    # --- D. 分析结果 ---
    df_res = pd.DataFrame(results)

    # 按照 夏普比率 排序
    df_best = df_res.sort_values(by="sharpe", ascending=False).head(10)

    print("\n🏆 Top 10 最优参数组合 (按夏普比率):")
    print(df_best.to_markdown(index=False, floatfmt=".2f"))

    # --- E. 可视化 (热力图) ---
    plot_heatmap(df_res, symbol, timeframe)


def plot_heatmap(df_res, symbol, timeframe):
    """
    绘制参数热力图: X轴=Slow, Y轴=Fast, 颜色=Sharpe
    """
    # 转换数据格式为矩阵 (Pivot Table)
    pivot_table = df_res.pivot(index="fast", columns="slow", values="sharpe")

    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")

    # 绘制热力图
    # cmap="RdYlGn" (红-黄-绿)
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        cbar_kws={"label": "Sharpe Ratio"},
    )

    plt.title(f"Parameter Heatmap (Sharpe) - {symbol} {timeframe}")
    plt.ylabel("Fast Period")
    plt.xlabel("Slow Period")
    # 翻转Y轴，让小的在下面，符合直觉
    plt.gca().invert_yaxis()

    # 保存图片
    save_path = PROJECT_ROOT / f"data/results/{symbol.replace('/', '_')}_heatmap.png"
    plt.savefig(save_path)
    print(f"\n🖼️  热力图已保存: {save_path}")
    # plt.show() # 如果是在服务器跑，注释掉这就行


if __name__ == "__main__":
    # 你可以在这里手动修改想跑的币种
    target_symbol = "BTC/USDT"
    target_tf = "4h"

    try:
        run_grid_search(target_symbol, target_tf, days=365)
    except KeyboardInterrupt:
        print("\n👋 停止优化")
