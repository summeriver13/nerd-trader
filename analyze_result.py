"""
analyze_result.py
读取回测生成的 CSV，计算资金曲线和绩效指标
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# 1. 解析参数：初始资金必须与引擎配置一致，否则收益率与回撤会失真
parser = argparse.ArgumentParser(
    description="从 data/results 最新成交 CSV 估算资金曲线与绩效"
)
parser.add_argument(
    "--capital",
    type=float,
    default=1_000_000.0,
    help="初始资金，默认 1000000 与 run.py 的引擎配置一致",
)
args = parser.parse_args()

# 2. 自动查找最新的结果文件
results_dir = Path("data/results")
# 按修改时间排序，找最新的 csv
csv_files = sorted(results_dir.glob("trades_*.csv"), key=lambda x: x.stat().st_mtime)

if not csv_files:
    print("❌ 没有找到成交记录文件")
    exit()

latest_csv = csv_files[-1]
print(f"Loading: {latest_csv}")

# 2. 读取数据
df = pd.read_csv(latest_csv)
df["datetime"] = pd.to_datetime(df["datetime"])

# 3. 计算盈亏 (简化版：逐笔对冲计算)
initial_balance = args.capital
df["pnl"] = 0.0
df["balance"] = initial_balance

# 简单的 FIFO 盈亏计算逻辑
# (注意：这只是一个近似估算，严谨的需要完整的 Account 模块)
position = 0
avg_price = 0.0
balance = initial_balance
pnl_history = []

print("\n>>> 开始计算资金曲线...")

for index, row in df.iterrows():
    price = row["price"]
    vol = row["volume"]
    direction = row["direction"]  # LONG / SHORT
    offset = row["offset"]  # OPEN / CLOSE

    trade_pnl = 0.0

    if offset == "OPEN":
        # 开仓：更新持仓均价
        if direction == "LONG":
            # 多头开仓
            total_val = (position * avg_price) + (vol * price)
            position += vol
            avg_price = total_val / position if position != 0 else 0
        else:
            # 空头开仓 (反向)
            total_val = (position * avg_price) - (vol * price)
            position -= vol  # 空头记为负数
            avg_price = abs(total_val / position) if position != 0 else 0

    elif offset == "CLOSE":
        # 平仓：计算盈亏
        if direction == "SHORT":
            # 卖出平多 (原持仓是正的)
            trade_pnl = (price - avg_price) * vol
            position -= vol
        elif direction == "LONG":
            # 买入平空 (原持仓是负的)
            trade_pnl = (avg_price - price) * vol
            position += vol

        balance += trade_pnl

    pnl_history.append(balance)

df["balance"] = pnl_history

# 4. 计算核心指标
total_return = (df["balance"].iloc[-1] - initial_balance) / initial_balance * 100
max_balance = df["balance"].cummax()
drawdown = (df["balance"] - max_balance) / max_balance * 100
max_drawdown = drawdown.min()

print("-" * 30)
print(f"初始资金: {initial_balance:,.2f}")
print(f"最终资金: {df['balance'].iloc[-1]:,.2f}")
print(f"总收益率: {total_return:.2f}%")
print(f"最大回撤: {max_drawdown:.2f}%")
print(f"交易笔数: {len(df)}")
print("-" * 30)

# 5. 画图
plt.figure(figsize=(12, 8))

# 子图1：资金曲线
plt.subplot(2, 1, 1)
plt.plot(df["datetime"], df["balance"], label="Total Equity", color="orange")
plt.title(f"Backtest Result: {total_return:.2f}% Return")
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2：成交点位分析 (如果有行情数据更好，这里只画成交价)
plt.subplot(2, 1, 2)
plt.plot(
    df["datetime"],
    df["price"],
    label="Trade Price",
    marker=".",
    linestyle=" ",
    alpha=0.5,
)
plt.title("Trade Execution Points")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
