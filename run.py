"""
run.py
系统启动入口：事件驱动模拟盘 demo（离线，缺数据自动生成）
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import CACHE_ROOT
from src.simulation.engine import SimulationEngine
from src.strategies.dual_ma import DualMaStrategy
from src.utils.console import force_utf8_stdout, setup_logging

# === demo 配置（测试数据与策略参数都从这里取，避免两处各写一份）===
SYMBOL = "BTC-USDT"
START_DATE = "2024-01-01"
DATA_FILE = CACHE_ROOT / f"{SYMBOL.replace('-', '_')}_1m_test.parquet"
INITIAL_CAPITAL = 1_000_000.0
SEED = 42

# 演示仓位：1000 份约 10 万名义额（初始资金的 10%），1 倍杠杆、不含手续费。
# 不是有依据的仓位选择——1 份的收益/回撤会被本金摊成 0.00%，看不出 demo 在做什么。
FIXED_SIZE = 1000.0
FAST_WINDOW = 10
SLOW_WINDOW = 30


def generate_dummy_data(path: Path) -> None:
    """
    [辅助] 如果没数据，自动生成一点 1分钟 K线数据用于测试
    生成一段正弦波走势，确保肯定能触发金叉死叉；固定随机种子便于结果对照
    """
    print(f"⚠️ 未找到数据文件，正在生成测试数据: {path}")

    dates = pd.date_range(start=START_DATE, periods=1000, freq="1min")
    x = np.linspace(0, 4 * np.pi, 1000)
    # 制造一个明显的波动，让均线交叉
    prices = 100 + 10 * np.sin(x) + np.random.default_rng(SEED).normal(0, 0.1, 1000)

    df = pd.DataFrame(
        {
            "datetime": dates,
            "symbol": SYMBOL,
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": 100.0,
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    print("✅ 测试数据生成完毕")


def main() -> None:
    force_utf8_stdout()  # Windows 重定向输出 emoji 会 GBK 报错，先转 utf-8
    setup_logging()  # 日志出口（data_loader 的取数诊断走 logger）

    # 1. 检查数据
    if not DATA_FILE.exists():
        generate_dummy_data(DATA_FILE)

    # 2. 初始化模拟引擎
    print(">>> 正在初始化模拟引擎...")
    engine = SimulationEngine(
        data_path=str(DATA_FILE), symbols=SYMBOL, initial_capital=INITIAL_CAPITAL
    )

    # 3. 策略配置：1 分钟线，双均线 10/30
    strategy_settings = {
        "fast_window": FAST_WINDOW,
        "slow_window": SLOW_WINDOW,
        "fixed_size": FIXED_SIZE,
    }

    # 4. 加载策略
    engine.load_strategy(DualMaStrategy, strategy_settings)

    # 5. 启动回放
    print("\n>>> 🔥 3... 2... 1... 启动回测！\n")
    engine.run()


if __name__ == "__main__":
    main()
