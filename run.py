"""
run.py
系统启动入口
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 1. 确保项目根目录在 Python 路径中
# 这样才能正常 import src.xxx
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# 2. 导入模块 (加上 noqa: E402 忽略 "不在文件头" 的警告)
from src.simulation.engine import SimulationEngine  # noqa: E402
from src.strategies.dual_ma import DualMaStrategy  # noqa: E402


def generate_dummy_data(path: str):
    """
    [辅助] 如果没数据，自动生成一点 1分钟 K线数据用于测试
    生成一段正弦波走势，确保肯定能触发金叉死叉
    """
    print(f"⚠️ 未找到数据文件，正在生成测试数据: {path}")

    dates = pd.date_range(start="2024-01-01", periods=1000, freq="1min")
    x = np.linspace(0, 4 * np.pi, 1000)
    # 制造一个明显的波动，让均线交叉
    prices = 100 + 10 * np.sin(x) + np.random.normal(0, 0.1, 1000)

    df = pd.DataFrame(
        {
            "datetime": dates,
            "symbol": "BTC-USDT",
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": 100.0,
        }
    )

    # 确保目录存在
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    print("✅ 测试数据生成完毕")


def main():
    # --- 配置区域 ---
    DATA_FILE = "data/cache/BTC_USDT_1m_test.parquet"
    SYMBOL = "BTC-USDT"

    # 1. 检查数据
    if not Path(DATA_FILE).exists():
        generate_dummy_data(DATA_FILE)

    # 2. 初始化模拟引擎
    print(">>> 正在初始化模拟引擎...")
    engine = SimulationEngine(
        start_date="2024-01-01", data_path=DATA_FILE, symbol=SYMBOL
    )

    # 3. 策略配置
    # 注意：我们的 Parquet 是 1分钟线
    # 这里的 window=5 代表 5根 1分钟线 (即5分钟均线)
    strategy_settings = {"fast_window": 10, "slow_window": 30, "fixed_size": 1.0}

    # 4. 加载策略
    engine.load_strategy(DualMaStrategy, strategy_settings)

    # 5. 启动回放
    print("\n>>> 🔥 3... 2... 1... 启动回测！\n")
    engine.run()


if __name__ == "__main__":
    main()
