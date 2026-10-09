"""防前视：第 t 根生成的信号只能影响第 t+1 根及之后的持仓。

前视偏差是回测里最贵的一类错误（结果会好看得离谱却无法复现），
这里用一根价格跳变把边界钉死。
"""

import pandas as pd

from src.research.domains.crypto import CryptoDomain
from src.research.engine import BacktestEngine


def _engine() -> BacktestEngine:
    return BacktestEngine(
        domain=CryptoDomain(fee_rate=0.0, slippage=0.0),
        initial_capital=10_000.0,
        leverage=1.0,
    )


def test_signal_is_executed_one_bar_later() -> None:
    """价格在第 5 根翻倍，信号在第 3 根出现：持仓必须从第 4 根才生效。"""
    index = pd.date_range("2024-01-01", periods=5, freq="1D")
    df = pd.DataFrame(
        {
            "close": [100.0, 100.0, 100.0, 100.0, 200.0],
            "signal": [0.0, 0.0, 1.0, 1.0, 0.0],
        },
        index=index,
    )
    result = _engine().run(df)
    out = result["df"]

    # 信号当天（第 3 根）不得有仓位，次日（第 4 根）才建仓
    assert out["pos"].iloc[2] == 0.0
    assert out["pos"].iloc[3] == 1.0
    # 第 5 根翻倍的收益落在持仓上：净收益 ≈ +100%
    assert out["net_ret"].iloc[4] == 1.0


def test_no_position_before_first_signal() -> None:
    """首个信号之前收益必须为 0：没有仓位就不该有任何盈亏。"""
    index = pd.date_range("2024-01-01", periods=6, freq="1D")
    df = pd.DataFrame(
        {
            "close": [100.0, 120.0, 80.0, 90.0, 90.0, 90.0],
            "signal": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        },
        index=index,
    )
    out = _engine().run(df)["df"]
    assert out["net_ret"].iloc[:3].abs().sum() == 0.0
