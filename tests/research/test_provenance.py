"""运行指纹的契约：同一命令 → 同一 hash；数据/配置/代码任一环变化 → hash 变化。

这是「结果可复现」这条判据的自动化部分：没有它，指纹只是个装饰。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.research import provenance

BASE_CONFIG: dict[str, Any] = {"bars": 200, "strategy": "布林带均值回归 (Bollinger MR)"}
BASE_DATA: dict[str, Any] = {
    "bars": {"BTC/USDT": "sha-abc"},
    "manifests": {"crypto_okx_1d": "sha-def"},
}
BASE_CODE: dict[str, Any] = {"head": "deadbeef", "dirty": False}


def _record(
    config: dict[str, Any], data: dict[str, Any], code: dict[str, Any]
) -> dict[str, Any]:
    return {"mode": "backtest", "config": config, "data": data, "code": code}


def test_same_inputs_give_same_hash() -> None:
    first = provenance.repro_hash(_record(BASE_CONFIG, BASE_DATA, BASE_CODE))
    second = provenance.repro_hash(_record(BASE_CONFIG, BASE_DATA, BASE_CODE))
    assert first == second


def test_config_change_changes_hash() -> None:
    other = provenance.repro_hash(
        _record({**BASE_CONFIG, "bars": 210}, BASE_DATA, BASE_CODE)
    )
    assert other != provenance.repro_hash(_record(BASE_CONFIG, BASE_DATA, BASE_CODE))


def test_data_change_changes_hash() -> None:
    moved = {**BASE_DATA, "bars": {"BTC/USDT": "sha-CHANGED"}}
    assert provenance.repro_hash(
        _record(BASE_CONFIG, moved, BASE_CODE)
    ) != provenance.repro_hash(_record(BASE_CONFIG, BASE_DATA, BASE_CODE))


def test_code_change_changes_hash() -> None:
    dirty = {**BASE_CODE, "dirty": True}
    assert provenance.repro_hash(
        _record(BASE_CONFIG, BASE_DATA, dirty)
    ) != provenance.repro_hash(_record(BASE_CONFIG, BASE_DATA, BASE_CODE))


def test_record_run_writes_same_file_and_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同一命令重复跑：写同一个文件、得到同一个 hash（否则「可复现」无从核对）。"""
    monkeypatch.setattr(provenance, "PROVENANCE_DIR", tmp_path)
    monkeypatch.setattr(provenance, "PROJECT_ROOT", tmp_path.parent)
    monkeypatch.setattr(
        provenance, "data_fingerprint", lambda symbols, timeframe: BASE_DATA
    )
    monkeypatch.setattr(provenance, "_git_fingerprint", lambda: BASE_CODE)

    first = provenance.record_run("backtest", BASE_CONFIG, ["BTC/USDT"], "1d")
    hash_first = provenance.repro_hash(_record(BASE_CONFIG, BASE_DATA, BASE_CODE))
    second = provenance.record_run("backtest", BASE_CONFIG, ["BTC/USDT"], "1d")

    assert first == second == tmp_path / "backtest_1d_200_BTC-USDT.json"
    assert hash_first in second.read_text(encoding="utf-8")
