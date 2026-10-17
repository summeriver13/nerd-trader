"""试验账本（诚实计数 N）的单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.research.validation.trials import (
    ORIGIN_BACKFILL,
    ORIGIN_RUN,
    Trial,
    TrialLedger,
    honest_trials,
)


def _trial(
    study: str = "factor_sweep",
    market: str = "spot",
    kind: str = "mom",
    params: dict[str, object] | None = None,
    origin: str = ORIGIN_RUN,
) -> Trial:
    return Trial(
        study=study,
        market=market,
        timeframe="1d",
        kind=kind,
        params=params if params is not None else {"window": 30, "horizon": 1},
        origin=origin,
    )


@pytest.fixture
def ledger(tmp_path: Path) -> TrialLedger:
    """空账本。用 tmp_path 隔离，避免测试写进仓库的 data/results/。"""
    return TrialLedger(tmp_path / "trials.jsonl")


def test_missing_ledger_counts_zero(ledger: TrialLedger) -> None:
    """没建账不等于没试过：返回 0，由调用方去报「下界」。"""
    assert ledger.count() == 0
    assert ledger.records() == []
    assert ledger.breakdown().empty


def test_record_is_idempotent(ledger: TrialLedger) -> None:
    """同一配置记两次只算一次试验——这是「重复记录不放大 N」的保证。"""
    assert ledger.record([_trial()]) == 1
    assert ledger.record([_trial()]) == 0
    assert ledger.count() == 1


def test_params_order_does_not_change_identity() -> None:
    first = _trial(params={"window": 30, "horizon": 1})
    second = _trial(params={"horizon": 1, "window": 30})
    assert first.identity == second.identity


def test_horizon_is_a_dimension_not_a_detail(ledger: TrialLedger) -> None:
    """换 horizon 是另一次试验。只数因子会把 N 数小、把门槛放低。"""
    ledger.record([_trial(params={"window": 30, "horizon": 1})])
    ledger.record([_trial(params={"window": 30, "horizon": 2})])
    assert ledger.count() == 2


def test_numpy_scalars_are_serializable_and_stable(ledger: TrialLedger) -> None:
    """从 DataFrame 取出来的参数常是 np.int64：不可序列化会让账本写不出去。"""
    numpy_trial = _trial(params={"window": np.int64(30), "horizon": np.int64(1)})
    python_trial = _trial(params={"window": 30, "horizon": 1})
    assert numpy_trial.identity == python_trial.identity
    assert ledger.record([numpy_trial]) == 1
    assert ledger.record([python_trial]) == 0


def test_float_noise_does_not_split_identity() -> None:
    """0.1 + 0.2 != 0.3——不做归一就会把一次试验记成两次。"""
    noisy = _trial(params={"threshold": 0.1 + 0.2})
    clean = _trial(params={"threshold": 0.3})
    assert noisy.identity == clean.identity


def test_ledger_only_appends(ledger: TrialLedger) -> None:
    """append-only 是实质：新记录不重写旧行（旧行的字面量必须原样还在）。"""
    ledger.record([_trial(kind="mom")])
    original = ledger.path.read_text(encoding="utf-8")
    ledger.record([_trial(kind="vol")])
    updated = ledger.path.read_text(encoding="utf-8")
    assert updated.startswith(original)
    assert len(updated.splitlines()) == 2


def test_count_filters_by_scope(ledger: TrialLedger) -> None:
    ledger.record(
        [
            _trial(market="spot"),
            _trial(market="swap"),
            _trial(market="swap", kind="vol"),
            _trial(study="sadf", market="swap"),
        ]
    )
    assert ledger.count() == 4
    assert ledger.count(market="swap") == 3
    assert ledger.count(study="sadf") == 1
    assert ledger.count(study="factor_sweep", market="swap") == 2
    assert ledger.count(timeframe="1h") == 0


def test_origin_separates_backfill_from_live(ledger: TrialLedger) -> None:
    """补录与实时记录必须能分开数：前者是重建出来的下界，后者是实测。"""
    ledger.record([_trial(kind="mom")])
    ledger.record([_trial(kind="vol", origin=ORIGIN_BACKFILL)])
    assert ledger.count() == 2
    assert ledger.count(origin=ORIGIN_RUN) == 1
    assert ledger.count(origin=ORIGIN_BACKFILL) == 1


def test_breakdown_sums_to_total_and_sorts(ledger: TrialLedger) -> None:
    ledger.record(
        [
            _trial(kind="mom"),
            _trial(kind="mom", params={"window": 7}),
            _trial(market="swap"),
        ]
    )
    table = ledger.breakdown()
    assert list(table.columns) == ["study", "market", "trials"]
    assert int(table["trials"].sum()) == ledger.count()
    assert int(table["trials"].iloc[0]) == 2  # 最大的排前面


def test_record_round_trips_through_json(ledger: TrialLedger) -> None:
    ledger.record([_trial(kind="sadf", params={"min_window": 60, "lags": 1})])
    line = ledger.path.read_text(encoding="utf-8").strip()
    record = json.loads(line)
    restored = Trial.from_record(record)
    assert restored == _trial(kind="sadf", params={"min_window": 60, "lags": 1})
    assert restored.params_canonical == {"lags": 1, "min_window": 60}


def test_honest_trials_reads_the_real_ledger_path() -> None:
    """默认路径必须是仓库的账本，而不是临时文件——否则接线会静默接空。"""
    assert TrialLedger().path.name == "trials.jsonl"
    assert honest_trials() == TrialLedger().count()
