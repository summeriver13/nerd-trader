"""
Deribit 期权链 raw 层：每日（或每轮）快照落 JSONL。

    data/storage/deribit/raw/options/<YYYY-MM-DD>.jsonl
    data/storage/deribit/raw/futures/<YYYY-MM-DD>.jsonl

业务键是 `(instrument, fetched_at)`：一天可多轮快照、各留一行，重复运行不产生重复行。
落盘字段只保留定价要用的东西——期权是 mark/IV/买卖价/持仓，期货只有 mark。
期货的 bid/ask/oi 现有链路一个也不消费（carry 只用 mark 推远期），故不入库；
将来真要用，后续快照自然会带上，不必现在猜。

通用存储逻辑（去重追加、manifest）在 `src/utils/jsonl_store.py`，
manifest 落 `data/manifests/deribit_raw_<dataset>.json`（容器 ADR-0006 自持要求）。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.config.paths import (
    DERIBIT_DATASETS,
    DERIBIT_FUTURES,
    DERIBIT_OPTIONS,
    DERIBIT_RAW_ROOT,
    MANIFEST_ROOT,
    deribit_dataset_dir,
    deribit_jsonl_path,
)
from src.utils.data_sources.deribit import DeribitChain, is_expiry_code
from src.utils.jsonl_store import append_jsonl, day_stamp, rewrite_manifest

SOURCE = "www.deribit.com"

# 各数据集的去重业务键与覆盖区间时间字段
KEY_FIELDS: dict[str, tuple[str, ...]] = {
    DERIBIT_OPTIONS: ("instrument", "fetched_at"),
    DERIBIT_FUTURES: ("instrument", "fetched_at"),
}
TIME_FIELDS: dict[str, str | None] = {
    DERIBIT_OPTIONS: "fetched_at",
    DERIBIT_FUTURES: "fetched_at",
}


def option_rows(chain: DeribitChain) -> list[dict[str, Any]]:
    """期权链 → 行：每行自带指数价（定价锚点），下游不必再取一次。

    mark_price ≤ 0 的档在采集侧丢掉：近到期无人报价的档上游会回 0（不是脏数据，
    但零价与零 IV 对定价无用），入库后反而会让验收闸每次都报。
    """
    stamp = _stamp(chain)
    return [
        {
            "instrument": quote.instrument,
            "expiry": quote.expiry,
            "kind": quote.kind,
            "strike": quote.strike,
            "mark_price": quote.mark_price,
            "mark_iv": quote.mark_iv,
            "bid": quote.bid,
            "ask": quote.ask,
            "open_interest": quote.open_interest,
            "underlying": quote.underlying,
            "currency": chain.currency,
            "index_price": chain.index_price,
            "fetched_at": stamp,
        }
        for quote in chain.options
        if quote.mark_price > 0
    ]


def future_rows(chain: DeribitChain) -> list[dict[str, Any]]:
    """期货 → 行；`expiry` 为到期码，永续（PERPETUAL）记空以区别于到期合约。"""
    stamp = _stamp(chain)
    return [
        {
            "instrument": name,
            "expiry": _future_expiry(name),
            "mark_price": mark,
            "currency": chain.currency,
            "index_price": chain.index_price,
            "fetched_at": stamp,
        }
        for name, mark in sorted(chain.futures.items())
        if mark > 0
    ]


def append_chain(
    chain: DeribitChain,
    *,
    day: str | None = None,
    root: Path = DERIBIT_RAW_ROOT,
) -> dict[str, tuple[int, int]]:
    """把整条链追加进当日文件，返回每个数据集的 (新增, 跳过)。

    分区日默认取**快照自己的时刻**（`chain.fetched_at`，UTC），不取墙上时钟：跨 UTC
    午夜重跑（或把历史快照补进来）时按运行时刻分区，会把数据落到错误的文件里。
    """
    name = day or day_stamp(chain.fetched_at.to_pydatetime())
    payloads: Mapping[str, list[dict[str, Any]]] = {
        DERIBIT_OPTIONS: option_rows(chain),
        DERIBIT_FUTURES: future_rows(chain),
    }
    return {
        dataset: append_jsonl(deribit_jsonl_path(dataset, name, root=root), rows, KEY_FIELDS[dataset])
        for dataset, rows in payloads.items()
    }


def update_manifest(
    dataset: str,
    *,
    touched: Iterable[str] = (),
    raw_root: Path = DERIBIT_RAW_ROOT,
    manifest_root: Path = MANIFEST_ROOT,
) -> Path:
    """重写该数据集 manifest；只对本次触及的文件重算，其余沿用旧条目。"""
    if dataset not in DERIBIT_DATASETS:
        raise ValueError(f"未知数据集 {dataset!r}，可选 {DERIBIT_DATASETS}")
    return rewrite_manifest(
        f"deribit_raw_{dataset}",
        directory=deribit_dataset_dir(dataset, root=raw_root),
        source=SOURCE,
        time_field=TIME_FIELDS.get(dataset),
        touched=touched,
        manifest_root=manifest_root,
    )


def _stamp(chain: DeribitChain) -> str:
    """快照时刻统一写成 ISO（business key 与覆盖区间都靠它）。"""
    return chain.fetched_at.tz_convert("UTC").isoformat()


def _future_expiry(instrument: str) -> str | None:
    """期货名的到期码；永续（BTC-PERPETUAL）返回 None，区别于到期合约。"""
    code = instrument.rsplit("-", maxsplit=1)[-1]
    return code if is_expiry_code(code) else None
