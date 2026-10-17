"""
src/utils/liquidation_store.py
OKX 强平事件流的 raw 层：按事件时间分日追加、跨轮次去重、缺口记录。

    data/storage/crypto/okx/liquidation/<YYYY-MM-DD>.jsonl
    data/storage/crypto/okx/liquidation/gaps/<YYYY-MM-DD>.jsonl
    data/manifests/crypto_liquidation.json

三条口径：

- **分区日取事件自身的 `ts`（UTC）**，不取墙上时钟：补跑/重放时历史事件仍落在它
  原本的日期文件里，日文件之间的边界因此是稳定的。
- **业务键是 `(instId, ts, side, posSide, bkPx, sz)`**：上游不给订单 id，同一毫秒
  同一方向的多笔强平只能靠价格与张数区分，缺一个字段就会把两笔并成一笔。
- **缺口与事件分开存**：缺口行的 schema 完全不同（prevTs/nextTs/lostMs），混在
  事件文件里会让「一行一条强平成交」这条契约失效，验收闸也就没法直接用。

`sizeCcy = sz × ctVal`、`notional = sizeCcy × bkPx`；`sz` 原样保留字符串，保证即使
将来对 `ctVal` 的口径判断有误，也能从原始字段重算（raw 只增不改，改不了）。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config.paths import (
    CRYPTO_ROOT,
    MANIFEST_ROOT,
    liquidation_dir,
    liquidation_gap_path,
    liquidation_jsonl_path,
)
from src.utils.data_sources.okx_liquidation import LiquidationEvent
from src.utils.jsonl_store import append_jsonl, read_jsonl, rewrite_manifest

SOURCE = "www.okx.com"
DATASET = "crypto_liquidation"
TIME_FIELD = "ts"
# `ts` 是毫秒（raw 保真，强平簇的时序就差在这几百毫秒上），而 jsonl_store 的覆盖
# 区间契约是 epoch 秒，所以 manifest 走 ISO 字段：直接喂毫秒会在 Windows 抛
# OSError（fromtimestamp 越界），在 Linux 则静默算成公元 5 万年。
TIME_ISO_FIELD = "tsIso"
KEY_FIELDS = ("instId", "ts", "side", "posSide", "bkPx", "sz")
GAP_KEY_FIELDS = ("instId", "prevTs", "nextTs")


def iso_of_ms(ts_ms: int) -> str:
    """毫秒时间戳 → ISO 8601（UTC），只为人读方便，分区与排序仍用毫秒字段。"""
    moment = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def day_of_ms(ts_ms: int) -> str:
    """事件时间所属的 UTC 日期。"""
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).strftime("%Y-%m-%d")


def event_row(event: LiquidationEvent, fetched_at: str) -> dict[str, Any]:
    """事件 → 落盘行：原始张数、换算面值、基础币数量与名义都留，方便事后复核。"""
    return {
        "instId": event.inst_id,
        "instFamily": event.inst_family,
        "side": event.side,
        "posSide": event.pos_side,
        "bkPx": event.price,
        "sz": f"{event.size_contracts:.12g}",
        "ctVal": event.ct_val,
        "sizeCcy": event.size_ccy,
        "notional": event.notional,
        "ts": event.ts_ms,
        "tsIso": iso_of_ms(event.ts_ms),
        "fetchedAt": fetched_at,
    }


def group_by_day(
    events: Sequence[LiquidationEvent], fetched_at: str
) -> dict[str, list[dict[str, Any]]]:
    """按事件 UTC 日分组（同一轮里的跨午夜事件会被分到两个文件）。"""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        buckets.setdefault(day_of_ms(event.ts_ms), []).append(
            event_row(event, fetched_at)
        )
    return buckets


def append_events(
    events: Sequence[LiquidationEvent], fetched_at: str, *, root: Path = CRYPTO_ROOT
) -> tuple[int, int, set[str]]:
    """按日追加去重，返回 (新增, 跳过, 触及的日期文件名)。"""
    added = 0
    skipped = 0
    touched: set[str] = set()
    for day, rows in group_by_day(events, fetched_at).items():
        path = liquidation_jsonl_path(day, root=root)
        day_added, day_skipped = append_jsonl(path, rows, KEY_FIELDS)
        added += day_added
        skipped += day_skipped
        touched.add(path.name)
    return added, skipped, touched


def load_last_seen(days: Iterable[str], *, root: Path = CRYPTO_ROOT) -> dict[str, int]:
    """从指定日期的文件里重建「各族已见最新 ts」，重启后缺口检测才能接上。"""
    last_seen: dict[str, int] = {}
    for day in days:
        path = liquidation_jsonl_path(day, root=root)
        if not path.exists():
            continue
        for record in read_jsonl(path):
            inst_id = record.get("instId")
            ts = record.get(TIME_FIELD)
            if not isinstance(inst_id, str) or not isinstance(ts, int):
                continue
            last_seen[inst_id] = max(last_seen.get(inst_id, 0), ts)
    return last_seen


def detect_gaps(
    events: Sequence[LiquidationEvent], last_seen: dict[str, int], slack_ms: int
) -> list[dict[str, Any]]:
    """找出「本轮最老事件比上次最新事件还晚 slack 以上」的标的，即环形缓冲已绕过。

    首次见到的标的（last_seen 里没有）不算缺口：那是这条线刚开始采，不是丢数据。
    """
    oldest: dict[str, int] = {}
    for event in events:
        current = oldest.get(event.inst_id)
        if current is None or event.ts_ms < current:
            oldest[event.inst_id] = event.ts_ms
    gaps: list[dict[str, Any]] = []
    for inst_id, first in oldest.items():
        previous = last_seen.get(inst_id)
        if previous is None or first <= previous + slack_ms:
            continue
        gaps.append(
            {
                "instId": inst_id,
                "prevTs": previous,
                "nextTs": first,
                "lostMs": first - previous,
                "detectedAt": iso_of_ms(int(datetime.now(tz=UTC).timestamp() * 1000)),
            }
        )
    return gaps


def append_gaps(gaps: Sequence[dict[str, Any]], *, root: Path = CRYPTO_ROOT) -> int:
    """缺口按发现日追加（同一天重跑不会重复写同一条缺口）。"""
    if not gaps:
        return 0
    today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    added, _ = append_jsonl(
        liquidation_gap_path(today, root=root), gaps, GAP_KEY_FIELDS
    )
    return added


def update_manifest(
    touched: Iterable[str] = (),
    *,
    root: Path = CRYPTO_ROOT,
    manifest_root: Path = MANIFEST_ROOT,
) -> Path:
    """重写数据集 manifest；只对本次触及的文件重算行数与 hash。"""
    return rewrite_manifest(
        DATASET,
        directory=liquidation_dir(root=root),
        source=SOURCE,
        time_field=TIME_ISO_FIELD,
        touched=touched,
        manifest_root=manifest_root,
    )
