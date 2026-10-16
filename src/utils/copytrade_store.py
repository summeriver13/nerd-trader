"""
OKX 跟单榜单 raw 层：按天落快照。

    data/storage/okx_copy/raw/lead_traders/<YYYY-MM-DD>.jsonl
    data/manifests/okx_copy_raw_lead_traders.json

业务键是 `(uniqueCode, dataVer)`：平台榜单版本没动时重跑不写重复行，版本动了就再留
一份——raw 只增不改，一天多跑几次既不污染也不会丢掉当天的旧版本。

每行除榜单标量外还带两样东西，口径不能混读（见 `docs/copy-trading/README.md`）：

- `history`：上游附带的日度收益序列（3~90 天），是**交易员自有投入资金**的累计
  收益率历史，也是不重取接口就能一次拿到的唯一净值历史；
- `investAmt` / `curCopyTraderPnl`：跟单统计（90 天滚动口径），前者是自有投入资金
  （`pnlRatio ≈ pnl / investAmt`），后者才是**跟单者**的累计盈亏。

上游的 `traderInsts` 只留个数（`instrumentNum`）：最长能挂 283 个标的、占整份文件
七成字节，对跟踪读数没有增量信息，真要用时后续快照也能拿到。

通用逻辑（去重追加、manifest 生成）在 `src/utils/jsonl_store.py`；这里只声明
跟单数据集自己的东西。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.config.paths import (
    MANIFEST_ROOT,
    OKX_COPY_RAW_ROOT,
    okx_copy_dataset_dir,
    okx_copy_jsonl_path,
)
from src.utils.data_sources.okx_copytrading import Leaderboard, TraderRank, TraderStats
from src.utils.jsonl_store import append_jsonl, read_jsonl, rewrite_manifest

SOURCE = "www.okx.com"
DATASET = "lead_traders"
MANIFEST_NAME = f"okx_copy_raw_{DATASET}"
# 榜单版本 + 带单员：同一次榜单快照重跑只写一遍
KEY_FIELDS: tuple[str, ...] = ("uniqueCode", "dataVer")
TIME_FIELD = "fetched_at"


def snapshot_rows(
    leaderboard: Leaderboard,
    stats: Mapping[str, TraderStats],
    *,
    fetched_at: str,
) -> list[dict[str, Any]]:
    """榜单 + 跟单统计 → 待落盘行；某个带单员统计缺失时只留榜单字段。"""
    return [
        snapshot_row(
            rank,
            stats.get(rank.unique_code),
            data_ver=leaderboard.data_ver,
            inst_type=leaderboard.inst_type,
            fetched_at=fetched_at,
        )
        for rank in leaderboard.ranks
    ]


def snapshot_row(
    rank: TraderRank,
    stats: TraderStats | None,
    *,
    data_ver: str,
    inst_type: str,
    fetched_at: str,
) -> dict[str, Any]:
    """单行落盘格式：业务键在前，两个口径的字段名保持上游拼写便于溯源。"""
    row: dict[str, Any] = {
        "uniqueCode": rank.unique_code,
        "dataVer": data_ver,
        "instType": inst_type,
        "fetched_at": fetched_at,
        "nickName": rank.nick_name,
        "ccy": rank.ccy,
        "aum": rank.aum,
        "pnl": rank.pnl,
        "pnlRatio": rank.pnl_ratio,
        "winRatio": rank.win_ratio,
        "leadDays": rank.lead_days,
        "copyTraderNum": rank.copy_trader_num,
        "accCopyTraderNum": rank.acc_copy_trader_num,
        "maxCopyTraderNum": rank.max_copy_trader_num,
        "instrumentNum": rank.instrument_num,
        "history": [[point.begin_ts, point.pnl_ratio] for point in rank.history],
    }
    if stats is not None:
        row.update(
            {
                "investAmt": stats.invest_amt,
                "curCopyTraderPnl": stats.cur_copy_trader_pnl,
                "avgSubPosNotional": stats.avg_sub_pos_notional,
                "profitDays": stats.profit_days,
                "lossDays": stats.loss_days,
                "statsWinRatio": stats.win_ratio,
            }
        )
    return row


def append_snapshot(
    rows: Sequence[dict[str, Any]],
    *,
    day: str,
    root: Path = OKX_COPY_RAW_ROOT,
) -> tuple[int, int]:
    """按业务键去重后追加进当日文件，返回 (新增, 跳过)。"""
    return append_jsonl(okx_copy_jsonl_path(day, root=root), rows, KEY_FIELDS)


def read_snapshot(day: str, *, root: Path = OKX_COPY_RAW_ROOT) -> list[dict[str, Any]]:
    """读某一天的快照行；文件不存在返回空表（验收闸要靠它区分「没跑」与「跑了但空」）。"""
    path = okx_copy_jsonl_path(day, root=root)
    return list(read_jsonl(path)) if path.exists() else []


def update_manifest(
    *,
    touched: Iterable[str] = (),
    raw_root: Path = OKX_COPY_RAW_ROOT,
    manifest_root: Path = MANIFEST_ROOT,
) -> Path:
    """重写该数据集 manifest；只对本次触及的文件重算，其余沿用旧条目。"""
    return rewrite_manifest(
        MANIFEST_NAME,
        directory=okx_copy_dataset_dir(root=raw_root),
        source=SOURCE,
        time_field=TIME_FIELD,
        touched=touched,
        manifest_root=manifest_root,
    )
