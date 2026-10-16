"""抓 OKX 跟单榜单快照（薄壳：取数在 src/utils/data_sources/okx_copytrading.py，落盘在 src/utils/copytrade_store.py）。

榜单是平台的「明星带单员」成绩单：`pnlRatio` 是交易员**自有投入资金**的累计收益率，
跟单者自己赚多少看 `curCopyTraderPnl`——两个口径不能混读，细节见 docs/copy-trading/README.md。
这条线只做前向快照：榜单会换人（爆仓/退榜的从榜上消失），不每天留档就没有分母。

用法：

    uv run python scripts/fetch_okx_copy.py                # 全榜 + 每人 90 天跟单统计
    uv run python scripts/fetch_okx_copy.py --list-only    # 只看榜单摘要，不落盘
    uv run python scripts/fetch_okx_copy.py --skip-stats   # 只落榜单标量，省 246 次请求

落盘：data/storage/okx_copy/raw/lead_traders/<YYYY-MM-DD>.jsonl
manifest：data/manifests/okx_copy_raw_lead_traders.json
验收：uv run python scripts/audit_okx_copy.py
"""

from __future__ import annotations

import argparse
import time

import requests

from src.config.paths import OKX_COPY_RAW_ROOT
from src.utils.copytrade_store import (
    DATASET,
    SOURCE,
    append_snapshot,
    snapshot_rows,
    update_manifest,
)
from src.utils.data_sources.base import create_session
from src.utils.data_sources.okx_copytrading import (
    INST_TYPE_SWAP,
    REQUEST_PAUSE,
    STATS_WINDOW_90D,
    Leaderboard,
    TraderStats,
    fetch_leaderboard,
    fetch_stats,
)
from src.utils.jsonl_store import day_stamp

TOP_PREVIEW = 5


def fetch_all_stats(
    session: requests.Session,
    leaderboard: Leaderboard,
    *,
    inst_type: str,
    last_days: int,
    pause: float,
) -> dict[str, TraderStats]:
    """逐个带单员取跟单统计；单人失败只记失败数，不中断整批落盘。"""
    stats: dict[str, TraderStats] = {}
    failures = 0
    for rank in leaderboard.ranks:
        try:
            stats[rank.unique_code] = fetch_stats(
                session, rank.unique_code, inst_type=inst_type, last_days=last_days
            )
        except Exception as error:  # 单人异常不该丢掉另外 245 人的榜单数据
            failures += 1
            print(f"    ! {rank.unique_code} 统计失败：{type(error).__name__}: {error}")
        time.sleep(pause)
    print(f"    跟单统计：成功 {len(stats):,} 人，失败 {failures:,} 人")
    return stats


def describe(leaderboard: Leaderboard) -> None:
    """榜单一页纸摘要：人数、跟单关系规模、AUM 前几名——够判断这份快照是否抓全。"""
    ranks = leaderboard.ranks
    followed = [rank for rank in ranks if rank.copy_trader_num > 0]
    top = sorted(ranks, key=lambda rank: rank.aum, reverse=True)[:TOP_PREVIEW]
    print(
        f"    榜单 {len(ranks):,} 人，其中 {len(followed):,} 人有现存跟单者，"
        f"合计跟单资金 {sum(rank.aum for rank in ranks):,.0f} USDT"
    )
    for rank in top:
        print(
            f"      {rank.nick_name[:18]:<18} AUM {rank.aum:>13,.0f}"
            f" | 跟单 {rank.copy_trader_num:>4} 人 | 自有收益率 {rank.pnl_ratio:>8.2%}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="抓 OKX 跟单榜单快照")
    parser.add_argument("--inst-type", default=INST_TYPE_SWAP)
    parser.add_argument("--stats-window", type=int, default=STATS_WINDOW_90D)
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument("--list-only", action="store_true", help="只打印摘要，不落盘")
    parser.add_argument("--sleep-seconds", type=float, default=REQUEST_PAUSE)
    args = parser.parse_args()

    session = create_session()
    leaderboard = fetch_leaderboard(session, inst_type=args.inst_type)
    print(f"[{args.inst_type}] dataVer {leaderboard.data_ver}")
    describe(leaderboard)
    if args.list_only:
        return 0

    stats: dict[str, TraderStats] = {}
    if not args.skip_stats:
        stats = fetch_all_stats(
            session,
            leaderboard,
            inst_type=args.inst_type,
            last_days=args.stats_window,
            pause=args.sleep_seconds,
        )

    # 分区日与快照时刻一致（UTC），不跟墙上时钟：补数/重放时不会落进错误日期。
    day = day_stamp(leaderboard.fetched_at.to_pydatetime())
    rows = snapshot_rows(
        leaderboard, stats, fetched_at=leaderboard.fetched_at.isoformat()
    )
    added, skipped = append_snapshot(rows, day=day)
    manifest = update_manifest(touched=[f"{day}.jsonl"])
    print(
        f"    {DATASET}：新增 {added:,} 行，跳过重复 {skipped:,} 行 →"
        f" {OKX_COPY_RAW_ROOT / DATASET / f'{day}.jsonl'}"
    )
    print(f"manifest 已更新：{manifest}")
    print(f"\n来源 {SOURCE}；验收：uv run python scripts/audit_okx_copy.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
