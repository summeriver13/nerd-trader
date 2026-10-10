"""
src/research/provenance.py
运行指纹：把「同一命令 → 同一 hash」变成可验证的事实。

为什么需要：只要数据、配置、代码三者任一漂移，历史结论就不再可比。
本模块把三者的指纹写进 data/results/provenance/<run_id>.json，文件名由
命令参数决定（同一命令覆盖同一个文件），repro_hash 覆盖：

- config：命令行参数 + 费率常量（改费率必须改数字，也要改 hash）
- data：本次真正读到的 parquet 的 sha256 + 对应 manifest 的 sha256
- code：git HEAD 与工作区是否脏

判据：同一命令在数据与代码不变时得到同一个 repro_hash；任一环变了 hash 就变。
stdout 里也会打印一行，回测扩展归档 runs.json 时会把指纹一起存下来。
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config.paths import (
    MANIFEST_ROOT,
    PROJECT_ROOT,
    RESULTS_ROOT,
    crypto_bar_path,
    symbol_market,
)
from src.core.costs import (
    MARKET_SLIPPAGE,
    PERP_MAKER_FEE,
    PERP_TAKER_FEE,
    SPOT_MAKER_FEE,
    SPOT_TAKER_FEE,
)

logger = logging.getLogger(__name__)

PROVENANCE_DIR = RESULTS_ROOT / "provenance"
BAR_SOURCE = "okx"  # 行情缓存来源
FUNDING_DATASET = "crypto_binance_funding_8h"  # 资金费率缓存（binance/funding/8h）
HASH_DISPLAY_LEN = 12
READ_CHUNK = 1 << 20


def _sha256_file(path: Path) -> str | None:
    """文件 sha256；文件不存在返回 None（数据缺失也要在指纹里看得见）。"""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(READ_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_fingerprint() -> dict[str, Any]:
    """代码版本：HEAD + 工作区是否脏（脏意味着结论对应的不是某个提交）。"""

    def run(*argv: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *argv], cwd=PROJECT_ROOT, capture_output=True, text=True
            )
        except FileNotFoundError:
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    return {
        "head": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def data_fingerprint(symbols: list[str], timeframe: str) -> dict[str, Any]:
    """本次读到的数据：逐标的行情文件 + 各市场 manifest 的 sha256。"""
    bars = {
        symbol: _sha256_file(crypto_bar_path(BAR_SOURCE, symbol, timeframe))
        for symbol in symbols
    }
    manifest_names = [
        f"crypto_{BAR_SOURCE}_{market}_{timeframe}"
        for market in sorted({symbol_market(symbol) for symbol in symbols})
    ]
    manifests = {
        name: _sha256_file(MANIFEST_ROOT / f"{name}.json")
        for name in (*manifest_names, FUNDING_DATASET)
    }
    return {"bars": bars, "manifests": manifests}


def costs_fingerprint() -> dict[str, float]:
    """费率快照：这些数字直接决定净收益，必须进指纹。"""
    return {
        "spot_maker": SPOT_MAKER_FEE,
        "spot_taker": SPOT_TAKER_FEE,
        "perp_maker": PERP_MAKER_FEE,
        "perp_taker": PERP_TAKER_FEE,
        "market_slippage": MARKET_SLIPPAGE,
    }


def repro_hash(record: dict[str, Any]) -> str:
    """指纹本体：对 config + data + code 做规范化 JSON 的 sha256（不含生成时间）。"""
    payload = {key: record[key] for key in ("mode", "config", "data", "code")}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def record_run(
    mode: str, config: dict[str, Any], symbols: list[str], timeframe: str
) -> Path:
    """写指纹文件并打印一行。同一命令 → 同一文件、同一 hash。"""
    # 符号里的 '/' 与 ':' 都不能进文件名（':' 是 Windows 的备用数据流分隔符）
    symbol_tags = "+".join(s.replace("/", "-").replace(":", "-") for s in symbols)
    run_id = f"{mode}_{timeframe}_{config.get('bars', 'na')}_{symbol_tags}"
    record: dict[str, Any] = {
        "run_id": run_id,
        "mode": mode,
        "generated_at": datetime.now(UTC).isoformat(),
        "command": " ".join(["uv run python research.py", *sys.argv[1:]]),
        "config": config,
        "costs": costs_fingerprint(),
        "data": data_fingerprint(symbols, timeframe),
        "code": _git_fingerprint(),
    }
    record["repro_hash"] = repro_hash(record)

    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    path = PROVENANCE_DIR / f"{run_id}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    head = record["code"]["head"]
    code_note = f"{(head or 'no-git')[:8]}{'（工作区有未提交改动）' if record['code']['dirty'] else ''}"
    print(
        f"🔖 复现指纹 {record['repro_hash'][:HASH_DISPLAY_LEN]}"
        f"｜数据 {len(symbols)} 个文件 sha256｜代码 {code_note}"
    )
    print(f"📄 指纹记录: {path.relative_to(PROJECT_ROOT)}")
    return path
