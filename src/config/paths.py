"""
src/config/paths.py
路径常量：数据与结果目录的唯一出处。

crypto 数据由本项目自持（不入容器数据湖），依据容器层 ADR-0006；
所有代码从这里取路径，不再各自拼 "data/storage"。
"""

from pathlib import Path

# 仓库根目录：src/config/paths.py 上溯三层
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data"
STORAGE_ROOT = DATA_ROOT / "storage"
CRYPTO_ROOT = STORAGE_ROOT / "crypto"
CACHE_ROOT = DATA_ROOT / "cache"
RESULTS_ROOT = DATA_ROOT / "results"
MANIFEST_ROOT = DATA_ROOT / "manifests"

# 数据集 manifest 存放处（覆盖区间/行数/来源/hash），格式对齐容器数据湖
MANIFEST_SUFFIX = ".json"

# === crypto 数据集的目录契约 ===
# <crypto>/<exchange>/<market>/<timeframe>/<标的>.parquet
# market 是数据集的语义维度而不是目录装饰：现货与永续的上市起点、杠杆与成本口径都不同，
# 且永续的 ccxt 符号带结算币后缀（BTC/USDT:USDT），冒号在 Windows 上不能进文件名。
MARKET_SPOT = "spot"
MARKET_SWAP = "swap"
MARKET_FUNDING = "funding"
SETTLEMENT_SEP = ":"

# === 预测市场（Polymarket）数据集契约（依据容器 ADR-0007）===
# <polymarket>/raw/<dataset>/<name>.jsonl
# events / markets 是每日快照（按日期命名），prices / trades 是逐市场明细（按 slug 命名）。
POLYMARKET_ROOT = STORAGE_ROOT / "polymarket"
POLYMARKET_RAW_ROOT = POLYMARKET_ROOT / "raw"
# 研究产物（链快照、对照表）落 results，与只增不改的数据湖分开放
POLYMARKET_RESULTS_ROOT = RESULTS_ROOT / "polymarket"
POLYMARKET_EVENTS = "events"
POLYMARKET_MARKETS = "markets"
POLYMARKET_PRICES = "prices"
POLYMARKET_TRADES = "trades"
POLYMARKET_DATASETS = (
    POLYMARKET_EVENTS,
    POLYMARKET_MARKETS,
    POLYMARKET_PRICES,
    POLYMARKET_TRADES,
)


def polymarket_dataset_dir(dataset: str, *, root: Path = POLYMARKET_RAW_ROOT) -> Path:
    """预测市场数据集目录：<polymarket>/raw/<dataset>。"""
    return root / dataset


def polymarket_jsonl_path(
    dataset: str, name: str, *, root: Path = POLYMARKET_RAW_ROOT
) -> Path:
    """JSONL 文件路径；name 为日期（YYYY-MM-DD）或市场 slug。"""
    if dataset not in POLYMARKET_DATASETS:
        raise ValueError(f"未知数据集 {dataset!r}，可选 {POLYMARKET_DATASETS}")
    return polymarket_dataset_dir(dataset, root=root) / f"{name}.jsonl"


# === Deribit 期权链数据集契约（crypto 数据自持，容器 ADR-0006）===
# <deribit>/raw/<dataset>/<YYYY-MM-DD>.jsonl，一日可多轮快照（键含 fetched_at）
DERIBIT_ROOT = STORAGE_ROOT / "deribit"
DERIBIT_RAW_ROOT = DERIBIT_ROOT / "raw"
DERIBIT_OPTIONS = "options"
DERIBIT_FUTURES = "futures"
DERIBIT_DATASETS = (DERIBIT_OPTIONS, DERIBIT_FUTURES)


def deribit_dataset_dir(dataset: str, *, root: Path = DERIBIT_RAW_ROOT) -> Path:
    """Deribit 数据集目录：<deribit>/raw/<dataset>。"""
    return root / dataset


def deribit_jsonl_path(
    dataset: str, name: str, *, root: Path = DERIBIT_RAW_ROOT
) -> Path:
    """JSONL 文件路径；name 为日期（YYYY-MM-DD）。"""
    if dataset not in DERIBIT_DATASETS:
        raise ValueError(f"未知数据集 {dataset!r}，可选 {DERIBIT_DATASETS}")
    return deribit_dataset_dir(dataset, root=root) / f"{name}.jsonl"


# === OKX 跟单榜单数据集契约 ===
# <okx_copy>/raw/lead_traders/<YYYY-MM-DD>.jsonl，按天快照（业务键 (uniqueCode, dataVer)）。
# 这是本项目自持的第三方榜单数据（不是加密行情），故与 crypto 数据根分开存放。
OKX_COPY_ROOT = STORAGE_ROOT / "okx_copy"
OKX_COPY_RAW_ROOT = OKX_COPY_ROOT / "raw"
OKX_COPY_TRADERS = "lead_traders"


def okx_copy_dataset_dir(*, root: Path = OKX_COPY_RAW_ROOT) -> Path:
    """OKX 跟单数据集目录：<okx_copy>/raw/lead_traders。"""
    return root / OKX_COPY_TRADERS


def okx_copy_jsonl_path(name: str, *, root: Path = OKX_COPY_RAW_ROOT) -> Path:
    """JSONL 文件路径；name 为日期（YYYY-MM-DD）。"""
    return okx_copy_dataset_dir(root=root) / f"{name}.jsonl"


# === OKX 强平事件流契约（crypto 数据自持，容器 ADR-0006）===
# <crypto>/okx/liquidation/<YYYY-MM-DD>.jsonl：按事件自身 UTC 日分区，一行一条强平成交。
# 上游每族只保留最近 100 条（约 9 小时）且无历史回补，故这是只增不改的前向事件流；
# 轮询缺口另落 gaps/<YYYY-MM-DD>.jsonl，与事件文件分开（schema 不同，混在一起会污染口径）。
MARKET_LIQUIDATION = "liquidation"
LIQUIDATION_EXCHANGE = "okx"
LIQUIDATION_GAP_DIR = "gaps"


def liquidation_dir(*, root: Path = CRYPTO_ROOT) -> Path:
    """强平事件流目录：<crypto>/<exchange>/liquidation。"""
    return root / LIQUIDATION_EXCHANGE / MARKET_LIQUIDATION


def liquidation_jsonl_path(day: str, *, root: Path = CRYPTO_ROOT) -> Path:
    """事件文件路径；day 为事件时间所属的 UTC 日期（YYYY-MM-DD）。"""
    return liquidation_dir(root=root) / f"{day}.jsonl"


def liquidation_gap_path(day: str, *, root: Path = CRYPTO_ROOT) -> Path:
    """缺口记录路径；day 为缺口被发现的 UTC 日期。"""
    return liquidation_dir(root=root) / LIQUIDATION_GAP_DIR / f"{day}.jsonl"


def symbol_market(symbol: str) -> str:
    """ccxt 符号 → 市场层级：带结算币后缀的是永续。"""
    return MARKET_SWAP if SETTLEMENT_SEP in symbol else MARKET_SPOT


def symbol_stem(symbol: str) -> str:
    """ccxt 符号 → 文件名主干：去掉结算币后缀并转成文件名安全形式。"""
    return symbol.split(SETTLEMENT_SEP)[0].replace("/", "-").upper()


def crypto_dataset_dir(
    exchange: str, market: str, timeframe: str, *, root: Path = CRYPTO_ROOT
) -> Path:
    """crypto 数据集目录：<crypto>/<exchange>/<market>/<timeframe>。"""
    return root / exchange / market / timeframe


def crypto_bar_path(
    exchange: str, symbol: str, timeframe: str, *, root: Path = CRYPTO_ROOT
) -> Path:
    """行情文件路径；市场层级由符号本身推断，调用方不必重复判断。"""
    return (
        crypto_dataset_dir(exchange, symbol_market(symbol), timeframe, root=root)
        / f"{symbol_stem(symbol)}.parquet"
    )
