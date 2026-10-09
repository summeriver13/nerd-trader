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
