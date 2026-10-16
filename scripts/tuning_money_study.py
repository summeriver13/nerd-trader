"""
第 9 章遗留 + 第 12 章：**CV 抬高的那点分数，能变成钱吗？**——把单次切分的「一个数」换成
组合式 purged 交叉验证（CPCV）的**路径分布**（机制见 `src/research/cpcv.py`）与三臂对照：
`全下（等权）`（无元标签过滤）／`默认参数`／`CV 选中参数`（训练窗内 purged K 折选出）。

口径：每条路径的每个组恰好被测试一次，故**一条路径覆盖全时段**，池化在路径内部合法（第十七节
第 5 小节的教训：折只覆盖部分时段，拼起来会把零仓位日按零收益算进均值）；训练集含测试组之后的
日期，这是**留出**不是时序外推，衡量的是超参选择的估计噪声；purge/embargo 各 `--vertical` 天。

用法：uv run python scripts/tuning_money_study.py --market swap [--no-save]
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from src.config.paths import RESULTS_ROOT
from src.research import cpcv, meta, observations, sizing, tuning
from src.research.factors.panel import Panel, load_panel
from src.research.observations import (
    FEATURE_FACTORS,
    PRIMARY_FACTOR,
)
from src.research.trials import Trial, TrialLedger

LABEL_SOURCE: Final[str] = observations.BARRIER_LABEL_COLUMN
OUTPUT_DIR: Final[Path] = RESULTS_ROOT / "breaks"
ALL_IN: Final[str] = "全下（等权）"
DEFAULT_ARM: Final[str] = "默认参数"
CV_ARM: Final[str] = "CV 选中参数"
MODEL_ARMS: Final[tuple[str, ...]] = (DEFAULT_ARM, CV_ARM)
ARM_COLUMNS: Final[pd.Index] = pd.Index(list(MODEL_ARMS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ch 9 遗留：分数能变成钱吗（CPCV 路径版）"
    )
    parser.add_argument("--market", default="swap")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--vertical", type=int, default=3, help="三重障碍的垂直屏障")
    parser.add_argument("--groups", type=int, default=8, help="CPCV 组数 N")
    parser.add_argument("--test-groups", type=int, default=2, help="每次做测试的组数 k")
    parser.add_argument(
        "--inner-splits", type=int, default=4, help="内层 purged K 折数"
    )
    parser.add_argument("--n-iter", type=int, default=8, help="随机搜索抽几个配置")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def _dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, Panel]:
    """与 `tuning_study.py` 完全同源的数据集：只把标签与权重换成模型要的列名。"""
    panel = load_panel(args.timeframe, market=args.market)
    frame = observations.build_meta_frame(
        panel,
        primary_key=PRIMARY_FACTOR,
        feature_keys=FEATURE_FACTORS,
        vertical=args.vertical,
    )
    trained = frame.copy()
    trained[meta.LABEL_COLUMN] = (frame[LABEL_SOURCE] > 0).astype(int)
    trained[meta.WEIGHT_COLUMN] = frame["weight_unique"]
    print(
        f"{args.market} {args.timeframe}：{panel.close.shape[1]} 只 × "
        f"{panel.close.shape[0]} 天｜观测 {len(trained):,}｜正类率 "
        f"{trained[meta.LABEL_COLUMN].mean():.4f}"
    )
    return trained, panel


def _problem(
    train: pd.DataFrame,
    features: list[str],
    base: meta.MetaLabelOptions,
    args: argparse.Namespace,
) -> tuning.TuningProblem:
    return tuning.TuningProblem(
        factory=lambda params: meta.build_classifier(replace(base, **params)),
        features=train[features].to_numpy(),
        labels=train[meta.LABEL_COLUMN].to_numpy(),
        groups=pd.factorize(train[meta.DATE_COLUMN], sort=True)[0],
        sample_weight=train[meta.WEIGHT_COLUMN].to_numpy(),
        cv=tuning.PurgedKFold(
            n_splits=args.inner_splits, purge=args.vertical, embargo=args.vertical
        ),
    )


def _probability(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    params: dict[str, Any],
    base: meta.MetaLabelOptions,
) -> np.ndarray:
    model = meta.build_classifier(replace(base, **params))
    model.fit(
        train[features].to_numpy(),
        train[meta.LABEL_COLUMN].to_numpy(),
        sample_weight=train[meta.WEIGHT_COLUMN].to_numpy(),
    )
    return np.asarray(model.predict_proba(test[features].to_numpy()))[:, 1]


def _split_probabilities(
    trained: pd.DataFrame,
    features: list[str],
    base: meta.MetaLabelOptions,
    args: argparse.Namespace,
    cv: cpcv.CombinatorialPurgedCv,
    row_positions: np.ndarray,
) -> tuple[dict[int, pd.DataFrame], list[dict[str, Any]]]:
    """逐切分：选超参 → 拟合 → 测试段概率（段外 NaN）；k 个测试组共用一个模型。"""
    per_split: dict[int, pd.DataFrame] = {}
    searched: list[dict[str, Any]] = []
    for index, split in enumerate(cv.split(int(row_positions.max()) + 1), start=1):
        train_mask = np.isin(row_positions, split.train_dates)
        test_mask = np.isin(row_positions, split.test_dates)
        train, test = trained.loc[train_mask], trained.loc[test_mask]
        result = tuning.search_hyperparameters(
            _problem(train, features, base, args),
            meta.RF_PARAM_SPACE,
            n_iter=args.n_iter,
            seed=args.seed,
        )
        searched.append(
            {
                "candidates": [
                    {
                        key: result.fold_scores[key][row]
                        for key in meta.RF_DEFAULT_PARAMS
                    }
                    for row in range(len(result.fold_scores))
                ]
            }
        )
        table = pd.DataFrame(np.nan, index=trained.index, columns=ARM_COLUMNS)
        for arm, params in (
            (DEFAULT_ARM, meta.RF_DEFAULT_PARAMS),
            (CV_ARM, result.best_params),
        ):
            table.loc[test_mask, arm] = _probability(
                train, test, features, params, base
            )
        per_split[index - 1] = table
        print(
            f"  切分 {index}/{cv.n_splits}｜测试组 {split.test_groups}｜训练 "
            f"{int(train_mask.sum()):,} / 测试 {int(test_mask.sum()):,}｜CV 最优 "
            f"{result.best_params}（{result.best_score:.4f}）"
        )
    return per_split, searched


def _arm_stats(
    frame: pd.DataFrame, probability: pd.Series, horizon: int, panel: Panel
) -> sizing.BetStats:
    """一臂的净夏普：概率 → 下注规模 → 逐日 P&L。`全下` 那一臂传常数 1.0 的概率即可。"""
    enriched = frame[[meta.DATE_COLUMN, "symbol", "side"]].copy()
    # pandas 桩把列取用推成联合类型，按仓库惯例用 cast 收窄
    enriched["side"] = cast(pd.Series, enriched["side"]).where(probability.notna(), 0.0)
    enriched["probability"] = probability
    side = enriched.pivot(index=meta.DATE_COLUMN, columns="symbol", values="side")
    probabilities = enriched.pivot(
        index=meta.DATE_COLUMN, columns="symbol", values="probability"
    )
    daily = panel.close.shift(-1).div(panel.close) - 1.0
    return sizing.simulate_positions(
        sizing.size_columns(probabilities, side).fillna(0.0),
        daily.reindex(index=side.index, columns=side.columns),
        sizing.PositionOptions(horizon=horizon),
    )


def _report(
    trained: pd.DataFrame,
    panel: Panel,
    args: argparse.Namespace,
    tables: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """逐路径的下注读数与差值分布：一条路径覆盖全时段，故池化在**路径内部**是合法的。"""
    cost = sizing.DEFAULT_ONE_WAY_COST * 1e4
    print(f"\n=== 下注规模（持有 {args.vertical} 天，单边成本 {cost:.1f}bp）===")
    rows: list[dict[str, Any]] = []
    for index, table in tables.items():
        arms: dict[str, pd.Series] = {ALL_IN: pd.Series(1.0, index=trained.index)}
        for name in MODEL_ARMS:
            arms[name] = cast(pd.Series, table[name])
        for name, probability in arms.items():
            stats = _arm_stats(trained, probability, args.vertical, panel)
            # 全下那一臂没有分类器输出，f1 无定义
            observed = probability.notna().to_numpy()
            f1 = tuning.positive_f1(
                trained.loc[observed, meta.LABEL_COLUMN].to_numpy(),
                (probability.to_numpy() >= 0.5).astype(int),
            )
            rows.append(
                {
                    "路径": index + 1,
                    "臂": name,
                    "毛夏普": stats.gross_sharpe,
                    "净夏普": stats.net_sharpe,
                    "换手": float(stats.turnover.mean()),
                    "毛 bp/期": stats.gross * 1e4,
                    "净 bp/期": stats.net * 1e4,
                    "平衡成本 bp": stats.breakeven_cost * 1e4,
                    "样本外 f1": f1 if name in MODEL_ARMS else float("nan"),
                }
            )
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    records = table.to_dict("records")
    net = {(int(row["路径"]), str(row["臂"])): float(row["净夏普"]) for row in records}
    paths = sorted({int(row["路径"]) for row in records})
    for arm in (ALL_IN, DEFAULT_ARM, CV_ARM):
        values = np.array([net[(path, arm)] for path in paths])
        span = f"[{values.min():+.3f}, {values.max():+.3f}]"
        width = values.max() - values.min()
        print(f"{arm}：中位 {np.median(values):+.3f}｜{span}｜宽度 {width:.3f}")
    deltas = np.array(
        [net[(path, CV_ARM)] - net[(path, DEFAULT_ARM)] for path in paths]
    )
    print(f"\n逐路径净夏普差（CV − 默认）：{'、'.join(f'{v:+.3f}' for v in deltas)}")
    span = f"[{deltas.min():+.3f}, {deltas.max():+.3f}]"
    positive = int((deltas > 0).sum())
    print(f"中位 {np.median(deltas):+.3f}｜区间 {span}｜正号 {positive}/{deltas.size}")
    return table


def _record_trials(
    ledger: TrialLedger, searched: list[dict[str, Any]], args: argparse.Namespace
) -> None:
    trials = [
        Trial(
            study="tuning",
            market=args.market,
            timeframe=args.timeframe,
            kind="rf_purged_kfold",  # 与分数研究同批配置：只换评估口径不计新试验
            params=dict(params),
        )
        for row in searched
        for params in row["candidates"]
    ]
    print(f"试验账本：本次新增 {ledger.record(trials)} 条，累计 N={ledger.count()}")


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore")
    trained, panel = _dataset(args)
    features = [*FEATURE_FACTORS, "age_days"]
    base = meta.MetaLabelOptions(seed=args.seed)
    cv = cpcv.CombinatorialPurgedCv(
        n_groups=args.groups,
        k_test=args.test_groups,
        purge=args.vertical,
        embargo=args.vertical,
    )
    days = trained[meta.DATE_COLUMN].to_numpy().astype("datetime64[D]")
    unique_days, row_positions = np.unique(days, return_inverse=True)
    groups = cv.group_dates(unique_days.size)
    print(
        f"CPCV：{cv.n_splits} 次切分（purge={cv.purge}、embargo={cv.embargo}）→ "
        f"{cv.n_paths} 条路径｜每组约 {unique_days.size // cv.n_groups} 天"
    )
    per_split, searched = _split_probabilities(
        trained, features, base, args, cv, row_positions
    )
    paths = cv.paths()
    tables: dict[int, pd.DataFrame] = {}
    for index in range(paths.shape[0]):
        table = pd.DataFrame(np.nan, index=trained.index, columns=ARM_COLUMNS)
        for group in range(paths.shape[1]):
            mask = np.isin(row_positions, groups[group])
            table.loc[mask, :] = per_split[int(paths[index, group])].loc[mask, :]
        tables[index] = table
    table = _report(trained, panel, args, tables)

    _record_trials(TrialLedger(), searched, args)
    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        target = (
            OUTPUT_DIR / f"tuning_money_cpcv_{args.market}_{args.timeframe}.parquet"
        )
        table.to_parquet(target)
        print(f"落盘：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
