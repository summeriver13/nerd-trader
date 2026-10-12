"""
scripts/tuning_money_study.py
第 9 章遗留：**CV 抬高的那点分数，能变成钱吗？**（分数读数见 README 第十七节）

把同一套折上的样本外概率喂进第 10 章的下注管线（`research.sizing`），比较三臂的净夏普：
`全下（等权）`（无元标签过滤）／`默认参数`（`meta.RF_DEFAULT_PARAMS`）／`CV 选中参数`
（训练窗内 purged K 折选出）。两臂的差别**只来自超参**：同一批观测、同一个 `side`、同一
套持仓与成本口径；调参只发生在训练窗内部，测试窗只吃收益。
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
from src.research import meta, observations, sizing, tuning
from src.research.factors.panel import Panel, load_panel
from src.research.trials import Trial, TrialLedger

PRIMARY_KEY: Final[str] = "rev_3"
FEATURE_KEYS: Final[tuple[str, ...]] = ("mom_7", "mom_30", "vol_30", "liq_30", "dth_60")
AGE_KEY: Final[str] = "age_days"
LABEL_SOURCE: Final[str] = observations.BARRIER_LABEL_COLUMN
WEIGHT_SOURCE: Final[str] = "weight_unique"
OUTPUT_DIR: Final[Path] = RESULTS_ROOT / "breaks"
ALL_IN: Final[str] = "全下（等权）"
DEFAULT_ARM: Final[str] = "默认参数"
CV_ARM: Final[str] = "CV 选中参数"
MODEL_ARMS: Final[tuple[str, ...]] = (DEFAULT_ARM, CV_ARM)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ch 9 遗留：分数能变成钱吗")
    parser.add_argument("--market", default="swap")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--vertical", type=int, default=3, help="三重障碍的垂直屏障")
    parser.add_argument("--min-train", type=int, default=20000)
    parser.add_argument("--test-span", type=int, default=240, help="每折测试窗天数")
    parser.add_argument("--embargo", type=int, default=200)
    parser.add_argument(
        "--folds", type=int, default=4, help="抽样几折（与分数研究同几折）"
    )
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
        primary_key=PRIMARY_KEY,
        feature_keys=FEATURE_KEYS,
        vertical=args.vertical,
    )
    trained = frame.copy()
    trained[meta.LABEL_COLUMN] = (frame[LABEL_SOURCE] > 0).astype(int)
    trained[meta.WEIGHT_COLUMN] = frame[WEIGHT_SOURCE]
    print(
        f"{args.market} {args.timeframe}：{panel.close.shape[1]} 只 × "
        f"{panel.close.shape[0]} 天｜可交易观测 {len(trained)}，"
        f"正类基准率 {trained[meta.LABEL_COLUMN].mean():.4f}"
    )
    return trained, panel


def _fold_spans(
    trained: pd.DataFrame, features: list[str], base: meta.MetaLabelOptions
) -> list[tuple[pd.Index, pd.Index]]:
    """滚动折的训练/测试行号（只借折的切法，不用它的模型）。"""
    prediction = meta.walk_forward_meta(trained, features, base)
    return [
        (trained.index[fitted.train_rows], trained.index[fitted.rows])
        for fitted in prediction.fits
    ]


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
    """在训练窗上拟合、在测试窗上给出正类概率（不碰测试窗的标签）。"""
    model = meta.build_classifier(replace(base, **params))
    model.fit(
        train[features].to_numpy(),
        train[meta.LABEL_COLUMN].to_numpy(),
        sample_weight=train[meta.WEIGHT_COLUMN].to_numpy(),
    )
    return np.asarray(model.predict_proba(test[features].to_numpy()))[:, 1]


def _pooled_f1(trained: pd.DataFrame, probability: pd.Series) -> float:
    """池化样本外 f1（阈值 0.5，与分数研究同口径）——确认两处读数是同一批预测。"""
    observed = probability.notna().to_numpy()
    predicted = (probability.to_numpy() >= 0.5).astype(int)
    return tuning.positive_f1(
        trained.loc[observed, meta.LABEL_COLUMN].to_numpy(), predicted
    )


def _arm_stats(
    frame: pd.DataFrame,
    probability: pd.Series,
    horizon: int,
    panel: Panel,
) -> sizing.BetStats:
    """一臂的净夏普：概率 → 下注规模 → 逐日 P&L（口径同第 10 章那节）。

    `全下` 那一臂传常数 1.0 的概率即可——`bet_size_from_probability` 在 p→1 时恰好给 1。
    """
    enriched = frame[[meta.DATE_COLUMN, "symbol", "side"]].copy()
    # pandas 桩把列取用推成联合类型，按仓库惯例用 cast 收窄
    enriched["side"] = cast(pd.Series, enriched["side"]).where(probability.notna(), 0.0)
    enriched["probability"] = probability
    side = enriched.pivot(index=meta.DATE_COLUMN, columns="symbol", values="side")
    probabilities = enriched.pivot(
        index=meta.DATE_COLUMN, columns="symbol", values="probability"
    )
    daily = panel.close.shift(-1).div(panel.close) - 1.0
    weights = sizing.size_columns(probabilities, side)
    return sizing.simulate_positions(
        weights.fillna(0.0),
        daily.reindex(index=side.index, columns=side.columns),
        sizing.PositionOptions(horizon=horizon),
    )


def _record_trials(
    ledger: TrialLedger, searched: list[dict[str, Any]], args: argparse.Namespace
) -> None:
    """本次搜索过的配置记进账本（与分数研究同一批配置，身份去重后通常为 0 条新增）。"""
    trials = [
        Trial(
            study="tuning",
            market=args.market,
            timeframe=args.timeframe,
            kind="rf_purged_kfold",
            params=dict(params),
        )
        for row in searched
        for params in row["candidates"]
    ]
    added = ledger.record(trials)
    print(f"试验账本：本次新增 {added} 条，累计 N={ledger.count()}")


def _fold_windows(trained: pd.DataFrame, pooled: pd.DataFrame) -> list[np.ndarray]:
    """把样本外日期切成连续的折窗口——按日期连续性切，不依赖回传的折号。"""
    observed = pooled.notna().to_numpy().any(axis=1)
    dates = trained.loc[observed, meta.DATE_COLUMN].drop_duplicates().sort_values()
    days = dates.to_numpy().astype("datetime64[D]").astype(int)
    return np.split(dates.to_numpy(), np.flatnonzero(np.diff(days) > 1) + 1)


def _print_arms(
    trained: pd.DataFrame,
    pooled: pd.DataFrame,
    args: argparse.Namespace,
    panel: Panel,
) -> pd.DataFrame:
    """逐折的下注读数（三臂）。

    **必须逐折算，不能先拼成一条**：仓位只在样本外那几天存在，拼起来后其余日子按「零收益」
    计入均值和波动——实测夏普掉到逐折读数的三分之一，连符号都会翻。
    """
    cost = sizing.DEFAULT_ONE_WAY_COST * 1e4
    print(f"\n=== 下注规模（持有 {args.vertical} 天，单边成本 {cost:.1f}bp）===")
    rows: list[dict[str, Any]] = []
    for fold, block in enumerate(_fold_windows(trained, pooled), start=1):
        mask = np.isin(trained[meta.DATE_COLUMN].to_numpy(), block)
        window = trained.loc[mask]
        arms: dict[str, pd.Series] = {ALL_IN: pd.Series(1.0, index=window.index)}
        for name in MODEL_ARMS:
            arms[name] = cast(pd.Series, pooled.loc[mask, name])
        for name, probability in arms.items():
            stats = _arm_stats(window, probability, args.vertical, panel)
            rows.append(
                {
                    "折": fold,
                    "臂": name,
                    "毛夏普": stats.gross_sharpe,
                    "净夏普": stats.net_sharpe,
                    "暴露": stats.exposure,
                    "换手": float(stats.turnover.mean()),
                    "毛 bp/期": stats.gross * 1e4,
                    "净 bp/期": stats.net * 1e4,
                    "平衡成本 bp": stats.breakeven_cost * 1e4,
                    # 全下那一臂没有分类器输出，f1 无定义
                    "样本外 f1": _pooled_f1(window, probability)
                    if name in MODEL_ARMS
                    else float("nan"),
                }
            )
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    net = {(row["折"], row["臂"]): row["净夏普"] for row in rows}
    deltas = [net[(f, CV_ARM)] - net[(f, DEFAULT_ARM)] for f in table["折"].unique()]
    detail = ", ".join(f"{value:+.3f}" for value in deltas)
    print(f"\n逐折净夏普差（CV − 默认）：{detail}｜均值 {np.mean(deltas):+.3f}")
    return table


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore")
    trained, panel = _dataset(args)
    features = [*FEATURE_KEYS, AGE_KEY]
    base = meta.MetaLabelOptions(
        min_train=args.min_train,
        test_span=args.test_span,
        embargo=args.embargo,
        seed=args.seed,
        collect_fits=True,  # 要拿每折的训练/测试行号，否则没有折的切法
    )
    spans = _fold_spans(trained, features, base)
    if not spans:
        print("没有可用折：请调小 --min-train 或 --test-span")
        return 1
    picks = min(args.folds, len(spans))
    positions = np.unique(np.linspace(0, len(spans) - 1, num=picks, dtype=int))
    pooled = pd.DataFrame(
        np.nan, index=trained.index, columns=pd.Index(list(MODEL_ARMS))
    )
    searched: list[dict[str, Any]] = []
    for position in positions:
        train_index, test_index = spans[int(position)]
        train, test = trained.loc[train_index], trained.loc[test_index]
        problem = _problem(train, features, base, args)
        result = tuning.search_hyperparameters(
            problem, meta.RF_PARAM_SPACE, n_iter=args.n_iter, seed=args.seed
        )
        candidates = [
            {key: result.fold_scores[key][index] for key in meta.RF_DEFAULT_PARAMS}
            for index in range(len(result.fold_scores))
        ]
        searched.append({"candidates": candidates})
        pooled.loc[test_index, DEFAULT_ARM] = _probability(
            train, test, features, meta.RF_DEFAULT_PARAMS, base
        )
        pooled.loc[test_index, CV_ARM] = _probability(
            train, test, features, result.best_params, base
        )
        print(
            f"  折{int(position) + 1}/{len(spans)}｜训练 {len(train_index)} / 测试 "
            f"{len(test_index)}｜CV 最优 {result.best_params}（{result.best_score:.4f}）"
            f"｜就是默认配置 {result.best_params == meta.RF_DEFAULT_PARAMS}"
        )

    table = _print_arms(trained, pooled, args, panel)

    _record_trials(TrialLedger(), searched, args)
    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        target = OUTPUT_DIR / f"tuning_money_{args.market}_{args.timeframe}.parquet"
        table.to_parquet(target)
        # 概率单独落盘（带 date/symbol 键，能直接回接）：换下注口径不必重跑 CV 搜索
        probabilities = OUTPUT_DIR / (
            f"tuning_money_prob_{args.market}_{args.timeframe}.parquet"
        )
        trained[[meta.DATE_COLUMN, "symbol"]].join(pooled).to_parquet(probabilities)
        print(f"落盘：{target}、{probabilities}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
