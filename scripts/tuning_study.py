"""
scripts/tuning_study.py
第 9 章实证：**CV 调出来的超参，收益活得到样本外吗？**

用法：
    uv run python scripts/tuning_study.py --market swap
    uv run python scripts/tuning_study.py --market spot --folds 3 --n-iter 6 --no-save

设计（为什么这么测）：
- **内层用 purged K 折选参，外层用仓库既有的滚动折**——调参发生在训练窗**内部**，
  测试窗从不参与选择。这正是上线时该有的用法，也是原文 Snippet 9.1 的结构。
- **每个候选都在测试窗上算一次分数**，于是可以问一个比「调参有没有用」更狠的问题：
  **CV 的排序能不能预测样本外的排序**（Kendall τ）。τ 接近 0 就说明 CV 选出的「最好」
  只是噪声里的最大值。
- **每个候选都记进试验账本**：在 K 个配置里挑最好的那个，与在 K 个策略里挑最好的
  那个没有区别，都是多重检验，DSR 的分母要认这笔账。
- 只跑候选中包含默认配置的**同一组**折，默认配置的 CV 分与样本外分都单独算，作为基线。
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import stats

from src.config.paths import RESULTS_ROOT
from src.research import meta, observations, tuning
from src.research.factors.panel import Panel, load_panel
from src.research.trials import Trial, TrialLedger

PRIMARY_KEY: Final[str] = "rev_3"
FEATURE_KEYS: Final[tuple[str, ...]] = ("mom_7", "mom_30", "vol_30", "liq_30", "dth_60")
AGE_KEY: Final[str] = "age_days"
LABEL_SOURCE: Final[str] = observations.BARRIER_LABEL_COLUMN
WEIGHT_COLUMN: Final[str] = "weight_unique"
OUTPUT_DIR: Final[Path] = RESULTS_ROOT / "breaks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ch 9 超参调优实证")
    parser.add_argument("--market", default="swap")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--vertical", type=int, default=3, help="三重障碍的垂直屏障")
    parser.add_argument("--min-train", type=int, default=20000)
    parser.add_argument("--test-span", type=int, default=240, help="每折测试窗天数")
    parser.add_argument(
        "--embargo", type=int, default=8, help="训练与测试之间挖掉几个交易日"
    )
    parser.add_argument("--folds", type=int, default=4, help="抽样几折做调参")
    parser.add_argument(
        "--inner-splits", type=int, default=4, help="内层 purged K 折数"
    )
    parser.add_argument("--n-iter", type=int, default=8, help="随机搜索抽几个配置")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def prepare(args: argparse.Namespace) -> tuple[pd.DataFrame, Panel]:
    """搭元标签数据集：与 `meta_label_study.py` 同口径，只少一个 `sadf`。

    少它是因为那份缓存的路径按市场写死；调参研究要的是**折结构**，不是复刻特征集。
    """
    panel = load_panel(args.timeframe, market=args.market)
    frame = observations.build_meta_frame(
        panel,
        primary_key=PRIMARY_KEY,
        feature_keys=FEATURE_KEYS,
        vertical=args.vertical,
    )
    print(
        f"{args.market} {args.timeframe}：{panel.close.shape[1]} 只 × "
        f"{panel.close.shape[0]} 天｜可交易观测 {len(frame)}，"
        f"{frame['date'].nunique()} 天"
    )
    return frame, panel


def _classifier_factory(base: meta.MetaLabelOptions) -> tuning.EstimatorFactory:
    """按参数造模型（未拟合）：超参定义仍只有 `meta.build_classifier` 一处。"""

    def factory(params: dict[str, Any]) -> Any:
        return meta.build_classifier(replace(base, **params))

    return factory


def _f1_on(model: Any, frame: pd.DataFrame, features: list[str]) -> float:
    """在给定观测上算正类 f1（阈值 0.5，与 `tuning.fold_scores` 同口径）。"""
    probability = np.asarray(model.predict_proba(frame[features].to_numpy()))[:, 1]
    return tuning.positive_f1(
        frame[meta.LABEL_COLUMN].to_numpy(), (probability >= 0.5).astype(int)
    )


def _fit(
    train: pd.DataFrame,
    features: list[str],
    params: dict[str, Any],
    base: meta.MetaLabelOptions,
) -> Any:
    """在训练窗上拟合一个候选。"""
    model = meta.build_classifier(replace(base, **params))
    model.fit(
        train[features].to_numpy(),
        train[meta.LABEL_COLUMN].to_numpy(),
        sample_weight=train[meta.WEIGHT_COLUMN].to_numpy(),
    )
    return model


def evaluate_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    base: meta.MetaLabelOptions,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """一折的完整读数：CV 分数、样本外分数、CV 与样本外的排序一致性。"""
    codes = pd.factorize(train["date"], sort=True)[0]
    problem = tuning.TuningProblem(
        factory=_classifier_factory(base),
        features=train[features].to_numpy(),
        labels=train[meta.LABEL_COLUMN].to_numpy(),
        groups=codes,
        sample_weight=train[meta.WEIGHT_COLUMN].to_numpy(),
        cv=tuning.PurgedKFold(
            n_splits=args.inner_splits, purge=args.vertical, embargo=args.vertical
        ),
    )
    result = tuning.search_hyperparameters(
        problem, meta.RF_PARAM_SPACE, n_iter=args.n_iter, seed=args.seed
    )
    scores = result.fold_scores
    candidates = [
        {key: scores[key][index] for key in meta.RF_DEFAULT_PARAMS}
        for index in range(len(scores))
    ]
    out_of_sample = [
        _f1_on(_fit(train, features, params, base), test, features)
        for params in candidates
    ]
    default_cv, _ = tuning.fold_scores(problem, meta.RF_DEFAULT_PARAMS)
    default_oos = _f1_on(
        _fit(train, features, meta.RF_DEFAULT_PARAMS, base), test, features
    )
    correlation = float(
        pd.Series(scores["score"].to_numpy(dtype=float)).corr(
            pd.Series(np.asarray(out_of_sample)), method="kendall"
        )
    )
    best_index = int(np.argmax(scores["score"].to_numpy(dtype=float)))
    return {
        "folds_scored": len(scores),
        "cv_default": default_cv,
        "cv_best": float(scores["score"].max()),
        "oos_default": default_oos,
        "oos_best_choice": float(out_of_sample[best_index]),
        "oos_oracle": float(np.max(out_of_sample)),
        "rank_of_cv_best": int(
            stats.rankdata([-value for value in out_of_sample])[best_index]
        ),
        "kendall_tau": float(correlation),
        "best_params": result.best_params,
        "candidates": candidates,
    }


def _record_trials(
    ledger: TrialLedger, rows: list[dict[str, Any]], args: argparse.Namespace
) -> int:
    """每个**评估过的配置**记一条：搜索是独立于策略的一条多重检验轴。"""
    trials = [
        Trial(
            study="tuning",
            market=args.market,
            timeframe=args.timeframe,
            kind="rf_purged_kfold",
            params=dict(params),
        )
        for row in rows
        for params in row["candidates"]
    ]
    added = ledger.record(trials)
    print(f"\n试验账本：本次新增 {added} 条，累计 N={ledger.count()}")
    return added


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore")
    frame, _ = prepare(args)
    base = meta.MetaLabelOptions(
        min_train=args.min_train,
        test_span=args.test_span,
        embargo=args.embargo,
        seed=args.seed,
        collect_fits=True,  # 要拿每折的训练/测试行号，否则没有折的切法
    )
    features = [*FEATURE_KEYS, AGE_KEY]
    trained = frame.copy()
    trained[meta.LABEL_COLUMN] = (frame[LABEL_SOURCE] > 0).astype(int)
    trained[meta.WEIGHT_COLUMN] = frame[WEIGHT_COLUMN]
    print(f"正类基准率 {trained[meta.LABEL_COLUMN].mean():.4f}")

    prediction = meta.walk_forward_meta(trained, features, base)
    print(f"滚动 {prediction.folds} 折（只取折的切法，不代表这里的模型）")
    spans = [
        (trained.index[fitted.train_rows], trained.index[fitted.rows])
        for fitted in prediction.fits
    ]
    if not spans:
        print("没有可用折：请调小 --min-train 或 --test-span")
        return 1
    picks = min(args.folds, len(spans))
    positions = np.unique(np.linspace(0, len(spans) - 1, num=picks, dtype=int))
    rows: list[dict[str, Any]] = []
    for position in positions:
        train_index, test_index = spans[int(position)]
        row = evaluate_fold(
            trained.loc[train_index], trained.loc[test_index], features, base, args
        )
        rows.append(row)
        print(
            f"  折{int(position) + 1}/{len(spans)}｜训练 {len(train_index)} / 测试 "
            f"{len(test_index)}｜ΔCV {row['cv_best'] - row['cv_default']:+.4f}｜"
            f"ΔOOS {row['oos_best_choice'] - row['oos_default']:+.4f}｜"
            f"τ {row['kendall_tau']:+.3f}｜CV 最优在样本外排第 {row['rank_of_cv_best']}"
        )

    table = pd.DataFrame(rows)
    print(
        f"\n=== 汇总（{len(rows)} 折 × {args.n_iter} 个候选，内层 "
        f"{args.inner_splits} 折 purged CV）==="
    )
    print(
        f"  CV 分数     默认 {table['cv_default'].mean():.4f} → "
        f"最优 {table['cv_best'].mean():.4f}"
        f"（平均抬高 {(table['cv_best'] - table['cv_default']).mean():+.4f}）"
    )
    print(
        f"  样本外分数   默认 {table['oos_default'].mean():.4f} → "
        f"CV 选中 {table['oos_best_choice'].mean():.4f}"
        f"（平均变化 {(table['oos_best_choice'] - table['oos_default']).mean():+.4f}）"
    )
    print(
        f"  事后最优（oracle，不可得）{table['oos_oracle'].mean():.4f}"
        f" → 说明候选里确实有更好的，只是 CV 挑不出来"
    )
    print(
        f"  CV 排序 vs 样本外排序：Kendall τ 平均 "
        f"{table['kendall_tau'].mean():+.3f}（逐折 "
        f"{', '.join(f'{value:+.2f}' for value in table['kendall_tau'])}）"
    )
    ledger = TrialLedger()
    _record_trials(ledger, rows, args)
    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        target = OUTPUT_DIR / f"tuning_study_{args.market}_{args.timeframe}.parquet"
        table.drop(columns=["candidates"]).to_parquet(target)
        print(f"落盘：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
