"""
三重障碍 + 唯一性权重 + 元标签：能不能胜过固定期限标签？

用三条对照把变量隔开（前两条是 de Prado Ch 3，第三条是 Ch 4）：

| 变体 | 训练标签 | 样本权重 | 隔离出的变量 |
| --- | --- | --- | --- |
| A | 固定期限（h 天收益符号） | 等权 | 基线，也是仓库现状 |
| B | 三重障碍 | 等权 | 只换标签 |
| C | 三重障碍 | 唯一性 | 只换权重 |

主模型是既有因子（只定方向），元模型回答「这一注要不要下」。
**评估用同一把尺**：三个变体都用固定 h 日前向收益衡量每注盈亏——
换标签只该改变分类器学到什么，不该顺带换掉打分标准。
三重障碍自己的实际退出收益另列一张表。

口径：

- 主模型用 rev_3（仓库 IC 最强的因子），side = sign(因子值)。
- 元标签 = 主模型那一注赚没赚钱。
- 特征 = 其余五个因子 + 上市天数 + SADF（不含 rev_3——它是主模型）。
- 滚动训练，按日期切分，训练与测试之间留 embargo（见 src/research/meta.py）。
- **唯一性权重在标的内部算**：t1 是标的内的根序号，长表行号是另一套坐标，
  混用会静默错位（踩过）。

用法：
    uv run python scripts/meta_label_study.py [--market swap] [--vertical 3] [--no-save]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.research import importance, meta, observations, sizing
from src.research.factors.library import FACTOR_LIBRARY
from src.research.factors.panel import (
    Panel,
    listing_age_days,
    load_panel,
    tradable_mask,
)
from src.research.trials import Trial, TrialLedger

PRIMARY_KEY = "rev_3"
FEATURE_KEYS = ("mom_7", "mom_30", "vol_30", "liq_30", "dth_60")
AGE_KEY = "age_days"
SADF_KEY = "sadf"
SADF_CACHE = Path("data/results/breaks/sadf_swap_1d.parquet")
OUTPUT_DIR = Path("data/results/breaks")
THRESHOLDS = (0.0, 0.5, 0.55, 0.6, 0.7)


def _stack(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """宽表转长表（date, symbol, name）。"""
    stacked = frame.stack(future_stack=True)
    stacked.index = stacked.index.set_names(["date", "symbol"])
    return pd.DataFrame({name: stacked}).reset_index()


def _feature_frame(panel: Panel) -> pd.DataFrame:
    """元模型的特征：长表（date, symbol, 各特征）。"""
    pieces = [_stack(FACTOR_LIBRARY[key].values(panel), key) for key in FEATURE_KEYS]
    pieces.append(_stack(listing_age_days(panel), AGE_KEY))
    cached = pd.read_parquet(SADF_CACHE)
    sadf = cached.reindex(index=panel.index, columns=panel.close.columns)
    pieces.append(_stack(sadf, SADF_KEY))
    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.merge(piece, on=["date", "symbol"], how="inner")
    return merged


def prepare(args: argparse.Namespace) -> tuple[pd.DataFrame, Panel]:
    panel = load_panel(args.timeframe, market=args.market)
    mask = tradable_mask(panel)
    print(
        f"{args.market} {args.timeframe}：{panel.close.shape[1]} 只 × {panel.close.shape[0]} 天"
    )

    side_frame = FACTOR_LIBRARY[PRIMARY_KEY].values(panel)
    table = observations.build_observations(panel, side_frame, args.vertical)
    tradable = pd.DataFrame({"tradable": mask.stack(future_stack=True)}).reset_index()
    tradable.columns = pd.Index(["date", "symbol", "tradable"])

    frame = table.merge(_feature_frame(panel), on=["date", "symbol"], how="inner")
    frame = frame.merge(tradable, on=["date", "symbol"], how="inner")
    frame = frame.loc[frame["tradable"].fillna(False)]
    frame = frame.dropna(subset=["side", "forward", "barrier_label"])
    frame = frame.loc[frame["side"] != 0].reset_index(drop=True)
    print(
        f"可交易观测 {len(frame)}，{frame['symbol'].nunique()} 只标的，"
        f"{frame['date'].nunique()} 天"
    )
    return frame, panel


def _wide(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """长表转宽表（index=date, columns=symbol）。"""
    return frame.pivot(index="date", columns="symbol", values=column)


def sizing_table(
    frame: pd.DataFrame,
    probability: pd.Series,
    horizon: int,
    *,
    panel: Panel,
) -> pd.DataFrame:
    """在同一批观测上比较下注方案（de Prado 第 10 章）。

    **P&L 用 1 日前向收益，不用 frame 里那列 h 日前向收益**：持有 h 天已经由持仓路径
    表达（过去 h 天期望仓位的均值），每期只该赚当天那一段。拿 h 日收益当 1 日用会把
    每期 P&L 放大 h 倍，并让相邻期共享 h-1 天内容——夏普被 √有效样本数 抬起来，
    而且各方案抬得不一样。

    只在**有样本外概率**的观测上比较：否则「全下」会连训练期一起下注，比较就偏了。
    """
    enriched = cast(pd.DataFrame, frame[["date", "symbol", "side"]].copy())
    # 按索引对齐（walk_forward_meta 保证概率与本 frame 逐行同序）
    observed = cast(pd.Series, probability.notna())
    enriched["side"] = cast(pd.Series, enriched["side"]).where(observed, 0.0)
    enriched["probability"] = probability
    side = _wide(enriched, "side")
    daily = cast(pd.DataFrame, panel.close.shift(-1).div(panel.close) - 1.0)
    options = sizing.SchemeOptions(horizon=horizon)
    table = sizing.compare_schemes(
        side,
        daily.reindex(index=side.index, columns=side.columns),
        _wide(enriched, "probability"),
        options=options,
    )
    print(
        f"\n=== 下注方案对比（持有 {options.horizon} 天，单边成本 "
        f"{options.cost_one_way * 1e4:.1f}bp）==="
    )
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    return table


def _record_trials(
    ledger: TrialLedger,
    label_column: str,
    weight_column: str,
    args: argparse.Namespace,
) -> None:
    """把本次评估过的每个「（标签口径 × 权重口径 × 阈值）」记进账本。"""
    added = ledger.record(
        [
            Trial(
                study="meta_label",
                market=args.market,
                timeframe=args.timeframe,
                kind=label_column,
                params={
                    "weight": weight_column,
                    "threshold": float(threshold),
                    "vertical": args.vertical,
                },
            )
            for threshold in THRESHOLDS
        ]
    )
    print(f"  试验账本：本次新增 {added} 条，累计 N={ledger.count()}")


def _print_importance(
    prediction: meta.MetaPrediction,
    trained: pd.DataFrame,
    features: list[str],
    options: meta.MetaLabelOptions,
) -> pd.DataFrame:
    """Ch 8：同一批折、同一批模型上算 MDA / 聚类 MDA / MDI / SFI，并做 PCA 正交对比。

    四者的分歧才是信息——MDI 是样本内的、会给噪声特征分重要性；逐特征 MDA 会被
    替代效应稀释；簇的联合 MDA 是「这一块信息值多少」；SFI 把每个特征单独喂一个
    模型，免疫替代效应但看不到联合效应。`pca_rank` 是**无监督**的（只看方差结构），
    它与 MDA 排序的 τ 回答的是「模型追的是不是数据的主结构」。
    """
    table = importance.importance_table(
        prediction.fits, trained, features, options=options
    )
    print()
    print(f"--- 特征重要性（Ch 8，{len(prediction.fits)} 折）---")
    print("mda_logloss = 打乱该列后样本外 log loss 的增加（越大越重要）")
    print(
        "sfi_skill   = 只用这一列时相对常数预测器的改进（越大越好，但看不到联合效应）"
    )
    print("pca_rank    = 无监督方差排名（1 = 主导），只看共变结构、不看标签")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    agreement = importance.rank_agreement(
        pd.Series(table["mda_logloss"].to_numpy(dtype=float)),
        pd.Series(table["pca_rank"].to_numpy(dtype=float)),
    )
    print(f"τ（MDA 排序 vs 逆 PCA 排名，原文 Snippet 8.6）= {agreement:+.4f}")
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description="三重障碍 + 元标签研究")
    parser.add_argument("--market", default="swap")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--vertical", type=int, default=3)
    parser.add_argument("--min-train", type=int, default=20000)
    parser.add_argument("--test-span", type=int, default=60)
    parser.add_argument("--embargo", type=int, default=200)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    # 特征重要性要额外重训 7 倍（SFI 逐特征、MDA 逐簇），占九成耗时；
    # 只关心下注方案（比如扫折数/树数的预算敏感性）时关掉它。
    parser.add_argument("--no-importance", action="store_true")
    args = parser.parse_args()

    frame, panel = prepare(args)
    unique = frame["weight_unique"].to_numpy(dtype=float)
    print(f"唯一性权重：均值 {unique.mean():.4f}  标准差 {unique.std():.4f}", end="")
    print("（固定期限标签下标准差恒为 0——那正是 Ch 4 之前接不动的原因）")

    outcomes = frame["side"].to_numpy() * frame["forward"].to_numpy()
    print(
        f"\n主模型（{PRIMARY_KEY}）全量下注：每注 {outcomes.mean() * 1e4:+.2f}bp  "
        f"命中率 {(outcomes > 0).mean():.1%}  共 {outcomes.size} 注"
    )
    exits = frame["barrier_ret"].to_numpy(dtype=float)
    print(f"三重障碍实际退出：每注 {np.nanmean(exits) * 1e4:+.2f}bp")

    variants = {
        "A 固定期限/等权": ("fixed_label", "weight_uniform"),
        "B 三重障碍/等权": ("barrier_label", "weight_uniform"),
        "C 三重障碍/唯一性": ("barrier_label", "weight_unique"),
    }
    options = meta.MetaLabelOptions(
        min_train=args.min_train,
        test_span=args.test_span,
        embargo=args.embargo,
        n_estimators=args.trees,
        seed=args.seed,
        collect_fits=True,
    )
    features = [*FEATURE_KEYS, AGE_KEY, SADF_KEY]
    results: dict[str, pd.DataFrame] = {}
    sizing_results: dict[str, pd.DataFrame] = {}
    importance_results: dict[str, pd.DataFrame] = {}
    ledger = TrialLedger()
    for name, (label_column, weight_column) in variants.items():
        # 元标签按定义是二元的：三重障碍的 label 是 {-1, 0, +1}，直接喂给分类器
        # 会变成三分类，predict_proba[:, 1] 取到的是「恰为 0」那一类（踩过）。
        binary = (frame[label_column] > 0).astype(int)
        trained = frame.copy()
        trained[meta.LABEL_COLUMN] = binary
        trained[meta.WEIGHT_COLUMN] = frame[weight_column]
        print(f"\n=== {name}：正类基准率 {binary.mean():.4f} ===")
        prediction = meta.walk_forward_meta(trained, features, options)
        print(f"  滚动 {prediction.folds} 折，样本外 {prediction.observations} 注")
        table = meta.filter_summary(
            frame["side"] * frame["forward"], prediction.probability, THRESHOLDS
        )
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        results[name] = table
        _record_trials(ledger, label_column, weight_column, args)
        sizing_results[name] = sizing_table(
            frame, prediction.probability, args.vertical, panel=panel
        )
        if not args.no_importance:
            importance_results[name] = _print_importance(
                prediction, trained, features, options
            )

    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        target = OUTPUT_DIR / f"meta_label_{args.market}_{args.timeframe}.parquet"
        pd.concat(results, names=["variant"]).reset_index(level=0).to_parquet(target)
        print(f"\n落盘：{target.resolve()}")
        sizing_target = (
            OUTPUT_DIR / f"bet_sizing_{args.market}_{args.timeframe}.parquet"
        )
        pd.concat(sizing_results, names=["variant"]).reset_index(level=0).to_parquet(
            sizing_target
        )
        print(f"落盘：{sizing_target.resolve()}")
        if importance_results:
            importance_target = (
                OUTPUT_DIR
                / f"feature_importance_{args.market}_{args.timeframe}.parquet"
            )
            pd.concat(importance_results, names=["variant"]).reset_index(
                level=0
            ).to_parquet(importance_target)
            print(f"落盘：{importance_target.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
