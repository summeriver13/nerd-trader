"""
趋势扫描标签与 side/size 联合学习（de Prado Ch 3.6/3.7）。

仓库现有策略的方向都来自一个因子（`rev_3` 分位），标签来自固定 h 日收益。Ch 3.6 给了
另一条路：标签由「前向窗口里 |t| 最大的趋势」给出（`src/research/labels.trend_scan`），
Ch 3.7 再把**方向与仓位都交给模型**。本脚本把它接到既有元标签管线上正面比一次。

**最容易搞错的一点（第一版就是错的）**：趋势扫描标签是用未来窗口算出来的，它不能当
可交易方向——拿它乘 h 日前向收益会得到每注 +445bp、命中率 73% 的幻觉。所以：

- 趋势扫描只当**训练目标**，方向由模型在样本外给（`p > 0.5` 做多、否则做空）；
- 标签自己的前向收益只报一次，名为 `trend_oracle`：它是「模型若能完美预测该标签」的
  **上界**，与可交易读数必须分开看。

口径（改代码前先读）：

1. **方向与收益同一持有期**：统一用 `forward`（h 日）算钱。
2. **对照臂只在趋势扫描可用的行上评估**：趋势扫描要 `span` 根暖机，对照臂不白拿那几行。
3. **行选取不许看标签**：带动态阈值的臂把「弱趋势」当**负类**（不是删掉那些行），
   否则「哪些行下注」本身就用了未来信息。
4. **窗口按有报价的根计数**（与三重障碍同一口径），缺口不补。
5. 账本：趋势扫描是新标签家族，按「标签口径 × 阈值」记账；对照臂写 `meta_label`
   （与第十七节同一批实验，不虚增 N）。
6. 下注方案（开关 vs 概率定仓）第十七节已答，本脚本不重复。

用法：uv run python scripts/trend_scan_study.py [--market swap] [--span 20] [--threshold 0.5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.research import labels, meta, observations
from src.research.factors.panel import Panel, load_panel
from src.research.observations import (
    AGE_COLUMN,
    FEATURE_FACTORS,
    PRIMARY_FACTOR,
    SADF_COLUMN,
)
from src.research.trials import Trial, TrialLedger

UNIFORM_WEIGHT = "weight_uniform"
OUTPUT_DIR = Path("data/results/breaks")
THRESHOLDS = (0.0, 0.5, 0.55, 0.6, 0.7)


def prepare(args: argparse.Namespace) -> tuple[pd.DataFrame, Panel]:
    """元标签观测表 + 趋势扫描标签（同一批行）。"""
    panel = load_panel(args.timeframe, market=args.market)
    frame = observations.build_meta_frame(
        panel,
        primary_key=PRIMARY_FACTOR,
        feature_keys=FEATURE_FACTORS,
        vertical=args.vertical,
    )
    cache = OUTPUT_DIR / f"sadf_{args.market}_{args.timeframe}.parquet"
    sadf = pd.read_parquet(cache).reindex(
        index=panel.index, columns=panel.close.columns
    )
    frame = frame.merge(
        observations.to_long(sadf, SADF_COLUMN), on=["date", "symbol"], how="inner"
    )
    frame = frame.merge(
        observations.trend_labels(panel, labels.TrendScanOptions(span=args.span)),
        on=["date", "symbol"],
        how="inner",
    )
    if args.threshold > 0.0:
        cut = observations.trend_labels(
            panel, labels.TrendScanOptions(span=args.span, threshold=args.threshold)
        )
        cut = cut[["date", "symbol", "trend_label"]]
        cut.columns = pd.Index(["date", "symbol", "cut_label"])
        frame = frame.merge(cut, on=["date", "symbol"], how="inner")
    print(
        f"{args.market} {args.timeframe}：{panel.close.shape[1]} 只 × "
        f"{panel.close.shape[0]} 天，共同观测 {len(frame)}，"
        f"{frame['symbol'].nunique()} 只，{frame['date'].nunique()} 天"
    )
    return frame.reset_index(drop=True), panel


def label_layer(frame: pd.DataFrame) -> pd.DataFrame:
    """标签层：趋势扫描标签的形态 + 两条「每注收益」（可交易 vs 上界）。"""
    forward = frame["forward"].to_numpy(dtype=float)
    trend = frame["trend_label"].to_numpy(dtype=float)
    side = frame["side"].to_numpy(dtype=float)
    spans = frame["trend_span"].to_numpy(dtype=float)
    magnitude = np.abs(frame["trend_t"].to_numpy(dtype=float))
    usable = np.isfinite(trend) & np.isfinite(forward)
    print("\n=== 趋势扫描标签的形态 ===")
    print(
        f"  可用行 {int(usable.sum())}（暖机丢弃 {int((~usable).sum())}）｜"
        f"窗口 中位 {np.nanmedian(spans):.0f}｜95 分位 {np.nanquantile(spans, 0.95):.0f}｜"
        f"|t| 中位 {np.nanmedian(magnitude):.1f}｜与主因子一致率 "
        f"{(trend[usable] == side[usable]).mean():.1%}"
    )
    rows: list[dict[str, object]] = []
    for name, direction in ((f"primary_{PRIMARY_FACTOR}", side), ("trend_oracle", trend)):
        taken = direction[usable] * forward[usable]
        taken = taken[direction[usable] != 0.0]
        rows.append(
            {
                "source": name,
                "bets": int(taken.size),
                "per_bet_bp": float(taken.mean() * 1e4),
                "hit_rate": float((taken > 0).mean()),
                "t_value": float(taken.mean() / taken.std() * np.sqrt(taken.size)),
            }
        )
    table = pd.DataFrame(rows)
    print(f"\n=== 标签层（共同样本 {int(usable.sum())} 行）===")
    print("（trend_oracle 用了未来窗口，**不是策略**，是模型预测该标签的上界）")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return table


def run_arm(
    name: str,
    study: str,
    kind: str,
    frame: pd.DataFrame,
    target: pd.Series,
    features: list[str],
    options: meta.MetaLabelOptions,
    ledger: TrialLedger,
    args: argparse.Namespace,
    *,
    exogenous: np.ndarray | None = None,
) -> pd.DataFrame:
    """一个臂：滚动训练 → 样本外概率 → 过滤表 + 账本。

    `exogenous` 给了就用它当方向（臂 A：方向来自主因子）；没给就用**模型预测**当方向
    （臂 B/C：Ch 3.7 的 side/size 联合学习）。两条路径都只用样本外概率。

    `study` 是账本口径：对照臂必须写 `meta_label`（与第十七节同一批实验，不能虚增 N），
    趋势扫描臂写 `trend_scan`（新标签家族）。
    """
    forward = frame["forward"].to_numpy(dtype=float)
    usable = np.isfinite(forward) & target.notna().to_numpy()
    if exogenous is not None:
        usable &= exogenous != 0.0
    subset = frame.loc[usable].reset_index(drop=True)
    trained = subset.copy()
    trained[meta.LABEL_COLUMN] = target.to_numpy(dtype=float)[usable].astype(int)
    trained[meta.WEIGHT_COLUMN] = subset[UNIFORM_WEIGHT]
    print(
        f"\n=== {name}：注数 {len(trained)}，正类基准率 "
        f"{trained[meta.LABEL_COLUMN].mean():.4f} ==="
    )
    prediction = meta.walk_forward_meta(trained, features, options)
    probability = prediction.probability
    observed = probability.notna().to_numpy()
    if exogenous is None:
        # Ch 3.7：方向也由模型给（样本外 p > 0.5 做多，否则做空）
        side = np.where(probability.to_numpy(dtype=float) > 0.5, 1.0, -1.0)
        side = np.where(observed, side, 0.0)
    else:
        side = np.where(observed, exogenous[usable], 0.0)
    auc = float(
        roc_auc_score(
            trained[meta.LABEL_COLUMN].to_numpy()[observed],
            probability.to_numpy(dtype=float)[observed],
        )
    )
    print(
        f"  滚动 {prediction.folds} 折，样本外 {prediction.observations} 注，AUC {auc:.4f}"
    )
    per_bet = pd.Series(side * forward[usable], index=subset.index, dtype=float)
    table = meta.filter_summary(per_bet, probability, THRESHOLDS)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    params: dict[str, object] = {"weight": UNIFORM_WEIGHT, "vertical": args.vertical}
    if kind != "fixed_label":
        params["span"] = args.span
    added = ledger.record(
        [
            Trial(
                study=study,
                market=args.market,
                timeframe=args.timeframe,
                kind=kind,
                params={**params, "threshold": float(value)},
            )
            for value in THRESHOLDS
        ]
    )
    print(f"  试验账本：本次新增 {added} 条，累计 N={ledger.count()}")
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description="趋势扫描标签 + side/size 联合学习")
    parser.add_argument("--market", default="swap")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--span", type=int, default=20, help="趋势扫描的最大窗口（根）")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="动态阈值分位（0 = 不设阈值）"
    )
    parser.add_argument("--vertical", type=int, default=3)
    parser.add_argument("--min-train", type=int, default=20000)
    parser.add_argument("--test-span", type=int, default=60)
    parser.add_argument("--embargo", type=int, default=8)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    frame, _ = prepare(args)
    label_result = label_layer(frame)

    options = meta.MetaLabelOptions(
        min_train=args.min_train,
        test_span=args.test_span,
        embargo=args.embargo,
        n_estimators=args.trees,
        seed=args.seed,
    )
    features = [*FEATURE_FACTORS, AGE_COLUMN, SADF_COLUMN]
    ledger = TrialLedger()
    arms = [
        (
            f"A 主因子 {PRIMARY_FACTOR} + 固定期限（对照）",
            "meta_label",
            "fixed_label",
            frame["fixed_label"],
            frame["side"].to_numpy(dtype=float),
        ),
        (
            f"B 趋势扫描 span={args.span}（模型学方向）",
            "trend_scan",
            "trend_label",
            (frame["trend_label"] > 0).astype(float),
            None,
        ),
    ]
    if "cut_label" in frame.columns:
        arms.append(
            (
                f"C 趋势扫描 + 阈值 {args.threshold:.2f}（弱趋势当负类）",
                "trend_scan",
                "trend_cut_label",
                (frame["cut_label"] > 0).astype(float),
                None,
            )
        )
    results: dict[str, pd.DataFrame] = {}
    for name, study, kind, target, direction in arms:
        results[name] = run_arm(
            name,
            study,
            kind,
            frame,
            target,
            features,
            options,
            ledger,
            args,
            exogenous=direction,
        )

    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        target_path = OUTPUT_DIR / f"trend_scan_{args.market}_{args.timeframe}.parquet"
        combined = pd.concat({"labels": label_result, **results}, names=["stage"])
        combined.reset_index(level=0).to_parquet(target_path)
        print(f"\n落盘：{target_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
