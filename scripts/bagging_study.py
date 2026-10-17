"""Ch 6：袋化集成——把 Ch 4 的顺序采样器接进集成，比同骨架的均匀袋强多少？

五个臂，每个只差一件事：

| 臂 | 袋子 | 想隔离的变量 |
| --- | --- | --- |
| `single` | 均匀、1 棵树 | 「袋化到底有没有用」——与 `rf` 只差树数 |
| `rf` | 均匀（sklearn） | 现状：随机森林（均匀 bootstrap + 逐分裂特征子抽样） |
| `et` | 无 bootstrap | 把随机性来源从「袋子」换成「切分点」（ExtraTrees） |
| `uni` | 均匀（手写） | 与 `seq` **同骨架**，只差袋子的抽法——唯一能归因到采样器的对照 |
| `seq` | 顺序（Ch 4） | 本研究的正题 |

口径（与 `scripts/meta_label_study.py` 逐项对齐，只换估计器）：

- 主模型、特征、三套标签口径（A 固定期限 / B 三重障碍 / C 三重障碍+唯一性）、滚动切分与
  embargo 全部沿用元标签管道；打分也沿用同一把尺（固定 h 日前向收益的每注 bp + 命中率）。
- **袋子按标的分层抽**（`research.bagging` 的口径）：跨度坐标必须与标签同一套坐标系——
  观测日在自己标的 `close.dropna()` 里的根号，配 `barrier_t1`。对齐错了不报错，只会让顺序袋
  低估并发、静默退化成均匀袋。
- **方差用种子之间的离散度衡量**，不用折间离散度：袋化降的是「模型对随机性有多敏感」，
  折间差异混着市场状态，读不出这一点。

用法：
    uv run python scripts/bagging_study.py [--market swap] [--variants B] [--arms rf,uni,seq]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from src.research import bagging, ensembles, meta, observations
from src.research.factors.panel import load_panel
from src.research.observations import (
    AGE_COLUMN,
    FEATURE_FACTORS,
    PRIMARY_FACTOR,
    SADF_COLUMN,
)
from src.research.trials import Trial, TrialLedger
from src.utils.study_cli import add_market_argument, add_no_save, add_timeframe_argument

THRESHOLDS = (0.0, 0.55, 0.6, 0.7)
SHARES = (0.05, 0.02)
"""固定预算口径要取的份额（概率最高的前 5% / 前 2%）——臂间唯一可比的那把尺。"""
ARMS = ("single", "rf", "et", "uni", "seq")
VARIANTS: dict[str, tuple[str, tuple[str, str]]] = {
    "A": ("固定期限/等权", ("fixed_label", "weight_uniform")),
    "B": ("三重障碍/等权", ("barrier_label", "weight_uniform")),
    "C": ("三重障碍/唯一性", ("barrier_label", "weight_unique")),
}
OUTPUT_DIR = Path("data/results/breaks")


@dataclass(frozen=True)
class RunSettings:
    """一次运行的滚动口径与超参（从命令行来，不进各函数的参数表）。"""

    min_train: int = 20000
    test_span: int = 250
    embargo: int = 8
    trees: int = 200
    seed: int = 0

    @property
    def meta_options(self) -> meta.MetaLabelOptions:
        return meta.MetaLabelOptions(
            min_train=self.min_train,
            test_span=self.test_span,
            embargo=self.embargo,
            n_estimators=self.trees,
            seed=self.seed,
            collect_fits=False,
        )


def _span_frame(market: str, timeframe: str, *, vertical: int) -> pd.DataFrame:
    """元标签长表 + 跨度坐标四列（`bag_group`/`bag_start`/`bag_end`/`bag_size`）。

    根号取「观测日在自己标的 `close.dropna()` 里的位置」——与 `observations` 算
    `barrier_t1` 用的是同一条序列；`bag_end` 直接就是 `barrier_t1`。对齐错了不会报错，
    只会让顺序袋把并发算小、静默变成均匀袋。
    """
    panel = load_panel(timeframe, market=market)
    frame = observations.build_meta_frame(
        panel,
        primary_key=PRIMARY_FACTOR,
        feature_keys=FEATURE_FACTORS,
        vertical=vertical,
    )
    cached = pd.read_parquet(OUTPUT_DIR / f"sadf_{market}_{timeframe}.parquet")
    sadf = cached.reindex(index=panel.index, columns=panel.close.columns)
    frame = frame.merge(
        observations.to_long(sadf, SADF_COLUMN), on=["date", "symbol"], how="inner"
    )

    frame = ensembles.attach_spans(frame, panel.close)
    frame = frame.dropna(subset=[bagging.END_COLUMN])
    frame[bagging.END_COLUMN] = frame[bagging.END_COLUMN].to_numpy(dtype="int64")
    return frame.reset_index(drop=True)


def _share_table(outcomes: pd.Series, probability: pd.Series) -> pd.DataFrame:
    """固定预算读数：取概率最高的前 q 份，切点就是各臂自己概率的 (1−q) 分位。

    阈值口径**臂间不可比**：阈值相同、各臂的概率分布不同，下注数会差出近一倍，于是
    「每注赚多少」比的是不同的组合。固定份额把预算钉死，只比选出来的那批好不好；
    `level` 记的是份额，切点本身不跨臂可比。指标定义不另写一份——把切点当阈值交给
    `meta.filter_summary`，命中率/夏普/lift 与阈值口径逐位同定义。
    """
    usable = probability.dropna()
    cutoffs = tuple(float(usable.quantile(1.0 - share)) for share in SHARES)
    table = meta.filter_summary(outcomes, probability, cutoffs)
    table["mode"] = "share"
    table["level"] = list(SHARES)
    return table.drop(columns=["threshold"])


def _run_arm(
    frame: pd.DataFrame,
    arm: str,
    variant: tuple[str, str],
    *,
    settings: RunSettings,
    fit: meta.FitFunction,
) -> pd.DataFrame:
    """一个臂 × 一套标签口径：滚动预测 + 两种口径的每注读数（阈值 / 固定预算）。"""
    label_column, weight_column = variant
    train = frame.copy()
    train[meta.LABEL_COLUMN] = (frame[label_column] > 0).astype(int)
    train[meta.WEIGHT_COLUMN] = frame[weight_column]
    features = [*FEATURE_FACTORS, AGE_COLUMN, SADF_COLUMN]
    prediction = meta.walk_forward_meta(train, features, settings.meta_options, fit=fit)
    outcomes = frame["side"] * frame["forward"]
    filtered = meta.filter_summary(outcomes, prediction.probability, THRESHOLDS)
    filtered["mode"] = "threshold"
    filtered["level"] = filtered["threshold"]
    table = pd.concat(
        [
            filtered.drop(columns=["threshold"]),
            _share_table(outcomes, prediction.probability),
        ],
        ignore_index=True,
    )
    table.insert(0, "arm", arm)
    table["folds"] = prediction.folds
    table["observations"] = prediction.observations
    return table


def _report(combined: pd.DataFrame) -> None:
    """固定预算的臂间对比 + 种子之间的离散度（袋化降的是后者，不是折间差异）。"""
    budget = combined.loc[combined["mode"] == "share"]
    pivot = budget.pivot_table(
        index=["variant", "arm"], columns="level", values="per_bet_bp", aggfunc="median"
    )
    print("\n=== 固定预算：每注 bp（逐种子取中位）===")
    print(pivot.to_string(float_format=lambda v: f"{v:.1f}"))
    picked = combined.loc[
        (combined["mode"] == "share") & (combined["level"] == SHARES[-1])
    ]
    if picked["seed"].nunique() < 2:
        print("\n只跑了一颗种子：种子离散度不可读（--seeds ≥ 2 才有意义）")
        return
    spread = picked.pivot_table(
        index=["variant", "arm"], values="per_bet_bp", aggfunc="std"
    )
    print("\n=== 种子之间的每注 bp 标准差（阈值 0.60）===")
    print(spread.to_string(float_format=lambda v: f"{v:.2f}"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Ch 6 袋化集成研究")
    add_market_argument(parser, default="swap")
    add_timeframe_argument(parser)
    parser.add_argument("--vertical", type=int, default=3)
    parser.add_argument("--min-train", type=int, default=20000)
    parser.add_argument("--test-span", type=int, default=250)
    parser.add_argument("--embargo", type=int, default=8)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=1, help="每个臂重复跑几颗种子")
    parser.add_argument("--arms", default="rf,uni,seq")
    parser.add_argument("--variants", default="B")
    add_no_save(parser)
    args = parser.parse_args()

    frame = _span_frame(args.market, args.timeframe, vertical=args.vertical)
    print(
        f"{args.market} {args.timeframe}：{frame['symbol'].nunique()} 只 × "
        f"{frame['date'].nunique()} 天，{len(frame)} 行观测"
    )
    per_symbol = frame.groupby("symbol").size()
    print(
        f"逐标的观测数：中位 {per_symbol.median():.0f}  p5 {per_symbol.quantile(0.05):.0f}  "
        f"max {per_symbol.max():.0f}"
    )
    settings = RunSettings(
        min_train=args.min_train,
        test_span=args.test_span,
        embargo=args.embargo,
        trees=args.trees,
        seed=args.seed,
    )
    print("\n首训练窗的袋唯一性（逐标的取均值）：")
    uniqueness = ensembles.bag_uniqueness(
        frame, rows=settings.min_train, seed=settings.seed
    )
    print("\n".join(f"{name}: {value:.4f}" for name, value in uniqueness.items()))

    fitters = ensembles.arm_fitters(args.trees)
    names = [name for name in args.arms.split(",") if name]
    unknown = [name for name in names if name not in fitters]
    if unknown:
        raise SystemExit(f"未知的臂：{unknown}（可选 {ARMS}）")
    chosen = {key: VARIANTS[key] for key in args.variants.split(",") if key}
    ledger = TrialLedger()
    pieces: list[pd.DataFrame] = []
    for key, (label, variant) in chosen.items():
        for arm in names:
            for offset in range(args.seeds):
                seed = args.seed + offset
                print(f"\n--- {key} {label} × {arm} × seed={seed} ---", flush=True)
                result = _run_arm(
                    frame,
                    arm,
                    variant,
                    settings=replace(settings, seed=seed),
                    fit=fitters[arm],
                )
                result.insert(1, "variant", f"{key} {label}")
                result.insert(3, "seed", seed)
                pieces.append(result)
                print(
                    result.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
                    flush=True,
                )
                ledger.record(
                    [
                        Trial(
                            study="bagging",
                            market=args.market,
                            timeframe=args.timeframe,
                            kind=variant[0],
                            params={
                                "arm": arm,
                                "weight": variant[1],
                                "threshold": float(threshold),
                                "vertical": args.vertical,
                            },
                        )
                        for threshold in THRESHOLDS
                    ]
                )
    combined = pd.concat(pieces, ignore_index=True)
    _report(combined)
    print(f"\n试验账本：累计 N={ledger.count()}")
    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        target = OUTPUT_DIR / f"bagging_{args.market}_{args.timeframe}.parquet"
        combined.to_parquet(target)
        print(f"落盘：{target.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
