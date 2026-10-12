"""试验账本：诚实地数 N。

de Prado 的第 7 条 pitfall 是「N 从未被诚实计数」。这个数字在研究中会以三种方式被
低估，三种都在本仓库发生过：

1. **只数本批候选**——`verify()` 的 `trials` 默认就是「本批候选数」；
2. **换 horizon 不算一次试验**（它算）；
3. **事后重建永远偏小**——失败的、放弃的、跑完没写下来的试验不留文件。

所以唯一可靠的做法是**在试验发生的时刻追加记录**，而不是事后统计。本模块因此是
一个 append-only 的 JSONL 账本。

口径（改代码前先读）：

- 一次 trial = 一个**被评估过的策略配置**，不是一次评估。同一配置在两个脚本里各跑过
  一次，仍然只算一次试验（身份由参数决定，重复记录幂等）。
- **N 的默认作用域是全库累计**，不是本批候选。窄作用域会让 DSR 变松，所以窄作用域
  必须由调用方显式要求，绝不能是默认值。
- **记的是「试过」，不是「好」。** 失败的试验同样要记——不记就会重蹈覆辙。
- 方向（`direction`）不进身份：它是从 IC 符号推出来的，不是我们选择的自由度。
  把它算进去会让同一配置因为行情变化而变成两次试验。

**账本不是防篡改的。** `data/` 整个目录被 `.gitignore` 忽略（`.gitignore:56`），
`git ls-files data/results` 为空——账本与同层的 `runs.json`、`provenance/` 同待遇，
删掉它是 `rm` 一下的事。所以「不可悄悄缩小」不靠改不动，靠**看得见**：
记录发生在代码路径里（调用方不必记得）、每次验证都把 N 打印出来（偏差会进日志）。
这比事后统计强，但不等于有人能拦住你删它——如果哪天需要真正的防篡改，
把 `trials.jsonl` 用 `git add -f` 单独入库即可，代价是它要随研究增长。

账本是**下界而不是真值**：建账之前的历史只能回填那些有据可查的（见
`scripts/trials_ledger.py`），更早的零星尝试无从重建。因为偏小而不是偏大，
它不会让结论变松。
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.config.paths import RESULTS_ROOT

LEDGER_PATH = RESULTS_ROOT / "trials.jsonl"
ORIGIN_RUN = "run"  # 试验发生时实时记录
ORIGIN_BACKFILL = "backfill"  # 事后按可查证据补录
FLOAT_PRECISION = 8


def _canonical(value: object) -> object:
    """把参数值压成稳定的 JSON 形态。

    两个坑：numpy 标量不是 JSON 可序列化的（`horizon` 从 DataFrame 里取出来常是
    `np.int64`），以及浮点表示噪声（`0.1 + 0.2 != 0.3`）——两者都会让同一配置算出
    两个身份、把 N 悄悄翻倍。
    """
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, (bool, str)):
        return value
    item = getattr(value, "item", None)  # numpy 标量
    if callable(item):
        return _canonical(item())
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return round(float(value), FLOAT_PRECISION)
    return str(value)


@dataclass(frozen=True)
class ResearchScope:
    """一次研究在账本里的位置。"""

    study: str
    market: str
    timeframe: str


@dataclass(frozen=True)
class Trial:
    """一个被评估过的策略配置。"""

    study: str
    market: str
    timeframe: str
    kind: str
    params: Mapping[str, object] = field(default_factory=dict)
    origin: str = ORIGIN_RUN
    source: str = ""

    @property
    def params_canonical(self) -> dict[str, object]:
        return {
            str(key): _canonical(value) for key, value in sorted(self.params.items())
        }

    @property
    def identity(self) -> str:
        payload = {
            "study": self.study,
            "market": self.market,
            "timeframe": self.timeframe,
            "kind": self.kind,
            "params": self.params_canonical,
        }
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()

    @property
    def label(self) -> str:
        detail = ", ".join(
            f"{key}={value}" for key, value in self.params_canonical.items()
        )
        return f"{self.study}/{self.market}/{self.timeframe}/{self.kind}({detail})"

    def as_record(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "study": self.study,
            "market": self.market,
            "timeframe": self.timeframe,
            "kind": self.kind,
            "params": self.params_canonical,
            "origin": self.origin,
            "source": self.source,
        }

    @staticmethod
    def from_record(record: Mapping[str, object]) -> Trial:
        params = record.get("params") or {}
        return Trial(
            study=str(record["study"]),
            market=str(record["market"]),
            timeframe=str(record["timeframe"]),
            kind=str(record["kind"]),
            params=params if isinstance(params, Mapping) else {},
            origin=str(record.get("origin", ORIGIN_RUN)),
            source=str(record.get("source", "")),
        )


def _matches(
    trial: Trial,
    study: str | None,
    market: str | None,
    timeframe: str | None,
    origin: str | None,
) -> bool:
    if study is not None and trial.study != study:
        return False
    if market is not None and trial.market != market:
        return False
    if timeframe is not None and trial.timeframe != timeframe:
        return False
    return origin is None or trial.origin == origin


class TrialLedger:
    """append-only 的试验账本。

    「append-only」是实质而非形式：`record` 只追加新行、从不重写文件，所以已有记录
    的消失只可能来自外部删改（git 会留下痕迹）。
    """

    def __init__(self, path: Path = LEDGER_PATH) -> None:
        self.path = Path(path)

    def records(self) -> list[Trial]:
        """读全部记录。文件不存在返回空表——没建账不等于没试过，调用方应据此报下界。"""
        if not self.path.exists():
            return []
        text = self.path.read_text(encoding="utf-8")
        return [
            Trial.from_record(json.loads(line))
            for line in text.splitlines()
            if line.strip()
        ]

    def identities(self) -> set[str]:
        return {trial.identity for trial in self.records()}

    def record(self, trials: Iterable[Trial]) -> int:
        """追加新试验（身份已存在则跳过），返回新增条数。"""
        known = self.identities()
        fresh: list[Trial] = []
        for trial in trials:
            marker = trial.identity
            if marker in known:
                continue
            known.add(marker)
            fresh.append(trial)
        if not fresh:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            for trial in fresh:
                line = json.dumps(trial.as_record(), ensure_ascii=False, sort_keys=True)
                handle.write(f"{line}\n")
        return len(fresh)

    def trials(
        self,
        *,
        study: str | None = None,
        market: str | None = None,
        timeframe: str | None = None,
        origin: str | None = None,
    ) -> list[Trial]:
        return [
            trial
            for trial in self.records()
            if _matches(trial, study, market, timeframe, origin)
        ]

    def count(
        self,
        *,
        study: str | None = None,
        market: str | None = None,
        timeframe: str | None = None,
        origin: str | None = None,
    ) -> int:
        return len(
            self.trials(study=study, market=market, timeframe=timeframe, origin=origin)
        )

    def breakdown(self, fields: Sequence[str] = ("study", "market")) -> pd.DataFrame:
        """按指定字段汇总试验数。空账本返回空表（不是 0 行合计为 1 的假表）。"""
        counts: Counter[tuple[str, ...]] = Counter()
        for trial in self.records():
            record = trial.as_record()
            counts[tuple(str(record[name]) for name in fields)] += 1
        rows = [
            {**dict(zip(fields, key, strict=True)), "trials": total}
            for key, total in sorted(counts.items(), key=lambda item: -item[1])
        ]
        return pd.DataFrame(rows)


def honest_trials(
    *,
    study: str | None = None,
    market: str | None = None,
    timeframe: str | None = None,
) -> int:
    """DSR 该用的 N：账本累计。

    返回 0 表示账本为空或没有匹配记录，调用方应退回「本批候选数」**并显式提示**，
    不能静默当作 N=0。
    """
    return TrialLedger().count(study=study, market=market, timeframe=timeframe)
