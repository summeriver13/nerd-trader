"""阈值阶梯的装载与口径自检（全部合成数据）。

抓的是真出过错的边界：slug 尾部的版本号被当成行权价（会造出 800/1000/2250 这类伪档）、
结算文本混用直引号与弯引号（“Low”）、模板同时写 "High" or "Low"（方向由标题定）。
"""

from __future__ import annotations

from src.research.polymarket.polymarket_ladder import (
    PRICE_TYPE_BOTH,
    WINDOW_CREATION,
    WINDOW_UNBOUNDED,
    ladder_row,
    load_ladders,
    parse_touch_rule,
    rule_warnings,
)

STRAIGHT = (
    'This market will immediately resolve to "Yes" if any Binance 1 minute candle for Bitcoin '
    "(BTC/USDT) between November 24, 2025, 14:00 and December 31, 2026, 23:59 in the ET timezone "
    'has a final "High" price equal to or greater than the price specified in the title.'
)
CURLY = (
    "This market will immediately resolve to “Yes” if any Binance 1 minute candle for Ethereum "
    "(ETH/USDT) from the creation of this market through 11:59 PM ET on the last day of the year "
    "specified in the title has a final “Low” price equal to or lower than the price specified."
)
BOTH = (
    'This market will immediately resolve to "Yes" if any Binance 1 minute candle for Bitcoin '
    "(BTC/USDT) between the creation of this market through 11:59 PM ET has a final "
    '"High" or "Low" price equal to the price specified in the title.'
)


def snapshot(
    slug: str, *, price: float = 0.5, volume: float = 1.0, text: str = STRAIGHT
) -> dict[str, object]:
    """造一条 markets 快照行；默认字段按真实快照的形态给（像字符串的 outcomePrices）。"""
    return {
        "slug": slug,
        "conditionId": f"0x{slug}",
        "outcomePrices": f'["{price}", "{1 - price}"]',
        "volumeNum": volume,
        "description": text,
        "createdAt": "2025-11-24T15:44:31Z",
        "endDate": "2027-01-01T00:00:00Z",
        "spread": "0.01",
        "bestBid": "0.499",
        "bestAsk": "0.501",
    }


def test_parse_touch_rule_reads_both_quote_styles() -> None:
    """上游同族市场混用直引号与弯引号，两种都必须认出来。"""
    straight = parse_touch_rule(STRAIGHT)
    curly = parse_touch_rule(CURLY)
    assert straight is not None and curly is not None
    assert (straight.price_type, straight.window_from) == ("High", "2025-11-24")
    assert (curly.price_type, curly.window_from) == ("Low", WINDOW_CREATION)
    assert straight.granularity_minutes == 1 and straight.immediate and curly.immediate


def test_parse_touch_rule_flags_ambiguous_price_type() -> None:
    """模板写 "High" or "Low"（方向由标题定）时不能当成矛盾，要标成 both。"""
    rule = parse_touch_rule(BOTH)
    assert rule is not None and rule.price_type == PRICE_TYPE_BOTH


def test_parse_touch_rule_returns_none_for_unknown_source() -> None:
    """换判定源就认不出来——宁可显式失败，也不能拿旧公式套新语义。"""
    assert parse_touch_rule(STRAIGHT.replace("Binance", "Coinbase")) is None
    assert parse_touch_rule("no window at all") is None


def test_ladder_row_ignores_version_suffix_as_strike() -> None:
    """slug 尾部的版本数字段不能被当成行权价，k 后缀要走千倍。"""
    row = ladder_row(
        snapshot(
            "will-bitcoin-dip-to-60k-by-december-31-2026-from-july-21",
            text=CURLY.replace("Ethereum", "Bitcoin"),
        )
    )
    assert row is not None and row.strike == 60_000.0 and row.asset == "bitcoin"
    assert row.direction == "down"
    assert row.window_from == WINDOW_CREATION


def test_ladder_row_prefers_hardcoded_window_when_text_says_so() -> None:
    """主阶梯（100k 那批）文本写死 November 24, 2025，不能被归成 creation。"""
    row = ladder_row(snapshot("will-bitcoin-reach-100000-by-december-31-2026-571-361"))
    assert row is not None and row.window_from == "2025-11-24"


def test_load_ladders_groups_by_asset_family_and_window() -> None:
    """分组键必须带资产：BTC 60k 与 ETH 2250 的窗口起点相同也不能混成一张阶梯。"""
    ladders = load_ladders(
        [
            snapshot("will-bitcoin-reach-90000-by-december-31-2026-113-862"),
            snapshot("will-bitcoin-dip-to-60000-by-december-31-2026-1-2"),
            snapshot("will-ethereum-dip-to-2250-by-december-31-2026", text=CURLY),
            snapshot("clarity-act-signed-into-law-in-2026"),
        ]
    )
    assert set(ladders) == {
        "bitcoin/reach/2025-11-24",
        "bitcoin/dip-to/2025-11-24",
        "ethereum/dip-to/creation",
    }
    assert [row.strike for row in ladders["bitcoin/reach/2025-11-24"]] == [90_000.0]


def test_load_ladders_keeps_largest_volume_per_strike() -> None:
    """同一档位有多次快照时只留成交额最大的那行，避免同价重复。"""
    ladders = load_ladders(
        [
            snapshot(
                "will-bitcoin-reach-90000-by-december-31-2026", price=0.4, volume=1.0
            ),
            snapshot(
                "will-bitcoin-reach-90000-by-december-31-2026", price=0.5, volume=9.0
            ),
        ]
    )
    rows = ladders["bitcoin/reach/2025-11-24"]
    assert len(rows) == 1 and rows[0].volume == 9.0 and rows[0].price == 0.5


def test_rule_warnings_catch_direction_conflict_and_accept_both() -> None:
    """文本与方向矛盾要报警；both 是已知模板，不算矛盾。"""
    down = ladder_row(snapshot("will-bitcoin-dip-to-55000-by-december-31-2026"))
    ambiguous = ladder_row(
        snapshot("will-bitcoin-dip-to-55000-by-december-31-2026", text=BOTH)
    )
    assert down is not None and ambiguous is not None
    assert any("down 档但文本写 High" in warning for warning in rule_warnings([down]))
    assert not rule_warnings([ambiguous])


def test_rule_warnings_flag_unknown_text() -> None:
    row = ladder_row(
        snapshot("will-bitcoin-reach-90000-by-december-31-2026", text="nope")
    )
    assert row is not None and row.window_from == WINDOW_UNBOUNDED
    assert any("认不出判定源" in warning for warning in rule_warnings([row]))
