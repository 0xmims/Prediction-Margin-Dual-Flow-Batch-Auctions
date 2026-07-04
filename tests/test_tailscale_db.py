"""Synthetic-fixture tests for the Tailscale replica adapter.

Nothing here touches a live database; fixtures mirror the replica's row shapes
(JSONB levels as ``{"price": str, "size": str}`` with signed diff deltas).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from pm_dfba_sim.data import tailscale_db as db
from pm_dfba_sim.run_tailscale_probe import ProbeStats, build_probe_report, summarize_for_report

UTC = timezone.utc
T0 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def minutes(count: float) -> timedelta:
    return timedelta(minutes=count)


# ---------------------------------------------------------------------------
# Query builders: every hypertable query must be time-bounded
# ---------------------------------------------------------------------------


WINDOWED_BUILDERS = [
    lambda start, end: db.daily_trade_aggregate_query(start, end),
    lambda start, end: db.capped_book_presence_query("m1", start, end),
    lambda start, end: db.daily_activity_query("orderbook_diffs", "m1", start, end),
    lambda start, end: db.hourly_diff_counts_query("m1", start, end),
    lambda start, end: db.window_snapshots_query("m1", start, end),
    lambda start, end: db.window_diffs_query("m1", start, end, limit=10),
    lambda start, end: db.window_trades_query("m1", start, end, limit=10),
    lambda start, end: db.diff_gap_query("m1", start, end),
    lambda start, end: db.minute_price_series_query("m1", start, end),
    lambda start, end: db.hourly_trade_counts_query("m1", start, end),
    lambda start, end: db.snapshot_cadence_query("m1", start, end),
    lambda start, end: db.snapshot_gap_query("m1", start, end, 1800.0),
    lambda start, end: db.resolution_query(["m1"], start, end),
]


@pytest.mark.parametrize("builder", WINDOWED_BUILDERS)
def test_builders_reject_missing_or_naive_bounds(builder):
    end = T0 + timedelta(days=1)
    with pytest.raises(db.TailscaleDBError):
        builder(None, end)
    with pytest.raises(db.TailscaleDBError):
        builder(T0, None)
    with pytest.raises(db.TailscaleDBError):
        builder(T0.replace(tzinfo=None), end)
    with pytest.raises(db.TailscaleDBError):
        builder(end, T0)  # reversed window


@pytest.mark.parametrize("builder", WINDOWED_BUILDERS)
def test_builders_emit_time_predicates_and_params(builder):
    end = T0 + timedelta(days=1)
    sql, params = builder(T0, end)
    assert "event_time" in sql
    assert sql.count("%(") >= 2
    time_params = [value for value in params.values() if isinstance(value, datetime)]
    assert len(time_params) >= 2


def test_daily_activity_rejects_unknown_table():
    with pytest.raises(db.TailscaleDBError):
        db.daily_activity_query("markets", "m1", T0, T0 + timedelta(days=1))
    with pytest.raises(db.TailscaleDBError):
        db.daily_activity_query("orderbook_diffs; DROP TABLE x", "m1", T0, T0 + timedelta(days=1))


def test_latest_snapshot_query_bounds_lookback():
    with pytest.raises(db.TailscaleDBError):
        db.latest_snapshot_query("m1", T0, timedelta(0))
    sql, params = db.latest_snapshot_query("m1", T0, timedelta(hours=48))
    assert params["start"] == T0 - timedelta(hours=48)
    assert "event_time <= %(at_time)s" in sql
    assert "event_time > %(start)s" in sql
    # The replica contract's tie-break: latest event_time, then highest seq.
    assert "ORDER BY event_time DESC, seq DESC" in sql


def test_snapshot_gap_query_guards_and_row_normalization():
    end = T0 + timedelta(days=1)
    with pytest.raises(db.TailscaleDBError):
        db.snapshot_gap_query("m1", T0, end, threshold_seconds=0)
    with pytest.raises(db.TailscaleDBError):
        db.snapshot_gap_query("m1", T0, end, 1800.0, limit=0)
    sql, params = db.snapshot_gap_query("m1", T0, end, 1800.0, limit=50)
    assert "LIMIT %(limit)s" in sql and params["limit"] == 50

    rows = [
        {"gap_start": T0 + minutes(120), "gap_end": T0 + minutes(180),
         "gap_seconds": Decimal("3600.0"), "previous_session_id": "s-1", "session_id": "s-1"},
        {"gap_start": T0, "gap_end": T0 + minutes(90),
         "gap_seconds": Decimal("5400.0"), "previous_session_id": "s-1", "session_id": "s-2"},
        {"gap_start": T0 + minutes(300), "gap_end": T0 + minutes(360),
         "gap_seconds": Decimal("3600.0"), "previous_session_id": None, "session_id": "s-2"},
    ]
    gaps = db.gap_rows_from_query(rows)
    assert [gap["gap_start"] for gap in gaps] == sorted(gap["gap_start"] for gap in gaps)
    assert gaps[0]["session_changed"] is True  # s-1 -> s-2
    assert gaps[1]["session_changed"] is False  # same session
    assert gaps[2]["session_changed"] is False  # NULL previous session is not a change
    assert all(isinstance(gap["gap_seconds"], float) for gap in gaps)


def test_limit_and_id_guards():
    end = T0 + timedelta(hours=1)
    with pytest.raises(db.TailscaleDBError):
        db.window_diffs_query("m1", T0, end, limit=0)
    with pytest.raises(db.TailscaleDBError):
        db.window_trades_query("m1", T0, end, limit=-5)
    with pytest.raises(db.TailscaleDBError):
        db.resolution_query([], T0, end)
    with pytest.raises(db.TailscaleDBError):
        db.markets_metadata_query([])


# ---------------------------------------------------------------------------
# Reconstruction: snapshot anchoring + signed deltas
# ---------------------------------------------------------------------------


def level(price: str, size: str) -> dict:
    return {"price": price, "size": size}


def test_signed_delta_application():
    side = db.book_side_from_json([level("0.50", "100"), level("0.49", "0.1")])
    assert side[Decimal("0.49")] == Decimal("0.1")

    db.apply_signed_diffs(side, [level("0.49", "0.2")])
    assert side[Decimal("0.49")] == Decimal("0.3")  # exact Decimal accumulation

    db.apply_signed_diffs(side, [level("0.50", "-40")])
    assert side[Decimal("0.50")] == Decimal("60")

    db.apply_signed_diffs(side, [level("0.50", "-60")])
    assert Decimal("0.50") not in side  # exact zero removes the level

    db.apply_signed_diffs(side, [level("0.49", "-99")])
    assert Decimal("0.49") not in side  # over-negative clamps to removal

    db.apply_signed_diffs(side, [level("0.44", "25")])
    assert side[Decimal("0.44")] == Decimal("25")


def test_snapshot_scoped_reconstruction_and_reanchoring():
    seed = {
        "event_time": T0,
        "seq": 100,
        "session_id": "s-1",
        "bids": [level("0.50", "100"), level("0.49", "200")],
        "asks": [level("0.52", "150")],
    }
    snapshots = [
        {
            "event_time": T0 + minutes(3),
            "seq": 200,
            "session_id": "s-2",
            "bids": [level("0.48", "300")],
            "asks": [level("0.53", "100")],
        }
    ]
    diffs = [
        # applies to the seed anchor
        {"event_time": T0 + minutes(1), "seq": 1, "snapshot_seq": 100,
         "bid_diffs": [level("0.50", "-40")], "ask_diffs": None, "session_id": "s-1"},
        # wrong anchor: must be skipped, never accumulated
        {"event_time": T0 + minutes(2), "seq": 2, "snapshot_seq": 99,
         "bid_diffs": [level("0.50", "999")], "ask_diffs": None, "session_id": "s-1"},
        # stale anchor arriving after the re-anchor: skipped
        {"event_time": T0 + minutes(4), "seq": 3, "snapshot_seq": 100,
         "bid_diffs": [level("0.48", "999")], "ask_diffs": None, "session_id": "s-1"},
        # applies to the new anchor
        {"event_time": T0 + minutes(5), "seq": 4, "snapshot_seq": 200,
         "bid_diffs": [level("0.49", "50")], "ask_diffs": [level("0.53", "-100")],
         "session_id": "s-2"},
    ]

    stats = db.FoldStats()
    states = [
        (state.event_time, state.best_bid, state.best_ask, state.anchor_seq)
        for state in db.fold_book_events(seed, snapshots, diffs, stats)
    ]

    # Yields: seed, diff seq 1, re-anchor snapshot, diff seq 4 (skips do not yield).
    assert [entry[3] for entry in states] == [100, 100, 200, 200]
    assert states[0][1:3] == (Decimal("0.50"), Decimal("0.52"))
    assert states[1][1:3] == (Decimal("0.50"), Decimal("0.52"))  # 0.50 reduced, still best
    assert states[2][1:3] == (Decimal("0.48"), Decimal("0.53"))  # full reset on re-anchor
    assert states[3][1:3] == (Decimal("0.49"), None)  # ask removed, bid level added

    assert stats.snapshots_applied == 2
    assert stats.diffs_applied == 2
    assert stats.diffs_skipped_wrong_anchor == 2
    assert stats.diffs_skipped_duplicate == 0


def test_fold_skips_duplicate_seq():
    seed = {"event_time": T0, "seq": 10, "session_id": None,
            "bids": [level("0.40", "10")], "asks": []}
    diff = {"event_time": T0 + minutes(1), "seq": 5, "snapshot_seq": 10,
            "bid_diffs": [level("0.40", "5")], "ask_diffs": None, "session_id": None}
    stats = db.FoldStats()
    list(db.fold_book_events(seed, [], [diff, dict(diff)], stats))
    assert stats.diffs_applied == 1
    assert stats.diffs_skipped_duplicate == 1


# ---------------------------------------------------------------------------
# Depth, imbalance, exit curves
# ---------------------------------------------------------------------------


def make_state() -> db.BookState:
    return db.BookState(
        bids={Decimal("0.50"): Decimal("100"), Decimal("0.46"): Decimal("50"),
              Decimal("0.41"): Decimal("25")},
        asks={Decimal("0.52"): Decimal("80"), Decimal("0.56"): Decimal("40"),
              Decimal("0.61"): Decimal("10")},
    )


def test_top_of_book_and_depth_bands():
    state = make_state()
    tob = db.top_of_book_metrics(state)
    assert tob["best_bid"] == Decimal("0.50")
    assert tob["best_ask"] == Decimal("0.52")
    assert tob["mid"] == Decimal("0.51")
    assert tob["spread"] == Decimal("0.02")

    depth = db.depth_metrics(state)
    assert depth["bid_depth_within_1c"] == Decimal("100")
    assert depth["ask_depth_within_1c"] == Decimal("80")
    assert depth["bid_depth_within_5c"] == Decimal("150")
    assert depth["ask_depth_within_5c"] == Decimal("120")
    assert depth["bid_depth_within_10c"] == Decimal("175")
    assert depth["ask_depth_within_10c"] == Decimal("130")
    assert depth["imbalance_5c"] == (Decimal("30") / Decimal("270"))


def test_depth_metrics_empty_book():
    depth = db.depth_metrics(db.BookState())
    assert depth["bid_depth_within_5c"] is None
    assert depth["imbalance_5c"] is None


def test_executable_sell_curve_partial_fill():
    state = make_state()
    curve = db.executable_exit_curve(state.bids, Decimal("1000"), is_sell=True)
    assert curve.side == "sell_yes"
    assert curve.filled_quantity == Decimal("175")
    assert curve.unfilled_quantity == Decimal("825")
    expected_value = (
        Decimal("100") * Decimal("0.50")
        + Decimal("50") * Decimal("0.46")
        + Decimal("25") * Decimal("0.41")
    )
    assert curve.executable_value == expected_value
    assert curve.worst_price == Decimal("0.41")
    assert curve.best_price == Decimal("0.50")
    assert curve.vwap_price == expected_value / Decimal("175")


def test_executable_buy_curve_full_fill():
    state = make_state()
    curve = db.executable_exit_curve(state.asks, Decimal("100"), is_sell=False)
    assert curve.side == "buy_yes"
    assert curve.filled_quantity == Decimal("100")
    assert curve.unfilled_quantity == Decimal("0")
    assert curve.executable_value == Decimal("80") * Decimal("0.52") + Decimal("20") * Decimal("0.56")
    assert curve.worst_price == Decimal("0.56")
    assert curve.levels_used == 2

    with pytest.raises(db.TailscaleDBError):
        db.executable_exit_curve(state.asks, Decimal("0"), is_sell=False)


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------


def test_detect_stream_gaps_and_session_change():
    rows = [
        (T0, "s-1"),
        (T0 + minutes(1), "s-1"),
        (T0 + minutes(61), "s-2"),  # one-hour hole with a session change
        (T0 + minutes(62), "s-2"),
    ]
    gaps = db.detect_stream_gaps(rows, timedelta(minutes=30))
    assert len(gaps) == 1
    assert gaps[0]["gap_seconds"] == 3600.0
    assert gaps[0]["session_changed"] is True

    assert db.detect_stream_gaps(rows, timedelta(hours=2)) == []


def test_day_coverage_summary_counts_internal_zero_days():
    rows = [
        {"day": datetime(2026, 6, 1, tzinfo=UTC)},
        {"day": datetime(2026, 6, 2, tzinfo=UTC)},
        {"day": datetime(2026, 6, 4, tzinfo=UTC)},
    ]
    summary = db.day_coverage_summary(rows)
    assert summary["days_present"] == 3
    assert summary["days_missing_between_first_last"] == 1
    assert summary["missing_days"][0].isoformat() == "2026-06-03"
    assert db.day_coverage_summary([])["days_present"] == 0


# ---------------------------------------------------------------------------
# Trade alignment
# ---------------------------------------------------------------------------


def test_normalize_probability_price():
    assert db.normalize_probability_price(Decimal("0.60")) == Decimal("0.60")
    assert db.normalize_probability_price(Decimal("83")) == Decimal("0.83")  # cent-quoted feed
    assert db.normalize_probability_price(None) is None


def test_align_trades_to_book_dual_conventions():
    series = [
        (T0, Decimal("0.50"), Decimal("0.52")),
        (T0 + minutes(1), Decimal("0.51"), Decimal("0.53")),
    ]
    trades = [
        {"event_time": T0 - minutes(1), "outcome": "yes", "price": Decimal("0.51"),
         "size": Decimal("10"), "taker_side": "buy"},  # before book: dropped
        {"event_time": T0 + timedelta(seconds=30), "outcome": "yes", "price": Decimal("0.52"),
         "size": Decimal("5"), "taker_side": "buy"},
        # NO print carrying a book-space price: identity fits, complement does not.
        {"event_time": T0 + minutes(2), "outcome": "no", "price": Decimal("0.52"),
         "size": Decimal("7"), "taker_side": "sell"},
        # NO print carrying a complement-space price: complement fits, identity does not.
        {"event_time": T0 + minutes(2), "outcome": "no", "price": Decimal("0.47"),
         "size": Decimal("7"), "taker_side": "buy"},
        {"event_time": T0 + minutes(3), "outcome": "TeamA", "price": Decimal("0.52"),
         "size": Decimal("1"), "taker_side": "buy"},  # non-binary: dropped
        {"event_time": T0 + minutes(3), "outcome": "yes", "price": Decimal("0.40"),
         "size": Decimal("2"), "taker_side": "sell"},  # far outside the spread
    ]
    aligned = db.align_trades_to_book(series, trades, half_tick=Decimal("0.005"))
    assert len(aligned) == 4

    first = aligned[0]
    assert first["best_bid"] == Decimal("0.50")
    assert first["in_spread_identity"] is True and first["at_touch_identity"] is True
    assert first["book_age_seconds"] == 30.0

    identity_no = aligned[1]
    assert identity_no["identity_price"] == Decimal("0.52")
    assert identity_no["complement_price"] == Decimal("0.48")
    assert identity_no["in_spread_identity"] is True
    assert identity_no["in_spread_complement"] is False

    complement_no = aligned[2]
    assert complement_no["complement_price"] == Decimal("0.53")
    assert complement_no["in_spread_identity"] is False
    assert complement_no["in_spread_complement"] is True

    outside = aligned[3]
    assert outside["in_spread_identity"] is False and outside["at_touch_identity"] is False


# ---------------------------------------------------------------------------
# Jump scanning
# ---------------------------------------------------------------------------


def test_detect_minute_jumps_and_labels():
    rows = [
        {"minute": T0 + minutes(index), "price_avg": "0.50", "trade_count": 3}
        for index in range(10)
    ]
    rows.append({"minute": T0 + minutes(10), "price_avg": "0.62", "trade_count": 9})
    rows.append({"minute": T0 + minutes(11), "price_avg": "0.63", "trade_count": 4})
    rows.append({"minute": T0 + minutes(40), "price_avg": "0.635", "trade_count": 1})

    jumps = db.detect_minute_jumps(rows, window_minutes=10)
    assert len(jumps) == 1  # consecutive flagged minutes collapse into one episode
    jump = jumps[0]
    assert jump["jump_size"] == Decimal("0.13")
    assert jump["direction"] == "up"
    assert jump["threshold_5c"] is True
    assert jump["threshold_10c"] is True
    assert jump["threshold_20c"] is False

    resolution_soon = jump["minute"] + timedelta(hours=3)
    resolution_far = jump["minute"] + timedelta(days=10)
    assert db.label_jump_timing(jump["minute"], resolution_soon, None) == (
        "terminal_near_resolution_candidate"
    )
    assert db.label_jump_timing(jump["minute"], resolution_far, None) == "interim_candidate"
    assert db.label_jump_timing(jump["minute"], None, None) == "unknown"


def test_detect_minute_jumps_quiet_series():
    rows = [
        {"minute": T0 + minutes(index), "price_avg": "0.50", "trade_count": 1}
        for index in range(30)
    ]
    assert db.detect_minute_jumps(rows) == []


def test_annotate_gaps_with_trades_flags_suspicious():
    gaps = [
        {"gap_start": T0, "gap_end": T0 + minutes(60), "gap_seconds": 3600.0,
         "session_changed": False},
        {"gap_start": T0 + minutes(120), "gap_end": T0 + minutes(180),
         "gap_seconds": 3600.0, "session_changed": False},
    ]
    trade_minutes = [T0 + minutes(30), T0 + minutes(31), T0 + minutes(200)]

    # Without diff hours: any trade inside a gap is suspicious (coarse mode).
    db.annotate_gaps_with_trades(gaps, trade_minutes)
    assert gaps[0]["trade_minutes_during_gap"] == 2
    assert gaps[0]["suspicious"] is True
    assert gaps[1]["trade_minutes_during_gap"] == 0
    assert gaps[1]["suspicious"] is False

    # With diff hours: trades during an hour that still produced diffs are not
    # a stall signal (T0 is 12:00, so the trades at 12:30/12:31 fall in hour T0).
    db.annotate_gaps_with_trades(gaps, trade_minutes, diff_hours={T0})
    assert gaps[0]["suspicious"] is False

    # Trades during a diff-free hour inside the silence remain suspicious.
    db.annotate_gaps_with_trades(gaps, trade_minutes, diff_hours={T0 + minutes(240)})
    assert gaps[0]["suspicious"] is True


# ---------------------------------------------------------------------------
# Classification and quotas
# ---------------------------------------------------------------------------


def test_classify_market_sports_via_event_title_and_prefix():
    sports = db.classify_market(
        "Will United States win on 2026-06-25?", "Türkiye vs. United States", None, "1897356"
    )
    assert sports.is_sports and sports.bucket == "sports"

    prefix = db.classify_market("Some market", None, "KXNFLGAME", "KXNFLGAME-26SEP01-KC")
    assert prefix.is_sports and prefix.matched.startswith("prefix:")

    cup = db.classify_market("Will Ecuador win the 2026 FIFA World Cup?", None, None, "558955")
    assert cup.is_sports


def test_classify_market_categories():
    assert db.classify_market("Will Democrats win the Senate in 2026?").category == "politics_election"
    assert db.classify_market("Fed rate cut in July?").category == "macro_finance_rates"
    assert db.classify_market("Bitcoin above $150k by year end?").category == "crypto_regulatory"
    assert db.classify_market("Will SCOTUS overturn the ruling?").category == "legal_court_policy"
    assert db.classify_market("Top Global Netflix Show on Jul 6?").category == "announcement_product"
    fallback = db.classify_market("Will it rain in NYC tomorrow?")
    assert fallback.category == "other" and fallback.bucket == "announcement_other"


def test_bucket_quotas_sum_to_max_markets():
    quotas = db.bucket_quotas(25)
    assert sum(quotas.values()) == 25
    assert quotas["quiet_baseline"] == 5
    assert quotas["politics"] == 8 and quotas["macro_finance_crypto"] == 8
    # The sum invariant must hold for every sample size, including tiny ones.
    for max_markets in range(1, 40):
        quotas = db.bucket_quotas(max_markets)
        assert sum(quotas.values()) == max_markets, max_markets
        assert all(quota >= 0 for quota in quotas.values()), max_markets
    with pytest.raises(db.TailscaleDBError):
        db.bucket_quotas(0)


# ---------------------------------------------------------------------------
# Config, credentials, and report safety
# ---------------------------------------------------------------------------


def test_dbconfig_from_env_and_redaction():
    secret = "sentinel-password-value"
    env = {
        db.ENV_HOST: "10.0.0.1",
        db.ENV_PORT: "5433",
        db.ENV_PASSWORD: secret,
    }
    config = db.DbConfig.from_env(env)
    assert config.port == 5433
    assert config.password == secret
    assert secret not in config.redacted_description()
    assert "10.0.0.1" not in config.redacted_description()
    assert config.connect_kwargs()["password"] == secret

    no_password = db.DbConfig.from_env({db.ENV_HOST: "10.0.0.1"})
    assert "password" not in no_password.connect_kwargs()  # defers to ~/.pgpass

    with pytest.raises(db.TailscaleDBError):
        db.DbConfig.from_env({})


def test_scan_outputs_for_secret(tmp_path):
    secret = "sentinel-password-value"
    clean = tmp_path / "clean.csv"
    clean.write_text("market_id,mid\nm1,0.5\n")
    dirty = tmp_path / "dirty.md"
    dirty.write_text(f"oops {secret} leaked")
    offending = db.scan_outputs_for_secret([clean, dirty, tmp_path / "missing.csv"], secret)
    assert offending == [dirty]
    assert db.scan_outputs_for_secret([dirty], None) == []


def _fake_result(market_id: str = "m1") -> dict:
    return {
        "market": {
            "market_id": market_id,
            "platform": "kalshi",
            "title": "Fed rate cut in July?",
            "category": "macro_finance_rates",
            "selected_bucket": "macro_finance_crypto",
            "contracts_volume": Decimal("5000"),
        },
        "snapshot_summary": {"total": 1200, "first_event": T0, "last_event": T0 + timedelta(days=20),
                             "max_sessions_per_day": 2, "days_present": 20, "days_missing": 1,
                             "missing_days": []},
        "diff_summary": {"total": 34000, "first_event": T0, "last_event": T0 + timedelta(days=20),
                         "max_sessions_per_day": None, "days_present": 20, "days_missing": 2,
                         "missing_days": []},
        "trade_summary": {"total": 900, "first_event": T0, "last_event": T0 + timedelta(days=20),
                          "max_sessions_per_day": None, "days_present": 18, "days_missing": 0,
                          "missing_days": []},
        "median_snapshot_cadence_seconds": 700.0,
        "heartbeat_gaps": [{"gap_start": T0, "gap_end": T0 + minutes(45),
                            "gap_seconds": 2700.0, "session_changed": True,
                            "trade_minutes_during_gap": 3, "suspicious": True}],
        "replay": None,
        "replay_day_gaps": [],
        "alignment": [
            {"trade_time": T0, "book_time": T0, "book_age_seconds": 1.5, "outcome": "yes",
             "taker_side": "buy", "raw_price": Decimal("0.5"),
             "identity_price": Decimal("0.5"), "complement_price": Decimal("0.5"),
             "size": Decimal("10"), "best_bid": Decimal("0.49"), "best_ask": Decimal("0.51"),
             "in_spread_identity": True, "at_touch_identity": False,
             "in_spread_complement": True, "at_touch_complement": False},
        ],
        "trades_in_replay_window": 1,
        "jumps": [
            {"minute": T0, "jump_size": Decimal("0.06"), "direction": "up",
             "price_before": Decimal("0.5"), "price_after": Decimal("0.56"), "trade_count": 4,
             "window_minutes": 10, "threshold_5c": True, "threshold_10c": False,
             "threshold_20c": False, "timing_label": "interim_candidate"},
        ],
        "resolution": {"outcome": "yes", "resolution_time": T0 + timedelta(days=25),
                       "close_time": T0 + timedelta(days=25)},
    }


def test_summarize_for_report_exit_slips_and_convention_tie():
    sell_curve = db.ExitCurve(
        side="sell_yes", quantity=Decimal("100"), filled_quantity=Decimal("100"),
        unfilled_quantity=Decimal("0"), executable_value=Decimal("48"),
        vwap_price=Decimal("0.48"), worst_price=Decimal("0.47"),
        best_price=Decimal("0.50"), levels_used=2,
    )
    buy_curve = db.ExitCurve(
        side="buy_yes", quantity=Decimal("100"), filled_quantity=Decimal("60"),
        unfilled_quantity=Decimal("40"), executable_value=Decimal("33"),
        vwap_price=Decimal("0.55"), worst_price=Decimal("0.58"),
        best_price=Decimal("0.52"), levels_used=1,
    )
    result = _fake_result()
    # Both alignment conventions agree on this market -> tie is indeterminate.
    result["replay"] = {
        "depth_rows": [
            {"bid_depth_within_5c": Decimal("150"), "ask_depth_within_5c": Decimal("120")},
        ],
        "exit_rows": [
            {"quantity_label": "100", "mid": Decimal("0.50"), "curve": sell_curve},
            {"quantity_label": "100", "mid": Decimal("0.50"), "curve": buy_curve},
        ],
        "fold_stats": db.FoldStats(),
    }
    summary = summarize_for_report([result])

    by_key = {(entry["quantity_label"], entry["side"]): entry for entry in summary["exit_fill_summary"]}
    sell = by_key[("100", "sell_yes")]
    buy = by_key[("100", "buy_yes")]
    # Selling below mid is positive slip: (0.50 - 0.48) * 100 = 2 cents.
    assert sell["median_slip_cents"] == pytest.approx(2.0)
    assert sell["full_fill_share"] == 1.0
    # Buying above mid is positive slip: (0.55 - 0.50) * 100 = 5 cents; partial fill.
    assert buy["median_slip_cents"] == pytest.approx(5.0)
    assert buy["full_fill_share"] == 0.0

    assert summary["depth_medians"]["bid_5c"] == pytest.approx(150.0)
    assert summary["alignment_summary"][0]["best_convention"] == "indeterminate"
    assert summary["alignment_summary"][0]["judged"] == 1


def test_report_generation_and_credential_hygiene():
    secret = "sentinel-password-value"
    results = [_fake_result()]
    context = {
        "generated_at": T0.isoformat(),
        "start": T0,
        "end": T0 + timedelta(days=30),
        "era_label": "monitored",
        "gap_threshold_seconds": 1800.0,
        "quotas": db.bucket_quotas(25),
        "results": results,
        "stats": ProbeStats(queries=42, db_seconds=12.5, rows_fetched=1000),
    }
    context.update(summarize_for_report(results))
    report = build_probe_report(context)

    assert secret not in report
    assert "PREDICTION_DB_PASSWORD" not in report
    for heading in (
        "## 1-2. Sampled markets",
        "## 3-4. Coverage windows and gaps",
        "## 6. Top-of-book reconstruction",
        "## 8. Executable exit curves",
        "## 9. Trade-to-book alignment",
        "## 10. Resolution join",
        "## 11. What this supports",
        "## What this probe can claim",
        "## What this probe cannot claim",
    ):
        assert heading in report
    assert "does not prove true stale-quote races" in report
    assert "requires order IDs" in report


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def test_parse_utc_timestamp_handles_zulu():
    parsed = db.parse_utc_timestamp("2026-05-27T00:00:00Z")
    assert parsed == datetime(2026, 5, 27, tzinfo=UTC)
    assert db.parse_utc_timestamp("2026-05-27T02:00:00+02:00") == datetime(2026, 5, 27, tzinfo=UTC)


def test_iterate_days_pages_cover_window():
    start = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    end = datetime(2026, 6, 4, 6, 0, tzinfo=UTC)
    pages = list(db.iterate_days(start, end))
    assert pages[0][0] == start
    assert pages[-1][1] == end
    for (_, page_end), (next_start, _) in zip(pages, pages[1:]):
        assert page_end == next_start
    assert all(page_end - page_start <= timedelta(days=1) for page_start, page_end in pages)
    with pytest.raises(db.TailscaleDBError):
        list(db.iterate_days(end, start))
