"""Synthetic-fixture tests for the event-window replay study. No live DB."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from pm_dfba_sim.data import event_study as es
from pm_dfba_sim.data import tailscale_db as db
from pm_dfba_sim.run_event_study import build_event_study_report
from pm_dfba_sim.run_tailscale_probe import ProbeStats

UTC = timezone.utc
ANCHOR = datetime(2026, 6, 17, 18, 0, tzinfo=UTC)
WINDOW = es.WindowSpec(
    baseline_start_minutes=-10,
    baseline_end_minutes=-2,
    impact_end_minutes=5,
    recovery_end_minutes=10,
    grid_seconds=60,
    trade_through_lags_seconds=(1.0, 30.0),
    exit_quantities=(Decimal("100"),),
    recovery_fraction=0.8,
)


def minutes(count: float) -> timedelta:
    return timedelta(minutes=count)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def base_config() -> dict:
    return {
        "window": {
            "baseline_start_minutes": -60,
            "baseline_end_minutes": -5,
            "impact_end_minutes": 15,
            "recovery_end_minutes": 60,
            "grid_seconds": 5,
            "trade_through_lags_seconds": [30, 1, 5],
            "exit_quantities": [100, 1000],
            "recovery_fraction": 0.8,
        },
        "events": [
            {
                "event_id": "evt-1",
                "name": "Test event",
                "anchor_time_utc": "2026-06-17T18:00:00Z",
                "category": "public_scheduled",
                "timestamp_basis": "test",
                "markets": [
                    {"market_id": "m1", "platform": "kalshi", "role": "terminal_leg"},
                    {"market_id": "m2", "platform": "polymarket", "role": "control"},
                ],
            }
        ],
    }


def test_load_event_config_normalizes_and_sorts_lags():
    window, events = es.load_event_config(base_config())
    assert window.trade_through_lags_seconds == (1.0, 5.0, 30.0)
    assert window.exit_quantities == (Decimal("100"), Decimal("1000"))
    assert events[0].anchor_time == ANCHOR
    assert events[0].markets[0].role == "terminal_leg"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda cfg: cfg["window"].update({"baseline_start_minutes": -5, "baseline_end_minutes": -10}),
        lambda cfg: cfg["window"].update({"impact_end_minutes": -1}),
        lambda cfg: cfg["window"].update({"grid_seconds": 0}),
        lambda cfg: cfg["window"].update({"trade_through_lags_seconds": [0]}),
        lambda cfg: cfg["window"].update({"exit_quantities": [-5]}),
        lambda cfg: cfg["window"].update({"recovery_fraction": 0}),
        lambda cfg: cfg["events"][0]["markets"][0].update({"role": "hero_leg"}),
        lambda cfg: cfg["events"][0].update({"markets": []}),
        lambda cfg: cfg.update({"events": []}),
    ],
)
def test_load_event_config_rejects_bad_input(mutate):
    config = base_config()
    mutate(config)
    with pytest.raises(es.EventStudyError):
        es.load_event_config(config)


def test_label_phase_boundaries():
    assert es.label_phase(ANCHOR + minutes(-11), ANCHOR, WINDOW) == "pre_baseline"
    assert es.label_phase(ANCHOR + minutes(-10), ANCHOR, WINDOW) == "baseline"
    assert es.label_phase(ANCHOR + minutes(-2), ANCHOR, WINDOW) == "pre_anchor_buffer"
    assert es.label_phase(ANCHOR, ANCHOR, WINDOW) == "impact"
    assert es.label_phase(ANCHOR + minutes(5), ANCHOR, WINDOW) == "recovery"
    assert es.label_phase(ANCHOR + minutes(10), ANCHOR, WINDOW) == "post"


# ---------------------------------------------------------------------------
# Timeline construction from a synthetic fold
# ---------------------------------------------------------------------------


def level(price: str, size: str) -> dict:
    return {"price": price, "size": size}


def synthetic_book():
    """A book that thins out just after the anchor and recovers later."""

    seed = {
        "event_time": ANCHOR + minutes(-12),
        "seq": 1,
        "session_id": "s",
        "bids": [level("0.50", "300"), level("0.48", "200")],
        "asks": [level("0.52", "300"), level("0.54", "200")],
    }
    diffs = [
        # Depth vanishes right after the anchor...
        {"event_time": ANCHOR + timedelta(seconds=5), "seq": 10, "snapshot_seq": 1,
         "bid_diffs": [level("0.50", "-290"), level("0.48", "-190")],
         "ask_diffs": [level("0.52", "-290")], "session_id": "s"},
        # ...and returns during recovery.
        {"event_time": ANCHOR + minutes(6), "seq": 11, "snapshot_seq": 1,
         "bid_diffs": [level("0.50", "290"), level("0.48", "190")],
         "ask_diffs": [level("0.52", "290")], "session_id": "s"},
    ]
    return seed, diffs


def test_build_book_timeline_and_phase_stats():
    seed, diffs = synthetic_book()
    timeline, tob_series = es.build_book_timeline(seed, [], diffs, ANCHOR, WINDOW)
    assert timeline, "grid rows expected"
    assert len(tob_series) == 3  # seed + two diffs

    # Grid rows exist for baseline, impact, and recovery phases.
    phases = {row["phase"] for row in timeline}
    assert {"baseline", "impact", "recovery"} <= phases

    stats = es.summarize_phases(timeline, WINDOW)
    # Baseline depth (5c both sides): 300+200 bids + 300 asks (0.54 is out of band
    # for mid 0.51... it is within 5c of mid on the ask side: 0.51+0.05=0.56 -> included)
    assert stats["baseline"]["depth_5c_median"] == pytest.approx(1000.0)
    # Impact-window book: bids 10 + 10, asks 10 + 200 -> depth 230.
    assert stats["impact"]["depth_5c_min"] == pytest.approx(230.0)
    assert stats["depth_decay_ratio"] == pytest.approx(0.23)
    assert stats["exit_value_haircut_ratio"] is not None
    assert stats["exit_value_haircut_ratio"] < 1.0
    # Exit quality: baseline sells 100 into 300@0.50 -> 50/(0.51*100) ~ 0.980;
    # impact book fills only 20 (10@0.50 + 10@0.48) -> 9.8/(0.51*100) ~ 0.192.
    assert stats["exit_quality_haircut_ratio"] == pytest.approx(0.196, abs=0.001)
    assert stats["spread_widening_ratio"] == pytest.approx(1.0)  # spread unchanged here

    # The anchor grid point itself samples the pre-jump book (as-of semantics:
    # the collapse diff lands 5s after the anchor).
    anchor_row = next(row for row in timeline if row["offset_minutes"] == 0.0)
    assert float(anchor_row["bid_depth_within_5c"]) == 500.0

    recovery = es.recovery_time_seconds(timeline, ANCHOR, WINDOW)
    assert recovery is not None
    assert 6 * 60 <= recovery <= 8 * 60  # depth returns at +6min, needs 2 grid points


def test_recovery_none_when_depth_never_returns():
    seed = {
        "event_time": ANCHOR + minutes(-12), "seq": 1, "session_id": "s",
        "bids": [level("0.50", "300")], "asks": [level("0.52", "300")],
    }
    diffs = [
        {"event_time": ANCHOR + timedelta(seconds=5), "seq": 2, "snapshot_seq": 1,
         "bid_diffs": [level("0.50", "-290")], "ask_diffs": [level("0.52", "-290")],
         "session_id": "s"},
    ]
    timeline, _ = es.build_book_timeline(seed, [], diffs, ANCHOR, WINDOW)
    assert es.recovery_time_seconds(timeline, ANCHOR, WINDOW) is None


# ---------------------------------------------------------------------------
# Trade-throughs against tau-lagged books
# ---------------------------------------------------------------------------


def test_trade_through_detects_lagged_violations_direction_agnostically():
    # Book displayed 0.50/0.52 until the anchor, then bid collapses to 0.40.
    tob_series = [
        (ANCHOR + minutes(-5), Decimal("0.50"), Decimal("0.52")),
        (ANCHOR + timedelta(seconds=2), Decimal("0.40"), Decimal("0.52")),
    ]
    trades = [
        # Baseline print inside the spread: no through under any lag.
        {"event_time": ANCHOR + minutes(-4), "price": Decimal("0.51"),
         "size": Decimal("10"), "outcome": "yes", "taker_side": "buy"},
        # Post-collapse sell print at 0.41: vs the 30s-lagged book (0.50 bid)
        # it is a through worth (0.50-0.41)*20; vs the 1s-lagged book (0.40 bid)
        # it is fine.
        {"event_time": ANCHOR + timedelta(seconds=3), "price": Decimal("0.41"),
         "size": Decimal("20"), "outcome": "yes", "taker_side": "sell"},
        # NO-labeled print above the lagged ask: buy-side through, proving the
        # detector needs no side semantics.
        {"event_time": ANCHOR + timedelta(seconds=4), "price": Decimal("0.60"),
         "size": Decimal("5"), "outcome": "no", "taker_side": "buy"},
        # Non-binary outcome ignored.
        {"event_time": ANCHOR + timedelta(seconds=5), "price": Decimal("0.41"),
         "size": Decimal("5"), "outcome": "TeamA", "taker_side": "buy"},
    ]
    per_trade, aggregates = es.trade_through_analysis(
        tob_series, trades, ANCHOR, WINDOW, half_tick=Decimal("0.005")
    )
    assert len(per_trade) == 3

    baseline_trade = per_trade[0]
    assert baseline_trade["through_lag_1s"] is False
    assert baseline_trade["through_lag_30s"] is False

    collapse_trade = per_trade[1]
    assert collapse_trade["through_lag_1s"] is False
    assert collapse_trade["through_lag_30s"] is True
    assert collapse_trade["stale_loss_lag_30s"] == Decimal("0.09") * 20

    no_print = per_trade[2]
    assert no_print["through_lag_30s"] is True
    assert no_print["stale_loss_lag_30s"] == Decimal("0.08") * 5

    impact_30 = aggregates[("impact", 30.0)]
    assert impact_30["judged"] == 2
    assert impact_30["throughs"] == 2
    assert impact_30["through_rate"] == 1.0
    baseline_30 = aggregates[("baseline", 30.0)]
    assert baseline_30["through_rate"] == 0.0


def test_trade_through_handles_trades_before_series():
    tob_series = [(ANCHOR, Decimal("0.50"), Decimal("0.52"))]
    trades = [
        {"event_time": ANCHOR - minutes(30), "price": Decimal("0.10"),
         "size": Decimal("1"), "outcome": "yes", "taker_side": "sell"},
    ]
    per_trade, aggregates = es.trade_through_analysis(tob_series, trades, ANCHOR, WINDOW)
    assert per_trade[0]["through_lag_1s"] is None
    assert aggregates[("pre_baseline", 1.0)]["judged"] == 0


# ---------------------------------------------------------------------------
# Suggestions and report
# ---------------------------------------------------------------------------


def fake_result(role: str, decay: float = 0.4, market_id: str = "m1") -> dict:
    return {
        "event_id": "evt-1",
        "event_name": "Test event",
        "category": "public_scheduled",
        "anchor_time": ANCHOR,
        "market_id": market_id,
        "platform": "kalshi",
        "role": role,
        "timeline": [],
        "phase_stats": {
            "exit_reference_quantity": "100",
            "baseline": {"rows": 10, "spread_median": 0.02, "spread_max": 0.03,
                         "depth_5c_median": 1000.0, "depth_5c_min": 900.0,
                         "exit_value_median": 49.0, "exit_value_min": 48.0,
                         "exit_quality_median": 0.98, "exit_quality_min": 0.96},
            "impact": {"rows": 5, "spread_median": 0.04, "spread_max": 0.08,
                       "depth_5c_median": 500.0, "depth_5c_min": 1000.0 * decay,
                       "exit_value_median": 40.0, "exit_value_min": 30.0,
                       "exit_quality_median": 0.8, "exit_quality_min": 0.6},
            "recovery": {"rows": 5, "spread_median": 0.02, "spread_max": 0.05,
                         "depth_5c_median": 900.0, "depth_5c_min": 700.0,
                         "exit_value_median": 47.0, "exit_value_min": 42.0,
                         "exit_quality_median": 0.94, "exit_quality_min": 0.84},
            "depth_decay_ratio": decay,
            "spread_widening_ratio": 4.0,
            "exit_value_haircut_ratio": 30.0 / 49.0,
            "exit_quality_haircut_ratio": 0.6 / 0.98,
        },
        "recovery_time_seconds": 420.0,
        "per_trade": [],
        "through_aggregates": {
            ("impact", 30.0): {"trades": 50, "judged": 45, "throughs": 9,
                               "through_rate": 0.2, "stale_loss_proxy": Decimal("12.5")},
            ("baseline", 30.0): {"trades": 40, "judged": 40, "throughs": 1,
                                 "through_rate": 0.025, "stale_loss_proxy": Decimal("0.5")},
        },
        "fold_stats": db.FoldStats(),
        "diffs_truncated": False,
        "trades_truncated": False,
        "trades_in_window": 90,
    }


def test_build_parameter_suggestions_separates_roles():
    window, _ = es.load_event_config(base_config())
    summaries = [
        fake_result("interim_leg", decay=0.4, market_id="m1"),
        fake_result("headline_target", decay=0.6, market_id="m2"),
        fake_result("terminal_leg", decay=0.2, market_id="m3"),
        fake_result("control", decay=0.95, market_id="m4"),
    ]
    suggestions = es.build_parameter_suggestions(summaries, window)
    assert suggestions["public_interim_candidates"]["markets"] == 2
    assert suggestions["terminal_leg_candidates"]["markets"] == 1
    assert suggestions["control_baseline"]["markets"] == 1
    assert suggestions["public_interim_candidates"]["depth_decay_ratio"]["p50"] in (0.4, 0.6)
    assert suggestions["data_limitations"]


def test_report_sections_and_credential_hygiene():
    window, _ = es.load_event_config(base_config())
    secret = "sentinel-password-value"
    context = {
        "generated_at": ANCHOR.isoformat(),
        "window": window,
        "results": [fake_result("interim_leg"), fake_result("control", decay=0.95, market_id="m4")],
        "stats": ProbeStats(queries=10, db_seconds=5.0, rows_fetched=100),
    }
    report = build_event_study_report(context)
    assert secret not in report
    for heading in (
        "## Per-event results",
        "## Falsification check",
        "## What this study can claim",
        "## What this study cannot claim",
    ):
        assert heading in report
    assert "does not prove latency races" in report
    assert "control replays at the same clock windows" in report
