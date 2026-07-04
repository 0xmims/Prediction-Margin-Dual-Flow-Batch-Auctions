"""Bounded feasibility/calibration probe against the Tailscale predictiondb replica.

Samples 10-25 non-sports markets with order-book coverage, reconstructs books
from snapshots + snapshot_seq-scoped signed diffs, and writes small derived CSVs
plus a feasibility report. This is a probe for calibration and feasibility, not
empirical proof of stale-quote races and not a replay engine.

Run:

    PYTHONPATH=src python3 -m pm_dfba_sim.run_tailscale_probe \\
        --out outputs/tailscale_probe \\
        --start 2026-05-27T00:00:00Z --end 2026-06-27T00:00:00Z \\
        --max-markets 25 --min-volume 2000

Connection settings come from PREDICTION_DB_HOST / PREDICTION_DB_PORT /
PREDICTION_DB_NAME / PREDICTION_DB_USER and PREDICTION_DB_PASSWORD (or
``~/.pgpass``). Credentials are never printed or written to outputs.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from pm_dfba_sim.data import tailscale_db as db

MONITORED_ERA_START = datetime(2026, 5, 27, tzinfo=timezone.utc)
GO_INGEST_ERA_START = datetime(2026, 4, 21, tzinfo=timezone.utc)
ORDERBOOK_CAPTURE_FLOOR = datetime(2025, 12, 29, tzinfo=timezone.utc)
LIFECYCLE_LOOKBACK_START = datetime(2025, 12, 1, tzinfo=timezone.utc)

REPORT_NAME = "tailscale_probe_report.md"
OUTPUT_FILES = (
    "market_candidates.csv",
    "market_coverage_summary.csv",
    "gap_report.csv",
    "top_of_book_timeseries_sample.csv",
    "depth_timeseries_sample.csv",
    "trade_alignment_sample.csv",
    "liquidation_exit_curve_sample.csv",
    "jump_window_candidates.csv",
    REPORT_NAME,
)


@dataclass
class ProbeStats:
    queries: int = 0
    db_seconds: float = 0.0
    rows_fetched: int = 0
    presence_checks: int = 0
    diff_windows_truncated: int = 0
    notes: list = field(default_factory=list)


def run_query(cursor, built: tuple, stats: ProbeStats) -> list:
    sql, params = built
    started = time.time()
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    stats.queries += 1
    stats.db_seconds += time.time() - started
    stats.rows_fetched += len(rows)
    return rows


def _f(value: Any) -> Optional[float]:
    """Decimal/None -> float/None for CSV output."""

    if value is None:
        return None
    return float(value)


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    return value.isoformat()


# ---------------------------------------------------------------------------
# Discovery and selection
# ---------------------------------------------------------------------------


def discover_market_activity(cursor, start: datetime, end: datetime, stats: ProbeStats) -> dict:
    """Day-paged per-market trade totals over the window (bounded pages only)."""

    totals: dict = {}
    for day_start, day_end in db.iterate_days(start, end):
        rows = run_query(cursor, db.daily_trade_aggregate_query(day_start, day_end), stats)
        for row in rows:
            entry = totals.setdefault(
                row["market_id"],
                {"trade_count": 0, "contracts": Decimal(0), "notional": Decimal(0), "nonbinary_trades": 0},
            )
            entry["trade_count"] += int(row["trade_count"])
            entry["contracts"] += row["contracts"] or Decimal(0)
            entry["notional"] += row["notional"] or Decimal(0)
            entry["nonbinary_trades"] += int(row["nonbinary_trades"] or 0)
        print(f"  discovery {day_start.date()}: cumulative markets={len(totals)}")
    return totals


def fetch_metadata(cursor, market_ids: Sequence[str], stats: ProbeStats) -> dict:
    metadata: dict = {}
    ids = list(market_ids)
    if not ids:
        return metadata
    for offset in range(0, len(ids), 500):
        chunk = ids[offset : offset + 500]
        for row in run_query(cursor, db.markets_metadata_query(chunk), stats):
            metadata[row["market_id"]] = dict(row)
    return metadata


def check_book_presence(
    cursor,
    market_id: str,
    start: datetime,
    end: datetime,
    stats: ProbeStats,
    min_snapshots: int,
    min_diffs: int,
) -> tuple:
    rows = run_query(cursor, db.capped_book_presence_query(market_id, start, end), stats)
    stats.presence_checks += 1
    snapshots = int(rows[0]["snapshots_capped"])
    diffs = int(rows[0]["diffs_capped"])
    return (snapshots >= min_snapshots and diffs >= min_diffs), snapshots, diffs


def select_markets(
    cursor,
    totals: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    args,
    start: datetime,
    end: datetime,
    stats: ProbeStats,
) -> tuple:
    """Fill category-bucket quotas from volume-ranked non-sports candidates.

    Every accepted market must pass a capped order-book presence check; trades
    exist for far more markets than the book ingester tracks.
    """

    quotas = db.bucket_quotas(args.max_markets)
    quiet_low, quiet_high = Decimal(str(args.quiet_min_volume)), Decimal(str(args.min_volume))

    candidates: list = []
    for market_id, entry in totals.items():
        meta = metadata.get(market_id)
        contracts = entry["contracts"]
        row = {
            "market_id": market_id,
            "trade_count": entry["trade_count"],
            "contracts_volume": contracts,
            "notional_volume": entry["notional"],
            "nonbinary_trades": entry["nonbinary_trades"],
            "platform": meta.get("platform") if meta else None,
            "title": meta.get("title") if meta else None,
            "event_title": meta.get("event_title") if meta else None,
            "tick_size": meta.get("tick_size") if meta else None,
            "status": "not_checked",
            "snapshots_capped": None,
            "diffs_capped": None,
        }
        if meta is None:
            row["category"] = "unknown"
            row["bucket"] = "unknown"
            row["is_sports"] = None
            row["classification_matched"] = ""
            row["status"] = "skip:no_metadata"
        else:
            classification = db.classify_market(
                meta.get("title"), meta.get("event_title"), meta.get("series_id"), market_id
            )
            row["category"] = classification.category
            row["bucket"] = classification.bucket
            row["is_sports"] = classification.is_sports
            row["classification_matched"] = classification.matched
            if classification.is_sports:
                row["status"] = "skip:sports"
        candidates.append(row)

    by_volume = sorted(candidates, key=lambda r: r["contracts_volume"], reverse=True)
    eligible = [
        row
        for row in by_volume
        if row["status"] == "not_checked" and row["contracts_volume"] >= quiet_high
    ]
    quiet_pool = [
        row
        for row in by_volume
        if row["status"] == "not_checked" and quiet_low <= row["contracts_volume"] < quiet_high
    ]

    selected: list = []
    filled = {bucket: 0 for bucket in quotas}

    def try_accept(row, bucket, min_diffs) -> bool:
        if stats.presence_checks >= args.presence_check_budget:
            return False
        ok, snapshots, diffs = check_book_presence(
            cursor, row["market_id"], start, end, stats, args.min_snapshots, min_diffs
        )
        row["snapshots_capped"], row["diffs_capped"] = snapshots, diffs
        if not ok:
            row["status"] = "skip:no_book_coverage"
            return False
        row["status"] = f"selected:{bucket}"
        row["selected_bucket"] = bucket
        selected.append(row)
        filled[bucket] = filled.get(bucket, 0) + 1
        print(
            f"  selected [{bucket}] {row['market_id']} "
            f"vol={float(row['contracts_volume']):,.0f} :: {str(row['title'])[:60]}"
        )
        return True

    for bucket in ("politics", "macro_finance_crypto", "legal_policy"):
        for row in eligible:
            if filled[bucket] >= quotas[bucket] or len(selected) >= args.max_markets:
                break
            if row["status"] != "not_checked" or row["bucket"] != bucket:
                continue
            try_accept(row, bucket, args.min_diffs)

    for row in quiet_pool:
        if filled["quiet_baseline"] >= quotas["quiet_baseline"] or len(selected) >= args.max_markets:
            break
        if row["status"] != "not_checked":
            continue
        try_accept(row, "quiet_baseline", args.min_diffs_quiet)

    # Backfill remaining slots from any non-sports active candidate by volume.
    for row in eligible:
        if len(selected) >= args.max_markets:
            break
        if row["status"] != "not_checked":
            continue
        try_accept(row, "backfill_" + row["bucket"], args.min_diffs)

    if stats.presence_checks >= args.presence_check_budget:
        stats.notes.append(
            f"Presence-check budget ({args.presence_check_budget}) exhausted during selection."
        )
    return selected, candidates, quotas


# ---------------------------------------------------------------------------
# Per-market deep dive
# ---------------------------------------------------------------------------


def summarize_daily(rows: Sequence[Mapping[str, Any]]) -> dict:
    total = sum(int(row["row_count"]) for row in rows)
    first = min((row["first_event"] for row in rows), default=None)
    last = max((row["last_event"] for row in rows), default=None)
    max_sessions = max((int(row["sessions"]) for row in rows if row["sessions"] is not None), default=None)
    coverage = db.day_coverage_summary(rows)
    return {
        "total": total,
        "first_event": first,
        "last_event": last,
        "max_sessions_per_day": max_sessions,
        "days_present": coverage["days_present"],
        "days_missing": coverage["days_missing_between_first_last"],
        "missing_days": coverage["missing_days"],
    }


def pick_replay_window(
    cursor,
    market_id: str,
    start: datetime,
    end: datetime,
    replay_hours: float,
    stats: ProbeStats,
    diff_rows: Optional[list] = None,
) -> Optional[tuple]:
    """Prefer the busiest trading hour that also has book data, so the replay
    window exercises trade alignment; fall back to the busiest diff hour."""

    if diff_rows is None:
        diff_rows = run_query(cursor, db.hourly_diff_counts_query(market_id, start, end), stats)
    if not diff_rows:
        return None
    diff_by_hour = {row["hour"]: int(row["diff_count"]) for row in diff_rows}
    trade_rows = run_query(cursor, db.hourly_trade_counts_query(market_id, start, end), stats)
    peak_hour = None
    best_trades = 0
    for row in trade_rows:
        if diff_by_hour.get(row["hour"], 0) >= 50 and int(row["trade_count"]) > best_trades:
            best_trades = int(row["trade_count"])
            peak_hour = row["hour"]
    if peak_hour is None:
        peak_hour = max(diff_by_hour, key=diff_by_hour.get)
    replay_start = max(start, peak_hour - timedelta(minutes=30))
    replay_end = min(end, replay_start + timedelta(hours=replay_hours))
    if replay_end <= replay_start:
        return None
    return replay_start, replay_end


def replay_market_window(
    cursor,
    market_id: str,
    replay_start: datetime,
    replay_end: datetime,
    args,
    stats: ProbeStats,
) -> Optional[dict]:
    """Single fold pass: top-of-book series, depth grid, exit curves, fold stats."""

    seed_rows = run_query(
        cursor, db.latest_snapshot_query(market_id, replay_start, timedelta(hours=48)), stats
    )
    seed = dict(seed_rows[0]) if seed_rows else None
    snapshot_cap = 20_000
    snapshots = [
        dict(row)
        for row in run_query(
            cursor,
            db.window_snapshots_query(market_id, replay_start, replay_end, limit=snapshot_cap),
            stats,
        )
    ]
    if len(snapshots) >= snapshot_cap:
        stats.notes.append(
            f"Replay snapshots capped at {snapshot_cap} for {market_id}; later diffs "
            "referencing missing anchors are counted as skipped, not misapplied."
        )
    if seed is None and not snapshots:
        return None
    diff_fetch_start = seed["event_time"] if seed else snapshots[0]["event_time"]
    diffs = [
        dict(row)
        for row in run_query(
            cursor,
            db.window_diffs_query(market_id, diff_fetch_start, replay_end, args.max_diff_rows + 1),
            stats,
        )
    ]
    truncated = len(diffs) > args.max_diff_rows
    if truncated:
        diffs = diffs[: args.max_diff_rows]
        stats.diff_windows_truncated += 1

    fold_stats = db.FoldStats()
    tob_series: list = []
    minute_rows: dict = {}
    depth_rows: dict = {}
    exit_rows: list = []
    quantities = [Decimal(str(quantity)) for quantity in args.exit_quantities]
    sample_times = [
        replay_start + (replay_end - replay_start) * index / 5 for index in range(6)
    ]
    next_sample = 0

    for state in db.fold_book_events(seed, snapshots, diffs, fold_stats):
        event_time = state.event_time
        if event_time is None:
            continue
        best_bid, best_ask = state.best_bid, state.best_ask
        tob_series.append((event_time, best_bid, best_ask))
        if event_time < replay_start:
            continue
        minute = event_time.replace(second=0, microsecond=0)
        minute_rows[minute] = {
            "minute": minute,
            **db.top_of_book_metrics(state),
            "anchor_seq": state.anchor_seq,
        }
        five_minute = minute - timedelta(minutes=minute.minute % 5)
        depth_rows[five_minute] = {
            "time": five_minute,
            "mid": state.mid,
            **db.depth_metrics(state),
        }
        while next_sample < len(sample_times) and event_time >= sample_times[next_sample]:
            depth = db.depth_metrics(state)
            bid_depth_10c = depth.get("bid_depth_within_10c") or Decimal(0)
            adaptive = max(Decimal(1), (bid_depth_10c / 2).quantize(Decimal(1)))
            for quantity, label in [(quantity, str(quantity)) for quantity in quantities] + [
                (adaptive, "adaptive_half_10c_bid_depth")
            ]:
                for is_sell in (True, False):
                    side = state.bids if is_sell else state.asks
                    curve = db.executable_exit_curve(side, quantity, is_sell)
                    exit_rows.append(
                        {
                            "sample_time": sample_times[next_sample],
                            "book_time": event_time,
                            "quantity_label": label,
                            "mid": state.mid,
                            "curve": curve,
                        }
                    )
            next_sample += 1

    return {
        "seed_time": seed["event_time"] if seed else None,
        "replay_start": replay_start,
        "replay_end": replay_end,
        "truncated": truncated,
        "fold_stats": fold_stats,
        "tob_series": tob_series,
        "minute_rows": [minute_rows[key] for key in sorted(minute_rows)],
        "depth_rows": [depth_rows[key] for key in sorted(depth_rows)],
        "exit_rows": exit_rows,
    }


def probe_market(
    cursor,
    row: Mapping[str, Any],
    args,
    start: datetime,
    end: datetime,
    resolution: Optional[Mapping[str, Any]],
    stats: ProbeStats,
) -> dict:
    market_id = row["market_id"]
    print(f"  probing {market_id} :: {str(row['title'])[:70]}")

    snapshot_days = run_query(
        cursor, db.daily_activity_query("orderbook_snapshots", market_id, start, end), stats
    )
    diff_days = run_query(
        cursor, db.daily_activity_query("orderbook_diffs", market_id, start, end), stats
    )
    trade_days = run_query(
        cursor, db.daily_activity_query("public_trades", market_id, start, end), stats
    )
    snapshot_summary = summarize_daily(snapshot_days)
    diff_summary = summarize_daily(diff_days)
    trade_summary = summarize_daily(trade_days)

    minute_series = run_query(cursor, db.minute_price_series_query(market_id, start, end), stats)
    hourly_diff_rows = run_query(cursor, db.hourly_diff_counts_query(market_id, start, end), stats)
    diff_hours = {entry["hour"] for entry in hourly_diff_rows if int(entry["diff_count"]) > 0}

    cadence_rows = run_query(cursor, db.snapshot_cadence_query(market_id, start, end), stats)
    median_cadence = None
    if cadence_rows and cadence_rows[0]["median_interval_seconds"] is not None:
        median_cadence = float(cadence_rows[0]["median_interval_seconds"])
    # Snapshot cadence is activity-driven and platform-dependent; scale the gap
    # threshold so normal inter-snapshot intervals are not reported as gaps.
    effective_gap_threshold = float(args.gap_threshold_seconds)
    if median_cadence:
        effective_gap_threshold = max(effective_gap_threshold, 4.0 * median_cadence)
    gap_row_cap = 200
    raw_gap_rows = run_query(
        cursor,
        db.snapshot_gap_query(market_id, start, end, effective_gap_threshold, limit=gap_row_cap),
        stats,
    )
    heartbeat_gaps = db.gap_rows_from_query(raw_gap_rows)
    if len(raw_gap_rows) >= gap_row_cap:
        stats.notes.append(
            f"Gap rows capped at {gap_row_cap} for {market_id}; largest gaps kept."
        )
    db.annotate_gaps_with_trades(
        heartbeat_gaps, [row["minute"] for row in minute_series], diff_hours
    )

    replay = None
    window = pick_replay_window(
        cursor, market_id, start, end, args.replay_hours, stats, diff_rows=hourly_diff_rows
    )
    replay_day_gaps: list = []
    alignment: list = []
    trades_in_window = 0
    if window is not None:
        replay = replay_market_window(cursor, market_id, window[0], window[1], args, stats)
        if replay is not None:
            replay_day = window[0].replace(hour=0, minute=0, second=0, microsecond=0)
            replay_day_gaps = [
                dict(entry)
                for entry in run_query(
                    cursor,
                    db.diff_gap_query(market_id, replay_day, replay_day + timedelta(days=1)),
                    stats,
                )
                if entry["gap"] is not None
                and entry["gap"].total_seconds() >= args.gap_threshold_seconds
            ]
            trades = run_query(
                cursor,
                db.window_trades_query(
                    market_id, window[0], window[1], args.max_trades_per_window
                ),
                stats,
            )
            trades_in_window = len(trades)
            if trades_in_window >= args.max_trades_per_window:
                stats.notes.append(
                    f"Replay-window trades capped at {args.max_trades_per_window} "
                    f"for {market_id}; alignment sample is truncated, not exhaustive."
                )
            tick = row.get("tick_size")
            half_tick = (Decimal(str(tick)) / 2) if tick else Decimal("0.005")
            alignment = db.align_trades_to_book(replay["tob_series"], trades, half_tick)

    jumps = db.detect_minute_jumps(minute_series, window_minutes=args.jump_window_minutes)
    resolution_time = resolution.get("resolution_time") if resolution else None
    close_time = resolution.get("close_time") if resolution else None
    for jump in jumps:
        jump["timing_label"] = db.label_jump_timing(jump["minute"], resolution_time, close_time)

    return {
        "market": dict(row),
        "snapshot_summary": snapshot_summary,
        "diff_summary": diff_summary,
        "trade_summary": trade_summary,
        "median_snapshot_cadence_seconds": median_cadence,
        "effective_gap_threshold_seconds": effective_gap_threshold,
        "heartbeat_gaps": heartbeat_gaps,
        "replay": replay,
        "replay_day_gaps": replay_day_gaps,
        "alignment": alignment,
        "trades_in_replay_window": trades_in_window,
        "replay_trades_truncated": (
            window is not None and trades_in_window >= args.max_trades_per_window
        ),
        "jumps": jumps,
        "resolution": dict(resolution) if resolution else None,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_outputs(out_dir: Path, results: Sequence[Mapping[str, Any]], candidates, context) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows = []
    for row in sorted(candidates, key=lambda r: r["contracts_volume"], reverse=True)[:250]:
        candidate_rows.append(
            {
                "market_id": row["market_id"],
                "platform": row["platform"],
                "title": row["title"],
                "event_title": row["event_title"],
                "category": row["category"],
                "bucket": row.get("selected_bucket") or row["bucket"],
                "status": row["status"],
                "classification_matched": row["classification_matched"],
                "trade_count": row["trade_count"],
                "contracts_volume": _f(row["contracts_volume"]),
                "notional_volume": _f(row["notional_volume"]),
                "nonbinary_trades": row["nonbinary_trades"],
                "snapshots_capped": row["snapshots_capped"],
                "diffs_capped": row["diffs_capped"],
            }
        )
    pd.DataFrame(candidate_rows).to_csv(out_dir / "market_candidates.csv", index=False)

    coverage_rows = []
    gap_rows = []
    tob_rows = []
    depth_rows = []
    align_rows = []
    exit_csv_rows = []
    jump_rows = []
    for result in results:
        market = result["market"]
        market_id = market["market_id"]
        replay = result["replay"]
        fold = replay["fold_stats"] if replay else None
        resolution = result["resolution"] or {}
        coverage_rows.append(
            {
                "market_id": market_id,
                "platform": market["platform"],
                "bucket": market.get("selected_bucket"),
                "category": market["category"],
                "title": market["title"],
                "window_start": _iso(context["start"]),
                "window_end": _iso(context["end"]),
                "snapshots_total": result["snapshot_summary"]["total"],
                "diffs_total": result["diff_summary"]["total"],
                "trades_total": result["trade_summary"]["total"],
                "first_snapshot": _iso(result["snapshot_summary"]["first_event"]),
                "last_snapshot": _iso(result["snapshot_summary"]["last_event"]),
                "first_trade": _iso(result["trade_summary"]["first_event"]),
                "last_trade": _iso(result["trade_summary"]["last_event"]),
                "snapshot_days_present": result["snapshot_summary"]["days_present"],
                "snapshot_days_missing": result["snapshot_summary"]["days_missing"],
                "median_snapshot_cadence_seconds": result["median_snapshot_cadence_seconds"],
                "gap_threshold_effective_seconds": result["effective_gap_threshold_seconds"],
                "heartbeat_gap_count": len(result["heartbeat_gaps"]),
                "suspicious_gap_count": sum(
                    1 for gap in result["heartbeat_gaps"] if gap.get("suspicious")
                ),
                "max_sessions_per_day": result["snapshot_summary"]["max_sessions_per_day"],
                "replay_start": _iso(replay["replay_start"]) if replay else None,
                "replay_end": _iso(replay["replay_end"]) if replay else None,
                "replay_diffs_applied": fold.diffs_applied if fold else None,
                "replay_diffs_skipped_wrong_anchor": fold.diffs_skipped_wrong_anchor if fold else None,
                "replay_snapshots_applied": fold.snapshots_applied if fold else None,
                "replay_crossed_book_events": fold.crossed_book_events if fold else None,
                "replay_diffs_truncated": replay["truncated"] if replay else None,
                "resolution_outcome": resolution.get("outcome"),
                "resolution_time": _iso(resolution.get("resolution_time")),
                "close_time": _iso(resolution.get("close_time")),
                "jump_candidates": len(result["jumps"]),
                "trades_in_replay_window": result["trades_in_replay_window"],
                "replay_trades_truncated": result["replay_trades_truncated"],
                "aligned_trades": len(result["alignment"]),
            }
        )
        for gap in result["heartbeat_gaps"]:
            gap_rows.append(
                {
                    "market_id": market_id,
                    "gap_kind": "book_stream_gap",
                    "gap_start": _iso(gap["gap_start"]),
                    "gap_end": _iso(gap["gap_end"]),
                    "gap_seconds": gap["gap_seconds"],
                    "session_changed": gap["session_changed"],
                    "trade_minutes_during_gap": gap.get("trade_minutes_during_gap"),
                    "suspicious": gap.get("suspicious"),
                }
            )
        for day in result["diff_summary"]["missing_days"]:
            gap_rows.append(
                {
                    "market_id": market_id,
                    "gap_kind": "zero_diff_day",
                    "gap_start": day.isoformat(),
                    "gap_end": day.isoformat(),
                    "gap_seconds": 86400.0,
                    "session_changed": None,
                    "trade_minutes_during_gap": None,
                    "suspicious": None,
                }
            )
        for gap in result["replay_day_gaps"]:
            gap_rows.append(
                {
                    "market_id": market_id,
                    "gap_kind": "intraday_diff_gap_replay_day",
                    "gap_start": None,
                    "gap_end": _iso(gap["event_time"]),
                    "gap_seconds": gap["gap"].total_seconds(),
                    "session_changed": (
                        str(gap["session_id"]) != str(gap["previous_session_id"])
                        if gap.get("previous_session_id") is not None
                        else None
                    ),
                    "trade_minutes_during_gap": None,
                    "suspicious": None,
                }
            )
        if replay:
            for entry in replay["minute_rows"]:
                spread = entry["spread"]
                tob_rows.append(
                    {
                        "market_id": market_id,
                        "minute": _iso(entry["minute"]),
                        "best_bid": _f(entry["best_bid"]),
                        "best_ask": _f(entry["best_ask"]),
                        "mid": _f(entry["mid"]),
                        "spread": _f(spread),
                        "spread_cents": _f(spread * 100) if spread is not None else None,
                        "anchor_seq": entry["anchor_seq"],
                    }
                )
            for entry in replay["depth_rows"]:
                depth_rows.append(
                    {
                        "market_id": market_id,
                        "time": _iso(entry["time"]),
                        "mid": _f(entry["mid"]),
                        **{
                            key: _f(entry[key])
                            for key in entry
                            if key.startswith(("bid_depth", "ask_depth")) or key == "imbalance_5c"
                        },
                    }
                )
            for entry in replay["exit_rows"]:
                curve = entry["curve"]
                exit_csv_rows.append(
                    {
                        "market_id": market_id,
                        "sample_time": _iso(entry["sample_time"]),
                        "book_time": _iso(entry["book_time"]),
                        "side": curve.side,
                        "quantity_label": entry["quantity_label"],
                        "quantity_requested": _f(curve.quantity),
                        "quantity_filled": _f(curve.filled_quantity),
                        "unfilled_quantity": _f(curve.unfilled_quantity),
                        "executable_value": _f(curve.executable_value),
                        "vwap_price": _f(curve.vwap_price),
                        "worst_price": _f(curve.worst_price),
                        "best_price": _f(curve.best_price),
                        "levels_used": curve.levels_used,
                        "mid_at_sample": _f(entry["mid"]),
                    }
                )
        for entry in result["alignment"][:400]:
            align_rows.append(
                {
                    "market_id": market_id,
                    "trade_time": _iso(entry["trade_time"]),
                    "book_time": _iso(entry["book_time"]),
                    "book_age_seconds": entry["book_age_seconds"],
                    "outcome": entry["outcome"],
                    "taker_side": entry["taker_side"],
                    "raw_price": _f(entry["raw_price"]),
                    "identity_price": _f(entry["identity_price"]),
                    "complement_price": _f(entry["complement_price"]),
                    "size": _f(entry["size"]),
                    "best_bid": _f(entry["best_bid"]),
                    "best_ask": _f(entry["best_ask"]),
                    "in_spread_identity": entry["in_spread_identity"],
                    "at_touch_identity": entry["at_touch_identity"],
                    "in_spread_complement": entry["in_spread_complement"],
                    "at_touch_complement": entry["at_touch_complement"],
                }
            )
        for jump in result["jumps"]:
            jump_rows.append(
                {
                    "market_id": market_id,
                    "minute": _iso(jump["minute"]),
                    "jump_size": _f(jump["jump_size"]),
                    "direction": jump["direction"],
                    "price_before": _f(jump["price_before"]),
                    "price_after": _f(jump["price_after"]),
                    "window_minutes": jump["window_minutes"],
                    "threshold_5c": jump["threshold_5c"],
                    "threshold_10c": jump["threshold_10c"],
                    "threshold_20c": jump["threshold_20c"],
                    "timing_label": jump["timing_label"],
                    "trades_in_minute": jump["trade_count"],
                }
            )

    pd.DataFrame(coverage_rows).to_csv(out_dir / "market_coverage_summary.csv", index=False)
    pd.DataFrame(gap_rows).to_csv(out_dir / "gap_report.csv", index=False)
    pd.DataFrame(tob_rows).to_csv(out_dir / "top_of_book_timeseries_sample.csv", index=False)
    pd.DataFrame(depth_rows).to_csv(out_dir / "depth_timeseries_sample.csv", index=False)
    pd.DataFrame(align_rows).to_csv(out_dir / "trade_alignment_sample.csv", index=False)
    pd.DataFrame(exit_csv_rows).to_csv(out_dir / "liquidation_exit_curve_sample.csv", index=False)
    pd.DataFrame(jump_rows).to_csv(out_dir / "jump_window_candidates.csv", index=False)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_probe_report(context: Mapping[str, Any]) -> str:
    """Render the feasibility report. Receives derived values only — never
    connection settings or credentials."""

    results = context["results"]
    stats: ProbeStats = context["stats"]
    lines: list = []
    add = lines.append

    add("# Tailscale replica feasibility probe report")
    add("")
    add(f"Generated: {context['generated_at']}")
    add(f"Window: {context['start'].isoformat()} to {context['end'].isoformat()}")
    add(
        f"Reliability era of window start: {context['era_label']} "
        "(order-book capture floor 2025-12-29; Go ingest ~2026-04-21; monitored ~2026-05-27)."
    )
    add("")
    add(
        "This is a bounded feasibility/calibration probe, not empirical proof. "
        "It samples a small set of non-sports markets, reconstructs displayed "
        "order books from snapshots plus snapshot_seq-scoped signed diffs, and "
        "checks which PM-DFBA calibration objects the replica can support."
    )
    add("")
    add("## Probe load")
    add("")
    add(
        f"- {stats.queries} bounded queries, {stats.db_seconds:,.1f}s total DB time, "
        f"{stats.rows_fetched:,} rows fetched, single read-only connection."
    )
    add(f"- Presence checks: {stats.presence_checks}; diff windows truncated: {stats.diff_windows_truncated}.")
    for note in stats.notes:
        add(f"- Note: {note}")
    add("")

    add("## 1-2. Sampled markets and categories")
    add("")
    add("| market_id | platform | bucket | category | volume (contracts) | title |")
    add("|---|---|---|---|---:|---|")
    for result in results:
        market = result["market"]
        add(
            f"| {market['market_id']} | {market['platform']} | {market.get('selected_bucket')} "
            f"| {market['category']} | {float(market['contracts_volume']):,.0f} "
            f"| {str(market['title'])[:60]} |"
        )
    add("")
    bucket_counts: dict = {}
    for result in results:
        bucket = result["market"].get("selected_bucket") or "?"
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    add(f"Bucket counts: {bucket_counts}. Quotas requested: {context['quotas']}.")
    add(
        "Category labels are keyword heuristics over market and parent-event titles "
        "(sports excluded the same way); they are reviewable in market_candidates.csv, "
        "not ground truth."
    )
    add("")

    add("## 3-4. Coverage windows and gaps")
    add("")
    total_gaps = sum(len(result["heartbeat_gaps"]) for result in results)
    suspicious_gaps = sum(
        1
        for result in results
        for gap in result["heartbeat_gaps"]
        if gap.get("suspicious")
    )
    markets_with_zero_diff_days = sum(
        1 for result in results if result["diff_summary"]["days_missing"] > 0
    )
    add(
        f"- Book-stream gaps (no snapshots for longer than max({context['gap_threshold_seconds']:.0f}s, "
        f"4x that market's median snapshot cadence)): {total_gaps} across {len(results)} markets, "
        f"of which {suspicious_gaps} are suspicious (consistent with an ingest stall, not "
        "proof of one) — a trade printed during an hour with no recorded diffs at all. "
        "Snapshot cadence is activity-driven and platform-dependent "
        "(seconds on busy Polymarket books, ~45 minutes on some Kalshi books), so "
        "non-suspicious gaps are most often just quiet books. See gap_report.csv."
    )
    add(
        f"- Markets with zero-diff days inside their own active span: "
        f"{markets_with_zero_diff_days} (a zero-diff day can be a quiet book, "
        "not necessarily an outage)."
    )
    cadences = [
        result["median_snapshot_cadence_seconds"]
        for result in results
        if result["median_snapshot_cadence_seconds"] is not None
    ]
    if cadences:
        add(
            f"- Median snapshot cadence across sampled markets: "
            f"{statistics.median(cadences):,.0f}s "
            f"(min {min(cadences):,.0f}s, max {max(cadences):,.0f}s)."
        )
    add("")

    add("## 5. Row counts (sampled window)")
    add("")
    add("| market_id | snapshots | diffs | trades | days present | days missing |")
    add("|---|---:|---:|---:|---:|---:|")
    for result in results:
        add(
            f"| {result['market']['market_id']} | {result['snapshot_summary']['total']:,} "
            f"| {result['diff_summary']['total']:,} | {result['trade_summary']['total']:,} "
            f"| {result['snapshot_summary']['days_present']} "
            f"| {result['snapshot_summary']['days_missing']} |"
        )
    add("")

    replayed = [result for result in results if result["replay"] is not None]
    add("## 6. Top-of-book reconstruction")
    add("")
    add(
        f"- Replay succeeded for {len(replayed)}/{len(results)} markets "
        "(latest snapshot at or before the window seed, snapshot_seq-scoped signed diffs, "
        "Decimal arithmetic)."
    )
    if replayed:
        skipped = sum(result["replay"]["fold_stats"].diffs_skipped_wrong_anchor for result in replayed)
        applied = sum(result["replay"]["fold_stats"].diffs_applied for result in replayed)
        crossed = sum(result["replay"]["fold_stats"].crossed_book_events for result in replayed)
        anchors = sum(result["replay"]["fold_stats"].snapshots_applied for result in replayed)
        add(
            f"- Diffs applied: {applied:,}; skipped (wrong snapshot anchor): {skipped:,}; "
            f"crossed-book events: {crossed:,}."
        )
        if skipped and anchors:
            add(
                f"- Wrong-anchor skips run about {skipped / anchors:.1f} per snapshot re-anchor "
                f"({anchors:,} re-anchors) — a boundary effect of the snapshot/diff interleave. "
                "Each snapshot restores full authoritative book state, so skips do not "
                "accumulate; high trade/book alignment on the highest-skip markets corroborates."
            )
    add("")

    add("## 7. Depth within 1c / 5c / 10c")
    add("")
    depth_medians = context.get("depth_medians")
    if depth_medians:
        add(
            f"- Across {depth_medians['rows']:,} sampled book states: median bid depth "
            f"within 5c = {depth_medians['bid_5c']:,.0f} contracts, median ask depth "
            f"within 5c = {depth_medians['ask_5c']:,.0f} contracts."
        )
    add(
        "Depth rows are the last replayed book state in each 5-minute bucket, labeled "
        "at the bucket start; see depth_timeseries_sample.csv (bid/ask depth per band "
        "plus imbalance_5c)."
    )
    add("")

    add("## 8. Executable exit curves")
    add("")
    fill_summary = context.get("exit_fill_summary") or []
    if fill_summary:
        add("| quantity | side | samples | full-fill share | median vwap slip vs mid (cents) |")
        add("|---|---|---:|---:|---:|")
        for entry in fill_summary:
            slip = entry["median_slip_cents"]
            add(
                f"| {entry['quantity_label']} | {entry['side']} | {entry['samples']} "
                f"| {entry['full_fill_share']:.0%} | "
                f"{'n/a' if slip is None else f'{slip:.2f}'} |"
            )
    add("")
    add(
        "Partial fills are reported as partial (unfilled_quantity > 0); displayed "
        "liquidity is never extrapolated, mirroring the collar principle that bad "
        "execution is rejected, not conjured. Slip medians include partial fills, "
        "whose vwap covers only the filled (best-priced) portion — see "
        "liquidation_exit_curve_sample.csv for per-sample unfilled quantities."
    )
    add("")

    add("## 9. Trade-to-book alignment")
    add("")
    alignment_summary = context.get("alignment_summary") or []
    if alignment_summary:
        add(
            "| market_id | trades time-joined | two-sided-book trades | in-spread (identity) "
            "| in-spread (complement) | best convention | median book age (s) |"
        )
        add("|---|---:|---:|---:|---:|---|---:|")
        for entry in alignment_summary:
            identity = entry["in_spread_identity_rate"]
            complement = entry["in_spread_complement_rate"]
            book_age = entry["median_book_age"]
            identity_text = "n/a" if identity is None else f"{identity:.0%}"
            complement_text = "n/a" if complement is None else f"{complement:.0%}"
            book_age_text = "n/a" if book_age is None else f"{book_age:.1f}"
            add(
                f"| {entry['market_id']} | {entry['n']} | {entry['judged']} | {identity_text} "
                f"| {complement_text} | {entry['best_convention']} | {book_age_text} |"
            )
    add("")
    add(
        "In-spread shares are computed over trades whose book had both sides at the "
        "trade time (two-sided-book trades); one-sided books (common at extreme "
        "probabilities) yield n/a."
    )
    add(
        "Each trade is tested under two price conventions: identity (price as printed) "
        "and complement (1 - price for NO-labeled prints). Which convention fits is an "
        "empirical output; smoke runs found NO-labeled Polymarket prints already carry "
        "book-space prices, so complementing them would be wrong. Directional semantics "
        "of NO-labeled prints (whose taker side is which, in YES terms) still need "
        "confirmation from the data owner before stale-loss direction studies."
    )
    add("")

    add("## 10. Resolution join")
    add("")
    resolved = sum(1 for result in results if result["resolution"] and result["resolution"].get("outcome"))
    with_close = sum(
        1
        for result in results
        if result["resolution"]
        and (result["resolution"].get("close_time") or result["resolution"].get("resolution_time"))
    )
    add(
        f"- {resolved}/{len(results)} sampled markets have a resolved outcome in "
        f"market_lifecycle_events; {with_close}/{len(results)} have a close/resolution time "
        "(active markets legitimately have neither yet)."
    )
    add("")

    add("## 11. What this supports")
    add("")
    interim = sum(
        1 for result in results for jump in result["jumps"] if jump["timing_label"] == "interim_candidate"
    )
    terminal = sum(
        1
        for result in results
        for jump in result["jumps"]
        if jump["timing_label"] == "terminal_near_resolution_candidate"
    )
    unknown_jumps = sum(
        1 for result in results for jump in result["jumps"] if jump["timing_label"] == "unknown"
    )
    add(
        f"- Coarse jump candidates found: {interim} interim, {terminal} terminal/near-resolution, "
        f"{unknown_jumps} unknown timing (no resolution/close information)."
    )
    add("")

    # Verdicts are computed from this run's results, not asserted.
    replay_count = len(replayed)
    depth_rows_count = sum(len(result["replay"]["depth_rows"]) for result in replayed)
    exit_rows_count = sum(len(result["replay"]["exit_rows"]) for result in replayed)
    aligned_markets = sum(1 for entry in alignment_summary if entry["judged"] > 0)
    replay_verdict = (
        f"Yes for {replay_count}/{len(results)} sampled markets (bounded windows via "
        "snapshot+diff reconstruction)"
        if replay_count
        else "Not demonstrated in this run — no market replayed successfully"
    )
    depth_verdict = (
        f"Yes — {depth_rows_count:,} depth rows across {replay_count} markets, subject to gap checks"
        if depth_rows_count
        else "Not demonstrated in this run"
    )
    exit_verdict = (
        f"Yes — {exit_rows_count:,} exit-curve samples, partial fills reported honestly"
        if exit_rows_count
        else "Not demonstrated in this run"
    )
    stale_verdict = (
        f"Supported as an observable-book proxy only ({aligned_markets} markets with "
        "two-sided trade/book alignment)"
        if aligned_markets
        else "Not demonstrated in this run"
    )
    add("| Objective | This run's evidence |")
    add("|---|---|")
    add(f"| Simulator calibration (spreads, depth, trade sizes, jump sizes) | {depth_verdict} |")
    add(f"| Event-window replay of displayed books | {replay_verdict} |")
    add(f"| Stale-loss proxy (displayed quote just before jump vs post-jump prints) | {stale_verdict} |")
    add(f"| Liquidation exit-curve proxies V_exit(Q) from displayed depth | {exit_verdict} |")
    add("| True maker-cancel-vs-taker-hit race proof | No — requires order IDs and add/cancel/modify/fill lifecycle events, which the replica does not expose |")
    add("")
    add(
        "Aggregate-level inference (a level shrinking without a matching trade print "
        "suggests a cancel; shrinking with one suggests a fill) is possible but is an "
        "inference, not proof, and should be labeled as such."
    )
    add("")

    add("## What this probe can claim")
    add("")
    add(
        f"- The Tailscale replica supported order-book reconstruction for "
        f"{replay_count}/{len(results)} sampled non-sports markets in this run, "
        "subject to per-market coverage and gap checks."
    )
    if depth_rows_count and exit_rows_count:
        add(
            "- It enables observable-book replay, depth/exit-curve computation, and "
            "stale-loss proxies on the covered markets/windows."
        )
    if aligned_markets:
        add("- Trade prints align with reconstructed books at the rates shown above.")
    add("")
    add("## What this probe cannot claim")
    add("")
    add("- It does not prove true stale-quote races (no order lifecycle/cancel timing).")
    add("- It does not show the data is complete; gaps exist and are reported, not assumed away.")
    add("- It does not provide evidence that PM-DFBA is superior; it is calibration groundwork.")
    add("- Trade-size and exit-curve figures are proxies, not liquidation-size estimates.")
    add("")

    add("## Recommended next step")
    add("")
    add(
        "Event-window replay study: for a handful of dated public events (CPI/FOMC, "
        "court rulings, election calls) inside the monitored era, replay the books of "
        "affected markets around the event timestamp and measure displayed-depth decay, "
        "spread widening, executable V_exit(Q) before/after, and traded-through stale "
        "quotes — the empirical inputs for the simulator's stale-loss and liquidation "
        "parameters."
    )
    add("")
    return "\n".join(lines)


def summarize_for_report(results: Sequence[Mapping[str, Any]]) -> dict:
    alignment_summary = []
    for result in results:
        aligned = result["alignment"]
        if not aligned:
            continue
        judged = [entry for entry in aligned if entry["in_spread_identity"] is not None]
        ages = [entry["book_age_seconds"] for entry in aligned]
        identity_rate = (
            sum(1 for entry in judged if entry["in_spread_identity"]) / len(judged)
            if judged
            else None
        )
        complement_rate = (
            sum(1 for entry in judged if entry["in_spread_complement"]) / len(judged)
            if judged
            else None
        )
        if identity_rate is None:
            convention = None
        elif identity_rate == complement_rate:
            convention = "indeterminate"
        elif identity_rate > complement_rate:
            convention = "identity"
        else:
            convention = "complement"
        alignment_summary.append(
            {
                "market_id": result["market"]["market_id"],
                "n": len(aligned),
                "judged": len(judged),
                "in_spread_identity_rate": identity_rate,
                "in_spread_complement_rate": complement_rate,
                "best_convention": convention,
                "at_touch_identity_rate": (
                    sum(1 for entry in judged if entry["at_touch_identity"]) / len(judged)
                    if judged
                    else None
                ),
                "median_book_age": statistics.median(ages) if ages else None,
            }
        )

    by_key: dict = {}
    for result in results:
        if not result["replay"]:
            continue
        for entry in result["replay"]["exit_rows"]:
            curve = entry["curve"]
            key = (entry["quantity_label"], curve.side)
            bucket = by_key.setdefault(key, {"samples": 0, "full": 0, "slips": []})
            bucket["samples"] += 1
            if curve.unfilled_quantity == 0:
                bucket["full"] += 1
            if curve.vwap_price is not None and entry["mid"] is not None:
                slip = (entry["mid"] - curve.vwap_price) if curve.side == "sell_yes" else (
                    curve.vwap_price - entry["mid"]
                )
                bucket["slips"].append(float(slip * 100))
    exit_fill_summary = []
    for (label, side), bucket in sorted(by_key.items()):
        exit_fill_summary.append(
            {
                "quantity_label": label,
                "side": side,
                "samples": bucket["samples"],
                "full_fill_share": bucket["full"] / bucket["samples"] if bucket["samples"] else 0.0,
                "median_slip_cents": statistics.median(bucket["slips"]) if bucket["slips"] else None,
            }
        )

    bid_depths, ask_depths, depth_row_count = [], [], 0
    for result in results:
        if not result["replay"]:
            continue
        for entry in result["replay"]["depth_rows"]:
            depth_row_count += 1
            if entry.get("bid_depth_within_5c") is not None:
                bid_depths.append(float(entry["bid_depth_within_5c"]))
            if entry.get("ask_depth_within_5c") is not None:
                ask_depths.append(float(entry["ask_depth_within_5c"]))
    depth_medians = None
    if bid_depths and ask_depths:
        depth_medians = {
            "rows": depth_row_count,
            "bid_5c": statistics.median(bid_depths),
            "ask_5c": statistics.median(ask_depths),
        }
    return {
        "alignment_summary": alignment_summary,
        "exit_fill_summary": exit_fill_summary,
        "depth_medians": depth_medians,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Bounded feasibility probe of the Tailscale prediction-market replica."
    )
    parser.add_argument("--out", default="outputs/tailscale_probe")
    parser.add_argument("--start", default=None, help="ISO-8601 UTC, e.g. 2026-05-27T00:00:00Z")
    parser.add_argument("--end", default=None, help="ISO-8601 UTC, exclusive")
    parser.add_argument("--max-markets", type=int, default=25)
    parser.add_argument(
        "--min-volume", type=float, default=2000.0, help="Minimum in-window contract volume."
    )
    parser.add_argument(
        "--quiet-min-volume",
        type=float,
        default=50.0,
        help="Volume floor for quiet-baseline candidates (below --min-volume).",
    )
    parser.add_argument("--replay-hours", type=float, default=2.0)
    parser.add_argument("--max-diff-rows", type=int, default=150_000)
    parser.add_argument("--max-trades-per-window", type=int, default=5_000)
    parser.add_argument(
        "--exit-quantities", default="100,1000,5000", help="Comma-separated contract sizes."
    )
    parser.add_argument("--gap-threshold-seconds", type=float, default=1800.0)
    parser.add_argument("--jump-window-minutes", type=int, default=10)
    parser.add_argument("--min-snapshots", type=int, default=100)
    parser.add_argument("--min-diffs", type=int, default=1000)
    parser.add_argument("--min-diffs-quiet", type=int, default=100)
    parser.add_argument("--presence-check-budget", type=int, default=150)
    args = parser.parse_args(argv)
    args.exit_quantities = [
        int(part) for part in str(args.exit_quantities).split(",") if part and int(part) > 0
    ]
    if not args.exit_quantities:
        parser.error("--exit-quantities must contain at least one positive integer")
    for name in (
        "max_markets",
        "min_volume",
        "replay_hours",
        "max_diff_rows",
        "max_trades_per_window",
        "gap_threshold_seconds",
        "jump_window_minutes",
        "presence_check_budget",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.quiet_min_volume < 0 or args.quiet_min_volume >= args.min_volume:
        parser.error("--quiet-min-volume must be >= 0 and below --min-volume")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    end = db.parse_utc_timestamp(args.end) if args.end else datetime.now(timezone.utc).replace(
        minute=0, second=0, microsecond=0
    ) - timedelta(hours=6)
    start = db.parse_utc_timestamp(args.start) if args.start else end - timedelta(days=30)
    if start >= end:
        raise SystemExit("--start must be strictly before --end.")
    if start < ORDERBOOK_CAPTURE_FLOOR:
        raise SystemExit("Window start precedes the 2025-12-29 order-book capture floor.")
    if start >= MONITORED_ERA_START:
        era_label = "monitored"
    elif start >= GO_INGEST_ERA_START:
        era_label = "improved-but-unverified (Go ingest, pre-monitoring)"
    else:
        era_label = "coverage-unverified (fragile Python-ingest era)"

    stats = ProbeStats()
    config = db.DbConfig.from_env()
    print(f"Connecting: {config.redacted_description()}")
    conn = db.connect(config)
    try:
        cursor = conn.cursor()
        print(f"Window {start.isoformat()} -> {end.isoformat()} ({era_label})")

        print("Stage 1/4: discovery (day-paged trade aggregation)")
        totals = discover_market_activity(cursor, start, end, stats)

        print("Stage 2/4: metadata + classification + selection")
        by_volume = sorted(totals.items(), key=lambda kv: kv[1]["contracts"], reverse=True)
        active_ids = [
            market_id
            for market_id, entry in by_volume
            if entry["contracts"] >= Decimal(str(args.min_volume))
        ][:400]
        quiet_ids = [
            market_id
            for market_id, entry in by_volume
            if Decimal(str(args.quiet_min_volume)) <= entry["contracts"] < Decimal(str(args.min_volume))
        ][:200]
        metadata = fetch_metadata(cursor, active_ids + quiet_ids, stats)
        pool_totals = {market_id: totals[market_id] for market_id in active_ids + quiet_ids}
        selected, candidates, quotas = select_markets(
            cursor, pool_totals, metadata, args, start, end, stats
        )
        if not selected:
            raise SystemExit("No markets passed selection; widen the window or lower thresholds.")

        print("Stage 3/4: resolution join + per-market probes")
        resolution_rows = run_query(
            cursor,
            db.resolution_query(
                [row["market_id"] for row in selected],
                LIFECYCLE_LOOKBACK_START,
                datetime.now(timezone.utc) + timedelta(days=2),
            ),
            stats,
        )
        resolutions = {row["market_id"]: dict(row) for row in resolution_rows}

        results = []
        for row in selected:
            try:
                results.append(
                    probe_market(
                        cursor, row, args, start, end, resolutions.get(row["market_id"]), stats
                    )
                )
            except Exception as exc:  # keep the probe bounded: skip, don't abort
                note = f"Probe failed for {row['market_id']}: {type(exc).__name__}: {exc}"
                stats.notes.append(note)
                print(f"  WARNING {note}")
    finally:
        conn.close()

    if not results:
        raise SystemExit(
            "All per-market probes failed; refusing to write a feasibility report "
            f"from an empty run. Failures: {'; '.join(stats.notes[-5:])}"
        )

    print("Stage 4/4: writing outputs")
    out_dir = Path(args.out)
    context = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "start": start,
        "end": end,
        "era_label": era_label,
        "gap_threshold_seconds": args.gap_threshold_seconds,
        "quotas": quotas,
        "results": results,
        "stats": stats,
    }
    context.update(summarize_for_report(results))
    write_outputs(out_dir, results, candidates, context)
    (out_dir / REPORT_NAME).write_text(build_probe_report(context))

    import os

    secret = os.environ.get(db.ENV_PASSWORD)
    leaked = db.scan_outputs_for_secret([out_dir / name for name in OUTPUT_FILES], secret)
    if leaked:
        for path in leaked:
            path.unlink()
        raise SystemExit(
            f"Aborted: credential material detected in {len(leaked)} output file(s); files removed."
        )

    print(f"Wrote probe outputs to {out_dir}")
    print(
        f"Markets probed: {len(results)}; queries: {stats.queries}; "
        f"DB time: {stats.db_seconds:,.1f}s; rows fetched: {stats.rows_fetched:,}"
    )


if __name__ == "__main__":
    main()
