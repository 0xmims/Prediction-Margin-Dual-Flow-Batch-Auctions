"""Event-window replay study CLI against the Tailscale predictiondb replica.

For each configured event (see ``configs/event_study.json``), replays the
displayed books of affected markets around the event anchor and writes
baseline/impact/recovery statistics, trade-through analyses, calibration
parameter suggestions, figures, and a report with explicit claim boundaries.

Run:

    PREDICTION_DB_HOST=<tailscale-ip> PYTHONPATH=src python3 -m pm_dfba_sim.run_event_study \\
        --config configs/event_study.json --out outputs/event_study

Credentials come from PREDICTION_DB_* environment variables or ``~/.pgpass``
and are never printed or written to outputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from pm_dfba_sim.data import event_study as es
from pm_dfba_sim.data import tailscale_db as db
from pm_dfba_sim.run_tailscale_probe import ProbeStats, _f, _iso, run_query

REPORT_NAME = "event_study_report.md"
OUTPUT_FILES = (
    "event_summary.csv",
    "event_timeline_sample.csv",
    "trade_through_sample.csv",
    "trade_through_aggregates.csv",
    "event_parameter_suggestions.json",
    REPORT_NAME,
)
LIFECYCLE_LOOKBACK_START = datetime(2025, 12, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Per event-market replay
# ---------------------------------------------------------------------------


def replay_event_market(
    cursor,
    event: es.EventSpec,
    market: es.EventMarket,
    window: es.WindowSpec,
    args,
    stats: ProbeStats,
    tick_size: Optional[Decimal],
) -> Optional[dict]:
    """One bounded fold pass for one market around one event anchor."""

    anchor = event.anchor_time
    warmup = timedelta(seconds=window.trade_through_lags_seconds[-1] + 60.0)
    series_start = anchor + window.series_start_offset
    series_end = anchor + window.series_end_offset
    fetch_start = series_start - warmup

    seed_rows = run_query(
        cursor, db.latest_snapshot_query(market.market_id, fetch_start, timedelta(hours=48)), stats
    )
    seed = dict(seed_rows[0]) if seed_rows else None
    snapshots = [
        dict(row)
        for row in run_query(
            cursor,
            db.window_snapshots_query(market.market_id, fetch_start, series_end, limit=20_000),
            stats,
        )
    ]
    if seed is None and not snapshots:
        return None
    diff_fetch_start = seed["event_time"] if seed else snapshots[0]["event_time"]
    diffs = [
        dict(row)
        for row in run_query(
            cursor,
            db.window_diffs_query(
                market.market_id, diff_fetch_start, series_end, args.max_diff_rows + 1
            ),
            stats,
        )
    ]
    truncated = len(diffs) > args.max_diff_rows
    if truncated:
        diffs = diffs[: args.max_diff_rows]
        stats.diff_windows_truncated += 1
        stats.notes.append(
            f"Diff rows truncated at {args.max_diff_rows} for {market.market_id} "
            f"({event.event_id}); series may end early."
        )

    fold_stats = db.FoldStats()
    timeline, tob_series = es.build_book_timeline(
        seed, snapshots, diffs, anchor, window, fold_stats
    )
    if not timeline:
        return None

    trades = run_query(
        cursor,
        db.window_trades_query(market.market_id, series_start, series_end, args.max_trades),
        stats,
    )
    trades_truncated = len(trades) >= args.max_trades
    if trades_truncated:
        stats.notes.append(
            f"Trades truncated at {args.max_trades} for {market.market_id} ({event.event_id})."
        )
    half_tick = (Decimal(str(tick_size)) / 2) if tick_size else Decimal("0.005")
    per_trade, through_aggregates = es.trade_through_analysis(
        tob_series, trades, anchor, window, half_tick
    )

    phase_stats = es.summarize_phases(timeline, window)
    recovery = es.recovery_time_seconds(timeline, anchor, window)

    return {
        "event_id": event.event_id,
        "event_name": event.name,
        "category": event.category,
        "anchor_time": anchor,
        "market_id": market.market_id,
        "platform": market.platform,
        "role": market.role,
        "timeline": timeline,
        "phase_stats": phase_stats,
        "recovery_time_seconds": recovery,
        "per_trade": per_trade,
        "through_aggregates": through_aggregates,
        "fold_stats": fold_stats,
        "diffs_truncated": truncated,
        "trades_truncated": trades_truncated,
        "trades_in_window": len(trades),
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_outputs(out_dir: Path, window: es.WindowSpec, results: Sequence[Mapping[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    timeline_rows = []
    trade_rows = []
    aggregate_rows = []
    for result in results:
        phase_stats = result["phase_stats"]
        base = {
            "event_id": result["event_id"],
            "market_id": result["market_id"],
            "platform": result["platform"],
            "role": result["role"],
            "anchor_time": _iso(result["anchor_time"]),
        }
        summary_rows.append(
            {
                **base,
                "category": result["category"],
                "depth_decay_ratio": phase_stats["depth_decay_ratio"],
                "spread_widening_ratio": phase_stats["spread_widening_ratio"],
                "exit_value_haircut_ratio": phase_stats["exit_value_haircut_ratio"],
                "exit_quality_haircut_ratio": phase_stats["exit_quality_haircut_ratio"],
                "exit_reference_quantity": phase_stats["exit_reference_quantity"],
                "recovery_time_seconds": result["recovery_time_seconds"],
                "trades_in_window": result["trades_in_window"],
                "trades_truncated": result["trades_truncated"],
                "diffs_truncated": result["diffs_truncated"],
                "replay_diffs_applied": result["fold_stats"].diffs_applied,
                "replay_snapshots_applied": result["fold_stats"].snapshots_applied,
                "replay_crossed_book_events": result["fold_stats"].crossed_book_events,
                **{
                    f"{phase}_{key}": phase_stats[phase][key]
                    for phase in es.PHASES
                    for key in (
                        "rows",
                        "spread_median",
                        "spread_max",
                        "depth_5c_median",
                        "depth_5c_min",
                        "exit_value_median",
                        "exit_value_min",
                        "exit_quality_median",
                        "exit_quality_min",
                    )
                },
            }
        )
        for row in result["timeline"]:
            timeline_rows.append(
                {
                    **base,
                    "time": _iso(row["time"]),
                    "book_time": _iso(row["book_time"]),
                    "offset_minutes": row["offset_minutes"],
                    "phase": row["phase"],
                    "best_bid": _f(row["best_bid"]),
                    "best_ask": _f(row["best_ask"]),
                    "mid": _f(row["mid"]),
                    "spread": _f(row["spread"]),
                    **{
                        key: _f(row[key])
                        for key in row
                        if key.startswith(("bid_depth", "ask_depth", "exit_")) or key == "imbalance_5c"
                    },
                }
            )
        for row in result["per_trade"][: 500]:
            trade_rows.append(
                {
                    **base,
                    "trade_time": _iso(row["trade_time"]),
                    "phase": row["phase"],
                    "price": _f(row["price"]),
                    "size": _f(row["size"]),
                    "outcome": row["outcome"],
                    "taker_side": row["taker_side"],
                    **{
                        key: (_f(value) if isinstance(value, Decimal) else value)
                        for key, value in row.items()
                        if key.startswith(("through_lag_", "stale_loss_lag_"))
                    },
                }
            )
        for (phase, lag), counts in sorted(result["through_aggregates"].items()):
            aggregate_rows.append(
                {
                    **base,
                    "phase": phase,
                    "lag_seconds": lag,
                    "trades": counts["trades"],
                    "judged": counts["judged"],
                    "throughs": counts["throughs"],
                    "through_rate": counts["through_rate"],
                    "stale_loss_proxy_total": _f(counts["stale_loss_proxy"]),
                }
            )

    pd.DataFrame(summary_rows).to_csv(out_dir / "event_summary.csv", index=False)
    pd.DataFrame(timeline_rows).to_csv(out_dir / "event_timeline_sample.csv", index=False)
    pd.DataFrame(trade_rows).to_csv(out_dir / "trade_through_sample.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(out_dir / "trade_through_aggregates.csv", index=False)


def write_figures(out_dir: Path, window: es.WindowSpec, results: Sequence[Mapping[str, Any]]) -> list:
    """Timeline figures for the flagship event plus a cross-event ratio chart."""

    import os
    import tempfile

    mpl_cache = Path(tempfile.gettempdir()) / "pm-dfba-matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []

    def timeline_figure(result, path):
        timeline = result["timeline"]
        offsets = [row["offset_minutes"] for row in timeline]
        reference = result["phase_stats"]["exit_reference_quantity"]
        panels = [
            ("mid", "Mid price", lambda row: _f(row["mid"])),
            ("spread", "Spread", lambda row: _f(row["spread"])),
            (
                "depth",
                "Depth within 5c (bid+ask)",
                lambda row: (
                    _f((row["bid_depth_within_5c"] or Decimal(0)) + (row["ask_depth_within_5c"] or Decimal(0)))
                    if row["bid_depth_within_5c"] is not None or row["ask_depth_within_5c"] is not None
                    else None
                ),
            ),
            (
                "exit",
                f"V_exit(sell {reference})",
                lambda row: _f(row.get(f"exit_sell_{reference}_value")),
            ),
        ]
        fig, axes = plt.subplots(len(panels), 1, figsize=(9, 10), sharex=True)
        for axis, (_, title, extract) in zip(axes, panels):
            axis.plot(offsets, [extract(row) for row in timeline], linewidth=1.0)
            axis.axvline(0, color="crimson", linestyle="--", linewidth=1.0, label="event anchor")
            axis.set_ylabel(title, fontsize=8)
            axis.grid(True, alpha=0.25)
        axes[-1].set_xlabel("Minutes from event anchor")
        axes[0].set_title(
            f"{result['event_name']} — {result['market_id']} ({result['role']})", fontsize=10
        )
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    # Flagship timelines: the largest-decay interim leg and one terminal leg.
    scored = [
        result
        for result in results
        if result["role"] in {"interim_leg", "headline_target"}
        and result["phase_stats"]["depth_decay_ratio"] is not None
    ]
    scored.sort(key=lambda result: result["phase_stats"]["depth_decay_ratio"])
    if scored:
        timeline_figure(scored[0], out_dir / "event_timeline_flagship_interim.png")
    terminal = [result for result in results if result["role"] == "terminal_leg"]
    if terminal:
        timeline_figure(terminal[0], out_dir / "event_timeline_flagship_terminal.png")

    labeled = [
        result for result in results if result["phase_stats"]["depth_decay_ratio"] is not None
    ]
    if labeled:
        fig, axis = plt.subplots(figsize=(10, 5))
        names = [f"{result['market_id'][:18]}\n({result['role']})" for result in labeled]
        decay = [result["phase_stats"]["depth_decay_ratio"] for result in labeled]
        haircut = [
            result["phase_stats"]["exit_value_haircut_ratio"] or float("nan") for result in labeled
        ]
        positions = range(len(labeled))
        axis.bar([p - 0.2 for p in positions], decay, width=0.4, label="depth decay ratio")
        axis.bar([p + 0.2 for p in positions], haircut, width=0.4, label="V_exit haircut ratio")
        axis.axhline(1.0, color="gray", linewidth=0.8)
        axis.set_xticks(list(positions))
        axis.set_xticklabels(names, fontsize=6)
        axis.set_ylabel("Impact extreme / baseline median")
        axis.set_title("Event-window depth decay and executable-exit haircuts")
        axis.legend(fontsize=8)
        axis.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        path = out_dir / "event_ratios_summary.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_event_study_report(context: Mapping[str, Any]) -> str:
    """Render the event-study report. Receives derived values only — never
    connection settings or credentials."""

    results = context["results"]
    window: es.WindowSpec = context["window"]
    stats: ProbeStats = context["stats"]
    lines: list = []
    add = lines.append

    add("# Event-window replay study report")
    add("")
    add(f"Generated: {context['generated_at']}")
    add(
        f"Windows per event: baseline [{window.baseline_start_minutes:g}, "
        f"{window.baseline_end_minutes:g}] min, impact (0, {window.impact_end_minutes:g}] min, "
        f"recovery ({window.impact_end_minutes:g}, {window.recovery_end_minutes:g}] min; "
        f"grid {window.grid_seconds:g}s; trade-through lags "
        f"{list(window.trade_through_lags_seconds)}s."
    )
    add("")
    add(
        "Observable-book study of displayed liquidity around dated public events. "
        "All quantities are proxies for the paper's venue-created stale-loss and "
        "liquidation-gap terms; nothing here attributes maker cancels vs taker hits "
        "(the replica exposes no order lifecycle data). Event timestamps and their "
        "bases are recorded in configs/event_study.json."
    )
    add("")
    add("## Probe load")
    add("")
    add(
        f"- {stats.queries} bounded queries, {stats.db_seconds:,.1f}s DB time, "
        f"{stats.rows_fetched:,} rows fetched, single read-only connection."
    )
    for note in stats.notes:
        add(f"- Note: {note}")
    add("")

    add("## Per-event results")
    for event_id in sorted({result["event_id"] for result in results}):
        event_results = [result for result in results if result["event_id"] == event_id]
        first = event_results[0]
        add("")
        add(f"### {first['event_name']} (`{event_id}`, {first['category']})")
        add("")
        add(f"Anchor: {first['anchor_time'].isoformat()}")
        add("")
        add(
            "| market | role | depth decay | spread widening | V_exit haircut "
            "| exit quality haircut | recovery (s) | impact through-rate (30s lag) |"
        )
        add("|---|---|---:|---:|---:|---:|---:|---:|")
        longest_lag = window.trade_through_lags_seconds[-1]
        for result in event_results:
            phase_stats = result["phase_stats"]
            through = result["through_aggregates"].get(("impact", longest_lag), {})
            through_rate = through.get("through_rate")

            def fmt(value, pattern="{:.2f}"):
                return "n/a" if value is None else pattern.format(value)

            add(
                f"| {result['market_id']} | {result['role']} "
                f"| {fmt(phase_stats['depth_decay_ratio'])} "
                f"| {fmt(phase_stats['spread_widening_ratio'])} "
                f"| {fmt(phase_stats['exit_value_haircut_ratio'])} "
                f"| {fmt(phase_stats['exit_quality_haircut_ratio'])} "
                f"| {fmt(result['recovery_time_seconds'], '{:.0f}')} "
                f"| {fmt(through_rate, '{:.1%}')} |"
            )
        add("")
        add(
            "Ratios compare impact-window extremes against baseline medians. "
            "Depth decay < 1 means displayed liquidity shrank; spread widening > 1 "
            "means the spread blew out. V_exit haircut uses raw collected value and "
            "is contaminated by price direction (a rally raises it); exit quality "
            "haircut divides by mid x Q first, isolating execution quality, and is "
            "the better liquidation-gap proxy."
        )

    add("")
    add("## Falsification check")
    add("")
    event_legs = [
        result for result in results if result["role"] in {"interim_leg", "headline_target"}
    ]
    controls = [result for result in results if result["role"] == "control"]
    decayed = [
        result
        for result in event_legs
        if result["phase_stats"]["depth_decay_ratio"] is not None
        and result["phase_stats"]["depth_decay_ratio"] < 0.8
    ]
    widened = [
        result
        for result in event_legs
        if result["phase_stats"]["spread_widening_ratio"] is not None
        and result["phase_stats"]["spread_widening_ratio"] > 1.5
    ]
    add(
        f"- Of {len(event_legs)} event-exposed legs: {len(decayed)} showed 5c-depth decay "
        f"below 0.8x baseline and {len(widened)} showed spread widening above 1.5x during "
        "the impact window."
    )
    control_decayed = [
        result
        for result in controls
        if result["phase_stats"]["depth_decay_ratio"] is not None
        and result["phase_stats"]["depth_decay_ratio"] < 0.8
    ]
    add(
        f"- Of {len(controls)} control replays at the same clock windows: "
        f"{len(control_decayed)} showed comparable depth decay."
    )
    longest_lag = window.trade_through_lags_seconds[-1]

    def median_rate(rows, phase):
        rates = sorted(
            entry["through_aggregates"][(phase, longest_lag)]["through_rate"]
            for entry in rows
            if (phase, longest_lag) in entry["through_aggregates"]
            and entry["through_aggregates"][(phase, longest_lag)]["through_rate"] is not None
        )
        if not rates:
            return None
        middle = len(rates) // 2
        return rates[middle] if len(rates) % 2 else (rates[middle - 1] + rates[middle]) / 2

    impact_rate = median_rate(event_legs, "impact")
    baseline_rate = median_rate(event_legs, "baseline")
    if impact_rate is not None and baseline_rate is not None:
        add(
            f"- Median trade-through rate against the {longest_lag:g}s-lagged displayed book "
            f"on event legs: {impact_rate:.1%} during impact vs {baseline_rate:.1%} at baseline."
        )
    add(
        "- If event-exposed legs behaved like controls, the venue-created stale-loss "
        "term would be small and the PM-DFBA premise would weaken; the numbers above "
        "are the test, whichever way they cut."
    )
    add("")

    add("## What this study can claim")
    add("")
    add(
        "- Displayed-book dynamics (spread, depth bands, executable V_exit(Q)) around "
        "the configured events, with baseline/impact/recovery decomposition and "
        "control comparisons."
    )
    add(
        "- Trade-through rates and stale-loss proxies against tau-lagged displayed "
        "books, direction-agnostic by construction."
    )
    add("- Bounded calibration candidates for simulator sensitivity ranges.")
    add("")
    add("## What this study cannot claim")
    add("")
    add(
        "- It does not prove latency races or maker-cancel losses; there is no order "
        "lifecycle data."
    )
    add(
        "- It does not attribute jumps to public vs private information beyond the "
        "documented event timestamps."
    )
    add("- It is not evidence that PM-DFBA is superior; it calibrates the question.")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Replay displayed books around dated public events (bounded)."
    )
    parser.add_argument("--config", default="configs/event_study.json")
    parser.add_argument("--out", default="outputs/event_study")
    parser.add_argument(
        "--event", action="append", default=None, help="Event id filter; may be repeated."
    )
    parser.add_argument("--max-diff-rows", type=int, default=250_000)
    parser.add_argument("--max-trades", type=int, default=20_000)
    args = parser.parse_args(argv)
    if args.max_diff_rows <= 0 or args.max_trades <= 0:
        parser.error("--max-diff-rows and --max-trades must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    window, events = es.load_event_config_file(args.config)
    if args.event:
        wanted = set(args.event)
        events = [event for event in events if event.event_id in wanted]
        if not events:
            raise SystemExit(f"No configured events match {sorted(wanted)}.")

    stats = ProbeStats()
    config = db.DbConfig.from_env()
    print(f"Connecting: {config.redacted_description()}")
    conn = db.connect(config)
    results = []
    try:
        cursor = conn.cursor()
        market_ids = sorted({market.market_id for event in events for market in event.markets})
        tick_rows = run_query(cursor, db.markets_metadata_query(market_ids), stats)
        ticks = {row["market_id"]: row["tick_size"] for row in tick_rows}

        for event in events:
            print(f"Event {event.event_id} ({event.anchor_time.isoformat()})")
            for market in event.markets:
                print(f"  replaying {market.market_id} [{market.role}]")
                try:
                    result = replay_event_market(
                        cursor, event, market, window, args, stats, ticks.get(market.market_id)
                    )
                except Exception as exc:  # bounded study: skip, don't abort
                    note = (
                        f"Replay failed for {market.market_id} ({event.event_id}): "
                        f"{type(exc).__name__}: {exc}"
                    )
                    stats.notes.append(note)
                    print(f"  WARNING {note}")
                    continue
                if result is None:
                    stats.notes.append(
                        f"No book data for {market.market_id} around {event.event_id}; skipped."
                    )
                    continue
                results.append(result)
    finally:
        conn.close()

    if not results:
        raise SystemExit("No event-market replays succeeded; nothing to report.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(out_dir, window, results)
    suggestions = es.build_parameter_suggestions(results, window)
    with (out_dir / "event_parameter_suggestions.json").open("w") as handle:
        json.dump(suggestions, handle, indent=2, default=str)
    figures = write_figures(out_dir, window, results)
    context = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": window,
        "results": results,
        "stats": stats,
    }
    (out_dir / REPORT_NAME).write_text(build_event_study_report(context))

    import os

    secret = os.environ.get(db.ENV_PASSWORD)
    leaked = db.scan_outputs_for_secret(
        [out_dir / name for name in OUTPUT_FILES] + list(figures), secret
    )
    if leaked:
        for path in leaked:
            path.unlink()
        raise SystemExit(
            f"Aborted: credential material detected in {len(leaked)} output file(s); files removed."
        )

    print(f"Wrote event-study outputs to {out_dir}")
    print(
        f"Event-market replays: {len(results)}; queries: {stats.queries}; "
        f"DB time: {stats.db_seconds:,.1f}s; rows fetched: {stats.rows_fetched:,}"
    )


if __name__ == "__main__":
    main()
