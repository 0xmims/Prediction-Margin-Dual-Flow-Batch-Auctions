"""Event-window replay analysis over the Tailscale replica.

Measures how displayed order books behave around dated public events:
baseline/impact/recovery statistics for spread, depth bands, and executable
exit value V_exit(Q); trade-throughs against tau-lagged displayed books; and
recovery times. These are observable-book quantities — proxies for the paper's
venue-created stale-loss and liquidation-gap terms, not proof of latency races.

Falsification framing is deliberate: if books show no depth decay, spread
widening, or excess trade-throughs around public events, the venue-created
stale-loss term is small and the PM-DFBA premise weakens.

Trade prices are used as printed (identity space), which the tailscale probe
validated per market against reconstructed books. Trade-throughs are detected
direction-agnostically (a print outside the tau-lagged displayed spread), so
the still-unconfirmed side semantics of NO-labeled prints are not load-bearing.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

from pm_dfba_sim.data import tailscale_db as db


class EventStudyError(ValueError):
    """Raised for invalid event-study configuration or inputs."""


VALID_ROLES = ("terminal_leg", "interim_leg", "headline_target", "control")
PHASES = ("baseline", "impact", "recovery")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventMarket:
    market_id: str
    platform: str
    role: str


@dataclass(frozen=True)
class EventSpec:
    event_id: str
    name: str
    anchor_time: datetime
    category: str
    timestamp_basis: str
    markets: tuple


@dataclass(frozen=True)
class WindowSpec:
    baseline_start_minutes: float = -60.0
    baseline_end_minutes: float = -5.0
    impact_end_minutes: float = 15.0
    recovery_end_minutes: float = 60.0
    grid_seconds: float = 5.0
    trade_through_lags_seconds: tuple = (1.0, 5.0, 30.0)
    exit_quantities: tuple = (Decimal("100"), Decimal("1000"), Decimal("5000"))
    recovery_fraction: float = 0.8

    @property
    def series_start_offset(self) -> timedelta:
        return timedelta(minutes=self.baseline_start_minutes)

    @property
    def series_end_offset(self) -> timedelta:
        return timedelta(minutes=self.recovery_end_minutes)


def load_event_config(payload: Mapping[str, Any]) -> tuple:
    """Validate and normalize the event-study config mapping."""

    window_raw = payload.get("window") or {}
    window = WindowSpec(
        baseline_start_minutes=float(window_raw.get("baseline_start_minutes", -60)),
        baseline_end_minutes=float(window_raw.get("baseline_end_minutes", -5)),
        impact_end_minutes=float(window_raw.get("impact_end_minutes", 15)),
        recovery_end_minutes=float(window_raw.get("recovery_end_minutes", 60)),
        grid_seconds=float(window_raw.get("grid_seconds", 5)),
        trade_through_lags_seconds=tuple(
            sorted(float(lag) for lag in window_raw.get("trade_through_lags_seconds", (1, 5, 30)))
        ),
        exit_quantities=tuple(
            Decimal(str(quantity)) for quantity in window_raw.get("exit_quantities", (100, 1000, 5000))
        ),
        recovery_fraction=float(window_raw.get("recovery_fraction", 0.8)),
    )
    if not (
        window.baseline_start_minutes
        < window.baseline_end_minutes
        <= 0
        < window.impact_end_minutes
        < window.recovery_end_minutes
    ):
        raise EventStudyError(
            "Window minutes must satisfy baseline_start < baseline_end <= 0 < "
            "impact_end < recovery_end."
        )
    if window.grid_seconds <= 0:
        raise EventStudyError("grid_seconds must be positive.")
    if any(lag <= 0 for lag in window.trade_through_lags_seconds):
        raise EventStudyError("Trade-through lags must be positive.")
    if any(quantity <= 0 for quantity in window.exit_quantities):
        raise EventStudyError("Exit quantities must be positive.")
    if not (0 < window.recovery_fraction <= 1):
        raise EventStudyError("recovery_fraction must be in (0, 1].")

    events = []
    for raw in payload.get("events") or ():
        anchor = db.parse_utc_timestamp(raw["anchor_time_utc"])
        if anchor.tzinfo is None:
            raise EventStudyError(f"Event {raw.get('event_id')} anchor must be timezone-aware.")
        markets = []
        for market in raw.get("markets") or ():
            role = market.get("role")
            if role not in VALID_ROLES:
                raise EventStudyError(
                    f"Event {raw.get('event_id')}: unknown role '{role}' "
                    f"(expected one of {VALID_ROLES})."
                )
            markets.append(
                EventMarket(
                    market_id=str(market["market_id"]),
                    platform=str(market.get("platform") or ""),
                    role=role,
                )
            )
        if not markets:
            raise EventStudyError(f"Event {raw.get('event_id')} has no markets.")
        events.append(
            EventSpec(
                event_id=str(raw["event_id"]),
                name=str(raw.get("name") or raw["event_id"]),
                anchor_time=anchor,
                category=str(raw.get("category") or "unknown"),
                timestamp_basis=str(raw.get("timestamp_basis") or ""),
                markets=tuple(markets),
            )
        )
    if not events:
        raise EventStudyError("Event-study config contains no events.")
    return window, events


def load_event_config_file(path) -> tuple:
    with Path(path).open() as handle:
        return load_event_config(json.load(handle))


# ---------------------------------------------------------------------------
# Phase labeling and timeline construction
# ---------------------------------------------------------------------------


def label_phase(moment: datetime, anchor: datetime, window: WindowSpec) -> str:
    offset_minutes = (moment - anchor).total_seconds() / 60.0
    if offset_minutes < window.baseline_start_minutes:
        return "pre_baseline"
    if offset_minutes < window.baseline_end_minutes:
        return "baseline"
    if offset_minutes < 0:
        return "pre_anchor_buffer"
    if offset_minutes < window.impact_end_minutes:
        return "impact"
    if offset_minutes < window.recovery_end_minutes:
        return "recovery"
    return "post"


def _exit_fields(state: db.BookState, quantities: Sequence[Decimal]) -> dict:
    fields: dict = {}
    for quantity in quantities:
        label = str(quantity)
        sell = db.executable_exit_curve(state.bids, quantity, is_sell=True)
        buy = db.executable_exit_curve(state.asks, quantity, is_sell=False)
        fields[f"exit_sell_{label}_value"] = sell.executable_value
        fields[f"exit_sell_{label}_vwap"] = sell.vwap_price
        fields[f"exit_sell_{label}_unfilled"] = sell.unfilled_quantity
        fields[f"exit_buy_{label}_value"] = buy.executable_value
        fields[f"exit_buy_{label}_vwap"] = buy.vwap_price
        fields[f"exit_buy_{label}_unfilled"] = buy.unfilled_quantity
    return fields


def build_book_timeline(
    seed: Optional[Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
    diffs: Sequence[Mapping[str, Any]],
    anchor: datetime,
    window: WindowSpec,
    fold_stats: Optional[db.FoldStats] = None,
) -> tuple:
    """Fold the window once, sampling *as-of* book states on a regular grid.

    Returns ``(timeline_rows, tob_series)``. Grid rows use the last book state
    at or before each grid time (checkpoint yields from the fold — exact as-of
    semantics, so a quiet baseline is never stamped with the post-jump book).
    ``book_time`` records the sampled state's actual time. ``tob_series`` holds
    ``(event_time, best_bid, best_ask)`` for every folded event and feeds the
    tau-lagged trade-through analysis.
    """

    grid_start = anchor + window.series_start_offset
    grid_end = anchor + window.series_end_offset
    step = timedelta(seconds=window.grid_seconds)
    grid_times = []
    cursor = grid_start
    while cursor <= grid_end:
        grid_times.append(cursor)
        cursor = cursor + step

    timeline: list = []
    tob_series: list = []
    states = db.fold_book_events(seed, snapshots, diffs, fold_stats, checkpoints=grid_times)
    for state in states:
        if state.checkpoint is not None:
            grid_time = state.checkpoint
            timeline.append(
                {
                    "time": grid_time,
                    "book_time": state.event_time,
                    "offset_minutes": (grid_time - anchor).total_seconds() / 60.0,
                    "phase": label_phase(grid_time, anchor, window),
                    **db.top_of_book_metrics(state),
                    **db.depth_metrics(state),
                    **_exit_fields(state, window.exit_quantities),
                }
            )
        elif state.event_time is not None:
            tob_series.append((state.event_time, state.best_bid, state.best_ask))
    return timeline, tob_series


# ---------------------------------------------------------------------------
# Phase statistics, ratios, recovery
# ---------------------------------------------------------------------------


def _median(values: Sequence) -> Optional[float]:
    cleaned = sorted(float(value) for value in values if value is not None)
    if not cleaned:
        return None
    middle = len(cleaned) // 2
    if len(cleaned) % 2:
        return cleaned[middle]
    return (cleaned[middle - 1] + cleaned[middle]) / 2.0


def _phase_rows(timeline: Sequence[Mapping[str, Any]], phase: str) -> list:
    return [row for row in timeline if row["phase"] == phase]


def _ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def summarize_phases(
    timeline: Sequence[Mapping[str, Any]],
    window: WindowSpec,
    exit_reference_quantity: Optional[Decimal] = None,
) -> dict:
    """Per-phase medians/extremes plus impact-vs-baseline ratios.

    ``depth_decay_ratio`` < 1 means displayed 5c depth shrank during impact;
    ``spread_widening_ratio`` > 1 means the spread widened;
    ``exit_value_haircut_ratio`` < 1 means a forced seller of the reference
    quantity would have collected less than baseline.
    """

    reference = exit_reference_quantity or (
        window.exit_quantities[1] if len(window.exit_quantities) > 1 else window.exit_quantities[0]
    )
    exit_key = f"exit_sell_{reference}_value"
    reference_float = float(reference)

    stats: dict = {"exit_reference_quantity": str(reference)}
    for phase in PHASES:
        rows = _phase_rows(timeline, phase)
        spreads = [row["spread"] for row in rows]
        depths = [
            (row["bid_depth_within_5c"] or Decimal(0)) + (row["ask_depth_within_5c"] or Decimal(0))
            if row["bid_depth_within_5c"] is not None or row["ask_depth_within_5c"] is not None
            else None
            for row in rows
        ]
        exits = [row.get(exit_key) for row in rows]
        # Exit quality = collected value / (mid x Q): execution quality against
        # the concurrent mark, so a price rally cannot masquerade as liquidity.
        qualities = [
            float(row[exit_key]) / (float(row["mid"]) * reference_float)
            if row.get(exit_key) is not None and row["mid"] is not None and row["mid"] > 0
            else None
            for row in rows
        ]
        cleaned_spreads = [float(value) for value in spreads if value is not None]
        cleaned_depths = [float(value) for value in depths if value is not None]
        cleaned_exits = [float(value) for value in exits if value is not None]
        cleaned_qualities = [value for value in qualities if value is not None]
        stats[phase] = {
            "rows": len(rows),
            "spread_median": _median(spreads),
            "spread_max": max(cleaned_spreads) if cleaned_spreads else None,
            "depth_5c_median": _median(depths),
            "depth_5c_min": min(cleaned_depths) if cleaned_depths else None,
            "exit_value_median": _median(exits),
            "exit_value_min": min(cleaned_exits) if cleaned_exits else None,
            "exit_quality_median": _median(cleaned_qualities),
            "exit_quality_min": min(cleaned_qualities) if cleaned_qualities else None,
        }

    baseline = stats["baseline"]
    impact = stats["impact"]
    stats["depth_decay_ratio"] = _ratio(impact["depth_5c_min"], baseline["depth_5c_median"])
    stats["spread_widening_ratio"] = _ratio(impact["spread_max"], baseline["spread_median"])
    stats["exit_value_haircut_ratio"] = _ratio(
        impact["exit_value_min"], baseline["exit_value_median"]
    )
    stats["exit_quality_haircut_ratio"] = _ratio(
        impact["exit_quality_min"], baseline["exit_quality_median"]
    )
    return stats


def recovery_time_seconds(
    timeline: Sequence[Mapping[str, Any]],
    anchor: datetime,
    window: WindowSpec,
    consecutive: int = 2,
) -> Optional[float]:
    """Seconds after the anchor until spread and 5c depth both hold near
    baseline for ``consecutive`` grid points. None if never within the window
    or if a baseline is unavailable."""

    baseline_rows = _phase_rows(timeline, "baseline")
    spread_baseline = _median([row["spread"] for row in baseline_rows])
    depth_baseline = _median(
        [
            (row["bid_depth_within_5c"] or Decimal(0)) + (row["ask_depth_within_5c"] or Decimal(0))
            if row["bid_depth_within_5c"] is not None or row["ask_depth_within_5c"] is not None
            else None
            for row in baseline_rows
        ]
    )
    if spread_baseline is None or depth_baseline is None:
        return None

    spread_limit = spread_baseline / window.recovery_fraction
    depth_floor = depth_baseline * window.recovery_fraction
    streak = 0
    for row in timeline:
        if row["time"] < anchor:
            continue
        spread = row["spread"]
        bid_depth = row["bid_depth_within_5c"]
        ask_depth = row["ask_depth_within_5c"]
        depth = None
        if bid_depth is not None or ask_depth is not None:
            depth = float((bid_depth or Decimal(0)) + (ask_depth or Decimal(0)))
        recovered = (
            spread is not None
            and float(spread) <= spread_limit
            and depth is not None
            and depth >= depth_floor
        )
        if recovered:
            streak += 1
            if streak >= consecutive:
                first_recovered_index = timeline.index(row) - (consecutive - 1)
                return (timeline[first_recovered_index]["time"] - anchor).total_seconds()
        else:
            streak = 0
    return None


# ---------------------------------------------------------------------------
# Trade-throughs against tau-lagged displayed books
# ---------------------------------------------------------------------------


def trade_through_analysis(
    tob_series: Sequence[tuple],
    trades: Sequence[Mapping[str, Any]],
    anchor: datetime,
    window: WindowSpec,
    half_tick: Decimal = Decimal("0.005"),
) -> tuple:
    """Flag prints executing outside the tau-lagged displayed spread.

    Direction-agnostic: a print below ``bid(t - tau) - half_tick`` or above
    ``ask(t - tau) + half_tick`` is a trade-through against the book displayed
    tau seconds earlier, and the violated distance times size is an
    observable-book stale-loss proxy for that print. This measures how much
    worse executions were than recently displayed liquidity; it cannot say
    whether a maker cancel lost a race (no order lifecycle data).

    Returns ``(per_trade_rows, aggregates)`` where aggregates are keyed by
    ``(phase, lag_seconds)``.
    """

    if not tob_series:
        return [], {}
    times = [entry[0] for entry in tob_series]
    per_trade: list = []
    aggregates: dict = {}

    def bucket(phase: str, lag: float) -> dict:
        return aggregates.setdefault(
            (phase, lag),
            {"trades": 0, "judged": 0, "throughs": 0, "stale_loss_proxy": Decimal(0)},
        )

    for trade in trades:
        trade_time = trade["event_time"]
        price = db.normalize_probability_price(trade.get("price"))
        outcome = str(trade.get("outcome") or "").strip().lower()
        if price is None or outcome not in {"yes", "no"}:
            continue
        size = Decimal(str(trade.get("size") or 0))
        phase = label_phase(trade_time, anchor, window)
        row = {
            "trade_time": trade_time,
            "phase": phase,
            "price": price,
            "size": size,
            "outcome": trade.get("outcome"),
            "taker_side": trade.get("taker_side"),
        }
        for lag in window.trade_through_lags_seconds:
            lagged_time = trade_time - timedelta(seconds=lag)
            index = bisect_right(times, lagged_time) - 1
            counts = bucket(phase, lag)
            counts["trades"] += 1
            if index < 0:
                row[f"through_lag_{lag:g}s"] = None
                row[f"stale_loss_lag_{lag:g}s"] = None
                continue
            _, lag_bid, lag_ask = tob_series[index]
            if lag_bid is None and lag_ask is None:
                row[f"through_lag_{lag:g}s"] = None
                row[f"stale_loss_lag_{lag:g}s"] = None
                continue
            counts["judged"] += 1
            loss = Decimal(0)
            if lag_bid is not None and price < lag_bid - half_tick:
                loss = (lag_bid - price) * size
            elif lag_ask is not None and price > lag_ask + half_tick:
                loss = (price - lag_ask) * size
            through = loss > 0
            row[f"through_lag_{lag:g}s"] = through
            row[f"stale_loss_lag_{lag:g}s"] = loss if through else Decimal(0)
            if through:
                counts["throughs"] += 1
                counts["stale_loss_proxy"] += loss
        per_trade.append(row)

    for counts in aggregates.values():
        judged = counts["judged"]
        counts["through_rate"] = (counts["throughs"] / judged) if judged else None
    return per_trade, aggregates


# ---------------------------------------------------------------------------
# Cross-event parameter suggestions
# ---------------------------------------------------------------------------


def _quantiles(values: Sequence[float], points=(0.25, 0.5, 0.75, 0.9)) -> dict:
    cleaned = sorted(value for value in values if value is not None)
    if not cleaned:
        return {f"p{int(point * 100):02d}": None for point in points}
    result = {}
    for point in points:
        index = min(len(cleaned) - 1, max(0, round(point * (len(cleaned) - 1))))
        result[f"p{int(point * 100):02d}"] = cleaned[index]
    return result


def build_parameter_suggestions(
    summaries: Sequence[Mapping[str, Any]],
    window: WindowSpec,
) -> dict:
    """Bounded calibration candidates from event-window summaries.

    Interim/headline legs feed the public-jump stale-loss and liquidation
    parameters; terminal legs are kept separate (terminal risk belongs to
    margin rules, not the matching engine); controls calibrate the no-event
    baseline. These are candidates for simulator sensitivity ranges, not
    empirical truth.
    """

    def rows_for(roles) -> list:
        return [summary for summary in summaries if summary["role"] in roles]

    def block(rows) -> dict:
        longest_lag = window.trade_through_lags_seconds[-1]
        return {
            "markets": len(rows),
            "depth_decay_ratio": _quantiles([row["phase_stats"]["depth_decay_ratio"] for row in rows]),
            "spread_widening_ratio": _quantiles(
                [row["phase_stats"]["spread_widening_ratio"] for row in rows]
            ),
            "exit_value_haircut_ratio": _quantiles(
                [row["phase_stats"]["exit_value_haircut_ratio"] for row in rows]
            ),
            "exit_quality_haircut_ratio": _quantiles(
                [row["phase_stats"]["exit_quality_haircut_ratio"] for row in rows]
            ),
            "recovery_time_seconds": _quantiles([row["recovery_time_seconds"] for row in rows]),
            "impact_through_rate_longest_lag": _quantiles(
                [
                    row["through_aggregates"].get(("impact", longest_lag), {}).get("through_rate")
                    for row in rows
                ]
            ),
        }

    return {
        "window": {
            "baseline_minutes": [window.baseline_start_minutes, window.baseline_end_minutes],
            "impact_end_minutes": window.impact_end_minutes,
            "recovery_end_minutes": window.recovery_end_minutes,
            "trade_through_lags_seconds": list(window.trade_through_lags_seconds),
        },
        "public_interim_candidates": block(rows_for({"interim_leg", "headline_target"})),
        "terminal_leg_candidates": block(rows_for({"terminal_leg"})),
        "control_baseline": block(rows_for({"control"})),
        "data_limitations": [
            "Observable-book proxies only; no order lifecycle data, so no "
            "maker-cancel-vs-taker-hit attribution.",
            "Event timestamps come from schedules or data-discovered jump minutes "
            "recorded in the config's timestamp_basis fields.",
            "Trade-throughs are direction-agnostic; NO-print side semantics are "
            "unconfirmed and deliberately not load-bearing.",
            "Ratios are per-market impact extremes vs baseline medians in one window; "
            "use as simulator sensitivity ranges, not point estimates.",
        ],
    }
