"""Read-only feasibility/calibration adapter for the Tailscale predictiondb replica.

This module is the bounded empirical bridge between the PM-DFBA simulator and the
PostgreSQL + TimescaleDB read replica of Polymarket/Kalshi market data. It is a
probe, not a replay engine.

Hard rules encoded here:

- Credentials come only from environment variables (or libpq ``~/.pgpass`` when
  ``PREDICTION_DB_PASSWORD`` is unset). Nothing here hardcodes, prints, or writes
  connection secrets.
- Every hypertable query builder requires an explicit timezone-aware time window;
  builders raise ``TailscaleDBError`` rather than emit an unbounded scan.
- Order-book reconstruction follows the replica contract exactly: seed from the
  latest full snapshot at or before the target time, then apply only diffs whose
  ``snapshot_seq`` matches that snapshot's ``seq``, in ``seq`` order, treating
  diff sizes as signed deltas. Book math uses ``Decimal`` end to end.
"""

from __future__ import annotations

import os
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

ZERO = Decimal("0")
ONE = Decimal("1")
DEPTH_BANDS = (Decimal("0.01"), Decimal("0.05"), Decimal("0.10"))
DEFAULT_TICK = Decimal("0.01")

ENV_HOST = "PREDICTION_DB_HOST"
ENV_PORT = "PREDICTION_DB_PORT"
ENV_NAME = "PREDICTION_DB_NAME"
ENV_USER = "PREDICTION_DB_USER"
ENV_PASSWORD = "PREDICTION_DB_PASSWORD"

#: Tables partitioned by event_time on the replica; unbounded scans are forbidden.
HYPERTABLES = frozenset(
    {
        "orderbook_snapshots",
        "orderbook_diffs",
        "public_trades",
        "market_lifecycle_events",
        "whale_trades",
    }
)
_BOOK_TABLES = frozenset({"orderbook_snapshots", "orderbook_diffs"})


class TailscaleDBError(ValueError):
    """Raised for misuse that could hurt the replica or leak configuration."""


# ---------------------------------------------------------------------------
# Connection configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DbConfig:
    """Connection settings sourced from the environment, never from code."""

    host: str
    port: int = 5432
    dbname: str = "predictiondb"
    user: str = "friend_ro"
    password: Optional[str] = None
    sslmode: str = "prefer"
    connect_timeout: int = 10
    application_name: str = "pm_dfba_tailscale_probe"

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "DbConfig":
        source = os.environ if env is None else env
        host = (source.get(ENV_HOST) or "").strip()
        if not host:
            raise TailscaleDBError(
                f"{ENV_HOST} is not set. Refusing to guess a database host; see the "
                "replica onboarding README for connection details."
            )
        password = source.get(ENV_PASSWORD) or None
        return cls(
            host=host,
            port=int(source.get(ENV_PORT) or 5432),
            dbname=(source.get(ENV_NAME) or "predictiondb").strip(),
            user=(source.get(ENV_USER) or "friend_ro").strip(),
            password=password,
        )

    def connect_kwargs(self) -> dict:
        kwargs = {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
            "sslmode": self.sslmode,
            "connect_timeout": self.connect_timeout,
            "application_name": self.application_name,
        }
        if self.password is not None:
            kwargs["password"] = self.password
        return kwargs

    def redacted_description(self) -> str:
        """Loggable description. Never includes host, port, or password."""

        return (
            f"dbname={self.dbname} user={self.user} "
            "(host/port from environment; password from environment or ~/.pgpass)"
        )


def connect(config: DbConfig):
    """Open a single read-only autocommit connection (replica allows only 3)."""

    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(
        cursor_factory=psycopg2.extras.RealDictCursor, **config.connect_kwargs()
    )
    conn.set_session(readonly=True, autocommit=True)
    return conn


# ---------------------------------------------------------------------------
# Query builders (all hypertable builders require explicit time windows)
# ---------------------------------------------------------------------------


def _require_window(start: datetime, end: datetime) -> None:
    for label, value in (("start", start), ("end", end)):
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise TailscaleDBError(
                f"Query window '{label}' must be a timezone-aware datetime; "
                "unbounded hypertable scans are not allowed on the replica."
            )
    if start >= end:
        raise TailscaleDBError("Query window start must be strictly before end.")


def _require_market_ids(market_ids: Sequence[str]) -> list:
    ids = [str(market_id) for market_id in market_ids if str(market_id).strip()]
    if not ids:
        raise TailscaleDBError("At least one market_id is required.")
    return ids


def daily_trade_aggregate_query(start: datetime, end: datetime) -> tuple:
    """Per-market trade totals for one bounded window (page discovery by day)."""

    _require_window(start, end)
    sql = """
        SELECT market_id,
               count(*) AS trade_count,
               sum(size) AS contracts,
               sum(price * size) AS notional,
               count(*) FILTER (WHERE lower(outcome) NOT IN ('yes', 'no')) AS nonbinary_trades
        FROM public_trades
        WHERE event_time >= %(start)s AND event_time < %(end)s
        GROUP BY market_id
    """
    return sql, {"start": start, "end": end}


def markets_metadata_query(market_ids: Sequence[str]) -> tuple:
    """Metadata plus parent-event title (needed to catch sports like 'X vs. Y')."""

    ids = _require_market_ids(market_ids)
    sql = """
        SELECT m.id AS market_id,
               m.platform,
               m.title,
               m.tick_size,
               m.minimum_order_size,
               m.active,
               m.start_date,
               m.json_object ->> 'eventId' AS event_id,
               m.json_object ->> 'seriesId' AS series_id,
               e.title AS event_title
        FROM markets m
        LEFT JOIN events e ON e.id = m.json_object ->> 'eventId'
        WHERE m.id = ANY(%(market_ids)s)
    """
    return sql, {"market_ids": ids}


def capped_book_presence_query(
    market_id: str,
    start: datetime,
    end: datetime,
    snapshot_cap: int = 1001,
    diff_cap: int = 5001,
) -> tuple:
    """Cheap capped existence counts used to screen candidates for book coverage."""

    _require_window(start, end)
    sql = """
        SELECT
            (SELECT count(*) FROM (
                SELECT 1 FROM orderbook_snapshots
                WHERE market_id = %(market_id)s
                  AND event_time >= %(start)s AND event_time < %(end)s
                LIMIT %(snapshot_cap)s) s) AS snapshots_capped,
            (SELECT count(*) FROM (
                SELECT 1 FROM orderbook_diffs
                WHERE market_id = %(market_id)s
                  AND event_time >= %(start)s AND event_time < %(end)s
                LIMIT %(diff_cap)s) d) AS diffs_capped
    """
    params = {
        "market_id": str(market_id),
        "start": start,
        "end": end,
        "snapshot_cap": int(snapshot_cap),
        "diff_cap": int(diff_cap),
    }
    return sql, params


def daily_activity_query(table: str, market_id: str, start: datetime, end: datetime) -> tuple:
    """Per-day row counts (plus sessions for book tables) for coverage/gap checks."""

    if table not in HYPERTABLES:
        raise TailscaleDBError(f"Table '{table}' is not an allowed hypertable.")
    _require_window(start, end)
    session_column = (
        "count(DISTINCT session_id) AS sessions," if table in _BOOK_TABLES else "NULL AS sessions,"
    )
    sql = f"""
        SELECT date_trunc('day', event_time) AS day,
               count(*) AS row_count,
               {session_column}
               min(event_time) AS first_event,
               max(event_time) AS last_event
        FROM {table}
        WHERE market_id = %(market_id)s
          AND event_time >= %(start)s AND event_time < %(end)s
        GROUP BY 1
        ORDER BY 1
    """
    return sql, {"market_id": str(market_id), "start": start, "end": end}


def hourly_diff_counts_query(market_id: str, start: datetime, end: datetime) -> tuple:
    _require_window(start, end)
    sql = """
        SELECT date_trunc('hour', event_time) AS hour, count(*) AS diff_count
        FROM orderbook_diffs
        WHERE market_id = %(market_id)s
          AND event_time >= %(start)s AND event_time < %(end)s
        GROUP BY 1
        ORDER BY 1
        LIMIT 2000
    """
    return sql, {"market_id": str(market_id), "start": start, "end": end}


def latest_snapshot_query(market_id: str, at_time: datetime, lookback: timedelta) -> tuple:
    """Latest full snapshot at or before ``at_time`` within a bounded lookback."""

    if not isinstance(lookback, timedelta) or lookback <= timedelta(0):
        raise TailscaleDBError("Snapshot lookback must be a positive timedelta.")
    start = at_time - lookback
    _require_window(start, at_time + timedelta(microseconds=1))
    sql = """
        SELECT event_time, seq, bids, asks, session_id
        FROM orderbook_snapshots
        WHERE market_id = %(market_id)s
          AND event_time <= %(at_time)s AND event_time > %(start)s
        ORDER BY event_time DESC, seq DESC
        LIMIT 1
    """
    return sql, {"market_id": str(market_id), "at_time": at_time, "start": start}


def window_snapshots_query(
    market_id: str, start: datetime, end: datetime, limit: int = 20_000
) -> tuple:
    """Re-anchoring snapshots inside a replay window, in apply order.

    The window bound is the primary control; the limit is a defensive cap for
    seconds-cadence books (a hit degrades gracefully — diffs referencing a
    missing anchor are counted and skipped, not misapplied).
    """

    _require_window(start, end)
    if limit <= 0:
        raise TailscaleDBError("Snapshot row limit must be positive.")
    sql = """
        SELECT event_time, seq, bids, asks, session_id
        FROM orderbook_snapshots
        WHERE market_id = %(market_id)s
          AND event_time > %(start)s AND event_time <= %(end)s
        ORDER BY event_time ASC, seq ASC
        LIMIT %(limit)s
    """
    return sql, {"market_id": str(market_id), "start": start, "end": end, "limit": int(limit)}


def snapshot_cadence_query(market_id: str, start: datetime, end: datetime) -> tuple:
    """Median inter-snapshot interval, computed server-side (one row back).

    Snapshot cadence is activity-driven (seconds on busy books, tens of minutes
    on quiet ones), so this median is needed to scale gap thresholds per market
    — and fetching a month of raw snapshot times client-side to compute it
    would be the largest uncapped fetch in the probe. The server does one pass.
    """

    _require_window(start, end)
    sql = """
        SELECT count(*) AS interval_count,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY gap_seconds)
                   AS median_interval_seconds
        FROM (
            SELECT extract(epoch FROM event_time - lag(event_time)
                       OVER (ORDER BY event_time)) AS gap_seconds
            FROM orderbook_snapshots
            WHERE market_id = %(market_id)s
              AND event_time >= %(start)s AND event_time < %(end)s
        ) intervals
        WHERE gap_seconds IS NOT NULL
    """
    return sql, {"market_id": str(market_id), "start": start, "end": end}


def snapshot_gap_query(
    market_id: str,
    start: datetime,
    end: datetime,
    threshold_seconds: float,
    limit: int = 200,
) -> tuple:
    """Book-stream gaps above a threshold, computed server-side (gap rows only).

    A hole in the snapshot stream is not proof of an outage — cadence is
    activity-driven — so callers should cross-check gaps against trade prints
    and diff activity before treating them as stalls.
    """

    _require_window(start, end)
    if threshold_seconds <= 0:
        raise TailscaleDBError("Gap threshold must be positive.")
    if limit <= 0:
        raise TailscaleDBError("Gap row limit must be positive.")
    sql = """
        SELECT gap_start, event_time AS gap_end, gap_seconds,
               previous_session_id, session_id
        FROM (
            SELECT event_time,
                   session_id,
                   lag(event_time) OVER (ORDER BY event_time) AS gap_start,
                   lag(session_id) OVER (ORDER BY event_time) AS previous_session_id,
                   extract(epoch FROM event_time - lag(event_time)
                       OVER (ORDER BY event_time)) AS gap_seconds
            FROM orderbook_snapshots
            WHERE market_id = %(market_id)s
              AND event_time >= %(start)s AND event_time < %(end)s
        ) gaps
        WHERE gap_seconds > %(threshold_seconds)s
        ORDER BY gap_seconds DESC
        LIMIT %(limit)s
    """
    params = {
        "market_id": str(market_id),
        "start": start,
        "end": end,
        "threshold_seconds": float(threshold_seconds),
        "limit": int(limit),
    }
    return sql, params


def gap_rows_from_query(rows: Sequence[Mapping[str, Any]]) -> list:
    """Normalize snapshot_gap_query rows into gap dicts, oldest first."""

    gaps = []
    for row in rows:
        previous_session = row.get("previous_session_id")
        session = row.get("session_id")
        gaps.append(
            {
                "gap_start": row["gap_start"],
                "gap_end": row["gap_end"],
                "gap_seconds": float(row["gap_seconds"]),
                "session_changed": (
                    previous_session is not None
                    and session is not None
                    and str(previous_session) != str(session)
                ),
            }
        )
    gaps.sort(key=lambda gap: gap["gap_start"])
    return gaps


def window_diffs_query(market_id: str, start: datetime, end: datetime, limit: int) -> tuple:
    """Diff stream for a replay window. Diff sizes are signed deltas."""

    _require_window(start, end)
    if limit <= 0:
        raise TailscaleDBError("Diff row limit must be positive.")
    sql = """
        SELECT event_time, seq, snapshot_seq, bid_diffs, ask_diffs, session_id
        FROM orderbook_diffs
        WHERE market_id = %(market_id)s
          AND event_time > %(start)s AND event_time <= %(end)s
        ORDER BY event_time ASC, seq ASC
        LIMIT %(limit)s
    """
    return sql, {"market_id": str(market_id), "start": start, "end": end, "limit": int(limit)}


def window_trades_query(market_id: str, start: datetime, end: datetime, limit: int) -> tuple:
    _require_window(start, end)
    if limit <= 0:
        raise TailscaleDBError("Trade row limit must be positive.")
    sql = """
        SELECT event_time, trade_time, outcome, price, size, taker_side
        FROM public_trades
        WHERE market_id = %(market_id)s
          AND event_time >= %(start)s AND event_time < %(end)s
        ORDER BY event_time ASC
        LIMIT %(limit)s
    """
    return sql, {"market_id": str(market_id), "start": start, "end": end, "limit": int(limit)}


def diff_gap_query(market_id: str, start: datetime, end: datetime, limit: int = 20) -> tuple:
    """Largest inter-diff time gaps in a bounded window (README gap hunt)."""

    _require_window(start, end)
    sql = """
        SELECT event_time,
               session_id,
               event_time - lag(event_time) OVER (ORDER BY seq) AS gap,
               lag(session_id) OVER (ORDER BY seq) AS previous_session_id
        FROM orderbook_diffs
        WHERE market_id = %(market_id)s
          AND event_time >= %(start)s AND event_time < %(end)s
        ORDER BY gap DESC NULLS LAST
        LIMIT %(limit)s
    """
    return sql, {"market_id": str(market_id), "start": start, "end": end, "limit": int(limit)}


def minute_price_series_query(market_id: str, start: datetime, end: datetime) -> tuple:
    """Minute-bucketed trade prices for coarse jump scanning.

    Prices are used as printed (identity space). Empirically, Polymarket
    NO-labeled prints already carry book-space prices — complementing them
    (1 - p) misaligns them against the reconstructed book and manufactures
    fake full-range jumps. The alignment stage measures both conventions per
    market so this assumption stays observable.
    """

    _require_window(start, end)
    sql = """
        SELECT date_trunc('minute', event_time) AS minute,
               avg(price) AS price_avg,
               count(*) AS trade_count,
               sum(size) AS contracts
        FROM public_trades
        WHERE market_id = %(market_id)s
          AND event_time >= %(start)s AND event_time < %(end)s
          AND lower(outcome) IN ('yes', 'no')
        GROUP BY 1
        ORDER BY 1
    """
    return sql, {"market_id": str(market_id), "start": start, "end": end}


def hourly_trade_counts_query(market_id: str, start: datetime, end: datetime) -> tuple:
    """Per-hour trade counts, for picking a replay window with actual trades."""

    _require_window(start, end)
    sql = """
        SELECT date_trunc('hour', event_time) AS hour, count(*) AS trade_count
        FROM public_trades
        WHERE market_id = %(market_id)s
          AND event_time >= %(start)s AND event_time < %(end)s
        GROUP BY 1
        ORDER BY 1
        LIMIT 2000
    """
    return sql, {"market_id": str(market_id), "start": start, "end": end}


def resolution_query(market_ids: Sequence[str], start: datetime, end: datetime) -> tuple:
    """Latest lifecycle resolution/close info per market (bounded; hypertable)."""

    ids = _require_market_ids(market_ids)
    _require_window(start, end)
    sql = """
        SELECT DISTINCT ON (market_id)
               market_id, stage, outcome, resolution_time, close_time, event_time, source
        FROM market_lifecycle_events
        WHERE market_id = ANY(%(market_ids)s)
          AND event_time >= %(start)s AND event_time < %(end)s
          AND (outcome IS NOT NULL OR resolution_time IS NOT NULL OR close_time IS NOT NULL)
        ORDER BY market_id, event_time DESC
    """
    return sql, {"market_ids": ids, "start": start, "end": end}


# ---------------------------------------------------------------------------
# Order-book reconstruction
# ---------------------------------------------------------------------------


def book_side_from_json(levels: Optional[Iterable[Mapping[str, Any]]]) -> dict:
    """Full snapshot side: absolute sizes keyed by Decimal price."""

    side: dict = {}
    for level in levels or ():
        price = Decimal(str(level["price"]))
        size = Decimal(str(level["size"]))
        if size > 0:
            side[price] = size
    return side


def apply_signed_diffs(side: dict, diffs: Optional[Iterable[Mapping[str, Any]]]) -> None:
    """Apply diff levels in place. ``size`` is a signed delta, never absolute."""

    for level in diffs or ():
        price = Decimal(str(level["price"]))
        delta = Decimal(str(level["size"]))
        new_size = side.get(price, ZERO) + delta
        if new_size <= 0:
            side.pop(price, None)
        else:
            side[price] = new_size


@dataclass
class BookState:
    """Mutable reconstructed book. Yielded in place by ``fold_book_events``.

    ``checkpoint`` is set only on checkpoint yields (see ``fold_book_events``):
    it marks that this yield represents the book *as of* that checkpoint time,
    with ``event_time`` still holding the last applied event's time.
    """

    bids: dict = field(default_factory=dict)
    asks: dict = field(default_factory=dict)
    anchor_seq: Optional[int] = None
    event_time: Optional[datetime] = None
    session_id: Optional[str] = None
    checkpoint: Optional[datetime] = None

    @property
    def best_bid(self) -> Optional[Decimal]:
        return max(self.bids) if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        return min(self.asks) if self.asks else None

    @property
    def mid(self) -> Optional[Decimal]:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    @property
    def spread(self) -> Optional[Decimal]:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return ask - bid

    def is_crossed(self) -> bool:
        bid, ask = self.best_bid, self.best_ask
        return bid is not None and ask is not None and bid >= ask


@dataclass
class FoldStats:
    snapshots_applied: int = 0
    diffs_applied: int = 0
    diffs_skipped_wrong_anchor: int = 0
    diffs_skipped_duplicate: int = 0
    crossed_book_events: int = 0


def fold_book_events(
    seed_snapshot: Optional[Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
    diffs: Sequence[Mapping[str, Any]],
    stats: Optional[FoldStats] = None,
    checkpoints: Optional[Sequence[datetime]] = None,
) -> Iterator[BookState]:
    """Replay a window: seed snapshot, re-anchor on each new snapshot, apply diffs.

    Diffs are applied only when their ``snapshot_seq`` matches the current
    anchor's ``seq``; anything else is counted and skipped, never accumulated
    across snapshot boundaries. The yielded ``BookState`` is mutated in place —
    callers must read metrics immediately, not store references.

    ``checkpoints`` (ascending datetimes) request extra *as-of* yields: before
    applying any event later than a pending checkpoint, the current book is
    yielded with ``state.checkpoint`` set to that checkpoint time. This gives
    exact last-state-at-or-before sampling without copying books. Checkpoints
    that precede the first book state are skipped.
    """

    stats = stats if stats is not None else FoldStats()
    state = BookState()
    last_diff_seq: Optional[int] = None
    checkpoint_list = sorted(checkpoints) if checkpoints else []
    checkpoint_index = 0

    def pending_checkpoints(before: Optional[datetime]) -> Iterator[datetime]:
        nonlocal checkpoint_index
        while checkpoint_index < len(checkpoint_list) and (
            before is None or checkpoint_list[checkpoint_index] < before
        ):
            checkpoint = checkpoint_list[checkpoint_index]
            checkpoint_index += 1
            if state.event_time is not None and state.event_time <= checkpoint:
                yield checkpoint

    def apply_snapshot(row: Mapping[str, Any]) -> None:
        nonlocal last_diff_seq
        state.bids = book_side_from_json(row.get("bids"))
        state.asks = book_side_from_json(row.get("asks"))
        state.anchor_seq = int(row["seq"])
        state.event_time = row.get("event_time")
        state.session_id = row.get("session_id")
        last_diff_seq = None
        stats.snapshots_applied += 1

    events = []
    for row in snapshots:
        events.append((row["event_time"], 0, int(row["seq"]), "snapshot", row))
    for row in diffs:
        events.append((row["event_time"], 1, int(row["seq"]), "diff", row))
    events.sort(key=lambda item: (item[0], item[1], item[2]))

    if seed_snapshot is not None:
        apply_snapshot(seed_snapshot)
        if state.is_crossed():
            stats.crossed_book_events += 1
        yield state

    for event_time, _, seq, kind, row in events:
        for checkpoint in pending_checkpoints(event_time):
            state.checkpoint = checkpoint
            yield state
            state.checkpoint = None
        if kind == "snapshot":
            apply_snapshot(row)
        else:
            if state.anchor_seq is None or int(row["snapshot_seq"]) != state.anchor_seq:
                stats.diffs_skipped_wrong_anchor += 1
                continue
            if last_diff_seq is not None and seq <= last_diff_seq:
                stats.diffs_skipped_duplicate += 1
                continue
            apply_signed_diffs(state.bids, row.get("bid_diffs"))
            apply_signed_diffs(state.asks, row.get("ask_diffs"))
            state.event_time = event_time
            state.session_id = row.get("session_id")
            last_diff_seq = seq
            stats.diffs_applied += 1
        if state.is_crossed():
            stats.crossed_book_events += 1
        yield state

    for checkpoint in pending_checkpoints(None):
        state.checkpoint = checkpoint
        yield state
        state.checkpoint = None


# ---------------------------------------------------------------------------
# Derived book metrics
# ---------------------------------------------------------------------------


def band_label(band: Decimal) -> str:
    cents = (band * 100).normalize()
    return f"{int(cents)}c"


def top_of_book_metrics(state: BookState) -> dict:
    return {
        "best_bid": state.best_bid,
        "best_ask": state.best_ask,
        "mid": state.mid,
        "spread": state.spread,
    }


def depth_within(side: Mapping[Decimal, Decimal], mid: Decimal, band: Decimal, is_bid: bool) -> Decimal:
    """Total size within ``band`` of mid on one side (bids below, asks above)."""

    total = ZERO
    if is_bid:
        floor = mid - band
        for price, size in side.items():
            if floor <= price <= mid:
                total += size
    else:
        ceiling = mid + band
        for price, size in side.items():
            if mid <= price <= ceiling:
                total += size
    return total


def depth_metrics(state: BookState, bands: Sequence[Decimal] = DEPTH_BANDS) -> dict:
    mid = state.mid
    metrics: dict = {}
    for band in bands:
        label = band_label(band)
        if mid is None:
            metrics[f"bid_depth_within_{label}"] = None
            metrics[f"ask_depth_within_{label}"] = None
        else:
            metrics[f"bid_depth_within_{label}"] = depth_within(state.bids, mid, band, True)
            metrics[f"ask_depth_within_{label}"] = depth_within(state.asks, mid, band, False)
    bid_5c = metrics.get("bid_depth_within_5c")
    ask_5c = metrics.get("ask_depth_within_5c")
    if bid_5c is None or ask_5c is None or (bid_5c + ask_5c) == 0:
        metrics["imbalance_5c"] = None
    else:
        metrics["imbalance_5c"] = (bid_5c - ask_5c) / (bid_5c + ask_5c)
    return metrics


@dataclass(frozen=True)
class ExitCurve:
    """Executable exit value for liquidating ``quantity`` against one book side."""

    side: str
    quantity: Decimal
    filled_quantity: Decimal
    unfilled_quantity: Decimal
    executable_value: Decimal
    vwap_price: Optional[Decimal]
    worst_price: Optional[Decimal]
    best_price: Optional[Decimal]
    levels_used: int


def executable_exit_curve(
    book_side: Mapping[Decimal, Decimal], quantity: Decimal, is_sell: bool
) -> ExitCurve:
    """Walk displayed liquidity: selling YES hits bids high→low, buying lifts asks low→high.

    A partial book yields a partial fill; nothing here invents liquidity beyond
    displayed levels (mirrors the collar principle: reject, don't conjure).
    """

    if quantity <= 0:
        raise TailscaleDBError("Exit-curve quantity must be positive.")
    levels = sorted(book_side.items(), key=lambda kv: kv[0], reverse=is_sell)
    remaining = quantity
    value = ZERO
    filled = ZERO
    worst: Optional[Decimal] = None
    best: Optional[Decimal] = None
    levels_used = 0
    for price, size in levels:
        if remaining <= 0:
            break
        take = size if size <= remaining else remaining
        value += take * price
        filled += take
        remaining -= take
        worst = price
        if best is None:
            best = price
        levels_used += 1
    return ExitCurve(
        side="sell_yes" if is_sell else "buy_yes",
        quantity=quantity,
        filled_quantity=filled,
        unfilled_quantity=quantity - filled,
        executable_value=value,
        vwap_price=(value / filled) if filled > 0 else None,
        worst_price=worst,
        best_price=best,
        levels_used=levels_used,
    )


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------


def detect_stream_gaps(
    rows: Sequence[tuple],
    threshold: timedelta,
) -> list:
    """Find inter-event gaps above ``threshold`` in (event_time, session_id) rows."""

    gaps = []
    previous_time: Optional[datetime] = None
    previous_session: Optional[str] = None
    for event_time, session_id in rows:
        if previous_time is not None:
            delta = event_time - previous_time
            session_changed = (
                previous_session is not None
                and session_id is not None
                and session_id != previous_session
            )
            if delta > threshold:
                gaps.append(
                    {
                        "gap_start": previous_time,
                        "gap_end": event_time,
                        "gap_seconds": delta.total_seconds(),
                        "session_changed": session_changed,
                    }
                )
        previous_time = event_time
        previous_session = session_id
    return gaps


def annotate_gaps_with_trades(
    gaps: Sequence[dict],
    trade_minutes: Sequence[datetime],
    diff_hours: Optional[set] = None,
) -> None:
    """Cross-check book-silent gaps against trade prints, in place.

    Snapshot cadence is activity-driven (and much slower on Kalshi than on
    Polymarket), so a silent snapshot stream can be a quiet market rather than
    an outage. A gap is marked ``suspicious`` only when a trade printed during
    an hour with no recorded diffs at all — trades imply book changes, so a
    diff-free trading hour inside the silence is a genuine stall signal. When
    ``diff_hours`` (hours having >= 1 diff) is not provided, any trade during
    the gap marks it suspicious, which overcounts on slow-cadence books.
    """

    minutes_sorted = sorted(trade_minutes)
    for gap in gaps:
        low = bisect_right(minutes_sorted, gap["gap_start"])
        high = bisect_left(minutes_sorted, gap["gap_end"])
        inside = minutes_sorted[low:high]
        gap["trade_minutes_during_gap"] = len(inside)
        if diff_hours is None:
            gap["suspicious"] = bool(inside)
        else:
            gap["suspicious"] = any(
                minute.replace(minute=0, second=0, microsecond=0) not in diff_hours
                for minute in inside
            )


def day_coverage_summary(day_rows: Sequence[Mapping[str, Any]]) -> dict:
    """Zero-days and day-boundary gaps from per-day activity rows.

    Missing days are counted between the market's own first and last active
    days, so a market that starts or resolves mid-window is not penalized.
    """

    days = sorted(row["day"].date() if isinstance(row["day"], datetime) else row["day"] for row in day_rows)
    if not days:
        return {"days_present": 0, "days_missing_between_first_last": 0, "missing_days": []}
    first_day, last_day = days[0], days[-1]
    expected = set()
    cursor = first_day
    while cursor <= last_day:
        expected.add(cursor)
        cursor = cursor + timedelta(days=1)
    missing = sorted(expected - set(days))
    return {
        "days_present": len(days),
        "days_missing_between_first_last": len(missing),
        "missing_days": missing,
    }


# ---------------------------------------------------------------------------
# Trade alignment
# ---------------------------------------------------------------------------


def normalize_probability_price(price: Any) -> Optional[Decimal]:
    """Decimal probability price; cent-quoted feeds (> 1.5) are scaled down."""

    if price is None:
        return None
    value = Decimal(str(price))
    if value > Decimal("1.5"):
        value = value / 100
    return value


def _spread_flags(
    price: Decimal, best_bid: Optional[Decimal], best_ask: Optional[Decimal], half_tick: Decimal
) -> tuple:
    if best_bid is None or best_ask is None:
        return None, None
    in_spread = (best_bid - half_tick) <= price <= (best_ask + half_tick)
    at_touch = abs(price - best_bid) <= half_tick or abs(price - best_ask) <= half_tick
    return in_spread, at_touch


def align_trades_to_book(
    book_top_series: Sequence[tuple],
    trades: Sequence[Mapping[str, Any]],
    half_tick: Decimal = Decimal("0.005"),
) -> list:
    """Join trades to the last known top-of-book at or before each trade time.

    Each trade is tested under two price conventions, because feed semantics
    for NO-labeled prints are not documented: ``identity`` (price as printed)
    and ``complement`` (1 - price for NO-labeled outcomes). Which convention
    fits is an empirical output, not an assumption — smoke runs showed
    Polymarket NO-labeled prints align in identity space, not complement.

    ``book_top_series`` is a time-ascending sequence of
    ``(event_time, best_bid, best_ask)`` tuples from a fold pass.
    """

    if not book_top_series:
        return []
    times = [row[0] for row in book_top_series]
    aligned = []
    for trade in trades:
        trade_time = trade["event_time"]
        index = bisect_right(times, trade_time) - 1
        if index < 0:
            continue
        outcome_text = str(trade.get("outcome") or "").strip().lower()
        if outcome_text not in {"yes", "no"}:
            continue
        identity_price = normalize_probability_price(trade.get("price"))
        if identity_price is None:
            continue
        complement_price = (ONE - identity_price) if outcome_text == "no" else identity_price
        book_time, best_bid, best_ask = book_top_series[index]
        in_spread_identity, at_touch_identity = _spread_flags(
            identity_price, best_bid, best_ask, half_tick
        )
        in_spread_complement, at_touch_complement = _spread_flags(
            complement_price, best_bid, best_ask, half_tick
        )
        aligned.append(
            {
                "trade_time": trade_time,
                "book_time": book_time,
                "book_age_seconds": (trade_time - book_time).total_seconds(),
                "outcome": trade.get("outcome"),
                "taker_side": trade.get("taker_side"),
                "raw_price": trade.get("price"),
                "identity_price": identity_price,
                "complement_price": complement_price,
                "size": trade.get("size"),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "in_spread_identity": in_spread_identity,
                "at_touch_identity": at_touch_identity,
                "in_spread_complement": in_spread_complement,
                "at_touch_complement": at_touch_complement,
            }
        )
    return aligned


# ---------------------------------------------------------------------------
# Coarse jump scanning (candidate labels only; no public/private attribution)
# ---------------------------------------------------------------------------

JUMP_THRESHOLDS = (Decimal("0.05"), Decimal("0.10"), Decimal("0.20"))


def detect_minute_jumps(
    minute_rows: Sequence[Mapping[str, Any]],
    window_minutes: int = 10,
    thresholds: Sequence[Decimal] = JUMP_THRESHOLDS,
) -> list:
    """Flag minutes whose price moved >= min(threshold) vs any prior minute in window.

    Consecutive flagged minutes within one window-length collapse into a single
    episode keeping the largest move.
    """

    min_threshold = min(thresholds)
    window = timedelta(minutes=window_minutes)
    series = [
        (row["minute"], Decimal(str(row["price_avg"])), row.get("trade_count"))
        for row in minute_rows
        if row.get("price_avg") is not None
    ]
    series.sort(key=lambda item: item[0])
    episodes: list = []
    for index, (minute, price, trade_count) in enumerate(series):
        best_move = None
        price_before = None
        for back in range(index - 1, -1, -1):
            earlier_minute, earlier_price, _ = series[back]
            if minute - earlier_minute > window:
                break
            move = price - earlier_price
            if best_move is None or abs(move) > abs(best_move):
                best_move = move
                price_before = earlier_price
        if best_move is None or abs(best_move) < min_threshold:
            continue
        jump = {
            "minute": minute,
            "jump_size": abs(best_move),
            "direction": "up" if best_move >= 0 else "down",
            "price_before": price_before,
            "price_after": price,
            "trade_count": trade_count,
            "window_minutes": window_minutes,
        }
        for threshold in thresholds:
            jump[f"threshold_{band_label(threshold)}"] = abs(best_move) >= threshold
        if episodes and minute - episodes[-1]["minute"] <= window:
            if jump["jump_size"] > episodes[-1]["jump_size"]:
                episodes[-1] = jump
        else:
            episodes.append(jump)
    return episodes


def label_jump_timing(
    minute: datetime,
    resolution_time: Optional[datetime],
    close_time: Optional[datetime],
    near_window: timedelta = timedelta(hours=24),
) -> str:
    """Coarse label only. Public vs private attribution needs event labels."""

    reference = resolution_time or close_time
    if reference is None:
        return "unknown"
    if minute >= reference - near_window:
        return "terminal_near_resolution_candidate"
    return "interim_candidate"


# ---------------------------------------------------------------------------
# Market classification (title/event heuristics; reviewable, not ground truth)
# ---------------------------------------------------------------------------

SPORTS_TICKER_PREFIXES = (
    "KXNFL", "KXNBA", "KXWNBA", "KXMLB", "KXNHL", "KXNCAA", "KXCFB", "KXCBB",
    "KXUFC", "KXMMA", "KXBOX", "KXATP", "KXWTA", "KXITF", "KXTENNIS", "KXPGA",
    "KXGOLF", "KXF1", "KXNASCAR", "KXINDY", "KXMLS", "KXEPL", "KXUCL", "KXUEL",
    "KXFIFA", "KXWC", "KXSOCCER", "KXLALIGA", "KXSERIEA", "KXBUNDES", "KXLIGUE",
    "KXCRICKET", "KXIPL", "KXRUGBY", "KXDARTS", "KXSNOOKER", "KXCYCL",
    "KXMARATHON", "KXOLYMPIC", "KXESPORTS", "KXLOL", "KXCSGO", "KXCS2",
    "KXDOTA", "KXVALORANT", "KXHOCKEY", "KXTOURDE", "KXSKI", "KXSUMO",
)

_SPORTS_PATTERNS = (
    r"world cup", r"\bfifa\b", r"\buefa\b", r"champions league", r"europa league",
    r"premier league", r"la liga", r"serie a", r"bundesliga", r"ligue 1", r"\bmls\b",
    r"\bepl\b", r"\bcopa\b", r"concacaf", r"olympic", r"super bowl", r"\bnfl\b",
    r"\bnba\b", r"\bwnba\b", r"\bmlb\b", r"\bnhl\b", r"\bncaa\b", r"march madness",
    r"stanley cup", r"world series", r"playoff", r"grand slam", r"wimbledon",
    r"french open", r"australian open", r"us open", r"\batp\b", r"\bwta\b",
    r"tennis", r"soccer", r"football", r"basketball", r"baseball", r"hockey",
    r"\bgolf\b", r"boxing", r"\bufc\b", r"\bmma\b", r"wrestl", r"cricket",
    r"rugby", r"formula 1", r"grand prix", r"\bf1\b", r"nascar", r"motogp",
    r"marathon", r"cycling", r"tour de france", r"esports", r"league of legends",
    r"\bcsgo\b", r"\bcs2\b", r"\bdota\b", r"valorant", r"heisman", r"ballon d'or",
    r"home run", r"touchdown", r"\bvs\.?\b", r"\bmatch\b", r"scorer", r"knockout",
)
SPORTS_REGEX = re.compile("|".join(_SPORTS_PATTERNS))

#: (category, regex) checked in priority order on "title | event_title" lowercase.
CATEGORY_PATTERNS = (
    (
        "politics_election",
        re.compile(
            r"election|senate|governor|president|congress|congressional|primary|"
            r"nominee|nomination|mayor|parliament|prime minister|midterm|ballot|"
            r"referendum|approval rating|impeach|cabinet|secretary of|veto|"
            r"government shutdown|speaker of|coalition|chancellor|\bpoll\b|"
            r"electoral|candidate|party leader|minister|white house|recall|\bveep\b"
        ),
    ),
    (
        "macro_finance_rates",
        re.compile(
            r"\bfed\b|fomc|\bcpi\b|inflation|\bgdp\b|interest rate|rate cut|"
            r"rate hike|recession|unemployment|jobs report|nonfarm|payroll|"
            r"treasury|tariff|trade deal|s&p|nasdaq|dow jones|\bstock\b|\bipo\b|"
            r"earnings|market cap|gas price|oil price|\bopec\b|gold price|"
            r"silver price|mortgage|powell|\becb\b|debt ceiling|basis points"
        ),
    ),
    (
        "crypto_regulatory",
        re.compile(
            r"bitcoin|\bbtc\b|ethereum|\beth\b|solana|\bsol\b|\bxrp\b|\bdoge\b|"
            r"crypto|stablecoin|\betf\b|\bsec\b|binance|coinbase|\bdefi\b|"
            r"\btoken\b|airdrop|satoshi|memecoin|halving"
        ),
    ),
    (
        "legal_court_policy",
        re.compile(
            r"court|scotus|ruling|verdict|\btrial\b|indict|lawsuit|judge|"
            r"conviction|convicted|sentenc|appeal|\bdoj\b|pardon|charges|\bplea\b|"
            r"extradit|deport|executive order|\bban\b|regulation|antitrust|injunction"
        ),
    ),
    (
        "announcement_product",
        re.compile(
            r"apple|iphone|openai|\bgpt\b|anthropic|claude|gemini|google|microsoft|"
            r"tesla|spacex|starship|launch|release date|announce|unveil|oscar|"
            r"grammy|emmy|nobel|time person|album|box office|netflix|spotify|"
            r"game of the year|wwdc|keynote|ai model"
        ),
    ),
)

CATEGORY_TO_BUCKET = {
    "politics_election": "politics",
    "macro_finance_rates": "macro_finance_crypto",
    "crypto_regulatory": "macro_finance_crypto",
    "legal_court_policy": "legal_policy",
    "announcement_product": "announcement_other",
    "other": "announcement_other",
}


@dataclass(frozen=True)
class MarketClassification:
    category: str
    bucket: str
    is_sports: bool
    matched: str


def classify_market(
    title: Optional[str],
    event_title: Optional[str] = None,
    series_id: Optional[str] = None,
    market_id: Optional[str] = None,
) -> MarketClassification:
    text = f"{title or ''} | {event_title or ''}".lower()
    for identifier in (series_id, market_id):
        upper = str(identifier or "").upper()
        for prefix in SPORTS_TICKER_PREFIXES:
            if upper.startswith(prefix):
                return MarketClassification("sports", "sports", True, f"prefix:{prefix}")
    sports_hit = SPORTS_REGEX.search(text)
    if sports_hit:
        return MarketClassification("sports", "sports", True, f"keyword:{sports_hit.group(0)}")
    for category, pattern in CATEGORY_PATTERNS:
        hit = pattern.search(text)
        if hit:
            return MarketClassification(
                category, CATEGORY_TO_BUCKET[category], False, f"keyword:{hit.group(0)}"
            )
    return MarketClassification("other", CATEGORY_TO_BUCKET["other"], False, "")


def bucket_quotas(max_markets: int, include_quiet: bool = True) -> dict:
    """Split the sample across preferred buckets (roughly 32/32/16/20 percent).

    Invariant: quotas always sum to exactly ``max_markets``.
    """

    if max_markets <= 0:
        raise TailscaleDBError("max_markets must be positive.")
    quiet = min(max(1, round(max_markets * 0.20)), max_markets) if include_quiet else 0
    remaining = max_markets - quiet
    politics = remaining * 2 // 5
    macro = remaining * 2 // 5
    legal = remaining - politics - macro
    return {
        "politics": politics,
        "macro_finance_crypto": macro,
        "legal_policy": legal,
        "quiet_baseline": quiet,
    }


# ---------------------------------------------------------------------------
# Output safety
# ---------------------------------------------------------------------------


def scan_outputs_for_secret(paths: Iterable, secret: Optional[str]) -> list:
    """Return output files that leak ``secret``. Used as a post-write guard."""

    if not secret:
        return []
    offending = []
    for path in paths:
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if secret in text:
            offending.append(path)
    return offending


def parse_utc_timestamp(value: str) -> datetime:
    """Parse ISO-8601 (with 'Z' allowed) into an aware UTC datetime."""

    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iterate_days(start: datetime, end: datetime) -> Iterator[tuple]:
    """Yield (day_start, day_end) UTC pages covering [start, end)."""

    _require_window(start, end)
    cursor = start
    while cursor < end:
        page_end = min(cursor + timedelta(days=1), end)
        yield cursor, page_end
        cursor = page_end
