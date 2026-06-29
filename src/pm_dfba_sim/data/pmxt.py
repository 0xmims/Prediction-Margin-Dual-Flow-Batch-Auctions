from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd


EVENT_TYPES_OF_INTEREST = ("book", "price_change", "last_trade_price", "tick_size_change")
DEPTH_RADII = (0.01, 0.05, 0.10)


class PMXTProbeError(ValueError):
    """Raised when a PMXT probe cannot load or inspect the requested file."""


@dataclass(frozen=True)
class PMXTProbeResult:
    schema_summary: dict[str, Any]
    event_type_counts: pd.DataFrame
    market_sample: pd.DataFrame
    top_of_book_timeseries: pd.DataFrame
    depth_timeseries: pd.DataFrame
    report_path: Path


def run_pmxt_probe(
    input_path: str | Path,
    out_dir: str | Path,
    max_rows: int = 200_000,
    max_markets: int = 3,
    source_label: str | None = None,
    url_diagnostics: dict[str, Any] | None = None,
) -> PMXTProbeResult:
    """Run a bounded feasibility probe over one PMXT parquet file."""

    path = Path(input_path)
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frame = read_parquet_sample(path, max_rows=max_rows)
    if frame.empty:
        raise PMXTProbeError(f"No rows loaded from {path}")

    event_type_column = detect_event_type_column(frame)
    timestamp_column = detect_timestamp_column(frame)
    market_column = detect_market_column(frame)
    trade_side_column = detect_trade_side_column(frame)

    event_counts = count_event_types(frame, event_type_column)
    market_sample = build_market_sample(frame, event_type_column, market_column, max_markets)
    top_of_book, depth = reconstruct_l2_timeseries(
        frame=frame,
        event_type_column=event_type_column,
        timestamp_column=timestamp_column,
        market_column=market_column,
        max_markets=max_markets,
    )

    schema_summary = build_schema_summary(
        frame=frame,
        source_label=source_label or str(path),
        event_type_column=event_type_column,
        timestamp_column=timestamp_column,
        market_column=market_column,
        trade_side_column=trade_side_column,
        event_counts=event_counts,
        top_of_book=top_of_book,
        depth=depth,
        max_rows=max_rows,
        max_markets=max_markets,
        url_diagnostics=url_diagnostics,
    )

    with (output_path / "schema_summary.json").open("w") as f:
        json.dump(schema_summary, f, indent=2, default=_json_default)
    if url_diagnostics is not None:
        write_url_diagnostics(output_path / "url_diagnostics.json", url_diagnostics)
    event_counts.to_csv(output_path / "event_type_counts.csv", index=False)
    market_sample.to_csv(output_path / "market_sample.csv", index=False)
    top_of_book.to_csv(output_path / "top_of_book_timeseries.csv", index=False)
    depth.to_csv(output_path / "depth_timeseries.csv", index=False)
    report_path = output_path / "pmxt_probe_report.md"
    write_probe_report(report_path, schema_summary)

    return PMXTProbeResult(
        schema_summary=schema_summary,
        event_type_counts=event_counts,
        market_sample=market_sample,
        top_of_book_timeseries=top_of_book,
        depth_timeseries=depth,
        report_path=report_path,
    )


def read_parquet_sample(path: str | Path, max_rows: int) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise PMXTProbeError(
            "Reading PMXT parquet files requires pyarrow. Install with `python3 -m pip install pyarrow`."
        ) from exc

    parquet_file = pq.ParquetFile(path)
    batches = parquet_file.iter_batches(batch_size=max_rows)
    try:
        return next(batches).to_pandas()
    except StopIteration:
        return pd.DataFrame()


def diagnose_pmxt_url(
    url: str,
    opener: Callable[..., Any] = urllib.request.urlopen,
    timeout: int = 20,
    attempted_at_utc: str | None = None,
) -> dict[str, Any]:
    """Perform a safe PMXT URL diagnostic without downloading the full parquet."""

    attempted_at = attempted_at_utc or datetime.now(timezone.utc).isoformat()
    notes: list[str] = []
    diagnostics: dict[str, Any] = {
        "url": url,
        "attempted_at_utc": attempted_at,
        "http_status": None,
        "content_type": None,
        "content_length": None,
        "accepts_ranges": None,
        "requires_api_key_guess": "unknown",
        "notes": notes,
    }

    head_request = urllib.request.Request(url, method="HEAD")
    try:
        with opener(head_request, timeout=timeout) as response:
            _merge_response_diagnostics(diagnostics, response)
            notes.append("HEAD request succeeded.")
            diagnostics["requires_api_key_guess"] = _api_key_guess(diagnostics["http_status"])
            return diagnostics
    except Exception as exc:  # noqa: BLE001 - diagnostic should report, not crash.
        notes.append(f"HEAD request failed: {_exception_summary(exc)}")
        _merge_exception_diagnostics(diagnostics, exc)

    range_request = urllib.request.Request(url, headers={"Range": "bytes=0-1023"}, method="GET")
    try:
        with opener(range_request, timeout=timeout) as response:
            _merge_response_diagnostics(diagnostics, response)
            response.read(1024)
            status = diagnostics["http_status"]
            if status == 206:
                notes.append("Range GET succeeded with HTTP 206 Partial Content.")
            else:
                notes.append(f"Range GET returned HTTP {status}; server may ignore range requests.")
            diagnostics["requires_api_key_guess"] = _api_key_guess(status)
    except Exception as exc:  # noqa: BLE001 - diagnostic should report, not crash.
        notes.append(f"Range GET failed: {_exception_summary(exc)}")
        _merge_exception_diagnostics(diagnostics, exc)
        diagnostics["requires_api_key_guess"] = _api_key_guess(diagnostics["http_status"])

    return diagnostics


def write_url_diagnostics(path: Path, diagnostics: dict[str, Any]) -> None:
    path.write_text(json.dumps(diagnostics, indent=2, default=_json_default) + "\n")


def detect_event_type_column(frame_or_columns: pd.DataFrame | Iterable[str]) -> str | None:
    return _choose_column(
        _columns(frame_or_columns),
        exact=("event_type", "type", "event", "message_type", "channel"),
        contains=("event_type", "event", "type"),
    )


def detect_timestamp_column(frame_or_columns: pd.DataFrame | Iterable[str]) -> str | None:
    return _choose_column(
        _columns(frame_or_columns),
        exact=("timestamp", "ts", "time", "datetime", "created_at", "event_timestamp", "received_at"),
        contains=("timestamp", "time"),
    )


def detect_market_column(frame_or_columns: pd.DataFrame | Iterable[str]) -> str | None:
    return _choose_column(
        _columns(frame_or_columns),
        exact=(
            "market_id",
            "condition_id",
            "conditionId",
            "asset_id",
            "token_id",
            "clob_token_id",
            "outcome_token_id",
            "market",
            "slug",
        ),
        contains=("market_id", "condition", "asset", "token"),
    )


def detect_trade_side_column(frame_or_columns: pd.DataFrame | Iterable[str]) -> str | None:
    return _choose_column(
        _columns(frame_or_columns),
        exact=("side", "taker_side", "trade_side", "direction", "aggressor_side"),
        contains=("side", "direction"),
    )


def count_event_types(
    frame: pd.DataFrame,
    event_type_column: str | None = None,
) -> pd.DataFrame:
    column = event_type_column or detect_event_type_column(frame)
    if column is None or column not in frame:
        return pd.DataFrame(columns=["event_type", "count"])
    return (
        frame[column]
        .fillna("__missing__")
        .astype(str)
        .value_counts()
        .rename_axis("event_type")
        .reset_index(name="count")
    )


def build_market_sample(
    frame: pd.DataFrame,
    event_type_column: str | None,
    market_column: str | None,
    max_markets: int,
) -> pd.DataFrame:
    if market_column is None or market_column not in frame:
        return pd.DataFrame(columns=["market_id", "row_count", *[f"has_{event}" for event in EVENT_TYPES_OF_INTEREST]])

    working = frame[[market_column] + ([event_type_column] if event_type_column else [])].copy()
    working["market_id"] = working[market_column].astype(str)
    if event_type_column:
        working["event_type_normalized"] = working[event_type_column].map(_normalize_event_type)
    else:
        working["event_type_normalized"] = ""

    rows: list[dict[str, Any]] = []
    for market_id, group in working.groupby("market_id", dropna=False):
        event_types = set(group["event_type_normalized"])
        row = {"market_id": market_id, "row_count": int(len(group))}
        for event in EVENT_TYPES_OF_INTEREST:
            row[f"has_{event}"] = event in event_types
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["market_id", "row_count", *[f"has_{event}" for event in EVENT_TYPES_OF_INTEREST]])
    sample = pd.DataFrame(rows).sort_values("row_count", ascending=False)
    return sample.head(max_markets).reset_index(drop=True)


def parse_book_payload(row: pd.Series | dict[str, Any]) -> dict[str, list[tuple[float, float]]]:
    data = dict(row)
    bids = _parse_side(_first_present(data, ("bids", "bid", "buy", "buys")))
    asks = _parse_side(_first_present(data, ("asks", "ask", "sell", "sells")))
    if bids or asks:
        return {"bids": _sort_bids(bids), "asks": _sort_asks(asks)}

    for column in ("book", "payload", "data", "message", "raw", "event_data"):
        payload = _jsonish(data.get(column))
        book = _extract_book(payload)
        if book["bids"] or book["asks"]:
            return book
    return {"bids": [], "asks": []}


def compute_top_of_book(book: dict[str, list[tuple[float, float]]]) -> dict[str, float | None]:
    bids = _sort_bids(book.get("bids", []))
    asks = _sort_asks(book.get("asks", []))
    best_bid = bids[0][0] if bids else None
    best_bid_size = bids[0][1] if bids else None
    best_ask = asks[0][0] if asks else None
    best_ask_size = asks[0][1] if asks else None
    midpoint = (best_bid + best_ask) / 2.0 if best_bid is not None and best_ask is not None else None
    spread = best_ask - best_bid if best_bid is not None and best_ask is not None else None
    return {
        "best_bid": best_bid,
        "best_bid_size": best_bid_size,
        "best_ask": best_ask,
        "best_ask_size": best_ask_size,
        "midpoint": midpoint,
        "spread": spread,
    }


def compute_depth_near_mid(
    book: dict[str, list[tuple[float, float]]],
    midpoint: float | None,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    bids = _sort_bids(book.get("bids", []))
    asks = _sort_asks(book.get("asks", []))
    for radius in DEPTH_RADII:
        suffix = _radius_suffix(radius)
        if midpoint is None:
            result[f"depth_bid_{suffix}"] = None
            result[f"depth_ask_{suffix}"] = None
            result[f"depth_total_{suffix}"] = None
            continue
        bid_depth = sum(size for price, size in bids if price >= midpoint - radius)
        ask_depth = sum(size for price, size in asks if price <= midpoint + radius)
        result[f"depth_bid_{suffix}"] = float(bid_depth)
        result[f"depth_ask_{suffix}"] = float(ask_depth)
        result[f"depth_total_{suffix}"] = float(bid_depth + ask_depth)
    return result


def reconstruct_l2_timeseries(
    frame: pd.DataFrame,
    event_type_column: str | None = None,
    timestamp_column: str | None = None,
    market_column: str | None = None,
    max_markets: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_col = event_type_column or detect_event_type_column(frame)
    timestamp_col = timestamp_column or detect_timestamp_column(frame)
    market_col = market_column or detect_market_column(frame)
    if event_col is None or timestamp_col is None or market_col is None:
        return _empty_top_of_book(), _empty_depth()

    working = frame.copy()
    working["_pmxt_event_type"] = working[event_col].map(_normalize_event_type)
    working["_pmxt_timestamp"] = _parse_timestamps(working[timestamp_col])
    working["_pmxt_market_id"] = working[market_col].astype(str)
    working = working.dropna(subset=["_pmxt_timestamp"])
    selected_markets = _select_reconstruction_markets(working, max_markets=max_markets)

    top_rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    for market_id in selected_markets:
        market_rows = working[working["_pmxt_market_id"] == market_id].sort_values("_pmxt_timestamp")
        book: dict[str, list[tuple[float, float]]] = {"bids": [], "asks": []}
        have_book = False
        for _, row in market_rows.iterrows():
            event_type = row["_pmxt_event_type"]
            if event_type == "book":
                parsed = parse_book_payload(row)
                if parsed["bids"] or parsed["asks"]:
                    book = parsed
                    have_book = True
                else:
                    continue
            elif event_type == "price_change":
                if not have_book:
                    continue
                updates = parse_price_change_payload(row)
                if updates:
                    book = apply_price_change_updates(book, updates)
            else:
                continue

            top = compute_top_of_book(book)
            timestamp = row["_pmxt_timestamp"]
            top_rows.append(
                {
                    "market_id": market_id,
                    "timestamp": timestamp,
                    "source_event_type": event_type,
                    **top,
                }
            )
            depth_rows.append(
                {
                    "market_id": market_id,
                    "timestamp": timestamp,
                    "source_event_type": event_type,
                    **top,
                    **compute_depth_near_mid(book, top["midpoint"]),
                }
            )

    top = pd.DataFrame(top_rows) if top_rows else _empty_top_of_book()
    depth = pd.DataFrame(depth_rows) if depth_rows else _empty_depth()
    return top, depth


def parse_price_change_payload(row: pd.Series | dict[str, Any]) -> list[tuple[str, float, float]]:
    data = dict(row)
    updates: list[tuple[str, float, float]] = []

    side = _normalize_side(data.get("side") or data.get("book_side"))
    price = _to_probability(data.get("price"))
    size = _to_float(_first_non_missing(data.get("size"), data.get("quantity"), data.get("qty")))
    if side and price is not None and size is not None:
        updates.append((side, price, size))

    for column in ("changes", "price_changes", "payload", "data", "message", "raw", "event_data"):
        payload = _jsonish(data.get(column))
        updates.extend(_extract_price_changes(payload))
    return updates


def apply_price_change_updates(
    book: dict[str, list[tuple[float, float]]],
    updates: Iterable[tuple[str, float, float]],
) -> dict[str, list[tuple[float, float]]]:
    bid_map = {_price_key(price): size for price, size in book.get("bids", [])}
    ask_map = {_price_key(price): size for price, size in book.get("asks", [])}
    for side, price, size in updates:
        target = bid_map if side == "bid" else ask_map
        key = _price_key(price)
        if size <= 0:
            target.pop(key, None)
        else:
            target[key] = size
    return {
        "bids": _sort_bids([(price, size) for price, size in bid_map.items()]),
        "asks": _sort_asks([(price, size) for price, size in ask_map.items()]),
    }


def build_schema_summary(
    frame: pd.DataFrame,
    source_label: str,
    event_type_column: str | None,
    timestamp_column: str | None,
    market_column: str | None,
    trade_side_column: str | None,
    event_counts: pd.DataFrame,
    top_of_book: pd.DataFrame,
    depth: pd.DataFrame,
    max_rows: int,
    max_markets: int,
    url_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event_types = set(event_counts.get("event_type", pd.Series(dtype=str)).astype(str).str.lower())
    has_depth = (
        not depth.empty
        and "depth_total_1c" in depth
        and pd.to_numeric(depth["depth_total_1c"], errors="coerce").notna().any()
    )
    order_lifecycle_columns = [
        column
        for column in frame.columns
        if any(token in column.lower() for token in ("order_id", "maker", "owner", "cancel", "lifecycle"))
    ]
    suitability = {
        "calibration": bool(not top_of_book.empty or not depth.empty),
        "event_window_replay": bool(not top_of_book.empty and timestamp_column and market_column),
        "stale_loss_proxy_construction": bool(not top_of_book.empty),
        "true_maker_taker_latency_race_proof": False,
    }
    hourly_interpretation = build_hourly_file_interpretation(
        frame=frame,
        event_counts=event_counts,
        timestamp_column=timestamp_column,
        market_column=market_column,
    )
    return {
        "source": source_label,
        "rows_loaded": int(len(frame)),
        "max_rows": int(max_rows),
        "max_markets": int(max_markets),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "event_type_column": event_type_column,
        "timestamp_column": timestamp_column,
        "market_column": market_column,
        "trade_side_column": trade_side_column,
        "event_types_present": sorted(event_types),
        "required_event_types": {
            event_type: event_type in event_types for event_type in EVENT_TYPES_OF_INTEREST
        },
        "market_ids_identified": market_column is not None,
        "timestamps_identified": timestamp_column is not None,
        "top_of_book_reconstructed": not top_of_book.empty,
        "depth_near_mid_computed": bool(has_depth),
        "trade_events_present": "last_trade_price" in event_types or any("trade" in event for event in event_types),
        "trade_side_available": trade_side_column is not None,
        "order_lifecycle_columns_observed": order_lifecycle_columns,
        "hourly_file_interpretation": hourly_interpretation,
        "url_diagnostics": url_diagnostics,
        "suitability": suitability,
        "caveats": [
            "This probe uses a bounded row sample from one PMXT parquet file.",
            "Derived outputs are feasibility artifacts, not empirical PM-DFBA evidence.",
            "True stale-quote race proof requires order lifecycle, cancellation, and taker-hit sequencing.",
            "Maker identity, account-level quote lifecycle, failed cancels, and unobserved orders are not established by this probe.",
        ],
    }


def build_hourly_file_interpretation(
    frame: pd.DataFrame,
    event_counts: pd.DataFrame,
    timestamp_column: str | None = None,
    market_column: str | None = None,
) -> dict[str, Any]:
    timestamp_col = timestamp_column or detect_timestamp_column(frame)
    market_col = market_column or detect_market_column(frame)
    event_type_counts = {
        str(row["event_type"]): int(row["count"]) for _, row in event_counts.iterrows()
    }
    normalized_event_types = {event_type.lower() for event_type in event_type_counts}
    expected_event_types = {
        event_type: event_type in normalized_event_types for event_type in EVENT_TYPES_OF_INTEREST
    }

    timestamp_values = pd.Series(dtype="datetime64[ns, UTC]")
    if timestamp_col is not None and timestamp_col in frame:
        timestamp_values = _parse_timestamps(frame[timestamp_col]).dropna()
    distinct_timestamps = int(timestamp_values.nunique()) if not timestamp_values.empty else 0
    min_timestamp = timestamp_values.min() if not timestamp_values.empty else None
    max_timestamp = timestamp_values.max() if not timestamp_values.empty else None
    span_seconds = (
        float((max_timestamp - min_timestamp).total_seconds())
        if min_timestamp is not None and max_timestamp is not None
        else None
    )
    distinct_markets = (
        int(frame[market_col].astype(str).nunique()) if market_col is not None and market_col in frame else 0
    )
    classification = classify_hourly_file(
        rows_loaded=len(frame),
        distinct_timestamps=distinct_timestamps,
        timestamp_span_seconds=span_seconds,
        event_type_counts=event_type_counts,
        distinct_markets=distinct_markets,
    )
    return {
        "classification": classification,
        "rows_loaded": int(len(frame)),
        "distinct_timestamps": distinct_timestamps,
        "min_timestamp": min_timestamp.isoformat() if min_timestamp is not None else None,
        "max_timestamp": max_timestamp.isoformat() if max_timestamp is not None else None,
        "timestamp_span_seconds": span_seconds,
        "event_type_counts": event_type_counts,
        "expected_event_types_present": expected_event_types,
        "distinct_markets": distinct_markets,
        "criteria": [
            "tick_level_hourly_partition: multiple timestamps plus tick/update event types such as price_change or last_trade_price.",
            "static_hourly_snapshot: very few timestamps and mostly book/snapshot records.",
            "unknown: insufficient timestamp/event diversity or missing schema fields.",
        ],
    }


def classify_hourly_file(
    rows_loaded: int,
    distinct_timestamps: int,
    timestamp_span_seconds: float | None,
    event_type_counts: dict[str, int],
    distinct_markets: int = 0,
) -> str:
    normalized_event_types = {event_type.lower() for event_type in event_type_counts}
    has_tick_events = bool(
        normalized_event_types.intersection({"price_change", "last_trade_price", "tick_size_change"})
    )
    has_multiple_event_types = len([count for count in event_type_counts.values() if count > 0]) > 1
    has_time_movement = distinct_timestamps > 1 and (timestamp_span_seconds or 0.0) > 0.0
    if rows_loaded >= 3 and distinct_timestamps >= 3 and has_time_movement and has_tick_events:
        return "tick_level_hourly_partition"
    if rows_loaded >= 3 and distinct_timestamps >= 2 and has_multiple_event_types and has_tick_events:
        return "tick_level_hourly_partition"
    if distinct_timestamps <= 1 and not has_tick_events:
        return "static_hourly_snapshot"
    if distinct_timestamps <= 2 and normalized_event_types.issubset({"book", "snapshot"}):
        return "static_hourly_snapshot"
    if distinct_markets > 1 and has_tick_events and has_multiple_event_types:
        return "tick_level_hourly_partition"
    return "unknown"


def write_probe_report(path: Path, summary: dict[str, Any]) -> None:
    required = summary["required_event_types"]
    suitability = summary["suitability"]
    hourly = summary["hourly_file_interpretation"]
    url_diagnostics = summary.get("url_diagnostics")
    lines = [
        "# PMXT v2 Feasibility Probe",
        "",
        "This is a bounded feasibility probe, not an empirical PM-DFBA result.",
        "",
        f"- Did the file load? {'Yes' if summary['rows_loaded'] else 'No'} ({summary['rows_loaded']} rows loaded).",
        f"- What columns exist? {', '.join(summary['columns'])}.",
        f"- What event types exist? {', '.join(summary['event_types_present']) or 'None detected'}.",
        f"- Are `book`, `price_change`, `last_trade_price`, and `tick_size_change` present? {required}.",
        f"- Can we identify market IDs? {'Yes' if summary['market_ids_identified'] else 'No'}.",
        f"- Can we identify timestamps? {'Yes' if summary['timestamps_identified'] else 'No'}.",
        f"- Can we reconstruct top-of-book for at least one market? {'Yes' if summary['top_of_book_reconstructed'] else 'No'}.",
        f"- Can we compute depth within 1c, 5c, and 10c of mid? {'Yes' if summary['depth_near_mid_computed'] else 'No'}.",
        f"- Are trade events present? {'Yes' if summary['trade_events_present'] else 'No'}.",
        f"- Are trade direction / taker side fields available? {'Yes' if summary['trade_side_available'] else 'No'}.",
        "",
        "## URL Diagnostics",
        "",
    ]
    if url_diagnostics:
        lines.extend(
            [
                f"- URL: {url_diagnostics.get('url')}",
                f"- Attempted at UTC: {url_diagnostics.get('attempted_at_utc')}",
                f"- HTTP status: {url_diagnostics.get('http_status')}",
                f"- Content type: {url_diagnostics.get('content_type')}",
                f"- Content length: {url_diagnostics.get('content_length')}",
                f"- Accepts ranges: {url_diagnostics.get('accepts_ranges')}",
                f"- Requires API key guess: {url_diagnostics.get('requires_api_key_guess')}",
                f"- Notes: {'; '.join(url_diagnostics.get('notes', []))}",
                "",
            ]
        )
    else:
        lines.extend(["No URL diagnostics were collected for this local-input run.", ""])
    lines.extend(
        [
            "## Hourly file interpretation",
            "",
            f"- Classification: `{hourly['classification']}`.",
            f"- Rows loaded: {hourly['rows_loaded']}.",
            f"- Distinct timestamps: {hourly['distinct_timestamps']}.",
            f"- Min timestamp: {hourly['min_timestamp']}.",
            f"- Max timestamp: {hourly['max_timestamp']}.",
            f"- Timestamp span seconds: {hourly['timestamp_span_seconds']}.",
            f"- Distinct markets: {hourly['distinct_markets']}.",
            f"- Event type counts: {hourly['event_type_counts']}.",
            f"- Expected event types present: {hourly['expected_event_types_present']}.",
            "",
            "A tick-level hourly partition should have many rows, multiple timestamps, and usually multiple event types such as `book`, `price_change`, or `last_trade_price`. A static hourly snapshot would likely have very few timestamps and mostly `book`/snapshot records.",
            "",
        ]
    )
    lines.extend(
        [
        "## Missing For True Latency-Race Proof",
        "",
        "This sample does not establish true maker-cancel-versus-taker-hit race proof. That would require exchange-side order lifecycle sequencing, cancellation timing, taker-hit timing, and maker/order identity or equivalent account-level quote lifecycle fields.",
        "",
        "Observed order-lifecycle-like columns: "
        + (", ".join(summary["order_lifecycle_columns_observed"]) or "none detected"),
        "",
        "## Suitability",
        "",
        f"- Calibration: {_yes_no(suitability['calibration'])}",
        f"- Event-window replay: {_yes_no(suitability['event_window_replay'])}",
        f"- Stale-loss proxy construction: {_yes_no(suitability['stale_loss_proxy_construction'])}",
        f"- True maker/taker latency-race proof: {_yes_no(suitability['true_maker_taker_latency_race_proof'])}",
        "",
        "## What this does and does not prove",
        "",
        "If PMXT contains tick-level book and trade events, it may support event-window replay, depth studies, liquidation exit-curve proxies, and stale-loss proxy construction.",
        "",
        "It still does not prove true maker-cancel-versus-taker-hit latency races unless exchange-side order lifecycle sequencing, cancel timing, taker-hit timing, and maker/order identity are present.",
        "",
        "PMXT may support L2 replay ingredients if top-of-book and depth reconstruct cleanly, but it should not be used to claim true stale-quote races unless order lifecycle and cancellation sequencing are visible.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _columns(frame_or_columns: pd.DataFrame | Iterable[str]) -> list[str]:
    if isinstance(frame_or_columns, pd.DataFrame):
        return list(frame_or_columns.columns)
    return list(frame_or_columns)


def _choose_column(
    columns: Iterable[str],
    exact: Iterable[str] = (),
    contains: Iterable[str] = (),
) -> str | None:
    by_lower = {column.lower(): column for column in columns}
    for candidate in exact:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    for token in contains:
        token_lower = token.lower()
        for column in columns:
            if token_lower in column.lower():
                return column
    return None


def _normalize_event_type(value: Any) -> str:
    return str(value).strip().lower() if not _missing(value) else ""


def _normalize_side(value: Any) -> str | None:
    if _missing(value):
        return None
    side = str(value).strip().lower()
    if side in {"bid", "bids", "buy", "buys", "yes", "yes_bid"}:
        return "bid"
    if side in {"ask", "asks", "sell", "sells", "offer", "offers", "no", "yes_ask"}:
        return "ask"
    return None


def _first_present(data: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data and not _missing(data[key]):
            return data[key]
    return None


def _first_non_missing(*values: Any) -> Any:
    for value in values:
        if not _missing(value):
            return value
    return None


def _jsonish(value: Any) -> Any:
    if _missing(value):
        return None
    if isinstance(value, (dict, list, tuple)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def _extract_book(payload: Any) -> dict[str, list[tuple[float, float]]]:
    payload = _jsonish(payload)
    if not isinstance(payload, dict):
        return {"bids": [], "asks": []}
    if "book" in payload:
        nested = _extract_book(payload["book"])
        if nested["bids"] or nested["asks"]:
            return nested
    bids = _parse_side(_first_present(payload, ("bids", "bid", "buy", "buys")))
    asks = _parse_side(_first_present(payload, ("asks", "ask", "sell", "sells")))
    return {"bids": _sort_bids(bids), "asks": _sort_asks(asks)}


def _extract_price_changes(payload: Any) -> list[tuple[str, float, float]]:
    payload = _jsonish(payload)
    if payload is None:
        return []
    if isinstance(payload, dict):
        if "changes" in payload:
            return _extract_price_changes(payload["changes"])
        updates: list[tuple[str, float, float]] = []
        for side_name, side in (("bids", "bid"), ("bid", "bid"), ("asks", "ask"), ("ask", "ask")):
            for price, size in _parse_side(payload.get(side_name)):
                updates.append((side, price, size))
        direct_side = _normalize_side(payload.get("side") or payload.get("book_side"))
        direct_price = _to_probability(payload.get("price"))
        direct_size = _to_float(
            _first_non_missing(payload.get("size"), payload.get("quantity"), payload.get("qty"))
        )
        if direct_side and direct_price is not None and direct_size is not None:
            updates.append((direct_side, direct_price, direct_size))
        return updates
    if isinstance(payload, (list, tuple)):
        updates = []
        for item in payload:
            if isinstance(item, dict):
                side = _normalize_side(item.get("side") or item.get("book_side"))
                price = _to_probability(item.get("price"))
                size = _to_float(
                    _first_non_missing(item.get("size"), item.get("quantity"), item.get("qty"))
                )
                if side and price is not None and size is not None:
                    updates.append((side, price, size))
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                side = _normalize_side(item[0])
                price = _to_probability(item[1])
                size = _to_float(item[2])
                if side and price is not None and size is not None:
                    updates.append((side, price, size))
        return updates
    return []


def _parse_side(value: Any) -> list[tuple[float, float]]:
    value = _jsonish(value)
    if value is None:
        return []
    levels: list[tuple[float, float]] = []
    if isinstance(value, dict):
        if "price" in value and any(key in value for key in ("size", "quantity", "qty")):
            price = _to_probability(value.get("price"))
            size = _to_float(
                _first_non_missing(value.get("size"), value.get("quantity"), value.get("qty"))
            )
            return [(price, size)] if price is not None and size is not None else []
        for price_raw, size_raw in value.items():
            price = _to_probability(price_raw)
            size = _to_float(size_raw)
            if price is not None and size is not None:
                levels.append((price, size))
        return levels
    if isinstance(value, (list, tuple)):
        for item in value:
            item = _jsonish(item)
            if isinstance(item, dict):
                price = _to_probability(_first_non_missing(item.get("price"), item.get("p")))
                size = _to_float(
                    _first_non_missing(
                        item.get("size"),
                        item.get("quantity"),
                        item.get("qty"),
                        item.get("s"),
                    )
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                price = _to_probability(item[0])
                size = _to_float(item[1])
            else:
                continue
            if price is not None and size is not None:
                levels.append((price, size))
    return levels


def _select_reconstruction_markets(frame: pd.DataFrame, max_markets: int) -> list[str]:
    rows = []
    for market_id, group in frame.groupby("_pmxt_market_id", dropna=False):
        event_types = set(group["_pmxt_event_type"])
        rows.append(
            {
                "market_id": market_id,
                "row_count": len(group),
                "has_book": "book" in event_types,
                "has_price_change": "price_change" in event_types,
            }
        )
    if not rows:
        return []
    markets = pd.DataFrame(rows)
    markets["priority"] = markets["has_book"].astype(int) * 2 + markets["has_price_change"].astype(int)
    markets = markets.sort_values(["priority", "row_count"], ascending=False)
    return markets["market_id"].astype(str).head(max_markets).tolist()


def _parse_timestamps(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        finite = numeric.dropna()
        if finite.empty:
            return pd.to_datetime(values, errors="coerce", utc=True)
        median = float(finite.median())
        unit = "s"
        if median > 1e17:
            unit = "ns"
        elif median > 1e14:
            unit = "us"
        elif median > 1e11:
            unit = "ms"
        return pd.to_datetime(numeric, errors="coerce", unit=unit, utc=True)
    return pd.to_datetime(values, errors="coerce", utc=True)


def _empty_top_of_book() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "market_id",
            "timestamp",
            "source_event_type",
            "best_bid",
            "best_bid_size",
            "best_ask",
            "best_ask_size",
            "midpoint",
            "spread",
        ]
    )


def _empty_depth() -> pd.DataFrame:
    columns = list(_empty_top_of_book().columns)
    for radius in DEPTH_RADII:
        suffix = _radius_suffix(radius)
        columns.extend([f"depth_bid_{suffix}", f"depth_ask_{suffix}", f"depth_total_{suffix}"])
    return pd.DataFrame(columns=columns)


def _radius_suffix(radius: float) -> str:
    return f"{int(round(radius * 100)):d}c"


def _sort_bids(levels: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    return sorted([(float(price), float(size)) for price, size in levels], key=lambda level: level[0], reverse=True)


def _sort_asks(levels: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    return sorted([(float(price), float(size)) for price, size in levels], key=lambda level: level[0])


def _price_key(price: float) -> float:
    return round(float(price), 10)


def _to_probability(value: Any) -> float | None:
    number = _to_float(value)
    if number is None:
        return None
    if abs(number) > 1.5:
        number = number / 100.0
    if math.isnan(number):
        return None
    return float(number)


def _to_float(value: Any) -> float | None:
    if _missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return abs(number)


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (dict, list, tuple)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _merge_response_diagnostics(diagnostics: dict[str, Any], response: Any) -> None:
    headers = getattr(response, "headers", {}) or {}
    status = getattr(response, "status", None) or getattr(response, "code", None)
    diagnostics["http_status"] = int(status) if status is not None else None
    diagnostics["content_type"] = _header_value(headers, "content-type")
    content_length = _header_value(headers, "content-length")
    diagnostics["content_length"] = int(content_length) if content_length and content_length.isdigit() else None
    diagnostics["accepts_ranges"] = _accepts_ranges(headers, status)


def _merge_exception_diagnostics(diagnostics: dict[str, Any], exc: Exception) -> None:
    if isinstance(exc, urllib.error.HTTPError):
        diagnostics["http_status"] = int(exc.code)
        diagnostics["content_type"] = _header_value(exc.headers, "content-type")
        content_length = _header_value(exc.headers, "content-length")
        diagnostics["content_length"] = (
            int(content_length) if content_length and content_length.isdigit() else diagnostics["content_length"]
        )
        diagnostics["accepts_ranges"] = _accepts_ranges(exc.headers, exc.code)


def _header_value(headers: Any, key: str) -> str | None:
    if headers is None:
        return None
    if hasattr(headers, "get"):
        value = headers.get(key) or headers.get(key.title()) or headers.get(key.lower())
        return str(value) if value is not None else None
    return None


def _accepts_ranges(headers: Any, status: int | None) -> bool | str:
    accept_ranges = (_header_value(headers, "accept-ranges") or "").lower()
    content_range = _header_value(headers, "content-range")
    if "bytes" in accept_ranges or content_range or status == 206:
        return True
    if status in {200, 404, 401, 403}:
        return False
    return "unknown"


def _api_key_guess(status: int | None) -> bool | str:
    if status in {401, 403}:
        return True
    if status in {200, 206, 404}:
        return False
    return "unknown"


def _exception_summary(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}: {exc.reason}"
    if isinstance(exc, urllib.error.URLError):
        return f"URL error: {exc.reason}"
    return f"{type(exc).__name__}: {exc}"
