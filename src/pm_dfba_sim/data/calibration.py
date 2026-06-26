from __future__ import annotations

import glob
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_TRADE_INPUTS = (
    "output/cache/trades_enriched_sample.parquet",
    "output/cache/trades_enriched_sample_v2.parquet",
    "data/polymarket/trump_winner_trades.parquet",
    "data/kalshi/trades/*.parquet",
    "data/polymarket/trades/*.parquet",
    "data/polymarket/legacy_trades/*.parquet",
)
DEFAULT_METADATA_INPUTS = (
    "output/cache/market_token_map.parquet",
    "kalshi_audit/data/all_markets.csv",
    "data/kalshi/markets/*.parquet",
    "data/polymarket/markets/*.parquet",
)
DEFAULT_EVENT_INPUTS = ("data/polymarket/trump_jumps.parquet",)
DEFAULT_WINDOWS = ("5min", "15min", "1h")
DEFAULT_THRESHOLDS = (0.05, 0.10, 0.20)


class CalibrationError(ValueError):
    """Raised when input data cannot be normalized for calibration."""


@dataclass(frozen=True)
class CalibrationResult:
    market_summary: pd.DataFrame
    trade_size_summary: pd.DataFrame
    jump_windows: pd.DataFrame
    jump_size_distribution: pd.DataFrame
    interim_jump_size_distribution: pd.DataFrame
    terminal_jump_size_distribution: pd.DataFrame
    simulator_parameter_suggestions: dict[str, Any]
    price_paths: pd.DataFrame
    normalized_trades: pd.DataFrame
    metadata: pd.DataFrame


def run_calibration(
    data_dir: str | Path,
    out_dir: str | Path,
    trade_inputs: Iterable[str] | None = None,
    metadata_inputs: Iterable[str] | None = None,
    event_inputs: Iterable[str] | None = None,
    max_rows_per_file: int = 100_000,
    max_files_per_glob: int = 9,
    windows: Iterable[str] = DEFAULT_WINDOWS,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    rolling_vwap_trades: int = 20,
    near_resolution_window: str = "24h",
) -> CalibrationResult:
    """Run a bounded local-data calibration pass and write derived outputs."""

    data_path = Path(data_dir).expanduser()
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trade_paths = expand_input_paths(
        data_path,
        tuple(trade_inputs or DEFAULT_TRADE_INPUTS),
        max_files_per_glob=max_files_per_glob,
    )
    metadata_paths = expand_input_paths(
        data_path,
        tuple(metadata_inputs or DEFAULT_METADATA_INPUTS),
        max_files_per_glob=max_files_per_glob,
    )
    event_paths = expand_input_paths(
        data_path,
        tuple(event_inputs or DEFAULT_EVENT_INPUTS),
        max_files_per_glob=max_files_per_glob,
    )

    trades = load_normalized_trades(trade_paths, max_rows_per_file=max_rows_per_file)
    metadata = load_normalized_metadata(metadata_paths, max_rows_per_file=max_rows_per_file)
    event_jumps = load_event_jump_labels(event_paths, max_rows_per_file=max_rows_per_file)

    price_paths = build_price_paths(trades, rolling_window=rolling_vwap_trades)
    jump_windows = detect_jump_windows(price_paths, windows=windows, thresholds=thresholds)
    jump_windows = label_near_resolution_jumps(
        jump_windows,
        metadata,
        near_resolution_window=near_resolution_window,
    )
    event_jumps = label_near_resolution_jumps(
        event_jumps,
        metadata,
        near_resolution_window=near_resolution_window,
    )
    jump_size_distribution = build_jump_size_distribution(jump_windows, event_jumps)
    interim_jump_size_distribution = jump_size_distribution[
        ~jump_size_distribution["near_resolution"].fillna(False)
    ].copy()
    terminal_jump_size_distribution = jump_size_distribution[
        jump_size_distribution["near_resolution"].fillna(False)
    ].copy()
    market_summary = build_market_summary(trades, price_paths, jump_windows, metadata)
    trade_size_summary = build_trade_size_summary(trades)
    suggestions = build_simulator_parameter_suggestions(
        trades=trades,
        price_paths=price_paths,
        jump_size_distribution=jump_size_distribution,
        interim_jump_size_distribution=interim_jump_size_distribution,
        terminal_jump_size_distribution=terminal_jump_size_distribution,
        metadata=metadata,
        trade_paths=trade_paths,
        metadata_paths=metadata_paths,
        event_paths=event_paths,
        max_rows_per_file=max_rows_per_file,
        near_resolution_window=near_resolution_window,
    )

    market_summary.to_csv(output_path / "market_summary.csv", index=False)
    trade_size_summary.to_csv(output_path / "trade_size_summary.csv", index=False)
    jump_windows.to_csv(output_path / "jump_windows.csv", index=False)
    jump_size_distribution.to_csv(output_path / "jump_size_distribution.csv", index=False)
    interim_jump_size_distribution.to_csv(
        output_path / "interim_jump_size_distribution.csv",
        index=False,
    )
    terminal_jump_size_distribution.to_csv(
        output_path / "terminal_jump_size_distribution.csv",
        index=False,
    )
    with (output_path / "simulator_parameter_suggestions.json").open("w") as f:
        json.dump(suggestions, f, indent=2, default=_json_default)

    write_calibration_plots(
        out_dir=output_path,
        price_paths=price_paths,
        jump_size_distribution=jump_size_distribution,
        trades=trades,
    )

    return CalibrationResult(
        market_summary=market_summary,
        trade_size_summary=trade_size_summary,
        jump_windows=jump_windows,
        jump_size_distribution=jump_size_distribution,
        interim_jump_size_distribution=interim_jump_size_distribution,
        terminal_jump_size_distribution=terminal_jump_size_distribution,
        simulator_parameter_suggestions=suggestions,
        price_paths=price_paths,
        normalized_trades=trades,
        metadata=metadata,
    )


def expand_input_paths(
    data_dir: Path,
    patterns: Iterable[str],
    max_files_per_glob: int = 9,
) -> list[Path]:
    """Resolve explicit paths and globs, sampling large glob expansions."""

    resolved: list[Path] = []
    for pattern in patterns:
        pattern_path = Path(pattern).expanduser()
        search_pattern = str(pattern_path if pattern_path.is_absolute() else data_dir / pattern)
        has_glob = any(char in search_pattern for char in "*?[]")
        matches = [Path(path) for path in glob.glob(search_pattern)]
        matches = [path for path in matches if path.is_file()]
        if not has_glob:
            candidate = Path(search_pattern)
            if candidate.exists() and candidate.is_file():
                matches = [candidate]
        if has_glob:
            matches = select_representative_paths(matches, max_files=max_files_per_glob)
        resolved.extend(matches)

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in resolved:
        canonical = path.resolve()
        if canonical not in seen:
            seen.add(canonical)
            unique.append(path)
    return unique


def select_representative_paths(paths: Iterable[Path], max_files: int = 9) -> list[Path]:
    ordered = sorted(paths, key=_natural_key)
    if len(ordered) <= max_files:
        return ordered
    first_n = max_files // 3
    middle_n = max_files // 3
    last_n = max_files - first_n - middle_n
    middle_start = max(0, len(ordered) // 2 - middle_n // 2)
    selected = (
        ordered[:first_n]
        + ordered[middle_start : middle_start + middle_n]
        + ordered[-last_n:]
    )
    return sorted(set(selected), key=_natural_key)


def load_normalized_trades(paths: Iterable[Path], max_rows_per_file: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for path in paths:
        try:
            frame = read_table_sample(path, max_rows=max_rows_per_file)
            if frame.empty:
                continue
            frames.append(normalize_trade_frame(frame, source_path=path))
        except CalibrationError as exc:
            errors.append(f"{path}: {exc}")

    if not frames:
        message = "No trade rows could be normalized."
        if errors:
            message += " Errors: " + "; ".join(errors[:5])
        raise CalibrationError(message)

    trades = pd.concat(frames, ignore_index=True)
    raw_normalized_rows = len(trades)
    trades = trades.dropna(subset=["timestamp", "yes_price"])
    trades = trades[(trades["yes_price"] >= 0.0) & (trades["yes_price"] <= 1.0)]
    trades = deduplicate_normalized_trades(trades)
    trades.attrs["raw_normalized_rows"] = raw_normalized_rows
    trades.attrs["deduplicated_rows_removed"] = raw_normalized_rows - len(trades)
    trades = trades.sort_values(["market_id", "timestamp"]).reset_index(drop=True)
    return trades


def normalize_trade_frame(frame: pd.DataFrame, source_path: str | Path = "<memory>") -> pd.DataFrame:
    columns = list(frame.columns)
    timestamp_col = choose_column(
        columns,
        exact=("trade_ts", "created_time", "timestamp", "time", "datetime", "minute", "date"),
        contains=("timestamp", "created", "trade_time"),
    )
    if timestamp_col is None:
        raise CalibrationError(
            f"Missing timestamp column in {source_path}; expected one of trade_ts, "
            "created_time, timestamp, time, datetime, minute, or date."
        )

    yes_price, raw_price_col, price_assumption, orientation_verified = construct_yes_equivalent_price(
        frame
    )
    if yes_price is None:
        raise CalibrationError(
            f"Missing price column in {source_path}; expected yes_price/no_price, "
            "price, trade_price, exec_price, yes_vwap, last_price, or bid/ask proxy."
        )

    market_col = choose_market_identifier_column(columns)
    size_col = choose_column(
        columns,
        exact=("size", "count", "quantity", "shares_amount", "shares", "amount"),
        contains=("quantity", "shares", "size"),
    )
    side_col = choose_column(columns, exact=("side", "taker_side", "direction", "outcome"))
    notional_col = choose_column(
        columns,
        exact=("notional", "usdc_amount", "volume_usd", "amount_usd"),
        contains=("notional", "usdc"),
    )
    trade_id_col = choose_column(
        columns,
        exact=("trade_id", "tx_hash", "transaction_hash", "signature", "hash"),
    )

    timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce", utc=True)
    size = (
        pd.to_numeric(frame[size_col], errors="coerce").abs()
        if size_col
        else pd.Series(1.0, index=frame.index)
    )
    notional = (
        pd.to_numeric(frame[notional_col], errors="coerce").abs()
        if notional_col
        else yes_price * size
    )
    market_id = (
        frame[market_col].astype(str)
        if market_col
        else pd.Series(f"__source__:{Path(str(source_path)).stem}", index=frame.index)
    )

    out = pd.DataFrame(
        {
            "source_path": str(source_path),
            "market_id": market_id,
            "timestamp": timestamps,
            "yes_price": normalize_probability_price(yes_price),
            "size": size,
            "side": frame[side_col].astype(str) if side_col else pd.NA,
            "notional": notional,
            "raw_price_column": raw_price_col,
            "price_assumption": price_assumption,
            "orientation_verified": orientation_verified,
            "trade_id": frame[trade_id_col].astype(str) if trade_id_col else pd.NA,
        }
    )
    midpoint = construct_midpoint_proxy(frame)
    if midpoint is not None:
        out["midpoint_proxy"] = midpoint
    return out.dropna(subset=["timestamp", "yes_price"])


def deduplicate_normalized_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate with stable trade IDs when present, else a conservative trade tuple."""

    if trades.empty:
        return trades
    with_ids = trades[
        trades.get("trade_id", pd.Series(pd.NA, index=trades.index)).notna()
        & (trades.get("trade_id", pd.Series("", index=trades.index)).astype(str) != "<NA>")
    ]
    without_ids = trades.drop(with_ids.index)
    deduped_parts: list[pd.DataFrame] = []
    if not with_ids.empty:
        deduped_parts.append(with_ids.drop_duplicates(subset=["source_path", "trade_id"]))
    if not without_ids.empty:
        deduped_parts.append(
            without_ids.drop_duplicates(subset=["market_id", "timestamp", "yes_price", "size"])
        )
    if not deduped_parts:
        return trades.iloc[0:0].copy()
    return pd.concat(deduped_parts, ignore_index=True)


def construct_yes_equivalent_price(
    frame: pd.DataFrame,
) -> tuple[pd.Series | None, str | None, str, bool]:
    columns = list(frame.columns)
    yes_col = choose_column(columns, exact=("yes_price", "yes_vwap", "yes_trade_price"))
    no_col = choose_column(columns, exact=("no_price", "no_trade_price"))
    if yes_col is not None:
        return (
            pd.to_numeric(frame[yes_col], errors="coerce"),
            yes_col,
            "YES-equivalent price from YES price column.",
            True,
        )
    if no_col is not None:
        no_price = normalize_probability_price(pd.to_numeric(frame[no_col], errors="coerce"))
        return 1.0 - no_price, no_col, "YES-equivalent price inferred as 1 - NO price.", True

    single_col = choose_column(
        columns,
        exact=("price", "trade_price", "exec_price", "last_price"),
        contains=("price",),
    )
    if single_col == "final_outcome_price":
        single_col = None
    if single_col is not None:
        return (
            pd.to_numeric(frame[single_col], errors="coerce"),
            single_col,
            "Single observed price used as YES-equivalent proxy; outcome orientation is unverified.",
            False,
        )

    midpoint = construct_midpoint_proxy(frame)
    if midpoint is not None:
        return (
            midpoint,
            "bid/ask midpoint",
            "Bid/ask midpoint used as YES-equivalent proxy; no trade price was present.",
            False,
        )
    return None, None, "", False


def construct_midpoint_proxy(frame: pd.DataFrame) -> pd.Series | None:
    columns = list(frame.columns)
    bid_col = choose_column(columns, exact=("yes_bid", "bid", "best_bid"))
    ask_col = choose_column(columns, exact=("yes_ask", "ask", "best_ask"))
    if bid_col is None or ask_col is None:
        return None
    bid = normalize_probability_price(pd.to_numeric(frame[bid_col], errors="coerce"))
    ask = normalize_probability_price(pd.to_numeric(frame[ask_col], errors="coerce"))
    return (bid + ask) / 2.0


def load_normalized_metadata(paths: Iterable[Path], max_rows_per_file: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = read_table_sample(path, max_rows=max_rows_per_file)
            normalized = normalize_metadata_frame(frame, source_path=path)
            if not normalized.empty:
                frames.append(normalized)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["market_id", "category", "close_time", "metadata_source"])
    metadata = pd.concat(frames, ignore_index=True)
    metadata = metadata.drop_duplicates(subset=["market_id"], keep="first")
    return metadata


def normalize_metadata_frame(frame: pd.DataFrame, source_path: str | Path = "<memory>") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    columns = list(frame.columns)
    market_col = choose_market_identifier_column(columns)
    if market_col is None:
        return pd.DataFrame()
    category_col = choose_column(
        columns,
        exact=("category", "series_ticker", "event_ticker", "market_type"),
        contains=("category",),
    )
    title_col = choose_column(columns, exact=("title", "question", "market_title", "event_title"))
    status_col = choose_column(columns, exact=("status", "market_status", "active", "closed"))
    result_col = choose_column(columns, exact=("result", "outcome", "outcome_label"))
    close_col = choose_column(
        columns,
        exact=(
            "close_time",
            "resolution_time",
            "resolved_time",
            "end_date",
            "expiration_time",
            "expiry",
            "expiration",
        ),
    )
    open_col = choose_column(columns, exact=("open_time", "created_time", "created_at"))

    out = pd.DataFrame(
        {
            "market_id": frame[market_col].astype(str),
            "category": frame[category_col].astype(str) if category_col else pd.NA,
            "title": frame[title_col].astype(str) if title_col else pd.NA,
            "status": frame[status_col].astype(str) if status_col else pd.NA,
            "result": frame[result_col].astype(str) if result_col else pd.NA,
            "close_time": parse_timestamp_column(frame[close_col]) if close_col else pd.NaT,
            "open_time": parse_timestamp_column(frame[open_col]) if open_col else pd.NaT,
            "metadata_source": str(source_path),
        }
    )
    midpoint = construct_midpoint_proxy(frame)
    if midpoint is not None:
        out["metadata_midpoint_proxy"] = midpoint
    return out


def load_event_jump_labels(paths: Iterable[Path], max_rows_per_file: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = read_table_sample(path, max_rows=max_rows_per_file)
            normalized = normalize_event_jump_frame(frame, source_path=path)
            if not normalized.empty:
                frames.append(normalized)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(
            columns=["market_id", "timestamp", "jump_size", "direction", "source", "threshold_5c", "threshold_10c", "threshold_20c"]
        )
    return pd.concat(frames, ignore_index=True)


def normalize_event_jump_frame(frame: pd.DataFrame, source_path: str | Path = "<memory>") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    columns = list(frame.columns)
    timestamp_col = choose_column(
        columns,
        exact=("timestamp", "minute", "time", "datetime", "date"),
        contains=("timestamp",),
    )
    market_col = choose_market_identifier_column(columns)
    direction_col = choose_column(columns, exact=("direction", "side"))
    notional_col = choose_column(columns, exact=("notional", "pre_notional_5m", "post_notional_5m"))

    if "dp_price" in frame:
        move = pd.to_numeric(frame["dp_price"], errors="coerce")
    elif {"pre_price", "post_price"}.issubset(frame.columns):
        move = pd.to_numeric(frame["post_price"], errors="coerce") - pd.to_numeric(
            frame["pre_price"], errors="coerce"
        )
    else:
        return pd.DataFrame()

    timestamp = (
        parse_timestamp_column(frame[timestamp_col])
        if timestamp_col
        else pd.Series(pd.NaT, index=frame.index)
    )
    market_id = (
        frame[market_col].astype(str)
        if market_col
        else pd.Series(f"__event_source__:{Path(str(source_path)).stem}", index=frame.index)
    )
    direction = np.where(move >= 0, "up", "down")
    if direction_col:
        direction = frame[direction_col].astype(str)

    out = pd.DataFrame(
        {
            "market_id": market_id,
            "timestamp": timestamp,
            "window": "event_label",
            "jump_size": move.abs(),
            "direction": direction,
            "threshold_5c": move.abs() >= 0.05,
            "threshold_10c": move.abs() >= 0.10,
            "threshold_20c": move.abs() >= 0.20,
            "max_threshold_met": move.abs().map(_max_threshold_met),
            "orientation_verified": True,
            "source": str(source_path),
            "notional": pd.to_numeric(frame[notional_col], errors="coerce") if notional_col else np.nan,
        }
    )
    return out.dropna(subset=["jump_size"])


def label_near_resolution_jumps(
    jumps: pd.DataFrame,
    metadata: pd.DataFrame,
    near_resolution_window: str = "24h",
) -> pd.DataFrame:
    if jumps.empty:
        labeled = jumps.copy()
        if "near_resolution" not in labeled:
            labeled["near_resolution"] = False
        return labeled

    labeled = jumps.copy()
    labeled["near_resolution_window"] = near_resolution_window
    labeled["near_resolution"] = False
    labeled["market_close_time"] = pd.NaT
    if metadata.empty or "close_time" not in metadata or "market_id" not in metadata:
        return labeled

    close_times = metadata[["market_id", "close_time"]].dropna(subset=["close_time"]).copy()
    if close_times.empty:
        return labeled
    close_times["market_id"] = close_times["market_id"].astype(str)
    close_times["market_close_time"] = pd.to_datetime(
        close_times["close_time"],
        errors="coerce",
        utc=True,
    )
    close_times = close_times.dropna(subset=["market_close_time"]).drop_duplicates(
        subset=["market_id"],
        keep="first",
    )
    labeled = labeled.drop(columns=["market_close_time"])
    labeled["market_id"] = labeled["market_id"].astype(str)
    labeled = labeled.merge(
        close_times[["market_id", "market_close_time"]],
        on="market_id",
        how="left",
    )

    window = pd.to_timedelta(near_resolution_window)
    timestamps = pd.to_datetime(labeled["timestamp"], errors="coerce", utc=True)
    close = pd.to_datetime(labeled["market_close_time"], errors="coerce", utc=True)
    labeled["near_resolution"] = (close.notna()) & (timestamps >= (close - window))
    return labeled


def build_price_paths(trades: pd.DataFrame, rolling_window: int = 20) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    required = {"market_id", "timestamp", "yes_price", "size"}
    missing = required - set(trades.columns)
    if missing:
        raise CalibrationError(f"Cannot build price paths; missing columns: {sorted(missing)}")

    paths = trades.sort_values(["market_id", "timestamp"]).copy()
    paths["last_trade_price"] = paths["yes_price"]
    paths["size_for_vwap"] = paths["size"].fillna(1.0)
    paths["price_times_size"] = paths["yes_price"] * paths["size"].fillna(1.0)
    denominator = paths.groupby("market_id")["size_for_vwap"].transform(
        lambda series: series.rolling(rolling_window, min_periods=1).sum()
    )
    numerator = paths.groupby("market_id")["price_times_size"].transform(
        lambda series: series.rolling(rolling_window, min_periods=1).sum()
    )
    paths["rolling_vwap"] = numerator / denominator.replace(0, np.nan)
    keep = [
        "source_path",
        "market_id",
        "timestamp",
        "last_trade_price",
        "rolling_vwap",
        "size",
        "notional",
        "side",
        "price_assumption",
        "orientation_verified",
        "trade_id",
    ]
    if "midpoint_proxy" in paths:
        keep.append("midpoint_proxy")
    return paths[keep].reset_index(drop=True)


def detect_jump_windows(
    price_paths: pd.DataFrame,
    windows: Iterable[str] = DEFAULT_WINDOWS,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    if price_paths.empty:
        return pd.DataFrame(
            columns=[
                "market_id",
                "timestamp",
                "window",
                "max_threshold_met",
                "jump_size",
                "direction",
                "price_before",
                "price_after",
                "threshold_5c",
                "threshold_10c",
                "threshold_20c",
                "orientation_verified",
            ]
        )
    required = {"market_id", "timestamp", "last_trade_price"}
    missing = required - set(price_paths.columns)
    if missing:
        raise CalibrationError(f"Cannot detect jumps; missing columns: {sorted(missing)}")

    threshold_values = tuple(float(threshold) for threshold in thresholds)
    min_threshold = min(threshold_values)
    rows: list[dict[str, Any]] = []

    for market_id, group in price_paths.groupby("market_id", sort=False):
        series_columns = ["timestamp", "last_trade_price"]
        if "orientation_verified" in group:
            series_columns.append("orientation_verified")
        series = (
            group[series_columns]
            .dropna()
            .sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
        )
        if len(series) < 2:
            continue
        series = series.set_index("timestamp")
        prices = pd.to_numeric(series["last_trade_price"], errors="coerce")
        for window in windows:
            shifted = prices.shift(1)
            prior_min = shifted.rolling(window, min_periods=1).min()
            prior_max = shifted.rolling(window, min_periods=1).max()
            up_move = prices - prior_min
            down_move = prior_max - prices
            jump_size = pd.concat([up_move, down_move], axis=1).max(axis=1)
            direction = np.where(up_move >= down_move, "up", "down")
            price_before = np.where(up_move >= down_move, prior_min, prior_max)
            hits = jump_size[jump_size >= min_threshold].dropna()
            for timestamp, size in hits.items():
                idx = prices.index.get_loc(timestamp)
                thresholds_met = [threshold for threshold in threshold_values if size >= threshold]
                row = {
                    "market_id": market_id,
                    "timestamp": timestamp,
                    "window": window,
                    "max_threshold_met": max(thresholds_met),
                    "jump_size": float(size),
                    "direction": str(direction[idx]),
                    "price_before": float(price_before[idx]),
                    "price_after": float(prices.loc[timestamp]),
                    "threshold_5c": bool(size >= 0.05),
                    "threshold_10c": bool(size >= 0.10),
                    "threshold_20c": bool(size >= 0.20),
                }
                if "orientation_verified" in series:
                    row["orientation_verified"] = bool(series.loc[timestamp, "orientation_verified"])
                else:
                    row["orientation_verified"] = True
                rows.append(row)
    return pd.DataFrame(rows)


def build_jump_size_distribution(
    jump_windows: pd.DataFrame,
    event_jumps: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not jump_windows.empty:
        detected = (
            jump_windows.sort_values(["market_id", "timestamp", "window", "jump_size"])
            .drop_duplicates(subset=["market_id", "timestamp", "window"])
            .copy()
        )
        detected["source"] = "trade_path_detector"
        detected["threshold_5c"] = detected["jump_size"] >= 0.05
        detected["threshold_10c"] = detected["jump_size"] >= 0.10
        detected["threshold_20c"] = detected["jump_size"] >= 0.20
        if "orientation_verified" not in detected:
            detected["orientation_verified"] = False
        if "near_resolution" not in detected:
            detected["near_resolution"] = False
        if "max_threshold_met" not in detected:
            detected["max_threshold_met"] = detected["jump_size"].map(_max_threshold_met)
        frames.append(
            detected[
                [
                    "market_id",
                    "timestamp",
                    "window",
                    "max_threshold_met",
                    "jump_size",
                    "direction",
                    "threshold_5c",
                    "threshold_10c",
                    "threshold_20c",
                    "near_resolution",
                    "orientation_verified",
                    "source",
                ]
            ]
        )
    if event_jumps is not None and not event_jumps.empty:
        event_jumps = event_jumps.copy()
        if "near_resolution" not in event_jumps:
            event_jumps["near_resolution"] = False
        if "orientation_verified" not in event_jumps:
            event_jumps["orientation_verified"] = True
        if "max_threshold_met" not in event_jumps:
            event_jumps["max_threshold_met"] = event_jumps["jump_size"].map(_max_threshold_met)
        frames.append(
            event_jumps[
                [
                    "market_id",
                    "timestamp",
                    "window",
                    "max_threshold_met",
                    "jump_size",
                    "direction",
                    "threshold_5c",
                    "threshold_10c",
                    "threshold_20c",
                    "near_resolution",
                    "orientation_verified",
                    "source",
                ]
            ]
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                "market_id",
                "timestamp",
                "window",
                "max_threshold_met",
                "jump_size",
                "direction",
                "threshold_5c",
                "threshold_10c",
                "threshold_20c",
                "near_resolution",
                "orientation_verified",
                "source",
            ]
        )
    return pd.concat(frames, ignore_index=True).sort_values("jump_size", ascending=False)


def build_market_summary(
    trades: pd.DataFrame,
    price_paths: pd.DataFrame,
    jump_windows: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    grouped = trades.groupby("market_id", dropna=False)
    summary = grouped.agg(
        trade_count=("yes_price", "count"),
        first_trade_timestamp=("timestamp", "min"),
        last_trade_timestamp=("timestamp", "max"),
        initial_price=("yes_price", "first"),
        last_price=("yes_price", "last"),
        mean_trade_price=("yes_price", "mean"),
        total_size=("size", "sum"),
        total_notional=("notional", "sum"),
        median_trade_size=("size", "median"),
    ).reset_index()

    if not jump_windows.empty:
        jump_counts = (
            jump_windows.groupby("market_id")
            .agg(jump_window_hits=("jump_size", "count"), max_jump_size=("jump_size", "max"))
            .reset_index()
        )
        summary = summary.merge(jump_counts, on="market_id", how="left")
    else:
        summary["jump_window_hits"] = 0
        summary["max_jump_size"] = 0.0
    summary["jump_window_hits"] = summary["jump_window_hits"].fillna(0).astype(int)
    summary["max_jump_size"] = summary["max_jump_size"].fillna(0.0)

    if not metadata.empty:
        summary = summary.merge(metadata, on="market_id", how="left")
        if "close_time" in summary:
            summary["time_to_resolution_days"] = (
                pd.to_datetime(summary["close_time"], errors="coerce", utc=True)
                - pd.to_datetime(summary["last_trade_timestamp"], errors="coerce", utc=True)
            ).dt.total_seconds() / 86_400.0
    else:
        summary["category"] = pd.NA
        summary["time_to_resolution_days"] = np.nan

    return summary.sort_values("trade_count", ascending=False).reset_index(drop=True)


def build_trade_size_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in ("size", "notional"):
        values = pd.to_numeric(trades[column], errors="coerce").dropna()
        if values.empty:
            continue
        quantiles = values.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        rows.append(
            {
                "metric": column,
                "count": int(values.count()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "p01": float(quantiles.loc[0.01]),
                "p05": float(quantiles.loc[0.05]),
                "p10": float(quantiles.loc[0.10]),
                "p25": float(quantiles.loc[0.25]),
                "p50": float(quantiles.loc[0.50]),
                "p75": float(quantiles.loc[0.75]),
                "p90": float(quantiles.loc[0.90]),
                "p95": float(quantiles.loc[0.95]),
                "p99": float(quantiles.loc[0.99]),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def build_simulator_parameter_suggestions(
    trades: pd.DataFrame,
    price_paths: pd.DataFrame,
    jump_size_distribution: pd.DataFrame,
    interim_jump_size_distribution: pd.DataFrame,
    terminal_jump_size_distribution: pd.DataFrame,
    metadata: pd.DataFrame,
    trade_paths: Iterable[Path],
    metadata_paths: Iterable[Path],
    event_paths: Iterable[Path],
    max_rows_per_file: int,
    near_resolution_window: str,
) -> dict[str, Any]:
    verified_trades = trades[_verified_orientation_mask(trades)].copy()
    unverified_count = int(len(trades) - len(verified_trades))
    verified_count = int(len(verified_trades))

    verified_interim_jumps = interim_jump_size_distribution[
        _verified_orientation_mask(interim_jump_size_distribution)
    ].copy()
    if not verified_interim_jumps.empty and "direction" in verified_interim_jumps:
        direction = verified_interim_jumps["direction"].astype(str).str.lower()
        adverse_proxy = float(direction.isin({"down", "-1", "sell", "no"}).mean())
    else:
        adverse_proxy = None

    category_distribution: dict[str, int] = {}
    if not metadata.empty and "category" in metadata:
        category_distribution = (
            metadata["category"].dropna().astype(str).value_counts().head(25).to_dict()
        )

    return {
        "inputs": {
            "trade_files": [str(path) for path in trade_paths],
            "metadata_files": [str(path) for path in metadata_paths],
            "event_files": [str(path) for path in event_paths],
            "max_rows_per_file": max_rows_per_file,
            "near_resolution_window": near_resolution_window,
            "raw_normalized_trade_rows": int(trades.attrs.get("raw_normalized_rows", len(trades))),
            "deduplicated_rows_removed": int(
                trades.attrs.get("deduplicated_rows_removed", 0)
            ),
            "normalized_trade_rows": int(len(trades)),
            "verified_orientation_trade_count": verified_count,
            "unverified_orientation_trade_count": unverified_count,
            "price_path_rows": int(len(price_paths)),
            "detected_jump_rows": int(len(jump_size_distribution)),
            "interim_jump_rows": int(len(interim_jump_size_distribution)),
            "terminal_jump_rows": int(len(terminal_jump_size_distribution)),
        },
        "unverified_orientation_share": (
            None if len(trades) == 0 else float(unverified_count / len(trades))
        ),
        "verified_orientation_trade_count": verified_count,
        "unverified_orientation_trade_count": unverified_count,
        "all_row_suggestions": _trade_based_suggestion_block(trades),
        "verified_orientation_only_suggestions": _trade_based_suggestion_block(verified_trades),
        "initial_price_candidates": _quantile_dict(
            verified_trades["yes_price"] if not verified_trades.empty else pd.Series(dtype=float),
            [0.10, 0.25, 0.50, 0.75, 0.90],
        ),
        "jump_size_interim_candidates": _jump_candidate_block(interim_jump_size_distribution),
        "jump_size_unfiltered_candidates": _jump_candidate_block(jump_size_distribution),
        "terminal_jump_size_candidates": _jump_candidate_block(terminal_jump_size_distribution),
        "jump_candidate_usage_note": (
            "Use jump_size_interim_candidates for public interim jump simulation. "
            "Unfiltered and terminal candidates include near-resolution/final-settlement moves "
            "and should not be used as generic public-jump max parameters."
        ),
        "adverse_jump_probability_proxy": adverse_proxy,
        "taker_trade_size_quantiles": _quantile_dict(
            trades["size"],
            [0.50, 0.75, 0.90, 0.95, 0.99],
        ),
        "liquidation_size_status": "unknown_from_trades_alone",
        "category_distribution": category_distribution,
        "public_private_jump_share": {
            "public_jump_share": None,
            "private_jump_share": None,
            "note": "Unspecified unless event labels distinguish public news from private/informed jumps.",
        },
        "data_limitations": [
            "Calibration uses trades, market metadata, and coarse event labels only.",
            "Cannot prove true stale-quote races without order add/cancel/modify/fill sequencing.",
            "Liquidation exit curves from trades are proxies, not displayed-book executable curves.",
            "Trade-size quantiles are taker-trade observations and are not liquidation-size estimates.",
            "Orientation-unverified rows are separated from verified-orientation suggestions.",
            "Near-resolution moves are labeled separately from interim jumps.",
            "Large parquet glob inputs are represented by first/middle/last samples unless explicit files are provided.",
        ],
    }


def _trade_based_suggestion_block(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        empty = pd.Series(dtype=float)
        return {
            "initial_price_candidates": _quantile_dict(empty, [0.10, 0.25, 0.50, 0.75, 0.90]),
            "taker_trade_size_quantiles": _quantile_dict(empty, [0.50, 0.75, 0.90, 0.95, 0.99]),
            "notional_volume_distribution": _quantile_dict(empty, [0.50, 0.75, 0.90, 0.95, 0.99]),
            "market_activity_distribution": _quantile_dict(
                empty,
                [0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
            ),
        }
    return {
        "initial_price_candidates": _quantile_dict(
            trades["yes_price"],
            [0.10, 0.25, 0.50, 0.75, 0.90],
        ),
        "taker_trade_size_quantiles": _quantile_dict(
            trades["size"],
            [0.50, 0.75, 0.90, 0.95, 0.99],
        ),
        "notional_volume_distribution": _quantile_dict(
            trades["notional"],
            [0.50, 0.75, 0.90, 0.95, 0.99],
        ),
        "market_activity_distribution": _quantile_dict(
            trades.groupby("market_id")["yes_price"].count(),
            [0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
        ),
    }


def _jump_candidate_block(jumps: pd.DataFrame) -> dict[str, float | None]:
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    if "jump_size" not in jumps:
        return _quantile_dict(pd.Series(dtype=float), quantiles)
    return _quantile_dict(
        pd.to_numeric(jumps["jump_size"], errors="coerce").dropna(),
        quantiles,
    )


def _verified_orientation_mask(frame: pd.DataFrame) -> pd.Series:
    if "orientation_verified" not in frame:
        return pd.Series(False, index=frame.index)
    return frame["orientation_verified"].fillna(False).astype(bool)


def write_calibration_plots(
    out_dir: Path,
    price_paths: pd.DataFrame,
    jump_size_distribution: pd.DataFrame,
    trades: pd.DataFrame,
) -> None:
    import os
    import tempfile

    mpl_cache = Path(tempfile.gettempdir()) / "pm-dfba-matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _plot_price_paths(price_paths, out_dir / "price_paths_sample.png", plt)
    _plot_histogram(
        pd.to_numeric(jump_size_distribution.get("jump_size"), errors="coerce").dropna(),
        out_dir / "jump_size_distribution.png",
        plt,
        title="Detected jump-size distribution",
        xlabel="Absolute price move",
    )
    _plot_histogram(
        pd.to_numeric(trades["size"], errors="coerce").dropna(),
        out_dir / "trade_size_distribution.png",
        plt,
        title="Trade-size distribution",
        xlabel="Trade size / count",
        log_x=True,
    )


def read_table_sample(path: Path, max_rows: int) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=max_rows, low_memory=False)
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise CalibrationError(
                "Reading parquet calibration inputs requires pyarrow. "
                "Install with `python3 -m pip install pyarrow`."
            ) from exc
        parquet_file = pq.ParquetFile(path)
        batches = parquet_file.iter_batches(batch_size=max_rows)
        try:
            return next(batches).to_pandas()
        except StopIteration:
            return pd.DataFrame()
    raise CalibrationError(f"Unsupported calibration input extension for {path}")


def choose_column(
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


def choose_market_identifier_column(columns: Iterable[str]) -> str | None:
    exact = (
        "market_id",
        "market_ticker",
        "ticker",
        "condition_id",
        "conditionId",
        "event_ticker",
        "market",
        "slug",
        "clob_token_id",
        "outcome_token_id",
        "token",
        "asset_id",
        "contract",
        "id",
    )
    return choose_column(columns, exact=exact, contains=("market_id", "condition"))


def normalize_probability_price(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    cent_mask = numeric.abs() > 1.5
    numeric = numeric.where(~cent_mask, numeric / 100.0)
    return numeric.clip(lower=0.0, upper=1.0)


def parse_timestamp_column(values: pd.Series) -> pd.Series:
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


def _plot_price_paths(price_paths: pd.DataFrame, path: Path, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    if not price_paths.empty:
        top_markets = (
            price_paths.groupby("market_id")["timestamp"]
            .count()
            .sort_values(ascending=False)
            .head(5)
            .index
        )
        for market_id in top_markets:
            group = price_paths[price_paths["market_id"] == market_id].sort_values("timestamp")
            if len(group) > 500:
                group = group.iloc[np.linspace(0, len(group) - 1, 500).astype(int)]
            label = str(market_id)
            if len(label) > 32:
                label = label[:29] + "..."
            ax.plot(group["timestamp"], group["rolling_vwap"], label=label, linewidth=1.2)
    ax.set_title("Sample rolling VWAP price paths")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("YES-equivalent price")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_histogram(
    values: pd.Series,
    path: Path,
    plt: Any,
    title: str,
    xlabel: str,
    log_x: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if not values.empty:
        plot_values = values[values > 0] if log_x else values
        if not plot_values.empty:
            ax.hist(plot_values, bins=50, alpha=0.85)
            if log_x:
                ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _quantile_dict(values: pd.Series, quantiles: Iterable[float]) -> dict[str, float | None]:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return {f"p{int(q * 100):02d}": None for q in quantiles}
    result = numeric.quantile(list(quantiles))
    return {f"p{int(q * 100):02d}": float(result.loc[q]) for q in quantiles}


def _max_threshold_met(value: float) -> float | None:
    if pd.isna(value):
        return None
    if value >= 0.20:
        return 0.20
    if value >= 0.10:
        return 0.10
    if value >= 0.05:
        return 0.05
    return None


def _natural_key(path: Path) -> list[Any]:
    text = str(path).lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)
