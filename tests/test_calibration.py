import json

import pandas as pd
import pytest

from pm_dfba_sim.data.calibration import (
    CalibrationError,
    detect_jump_windows,
    normalize_trade_frame,
    run_calibration,
)


def test_toy_trade_data_creates_yes_equivalent_prices():
    yes_frame = pd.DataFrame(
        {
            "ticker": ["MKT-YES"],
            "created_time": ["2026-01-01T00:00:00Z"],
            "yes_price": [62],
            "count": [10],
        }
    )
    no_frame = pd.DataFrame(
        {
            "ticker": ["MKT-NO"],
            "created_time": ["2026-01-01T00:00:00Z"],
            "no_price": [37],
            "count": [10],
        }
    )
    generic_frame = pd.DataFrame(
        {
            "market_id": ["MKT-GENERIC"],
            "timestamp": ["2026-01-01T00:00:00Z"],
            "price": [0.44],
            "size": [5],
        }
    )

    yes = normalize_trade_frame(yes_frame)
    no = normalize_trade_frame(no_frame)
    generic = normalize_trade_frame(generic_frame)

    assert yes["yes_price"].iloc[0] == pytest.approx(0.62)
    assert no["yes_price"].iloc[0] == pytest.approx(0.63)
    assert generic["yes_price"].iloc[0] == pytest.approx(0.44)
    assert "orientation is unverified" in generic["price_assumption"].iloc[0]


def test_jump_detector_finds_5c_10c_20c_jumps():
    price_paths = pd.DataFrame(
        {
            "market_id": ["MKT"] * 4,
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:01:00Z",
                    "2026-01-01T00:02:00Z",
                    "2026-01-01T00:03:00Z",
                ]
            ),
            "last_trade_price": [0.50, 0.56, 0.67, 0.88],
        }
    )

    jumps = detect_jump_windows(price_paths, windows=("5min",), thresholds=(0.05, 0.10, 0.20))

    assert {0.05, 0.10, 0.20}.issubset(set(jumps["threshold"]))
    assert jumps["jump_size"].max() == pytest.approx(0.38)


def test_calibration_summaries_are_non_empty_on_toy_data(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "outputs"
    data_dir.mkdir()
    _write_toy_files(data_dir)

    result = run_calibration(
        data_dir=data_dir,
        out_dir=out_dir,
        trade_inputs=("trades.csv",),
        metadata_inputs=("markets.csv",),
        event_inputs=("events.csv",),
        windows=("5min",),
        max_rows_per_file=100,
    )

    assert not result.market_summary.empty
    assert not result.trade_size_summary.empty
    assert not result.jump_windows.empty
    assert not result.jump_size_distribution.empty
    assert (out_dir / "market_summary.csv").exists()
    assert (out_dir / "trade_size_summary.csv").exists()
    assert (out_dir / "jump_windows.csv").exists()
    assert (out_dir / "jump_size_distribution.csv").exists()
    assert (out_dir / "price_paths_sample.png").exists()


def test_missing_timestamp_or_price_columns_produce_clear_errors():
    missing_timestamp = pd.DataFrame({"market_id": ["MKT"], "price": [0.5], "size": [1]})
    missing_price = pd.DataFrame(
        {"market_id": ["MKT"], "timestamp": ["2026-01-01T00:00:00Z"], "size": [1]}
    )

    with pytest.raises(CalibrationError, match="Missing timestamp column"):
        normalize_trade_frame(missing_timestamp)
    with pytest.raises(CalibrationError, match="Missing price column"):
        normalize_trade_frame(missing_price)


def test_simulator_parameter_suggestions_json_has_expected_keys(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "outputs"
    data_dir.mkdir()
    _write_toy_files(data_dir)

    run_calibration(
        data_dir=data_dir,
        out_dir=out_dir,
        trade_inputs=("trades.csv",),
        metadata_inputs=("markets.csv",),
        event_inputs=("events.csv",),
        windows=("5min",),
        max_rows_per_file=100,
    )

    suggestions = json.loads((out_dir / "simulator_parameter_suggestions.json").read_text())
    expected_keys = {
        "inputs",
        "initial_price_candidates",
        "jump_size_min_candidates",
        "jump_size_max_candidates",
        "adverse_jump_probability_proxy",
        "quantity_liquidation_size_ranges",
        "notional_volume_distribution",
        "market_activity_distribution",
        "category_distribution",
        "public_private_jump_share",
        "data_limitations",
    }

    assert expected_keys.issubset(suggestions)
    assert suggestions["public_private_jump_share"]["public_jump_share"] is None
    assert "Cannot prove true stale-quote races" in " ".join(suggestions["data_limitations"])


def _write_toy_files(data_dir):
    trades = pd.DataFrame(
        {
            "ticker": ["MKT1"] * 5 + ["MKT2"] * 3,
            "created_time": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:01:00Z",
                    "2026-01-01T00:02:00Z",
                    "2026-01-01T00:03:00Z",
                    "2026-01-01T00:04:00Z",
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:02:00Z",
                    "2026-01-01T00:04:00Z",
                ]
            ),
            "yes_price": [50, 56, 67, 88, 82, 35, 30, 42],
            "count": [10, 20, 30, 40, 50, 5, 6, 7],
            "taker_side": ["yes", "yes", "yes", "yes", "yes", "no", "no", "yes"],
        }
    )
    markets = pd.DataFrame(
        {
            "ticker": ["MKT1", "MKT2"],
            "series_ticker": ["TEST", "TEST"],
            "title": ["Toy market 1", "Toy market 2"],
            "close_time": [1_767_225_600, 1_767_312_000],
            "result": ["yes", "no"],
        }
    )
    events = pd.DataFrame(
        {
            "market_id": ["MKT1"],
            "timestamp": ["2026-01-01T00:03:00Z"],
            "pre_price": [0.67],
            "post_price": [0.88],
            "direction": ["up"],
            "notional": [1000],
        }
    )
    trades.to_csv(data_dir / "trades.csv", index=False)
    markets.to_csv(data_dir / "markets.csv", index=False)
    events.to_csv(data_dir / "events.csv", index=False)
