import json
from pathlib import Path

import pandas as pd
import pytest

from pm_dfba_sim.data.calibration import (
    CalibrationError,
    detect_jump_windows,
    expand_input_paths,
    normalize_probability_price,
    normalize_trade_frame,
    read_table_sample,
    run_calibration,
    select_representative_paths,
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
    assert bool(yes["orientation_verified"].iloc[0])
    assert bool(no["orientation_verified"].iloc[0])
    assert not bool(generic["orientation_verified"].iloc[0])


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

    assert jumps["threshold_5c"].any()
    assert jumps["threshold_10c"].any()
    assert jumps["threshold_20c"].any()
    assert jumps["max_threshold_met"].max() == pytest.approx(0.20)
    assert jumps["jump_size"].max() == pytest.approx(0.38)
    assert not jumps.duplicated(subset=["market_id", "timestamp", "window"]).any()


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
    assert not result.interim_jump_size_distribution.empty
    assert not result.terminal_jump_size_distribution.empty
    assert (out_dir / "market_summary.csv").exists()
    assert (out_dir / "trade_size_summary.csv").exists()
    assert (out_dir / "jump_windows.csv").exists()
    assert (out_dir / "jump_size_distribution.csv").exists()
    assert (out_dir / "interim_jump_size_distribution.csv").exists()
    assert (out_dir / "terminal_jump_size_distribution.csv").exists()
    assert (out_dir / "price_paths_sample.png").exists()


def test_near_resolution_jumps_are_labeled_and_split(tmp_path):
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
        near_resolution_window="1h",
    )

    assert "near_resolution" in result.jump_windows
    assert result.jump_windows["near_resolution"].any()
    assert not result.interim_jump_size_distribution["near_resolution"].any()
    assert result.terminal_jump_size_distribution["near_resolution"].all()


def test_missing_timestamp_or_price_columns_produce_clear_errors():
    missing_timestamp = pd.DataFrame({"market_id": ["MKT"], "price": [0.5], "size": [1]})
    missing_price = pd.DataFrame(
        {"market_id": ["MKT"], "timestamp": ["2026-01-01T00:00:00Z"], "size": [1]}
    )

    with pytest.raises(CalibrationError, match="Missing timestamp column"):
        normalize_trade_frame(missing_timestamp)
    with pytest.raises(CalibrationError, match="Missing price column"):
        normalize_trade_frame(missing_price)


def test_final_outcome_price_is_not_treated_as_trade_price():
    frame = pd.DataFrame(
        {
            "market_id": ["MKT"],
            "timestamp": ["2026-01-01T00:00:00Z"],
            "final_outcome_price": [1.0],
            "size": [1],
        }
    )

    with pytest.raises(CalibrationError, match="Missing price column"):
        normalize_trade_frame(frame)


def test_cent_normalization_is_elementwise_and_idempotent():
    values = pd.Series([0.62, 62, 1.0, 100])
    normalized = normalize_probability_price(values)
    normalized_again = normalize_probability_price(normalized)

    assert normalized.to_list() == pytest.approx([0.62, 0.62, 1.0, 1.0])
    assert normalized_again.to_list() == pytest.approx(normalized.to_list())


def test_bid_ask_midpoint_branch_is_unverified_orientation():
    frame = pd.DataFrame(
        {
            "ticker": ["MKT"],
            "created_time": ["2026-01-01T00:00:00Z"],
            "yes_bid": [40],
            "yes_ask": [60],
            "count": [10],
        }
    )

    normalized = normalize_trade_frame(frame)

    assert normalized["yes_price"].iloc[0] == pytest.approx(0.50)
    assert normalized["midpoint_proxy"].iloc[0] == pytest.approx(0.50)
    assert not bool(normalized["orientation_verified"].iloc[0])


def test_parquet_reader_path_and_max_rows_cap_if_pyarrow_available(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "toy.parquet"
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_parquet(path)

    sample = read_table_sample(path, max_rows=2)

    assert len(sample) == 2
    assert sample["a"].to_list() == [1, 2]


def test_csv_reader_honors_max_rows_cap(tmp_path):
    path = tmp_path / "toy.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(path, index=False)

    sample = read_table_sample(path, max_rows=2)

    assert len(sample) == 2


def test_representative_path_selection_uses_first_middle_last():
    paths = [Path(f"trades_{i * 10000}_{(i + 1) * 10000}.parquet") for i in range(12)]

    selected = select_representative_paths(paths, max_files=9)

    assert selected[:3] == paths[:3]
    assert paths[5] in selected
    assert paths[6] in selected
    assert paths[-3:] == selected[-3:]


def test_expand_input_paths_samples_large_globs(tmp_path):
    for i in range(12):
        (tmp_path / f"f_{i}.csv").write_text("a\n1\n")

    selected = expand_input_paths(tmp_path, ("*.csv",), max_files_per_glob=9)

    assert len(selected) == 9


def test_unverified_orientation_rows_do_not_feed_adverse_proxy(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "outputs"
    data_dir.mkdir()
    trades = pd.DataFrame(
        {
            "market_id": ["MKT"] * 3,
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:01:00Z",
                    "2026-01-01T00:02:00Z",
                ]
            ),
            "price": [0.80, 0.50, 0.20],
            "size": [10, 10, 10],
        }
    )
    metadata = pd.DataFrame(
        {
            "market_id": ["MKT"],
            "close_time": ["2026-01-02T00:00:00Z"],
        }
    )
    trades.to_csv(data_dir / "trades.csv", index=False)
    metadata.to_csv(data_dir / "markets.csv", index=False)

    result = run_calibration(
        data_dir=data_dir,
        out_dir=out_dir,
        trade_inputs=("trades.csv",),
        metadata_inputs=("markets.csv",),
        event_inputs=(),
        windows=("5min",),
        max_rows_per_file=100,
    )

    assert result.simulator_parameter_suggestions["adverse_jump_probability_proxy"] is None
    assert result.simulator_parameter_suggestions["unverified_orientation_share"] == 1.0


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
        "unverified_orientation_share",
        "verified_orientation_trade_count",
        "unverified_orientation_trade_count",
        "all_row_suggestions",
        "verified_orientation_only_suggestions",
        "initial_price_candidates",
        "jump_size_interim_candidates",
        "jump_size_unfiltered_candidates",
        "terminal_jump_size_candidates",
        "jump_candidate_usage_note",
        "adverse_jump_probability_proxy",
        "taker_trade_size_quantiles",
        "liquidation_size_status",
        "category_distribution",
        "public_private_jump_share",
        "data_limitations",
    }

    assert expected_keys.issubset(suggestions)
    assert "quantity_liquidation_size_ranges" not in suggestions
    assert suggestions["liquidation_size_status"] == "unknown_from_trades_alone"
    assert suggestions["public_private_jump_share"]["public_jump_share"] is None
    assert "Cannot prove true stale-quote races" in " ".join(suggestions["data_limitations"])


def test_simulator_parameter_suggestions_json_is_reproducible(tmp_path):
    data_dir = tmp_path / "data"
    first_out = tmp_path / "first"
    second_out = tmp_path / "second"
    data_dir.mkdir()
    _write_toy_files(data_dir)

    run_calibration(
        data_dir=data_dir,
        out_dir=first_out,
        trade_inputs=("trades.csv",),
        metadata_inputs=("markets.csv",),
        event_inputs=("events.csv",),
        windows=("5min",),
        max_rows_per_file=100,
    )
    run_calibration(
        data_dir=data_dir,
        out_dir=second_out,
        trade_inputs=("trades.csv",),
        metadata_inputs=("markets.csv",),
        event_inputs=("events.csv",),
        windows=("5min",),
        max_rows_per_file=100,
    )

    first = json.loads((first_out / "simulator_parameter_suggestions.json").read_text())
    second = json.loads((second_out / "simulator_parameter_suggestions.json").read_text())

    assert first == second


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
            "close_time": ["2026-01-01T00:05:00Z", "2026-01-03T00:00:00Z"],
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
