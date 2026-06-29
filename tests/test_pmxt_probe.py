import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from pm_dfba_sim.data.pmxt import (
    compute_depth_near_mid,
    compute_top_of_book,
    count_event_types,
    parse_book_payload,
)


def test_event_type_counting_from_toy_dataframe():
    frame = pd.DataFrame({"event_type": ["book", "book", "price_change", "last_trade_price"]})

    counts = count_event_types(frame)

    assert dict(zip(counts["event_type"], counts["count"])) == {
        "book": 2,
        "price_change": 1,
        "last_trade_price": 1,
    }


def test_top_of_book_from_simple_book_payload():
    book = parse_book_payload(
        {
            "bids": json.dumps([{"price": 0.49, "size": 10}, {"price": 0.48, "size": 20}]),
            "asks": json.dumps([{"price": 0.51, "size": 15}, {"price": 0.52, "size": 25}]),
        }
    )

    top = compute_top_of_book(book)

    assert top["best_bid"] == pytest.approx(0.49)
    assert top["best_bid_size"] == pytest.approx(10)
    assert top["best_ask"] == pytest.approx(0.51)
    assert top["best_ask_size"] == pytest.approx(15)
    assert top["midpoint"] == pytest.approx(0.50)
    assert top["spread"] == pytest.approx(0.02)


def test_depth_within_1c_5c_10c_from_toy_ladder():
    book = {
        "bids": [(0.495, 10), (0.47, 20), (0.39, 100)],
        "asks": [(0.505, 15), (0.54, 25), (0.62, 100)],
    }

    depth = compute_depth_near_mid(book, midpoint=0.50)

    assert depth["depth_total_1c"] == pytest.approx(25)
    assert depth["depth_total_5c"] == pytest.approx(70)
    assert depth["depth_total_10c"] == pytest.approx(70)


def test_unparseable_depth_ladder_is_graceful():
    book = parse_book_payload({"bids": "not-json", "asks": None})
    top = compute_top_of_book(book)
    depth = compute_depth_near_mid(book, top["midpoint"])

    assert book == {"bids": [], "asks": []}
    assert top["best_bid"] is None
    assert top["best_ask"] is None
    assert depth["depth_total_1c"] is None


def test_pmxt_probe_cli_writes_expected_files_from_local_fixture(tmp_path):
    pytest.importorskip("pyarrow")
    repo_root = Path(__file__).resolve().parents[1]
    fixture = tmp_path / "pmxt_fixture.parquet"
    out_dir = tmp_path / "probe_outputs"
    frame = pd.DataFrame(
        [
            {
                "event_type": "book",
                "timestamp": "2026-01-01T00:00:00Z",
                "market_id": "M1",
                "bids": json.dumps([{"price": 0.49, "size": 10}, {"price": 0.48, "size": 20}]),
                "asks": json.dumps([{"price": 0.51, "size": 15}, {"price": 0.52, "size": 25}]),
                "changes": None,
                "side": None,
                "price": None,
            },
            {
                "event_type": "price_change",
                "timestamp": "2026-01-01T00:00:01Z",
                "market_id": "M1",
                "bids": None,
                "asks": None,
                "changes": json.dumps([{"side": "bid", "price": 0.50, "size": 5}]),
                "side": None,
                "price": None,
            },
            {
                "event_type": "last_trade_price",
                "timestamp": "2026-01-01T00:00:02Z",
                "market_id": "M1",
                "bids": None,
                "asks": None,
                "changes": None,
                "side": "buy",
                "price": 0.52,
            },
            {
                "event_type": "tick_size_change",
                "timestamp": "2026-01-01T00:00:03Z",
                "market_id": "M1",
                "bids": None,
                "asks": None,
                "changes": None,
                "side": None,
                "price": None,
            },
        ]
    )
    frame.to_parquet(fixture)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pm_dfba_sim.run_pmxt_probe",
            "--input",
            str(fixture),
            "--out",
            str(out_dir),
            "--max-rows",
            "100",
            "--max-markets",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    expected = {
        "schema_summary.json",
        "event_type_counts.csv",
        "market_sample.csv",
        "top_of_book_timeseries.csv",
        "depth_timeseries.csv",
        "pmxt_probe_report.md",
    }
    for filename in expected:
        path = out_dir / filename
        assert path.exists()
        assert path.stat().st_size > 0

    top = pd.read_csv(out_dir / "top_of_book_timeseries.csv")
    depth = pd.read_csv(out_dir / "depth_timeseries.csv")
    assert len(top) >= 2
    assert "depth_total_1c" in depth
