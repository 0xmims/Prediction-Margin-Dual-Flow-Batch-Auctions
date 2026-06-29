import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from pm_dfba_sim.data.pmxt import (
    build_hourly_file_interpretation,
    build_schema_summary,
    classify_hourly_file,
    compute_depth_near_mid,
    compute_top_of_book,
    count_event_types,
    diagnose_pmxt_url,
    parse_book_payload,
    write_probe_report,
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


def test_url_diagnostics_object_creation_with_mocked_response():
    class FakeResponse:
        status = 200
        headers = {
            "Content-Type": "application/octet-stream",
            "Content-Length": "12345",
            "Accept-Ranges": "bytes",
        }

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size=-1):
            return b"PAR1"

    def fake_opener(request, timeout):
        assert request.get_method() == "HEAD"
        assert timeout == 20
        return FakeResponse()

    diagnostics = diagnose_pmxt_url(
        "https://example.test/hour.parquet",
        opener=fake_opener,
        attempted_at_utc="2026-01-01T00:00:00+00:00",
    )

    assert diagnostics["url"] == "https://example.test/hour.parquet"
    assert diagnostics["http_status"] == 200
    assert diagnostics["content_type"] == "application/octet-stream"
    assert diagnostics["content_length"] == 12345
    assert diagnostics["accepts_ranges"] is True
    assert diagnostics["requires_api_key_guess"] is False
    assert diagnostics["notes"] == ["HEAD request succeeded."]


def test_hourly_interpretation_classifies_tick_level_partition():
    frame = pd.DataFrame(
        {
            "event_type": ["book", "price_change", "last_trade_price", "price_change"],
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="s"),
            "market_id": ["M1", "M1", "M2", "M2"],
        }
    )
    counts = count_event_types(frame)

    interpretation = build_hourly_file_interpretation(frame, counts)

    assert interpretation["classification"] == "tick_level_hourly_partition"
    assert interpretation["distinct_timestamps"] == 4
    assert interpretation["distinct_markets"] == 2


def test_hourly_interpretation_classifies_static_snapshot():
    classification = classify_hourly_file(
        rows_loaded=2,
        distinct_timestamps=1,
        timestamp_span_seconds=0,
        event_type_counts={"book": 2},
        distinct_markets=2,
    )

    assert classification == "static_hourly_snapshot"


def test_report_includes_hourly_file_interpretation_section(tmp_path):
    frame = pd.DataFrame(
        {
            "event_type": ["book", "price_change", "last_trade_price"],
            "timestamp": pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="s"),
            "market_id": ["M1", "M1", "M1"],
        }
    )
    counts = count_event_types(frame)
    summary = build_schema_summary(
        frame=frame,
        source_label="toy",
        event_type_column="event_type",
        timestamp_column="timestamp",
        market_column="market_id",
        trade_side_column=None,
        event_counts=counts,
        top_of_book=pd.DataFrame({"depth_total_1c": [1.0]}),
        depth=pd.DataFrame({"depth_total_1c": [1.0]}),
        max_rows=100,
        max_markets=1,
    )
    report_path = tmp_path / "report.md"

    write_probe_report(report_path, summary)

    report = report_path.read_text()
    assert "## Hourly file interpretation" in report
    assert "## What this does and does not prove" in report
    assert "tick_level_hourly_partition" in report


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
    report = (out_dir / "pmxt_probe_report.md").read_text()
    assert len(top) >= 2
    assert "depth_total_1c" in depth
    assert "## Hourly file interpretation" in report
