from __future__ import annotations

import argparse
from pathlib import Path

from pm_dfba_sim.data.calibration import (
    DEFAULT_EVENT_INPUTS,
    DEFAULT_METADATA_INPUTS,
    DEFAULT_TRADE_INPUTS,
    CalibrationError,
    run_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate PM-DFBA simulator parameter ranges from local audited data."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Local root directory for audited prediction-market data.",
    )
    parser.add_argument(
        "--out",
        default="outputs/calibration",
        help="Output directory for derived calibration artifacts.",
    )
    parser.add_argument(
        "--trade-input",
        action="append",
        default=None,
        help=(
            "Trade CSV/parquet path or glob, relative to --data-dir unless absolute. "
            "May be supplied multiple times."
        ),
    )
    parser.add_argument(
        "--metadata-input",
        action="append",
        default=None,
        help=(
            "Market metadata CSV/parquet path or glob, relative to --data-dir unless absolute. "
            "May be supplied multiple times."
        ),
    )
    parser.add_argument(
        "--event-input",
        action="append",
        default=None,
        help=(
            "Event/jump label CSV/parquet path or glob, relative to --data-dir unless absolute. "
            "May be supplied multiple times."
        ),
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=100_000,
        help="Maximum rows read from any one CSV/parquet file.",
    )
    parser.add_argument(
        "--max-files-per-glob",
        type=int,
        default=9,
        help="Maximum representative files read from each glob expansion.",
    )
    parser.add_argument(
        "--window",
        action="append",
        default=None,
        help="Jump-detection window such as 5min, 15min, or 1h. May be repeated.",
    )
    parser.add_argument(
        "--near-resolution-window",
        default="24h",
        help=(
            "Label jumps at or after market close minus this window as near-resolution, "
            "for example 24h, 6h, or 1h."
        ),
    )
    args = parser.parse_args()

    try:
        result = run_calibration(
            data_dir=args.data_dir,
            out_dir=args.out,
            trade_inputs=args.trade_input or DEFAULT_TRADE_INPUTS,
            metadata_inputs=args.metadata_input or DEFAULT_METADATA_INPUTS,
            event_inputs=args.event_input or DEFAULT_EVENT_INPUTS,
            max_rows_per_file=args.max_rows_per_file,
            max_files_per_glob=args.max_files_per_glob,
            windows=args.window or ("5min", "15min", "1h"),
            near_resolution_window=args.near_resolution_window,
        )
    except CalibrationError as exc:
        raise SystemExit(f"Calibration failed: {exc}") from exc

    out_dir = Path(args.out)
    print(f"Wrote calibration outputs to {out_dir}")
    print(f"Normalized trade rows: {len(result.normalized_trades):,}")
    print(f"Markets summarized: {len(result.market_summary):,}")
    print(f"Jump-window rows: {len(result.jump_windows):,}")


if __name__ == "__main__":
    main()
