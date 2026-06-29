from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
from pathlib import Path

from pm_dfba_sim.data.pmxt import PMXTProbeError, run_pmxt_probe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a bounded feasibility probe over one PMXT v2 parquet file."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Local PMXT parquet file.")
    source.add_argument("--url", help="Remote PMXT hourly parquet URL. Downloads only this file.")
    parser.add_argument("--out", default="outputs/pmxt_probe", help="Directory for derived probe outputs.")
    parser.add_argument("--max-rows", type=int, default=200_000, help="Maximum parquet rows to load.")
    parser.add_argument("--max-markets", type=int, default=3, help="Maximum markets to reconstruct.")
    args = parser.parse_args()

    try:
        if args.url:
            with tempfile.TemporaryDirectory(prefix="pmxt_probe_") as tmp_dir:
                local_path = _download_one_pmxt_file(args.url, Path(tmp_dir))
                result = run_pmxt_probe(
                    input_path=local_path,
                    out_dir=args.out,
                    max_rows=args.max_rows,
                    max_markets=args.max_markets,
                    source_label=args.url,
                )
        else:
            result = run_pmxt_probe(
                input_path=args.input,
                out_dir=args.out,
                max_rows=args.max_rows,
                max_markets=args.max_markets,
                source_label=args.input,
            )
    except PMXTProbeError as exc:
        raise SystemExit(f"PMXT probe failed: {exc}") from exc

    print(f"Wrote PMXT probe outputs to {Path(args.out)}")
    print(f"Rows loaded: {result.schema_summary['rows_loaded']:,}")
    print(f"Top-of-book rows: {len(result.top_of_book_timeseries):,}")
    print(f"Depth rows: {len(result.depth_timeseries):,}")


def _download_one_pmxt_file(url: str, tmp_dir: Path) -> Path:
    filename = Path(url.split("?")[0]).name or "pmxt_hourly.parquet"
    if not filename.endswith(".parquet"):
        filename += ".parquet"
    destination = tmp_dir / filename
    with urllib.request.urlopen(url) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)
    return destination


if __name__ == "__main__":
    main()
