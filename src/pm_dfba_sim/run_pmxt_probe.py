from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
from pathlib import Path

from pm_dfba_sim.data.pmxt import (
    PMXTProbeError,
    diagnose_pmxt_url,
    run_pmxt_probe,
    write_url_diagnostics,
)


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
    parser.add_argument(
        "--diagnose-url",
        action="store_true",
        help="For --url, run HEAD/range diagnostics only and do not download the full parquet.",
    )
    args = parser.parse_args()

    try:
        if args.url:
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            url_diagnostics = diagnose_pmxt_url(args.url)
            write_url_diagnostics(out_dir / "url_diagnostics.json", url_diagnostics)
            if args.diagnose_url:
                _print_url_diagnostics(out_dir, url_diagnostics)
                return
            status = url_diagnostics.get("http_status")
            if status not in {200, 206}:
                raise PMXTProbeError(
                    f"URL diagnostic returned HTTP {status}; refusing full download. "
                    f"See {out_dir / 'url_diagnostics.json'}."
                )
            with tempfile.TemporaryDirectory(prefix="pmxt_probe_") as tmp_dir:
                local_path = _download_one_pmxt_file(args.url, Path(tmp_dir))
                result = run_pmxt_probe(
                    input_path=local_path,
                    out_dir=args.out,
                    max_rows=args.max_rows,
                    max_markets=args.max_markets,
                    source_label=args.url,
                    url_diagnostics=url_diagnostics,
                )
        else:
            if args.diagnose_url:
                raise PMXTProbeError("--diagnose-url can only be used with --url.")
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


def _print_url_diagnostics(out_dir: Path, diagnostics: dict) -> None:
    print(f"Wrote PMXT URL diagnostics to {out_dir / 'url_diagnostics.json'}")
    print(f"HTTP status: {diagnostics.get('http_status')}")
    print(f"Content length: {diagnostics.get('content_length')}")
    print(f"Accepts ranges: {diagnostics.get('accepts_ranges')}")
    print(f"Requires API key guess: {diagnostics.get('requires_api_key_guess')}")


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
