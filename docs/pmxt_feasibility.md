# PMXT Feasibility Probe

PMXT v2 is being investigated as a possible public L2 replay source for PM-DFBA research. The reported archive consists of hourly Parquet files of the Polymarket CLOB WebSocket event stream, including event types such as `book`, `price_change`, `last_trade_price`, and `tick_size_change`.

This probe is intentionally bounded. It is designed to inspect one local parquet file or one explicitly requested remote hourly parquet URL. Do not download the full archive for this probe, and do not commit raw PMXT data.

## Run

Local file:

```bash
PYTHONPATH=src python3 -m pm_dfba_sim.run_pmxt_probe \
  --input /path/to/polymarket_orderbook_hour.parquet \
  --out outputs/pmxt_probe \
  --max-rows 200000 \
  --max-markets 3
```

One remote hourly file:

```bash
PYTHONPATH=src python3 -m pm_dfba_sim.run_pmxt_probe \
  --url https://r2v2.pmxt.dev/polymarket_orderbook_YYYY-MM-DDTHH.parquet \
  --out outputs/pmxt_probe \
  --max-rows 200000 \
  --max-markets 3
```

Remote mode downloads only the requested parquet file to a temporary directory, then writes derived outputs under ignored `outputs/pmxt_probe/`.

URL diagnostics without downloading the full parquet:

```bash
PYTHONPATH=src python3 -m pm_dfba_sim.run_pmxt_probe \
  --url https://r2v2.pmxt.dev/polymarket_orderbook_2026-04-21T00.parquet \
  --out outputs/pmxt_probe \
  --max-rows 200000 \
  --max-markets 3 \
  --diagnose-url
```

Diagnostic mode attempts a HEAD request first. If HEAD fails, it tries a safe `Range: bytes=0-1023` GET request. It writes `outputs/pmxt_probe/url_diagnostics.json` and does not download the full parquet.

## Observed Remote Access Result

On 2026-06-29, the diagnostic command against:

```text
https://r2v2.pmxt.dev/polymarket_orderbook_2026-04-21T00.parquet
```

returned:

- HTTP status: `403`
- Content type: `text/plain; charset=UTF-8`
- Content length: `17`
- Accepts ranges: `false`
- Requires API key guess: `true`
- Notes: `HEAD request failed: HTTP 403: Forbidden`; `Range GET failed: HTTP 403: Forbidden`

This looks like an access-control/API-key restriction rather than a successful public range-readable parquet response. It does not verify whether the file is a tick-level hourly partition or a static hourly snapshot.

If remote access fails, download one PMXT parquet manually into `~/Downloads`, then run:

```bash
PYTHONPATH=src python3 -m pm_dfba_sim.run_pmxt_probe \
  --input ~/Downloads/polymarket_orderbook_2026-04-21T00.parquet \
  --out outputs/pmxt_probe \
  --max-rows 200000 \
  --max-markets 3
```

## Outputs

- `outputs/pmxt_probe/url_diagnostics.json` when probing a URL
- `outputs/pmxt_probe/schema_summary.json`
- `outputs/pmxt_probe/event_type_counts.csv`
- `outputs/pmxt_probe/market_sample.csv`
- `outputs/pmxt_probe/top_of_book_timeseries.csv`
- `outputs/pmxt_probe/depth_timeseries.csv`
- `outputs/pmxt_probe/pmxt_probe_report.md`

These are derived feasibility artifacts. Raw PMXT parquet files should not be committed.

## What PMXT May Support

If the event stream schema contains parseable full `book` snapshots and later `price_change` updates, PMXT may support:

- top-of-book reconstruction;
- best bid, best ask, midpoint, and spread time series;
- depth within 1c, 5c, and 10c of midpoint;
- event-window replay for one or more markets;
- stale-loss proxy construction;
- liquidation exit-curve calibration ingredients.

If `last_trade_price` events are present with usable timestamps and market identifiers, PMXT may also help align trade-price moves to reconstructed book states.

The report classifies each inspected file as one of:

- `tick_level_hourly_partition`
- `static_hourly_snapshot`
- `unknown`

The classification uses only observable sample properties: row count, distinct timestamp count, timestamp span, event type counts, and number of markets. It is a feasibility signal, not a guarantee that all archive partitions behave the same way.

## What PMXT Does Not Prove By Itself

PMXT should not be treated as proof of true stale-quote races unless exchange-side order lifecycle sequencing is visible. True maker-cancel-versus-taker-hit proof requires order add/cancel/modify/fill sequencing, cancellation timing, taker-hit timing, and maker/order identity or equivalent account-level quote lifecycle fields.

Without those fields, PMXT can support replay ingredients and stale-loss proxies, but not definitive latency-race attribution.
