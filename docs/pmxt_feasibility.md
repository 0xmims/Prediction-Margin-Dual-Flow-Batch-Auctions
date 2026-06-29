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

## Outputs

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

## What PMXT Does Not Prove By Itself

PMXT should not be treated as proof of true stale-quote races unless exchange-side order lifecycle sequencing is visible. True maker-cancel-versus-taker-hit proof requires order add/cancel/modify/fill sequencing, cancellation timing, taker-hit timing, and maker/order identity or equivalent account-level quote lifecycle fields.

Without those fields, PMXT can support replay ingredients and stale-loss proxies, but not definitive latency-race attribution.
