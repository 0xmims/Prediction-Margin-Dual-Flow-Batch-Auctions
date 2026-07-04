# Tailscale Replica Feasibility Probe

This module probes the read-only PostgreSQL + TimescaleDB replica of live
Polymarket/Kalshi market data (database `predictiondb`, reachable only over the
owner's Tailscale network). It answers one question: can this replica support
the empirical objects the PM-DFBA paper needs — top-of-book time series, L2
depth near mid, trade/book alignment, resolution joins, event-window replay,
liquidation exit-curve proxies, and stale-loss proxies — for a bounded sample of
non-sports markets?

It is a feasibility/calibration probe, not a replay engine and not empirical
proof that PM-DFBA reduces stale-quote losses.

## Connection And Safety

Connection settings come from environment variables only:

- `PREDICTION_DB_HOST` (required; the replica's Tailscale IP)
- `PREDICTION_DB_PORT` (default `5432`)
- `PREDICTION_DB_NAME` (default `predictiondb`)
- `PREDICTION_DB_USER` (default `friend_ro`)
- `PREDICTION_DB_PASSWORD` (optional; when unset, libpq reads `~/.pgpass`)

Nothing in the codebase hardcodes hosts or credentials, prints them, or writes
them into outputs. After writing outputs, the runner scans every output file
for the password value (when set via the environment) and aborts if it ever
appears. The replica enforces `statement_timeout = 120s`, a 3-connection limit,
and read-only transactions; the probe uses a single connection and only
time-bounded queries — every hypertable query builder in
`src/pm_dfba_sim/data/tailscale_db.py` raises if a timezone-aware window is
missing.

## Order-Book Reconstruction Contract

The replica stores the book as periodic full snapshots plus diff streams:

1. Seed from the latest snapshot with `event_time <= T` (bounded lookback),
   breaking ties by `seq DESC`.
2. Apply only diffs whose `snapshot_seq` equals that snapshot's `seq`, in `seq`
   order. Diff sizes are signed deltas, never absolute sizes.
3. Every new snapshot fully re-anchors the book; diffs referencing another
   anchor are counted and skipped, never accumulated across boundaries.

Book math uses `Decimal` end to end. The fold reports `diffs_applied`,
`diffs_skipped_wrong_anchor`, `diffs_skipped_duplicate`, and crossed-book
events so reconstruction quality is measurable, not assumed.

## Running

```bash
PREDICTION_DB_HOST=<tailscale-ip> PYTHONPATH=src python3 -m pm_dfba_sim.run_tailscale_probe \
  --out outputs/tailscale_probe \
  --start 2026-05-27T00:00:00Z \
  --end 2026-06-27T00:00:00Z \
  --max-markets 25 \
  --min-volume 2000
```

The probe:

1. Aggregates per-market trade activity day by day (discovery is paged; no
   unbounded scans).
2. Classifies candidates by keyword heuristics over market and parent-event
   titles (`markets.json_object->>'eventId'` joined to `events`), excludes
   sports (the top-volume markets are World Cup match markets whose own titles
   contain no sports keyword — the event title catches them), and fills bucket
   quotas: politics, macro/finance/crypto, legal/policy, plus a quiet-baseline
   bucket sampled below `--min-volume`.
3. Verifies order-book presence with capped counts before accepting a market —
   trade prints exist for far more markets than the book ingester tracks.
4. Per market: daily coverage and zero-day detection, book-stream gap hunting
   cross-checked against trade prints, a replay of the busiest trading window
   (top-of-book per minute, depth bands on a 5-minute grid, executable exit
   curves at sampled times), trade-to-book alignment, a coarse jump scan, and a
   resolution join from `market_lifecycle_events`.

## Outputs (derived, small, gitignored)

- `market_candidates.csv` — ranked candidates with classification and status
- `market_coverage_summary.csv` — per-market coverage, replay, and fold stats
- `gap_report.csv` — book-stream gaps (with `suspicious` flag), zero-diff days,
  replay-day intra-day gaps
- `top_of_book_timeseries_sample.csv` — per-minute best bid/ask/mid/spread
- `depth_timeseries_sample.csv` — depth within 1c/5c/10c and `imbalance_5c`
- `trade_alignment_sample.csv` — trades vs reconstructed book, both price
  conventions
- `liquidation_exit_curve_sample.csv` — executable sell/buy value for
  Q in {100, 1000, 5000} plus an adaptive size, with honest partial fills
- `jump_window_candidates.csv` — coarse 5c/10c/20c move candidates with
  interim/terminal/unknown timing labels
- `tailscale_probe_report.md` — the feasibility report

## Empirical Findings Encoded In The Probe

- **Trade prints are identity-space in every market sampled so far.**
  Polymarket NO-labeled prints already carry book-space prices; complementing
  them (`1 - p`) misaligned every sampled NO-labeled print and manufactured
  fake full-range jumps. The aligner therefore tests both conventions per
  trade and reports which fits per market rather than assuming; the jump
  scanner uses prices as printed. Directional semantics of NO-labeled prints
  still need confirmation from the data owner.
- **Snapshot cadence is activity-driven** (seconds on busy markets, tens of
  minutes on quiet Kalshi books), so a silent book stream is not automatically
  an outage. The gap threshold scales per market (at least 4x the median
  snapshot cadence, computed server-side), and a gap is flagged `suspicious` —
  consistent with an ingest stall, not proof of one — only when a trade
  printed during an hour with no recorded diffs at all.
- **Book coverage is a subset of trade coverage.** Selection must verify book
  presence per market; "no diffs" never means "no market activity."

## Data Reliability Eras (from the replica onboarding README)

- Order-book capture floor: 2025-12-29; trade prints: 2025-12-23.
- Before ~2026-04-21 (in-trader Python ingest): unmonitored, frequent silent
  gaps — treat as coverage-unverified.
- ~2026-04-21 to ~2026-05-27 (Go ingest, unmonitored): improved-but-unverified.
- From ~2026-05-27: monitored; occasional short stalls are recorded.

The probe labels the window's era in its report and refuses windows before the
capture floor. Prefer the monitored era for anything paper-facing.

## What This Probe Can And Cannot Claim

Can claim:

- The replica supports order-book reconstruction for sampled non-sports
  markets, subject to per-market coverage and gap checks.
- It enables observable-book replay, depth and executable exit-curve
  computation, trade/book alignment, and stale-loss proxies.

Cannot claim:

- True maker-cancel-vs-taker-hit race proof. That requires order IDs and
  add/cancel/modify/fill lifecycle events, which the replica does not expose.
  Aggregate-level inference (level shrinks without a trade print ≈ cancel) is
  possible but is inference, not proof.
- Completeness. Gaps exist and are reported, not assumed away.
- Anything about PM-DFBA superiority. This is calibration groundwork.

## Tests

`tests/test_tailscale_db.py` uses synthetic fixtures only (no live DB): query
builders must demand time bounds, reconstruction must respect snapshot
anchoring and signed deltas, depth/exit-curve/gap/alignment/jump math is
verified exactly with `Decimal`, and the report generator is checked to never
contain credential material.
