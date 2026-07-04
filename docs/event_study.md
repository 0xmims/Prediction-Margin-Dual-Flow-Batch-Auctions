# Event-Window Replay Study

Replays displayed order books around dated public events on the Tailscale
replica and measures the observable-book quantities behind the paper's
venue-created stale-loss and liquidation-gap terms. It consumes the
reconstruction adapter from `docs/tailscale_db_probe.md` and inherits all of
its safety rules (env-var credentials, time-bounded queries only, single
read-only connection, post-write credential scan).

## Design

Events live in `configs/event_study.json`, each with an explicit anchor
timestamp and a `timestamp_basis` recording where that timestamp came from
(a published schedule, a market resolution time, or a data-discovered jump
minute from the tailscale probe). Every market in an event carries a role:

- `terminal_leg` — the event resolves this market (terminal jump; belongs to
  margin rules, not the matching engine).
- `interim_leg` — public information moves probability; resolution is later.
  This is PM-DFBA's target case.
- `headline_target` — unscheduled but dated public headline.
- `control` — a quiet market replayed at the same clock window, calibrating
  what "nothing happened" looks like.

The flagship design point: the June 17 FOMC decision is simultaneously a
terminal event for the June Fed markets and an interim public jump for the
July Fed markets on both platforms — the paper's jump taxonomy inside one
natural experiment.

Each event-market window is decomposed into baseline `[-60, -5)` min, impact
`[0, 15)` min, and recovery `[15, 60)` min around the anchor (configurable).
The replay folds one bounded snapshot+diff window and samples **as-of** book
states on a 5-second grid (the last state at or before each grid time —
checkpoint yields from the fold, so quiet baselines are never stamped with
post-jump books).

## Metrics

- Per grid point: best bid/ask, mid, spread, depth within 1c/5c/10c,
  imbalance, and executable exit value `V_exit(Q)` for the configured sizes,
  both sides, with honest partial fills.
- Per phase: medians and extremes of spread, 5c depth, and `V_exit`; then
  impact-vs-baseline ratios — `depth_decay_ratio`, `spread_widening_ratio`,
  `exit_value_haircut_ratio` — and `recovery_time_seconds` (spread and depth
  both back near baseline for two consecutive grid points).
- Trade-throughs: every print is compared against the book displayed tau
  seconds earlier (tau = 1/5/30 by default). A print outside the tau-lagged
  spread is a trade-through, and the violated distance times size is an
  observable-book stale-loss proxy. Detection is **direction-agnostic** (either
  side of the lagged spread), so the unconfirmed side semantics of NO-labeled
  Polymarket prints are not load-bearing.

## Running

```bash
PREDICTION_DB_HOST=<tailscale-ip> PYTHONPATH=src python3 -m pm_dfba_sim.run_event_study \
  --config configs/event_study.json --out outputs/event_study
```

Outputs (derived, gitignored): `event_summary.csv`,
`event_timeline_sample.csv`, `trade_through_sample.csv`,
`trade_through_aggregates.csv`, `event_parameter_suggestions.json`, timeline
and ratio figures, and `event_study_report.md`.

## Interpretation guardrails

This is a falsification-style test, and the report says which way it cut: if
event-exposed legs behave like controls (no depth decay, no spread widening,
no excess trade-throughs), the venue-created stale-loss term is small and the
PM-DFBA premise weakens.

It cannot claim: maker-cancel-vs-taker-hit attribution (no order lifecycle
data — trade-throughs measure execution against recently displayed liquidity,
not who lost a race); public-vs-private information attribution beyond the
documented event timestamps; or anything about PM-DFBA superiority. The
parameter suggestions are bounded sensitivity ranges for the simulator, not
point estimates.
