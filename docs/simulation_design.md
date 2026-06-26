# Simulation Design

## Research Question

The simulator asks whether prediction-market dual-flow batch auctions can reduce venue-created stale-quote loss and liquidation shortfall enough to improve marginability relative to a continuous limit order book, holding the event-risk process fixed.

PM-DFBA should not be expected to eliminate fundamental event risk. A terminal resolution from `0.80` to `0.00` can still create bad debt. The venue design question is whether public repricing events and liquidation execution become less damaging when serial stale-quote races are reduced.

## Marginability Metrics

The scaffold reports:

- bad-debt probability;
- bad-debt mean;
- bad-debt expected shortfall at 95 percent and 99 percent;
- liquidation shortfall mean;
- stale quote loss mean;
- public stale quote loss mean;
- maker loss placeholder mean;
- liquidation trigger rate;
- effective liquidation depth;
- simplified taker delay cost;
- implied and sampled CLOB race probability in latency sweeps;
- safe leverage at the configured bad-debt tolerance.

Safe leverage is reported as the highest tested leverage whose bad-debt probability is less than or equal to the configured tolerance. If no leverage satisfies the tolerance, the output uses a null value.

## Venue Designs

`CLOB` is a continuous matching abstraction. Public news can create a race where takers try to hit stale quotes and makers try to cancel or update. Stale loss occurs only when taker latency beats maker latency.

`FBA` uses a uniform batch-clearing abstraction. It reduces time-priority race value but does not segregate maker and taker flow.

`DFBA` adds maker/taker segregation on top of uniform batch clearing. The MVP represents this with configurable stale-loss and depth multipliers.

`PM_DFBA` normalizes YES/NO contracts onto one probability axis, uses maker/taker segregation, applies a simplified volatility-call protection for large public jumps when `volatility_call_enabled` is true, and models liquidation orders as auction-only, price-collared flow that can access committed backstop liquidity.

The liquidation collar is a limit, not a price guarantee. Primary liquidation liquidity can fill only up to the quantity executable at or above the collar. If the remaining executable liquidity is worse than the collar, the MVP either routes the remainder to configured backstop depth or leaves it partially or fully unfilled. Unfilled quantity contributes no liquidation proceeds and can increase shortfall or bad debt.

The `collar_mode` config controls how primary depth is checked:

- `vwap`: current default. The collar is applied to the average executable primary fill price.
- `marginal`: stricter. The collar is applied to the final executable unit, so it can reduce the primary fill quantity relative to `vwap`.

## Loss Decomposition

The intended decomposition is:

```text
total loss =
    fundamental jump loss
  + venue-created stale-quote loss
  + liquidation gap loss
  + creep / delay slippage
```

The MVP directly tracks stale quote loss, public stale quote loss, liquidation shortfall, bad debt, `maker_loss_placeholder`, and taker delay cost. Fundamental jump risk is represented by synthetic probability jumps and terminal resolution events.

`maker_loss_placeholder` is an explicit placeholder alias for `stale_quote_loss`. The scaffold keeps it as a separate output column because later versions should distinguish maker inventory losses, picked-off stale quotes, and losses borne by financiers or liquidation engines. In the MVP, do not interpret `maker_loss_placeholder_mean` as an independently estimated maker PnL series.

## MVP Simplifications

- The order book is not fully simulated.
- Agent strategies are parameterized rather than strategic.
- Stale quote loss is a transparent function of jump size, displayed depth, venue multiplier, and, for CLOB, the taker-versus-maker latency race.
- Liquidation exit quality is represented by venue-specific depth and slippage multipliers.
- PM-DFBA volatility-call behavior is represented by a configurable public-jump threshold, stale-loss multiplier, and delay cost. `PM_DFBA_NO_VOL_CALL` disables both the public stale-loss benefit and the added volatility-call delay.
- PM-DFBA price collars are modeled as a minimum sell execution price for liquidation flow, with `vwap` and `marginal` collar modes.
- Batch interval sweeps report stale quote loss and simplified taker delay cost. They are useful for sensitivity checks, but they are not a full auction state-machine model.
- Latency sweeps vary maker and taker exponential latency means. The implied CLOB race probability is `maker_latency_mean_ms / (maker_latency_mean_ms + taker_latency_mean_ms)`.

These assumptions are synthetic placeholders, not empirical estimates.

## Ablation Framework

The ablation framework adds stress scenarios that are meant to identify load-bearing assumptions. Some scenarios are closer to structural ablations because they toggle mechanism behavior, while others remain synthetic parameter-sensitivity tests. The runner compares the core venues with PM-DFBA variants:

- `PM_DFBA_FULL`: baseline PM-DFBA assumptions.
- `PM_DFBA_NO_VOL_CALL`: public-jump volatility-call stale-loss protection and delay cost removed.
- `PM_DFBA_NO_BACKSTOP_ONLY`: only committed backstop depth removed.
- `PM_DFBA_NO_MAKER_TAKER_SEGREGATION`: PM-DFBA weakened toward FBA-like flow assumptions.
- `PM_DFBA_TOXIC_ONLY`: only toxic-flow stale-loss classification degraded.
- `PM_DFBA_TOXIC_WITH_BACKSTOP`: toxic classification plus adverse primary depth/slippage, but with backstop retained.
- `PM_DFBA_ADVERSE_DEPTH_ONLY`: only primary depth/slippage weakened.
- `PM_DFBA_ADVERSE_STACK`: combined toxic classification, adverse primary depth/slippage, and no backstop.
- `TERMINAL_JUMP_STRESS`: PM-DFBA under elevated terminal instant-resolution jump probability.

The ablation runner sweeps public jump share, private jump share, terminal jump probability, batch interval, backstop depth, maker/taker latency means, and leverage. These values live in `configs/baseline.json`.

Interpretation rules:

- PM-DFBA should help most when public jumps dominate and volatility-call protection is active.
- PM-DFBA should help less when private information dominates because the public-jump protection is less relevant.
- Removing backstop depth should worsen or leave unchanged liquidation shortfall. It should not improve liquidation outcomes.
- Removing volatility-call protection should increase public stale-quote loss while removing the volatility-call delay cost.
- Lower CLOB race probability should shrink PM-DFBA's stale-loss advantage because there are fewer stale quotes for CLOB takers to pick off.
- Toxic-flow misclassification should shrink or reverse PM-DFBA's stale-loss advantage, while adverse depth/backstop variants should identify liquidation-specific failure modes.
- Terminal instant-resolution jumps should still create bad debt at sufficient leverage.

These are falsification-oriented synthetic checks. Passing them does not prove PM-DFBA works in real markets; it only shows the scaffold can express wins, losses, and weakened-mechanism cases.

## Invalidating Cases

The mechanism can fail or lose its advantage under:

- terminal instant-resolution jumps;
- private-information dominance;
- thin backstop liquidity;
- bad maker/taker classification;
- role gaming;
- extreme illiquidity;
- incorrect or overly optimistic parameterization.

The simulator should preserve these failure modes. PM-DFBA must not be made superior by conditional logic that simply zeroes out bad debt or stale loss.

## Future Ablations

- Vary public versus private jump frequency.
- Vary terminal resolution probability.
- Sweep maker and taker latency distributions.
- Remove backstop liquidity from PM-DFBA.
- Vary volatility-call thresholds and multipliers.
- Stress liquidation collars under shallow liquidity.
- Compare gross versus net YES/NO margining.
- Add empirical replay from Kalshi and Polymarket data.
