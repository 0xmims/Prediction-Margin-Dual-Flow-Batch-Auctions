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
- maker loss mean;
- liquidation trigger rate;
- effective liquidation depth;
- simplified taker delay cost;
- safe leverage at the configured bad-debt tolerance.

Safe leverage is reported as the highest tested leverage whose bad-debt probability is less than or equal to the configured tolerance. If no leverage satisfies the tolerance, the output uses a null value.

## Venue Designs

`CLOB` is a continuous matching abstraction. Public news can create a race where takers try to hit stale quotes and makers try to cancel or update. Stale loss occurs only when taker latency beats maker latency.

`FBA` uses a uniform batch-clearing abstraction. It reduces time-priority race value but does not segregate maker and taker flow.

`DFBA` adds maker/taker segregation on top of uniform batch clearing. The MVP represents this with configurable stale-loss and depth multipliers.

`PM_DFBA` normalizes YES/NO contracts onto one probability axis, uses maker/taker segregation, applies a simplified volatility-call protection for large public jumps, and models liquidation orders as auction-only, price-collared flow that can access committed backstop liquidity.

## Loss Decomposition

The intended decomposition is:

```text
total loss =
    fundamental jump loss
  + venue-created stale-quote loss
  + liquidation gap loss
  + creep / delay slippage
```

The MVP directly tracks stale quote loss, public stale quote loss, liquidation shortfall, bad debt, maker loss, and taker delay cost. Fundamental jump risk is represented by synthetic probability jumps and terminal resolution events.

## MVP Simplifications

- The order book is not fully simulated.
- Agent strategies are parameterized rather than strategic.
- Stale quote loss is a transparent function of jump size, displayed depth, venue multiplier, and, for CLOB, the taker-versus-maker latency race.
- Liquidation exit quality is represented by venue-specific depth and slippage multipliers.
- PM-DFBA volatility-call behavior is represented by a configurable public-jump threshold and stale-loss multiplier.
- PM-DFBA price collars are modeled as a minimum sell execution price for liquidation flow.

These assumptions are synthetic placeholders, not empirical estimates.

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
