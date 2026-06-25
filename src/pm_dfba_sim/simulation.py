from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pm_dfba_sim.margin import (
    LeveragedLongYes,
    liquidation_barrier,
    liquidation_shortfall,
    liquidation_triggers,
    long_yes_bad_debt,
)
from pm_dfba_sim.probability import ProbabilityJump, generate_probability_jump
from pm_dfba_sim.types import MarketConfig, TrialResult, VenueType
from pm_dfba_sim.venues import (
    effective_liquidation_depth,
    liquidation_exit_price,
    stale_quote_loss,
    taker_delay_cost,
)


def load_config(path: str | Path) -> MarketConfig:
    with Path(path).open() as f:
        return MarketConfig.from_dict(json.load(f))


def simulate_experiment(config: MarketConfig) -> list[TrialResult]:
    rng = np.random.default_rng(config.seed)
    results: list[TrialResult] = []

    for trial_id in range(config.n_trials):
        event = generate_probability_jump(config, rng)
        maker_latency_ms = float(rng.exponential(config.maker_latency_mean_ms))
        taker_latency_ms = float(rng.exponential(config.taker_latency_mean_ms))

        for leverage in config.leverage_values:
            for venue in VenueType:
                results.append(
                    simulate_venue_trial(
                        config=config,
                        event=event,
                        maker_latency_ms=maker_latency_ms,
                        taker_latency_ms=taker_latency_ms,
                        trial_id=trial_id,
                        leverage=leverage,
                        venue=venue,
                    )
                )

    return results


def simulate_venue_trial(
    config: MarketConfig,
    event: ProbabilityJump,
    maker_latency_ms: float,
    taker_latency_ms: float,
    trial_id: int,
    leverage: float,
    venue: VenueType,
) -> TrialResult:
    position = LeveragedLongYes(
        p0=config.initial_price,
        quantity=config.quantity,
        leverage=leverage,
    )
    exit_price = liquidation_exit_price(venue, event, config)
    barrier = liquidation_barrier(config.initial_price, leverage, config.maintenance_buffer)
    triggered = liquidation_triggers(exit_price, barrier)

    if triggered:
        bad_debt = long_yes_bad_debt(position, exit_price)
        shortfall = liquidation_shortfall(config.quantity, barrier, exit_price)
    else:
        bad_debt = 0.0
        shortfall = 0.0

    stale_loss = stale_quote_loss(
        venue=venue,
        event=event,
        config=config,
        maker_latency_ms=maker_latency_ms,
        taker_latency_ms=taker_latency_ms,
    )

    return TrialResult(
        venue=venue,
        leverage=leverage,
        trial_id=trial_id,
        p0=event.p0,
        p_post=event.p_post,
        jump_size=event.jump_size,
        public_jump=event.public_jump,
        terminal_jump=event.terminal_jump,
        stale_quote_loss=stale_loss,
        public_stale_quote_loss=stale_loss if event.public_jump else 0.0,
        liquidation_triggered=triggered,
        liquidation_exit_price=exit_price,
        liquidation_shortfall=shortfall,
        bad_debt=bad_debt,
        maker_loss=stale_loss,
        effective_liquidation_depth=effective_liquidation_depth(venue, config),
        taker_delay_cost=taker_delay_cost(venue, event, config),
    )
