from dataclasses import replace

import pytest

from pm_dfba_sim.probability import ProbabilityJump
from pm_dfba_sim.simulation import load_config, simulate_experiment
from pm_dfba_sim.simulation import simulate_venue_trial
from pm_dfba_sim.types import VenueType


def test_baseline_simulation_produces_all_venues_and_leverages():
    config = replace(load_config("configs/baseline.json"), n_trials=5)

    results = simulate_experiment(config)

    assert results
    assert {result.venue for result in results} == set(VenueType)
    assert {result.leverage for result in results} == set(config.leverage_values)


def test_pm_dfba_collar_blocked_liquidation_uses_actual_zero_proceeds():
    config = replace(
        load_config("configs/baseline.json"),
        quantity=1000,
        base_liquidation_depth=100,
        pm_dfba_depth_multiplier=1.0,
        backstop_depth_multiplier=0.0,
        pm_dfba_liquidation_slippage_multiplier=1.0,
        liquidation_collar_buffer=0.0,
    )
    event = ProbabilityJump(
        p0=0.60,
        p_post=0.40,
        jump_size=0.20,
        public_jump=True,
        private_jump=False,
        terminal_jump=False,
        adverse_jump=True,
    )

    result = simulate_venue_trial(
        config=config,
        event=event,
        maker_latency_ms=50,
        taker_latency_ms=20,
        trial_id=0,
        leverage=3.0,
        venue=VenueType.PM_DFBA,
    )

    assert result.liquidation_triggered
    assert result.liquidation_exit_price == 0
    assert result.liquidation_executed_quantity == 0
    assert result.liquidation_unfilled_quantity == config.quantity
    assert result.liquidation_collar_breached
    assert result.bad_debt == pytest.approx(400.0)
