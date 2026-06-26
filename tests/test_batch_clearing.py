from dataclasses import replace

import pytest

from pm_dfba_sim.probability import ProbabilityJump
from pm_dfba_sim.simulation import load_config
from pm_dfba_sim.types import FlowType, NormalizedOrder, Side, VenueType
from pm_dfba_sim.venues import execute_liquidation, sell_collar_breached, uniform_batch_clear


def test_uniform_batch_clear_crossed_orders_clear_at_one_price():
    result = uniform_batch_clear(
        [
            NormalizedOrder(Side.BUY, price=0.62, quantity=100, flow_type=FlowType.TAKER),
            NormalizedOrder(Side.SELL, price=0.58, quantity=40, flow_type=FlowType.MAKER),
        ]
    )

    assert result.fill_quantity == 40
    assert result.clearing_price == pytest.approx(0.60)


def test_uniform_batch_clear_no_cross_no_trade():
    result = uniform_batch_clear(
        [
            NormalizedOrder(Side.BUY, price=0.57, quantity=100, flow_type=FlowType.TAKER),
            NormalizedOrder(Side.SELL, price=0.58, quantity=40, flow_type=FlowType.MAKER),
        ]
    )

    assert result.fill_quantity == 0
    assert result.clearing_price is None


def test_liquidation_sell_collar_rejects_prices_below_collar():
    assert sell_collar_breached(executable_price=0.32, collar_price=0.35)
    assert not sell_collar_breached(executable_price=0.36, collar_price=0.35)


def test_pm_dfba_liquidation_cannot_clear_below_collar_without_backstop():
    config = replace(
        load_config("configs/baseline.json"),
        quantity=1000,
        base_liquidation_depth=100,
        pm_dfba_depth_multiplier=1.0,
        backstop_depth_multiplier=0.0,
        pm_dfba_liquidation_slippage_multiplier=1.0,
        liquidation_collar_buffer=0.05,
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

    execution = execute_liquidation(VenueType.PM_DFBA, event, config)

    assert execution.executable_price is None
    assert execution.executed_quantity == 0
    assert execution.unfilled_quantity == config.quantity
    assert execution.collar_breached
    assert execution.used_backstop_depth == 0


def test_pm_dfba_liquidation_uses_backstop_before_improving_to_collar():
    config = replace(
        load_config("configs/baseline.json"),
        quantity=1000,
        base_liquidation_depth=100,
        pm_dfba_depth_multiplier=1.0,
        backstop_depth_multiplier=0.25,
        pm_dfba_liquidation_slippage_multiplier=1.0,
        liquidation_collar_buffer=0.05,
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

    execution = execute_liquidation(VenueType.PM_DFBA, event, config)

    assert execution.executable_price == pytest.approx(0.35)
    assert execution.executed_quantity == 25
    assert execution.unfilled_quantity == 975
    assert execution.collar_breached
    assert execution.used_backstop_depth == 25
