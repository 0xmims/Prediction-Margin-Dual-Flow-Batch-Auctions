import pytest

from pm_dfba_sim.types import FlowType, NormalizedOrder, Side
from pm_dfba_sim.venues import apply_sell_collar, uniform_batch_clear


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


def test_liquidation_sell_collar_prevents_execution_below_collar():
    assert apply_sell_collar(exit_price=0.32, collar_price=0.35) == pytest.approx(0.35)
