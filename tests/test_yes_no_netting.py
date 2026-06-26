import pytest

from pm_dfba_sim.types import FlowType, Order, Outcome, Side, normalize_order, yes_no_netting


@pytest.mark.parametrize(
    ("outcome", "side", "price", "expected_side", "expected_price"),
    [
        (Outcome.YES, Side.BUY, 0.60, Side.BUY, 0.60),
        (Outcome.YES, Side.SELL, 0.60, Side.SELL, 0.60),
        (Outcome.NO, Side.BUY, 0.40, Side.SELL, 0.60),
        (Outcome.NO, Side.SELL, 0.40, Side.BUY, 0.60),
    ],
)
def test_yes_no_normalization(outcome, side, price, expected_side, expected_price):
    normalized = normalize_order(
        Order(
            outcome=outcome,
            side=side,
            price=price,
            quantity=10,
            flow_type=FlowType.TAKER,
        )
    )

    assert normalized.side_on_yes_axis == expected_side
    assert normalized.price == pytest.approx(expected_price)


@pytest.mark.parametrize(
    ("yes_qty", "no_qty", "riskless_sets", "net_yes"),
    [
        (100, 40, 40, 60),
        (25, 70, 25, -45),
    ],
)
def test_yes_no_netting(yes_qty, no_qty, riskless_sets, net_yes):
    exposure = yes_no_netting(yes_qty, no_qty)

    assert exposure.riskless_sets == riskless_sets
    assert exposure.net_yes_exposure == net_yes
