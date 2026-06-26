import pytest

from pm_dfba_sim.margin import (
    bad_debt_amount,
    liquidation_barrier,
    liquidation_triggers,
    zero_equity_price,
)


def test_long_yes_zero_equity_price():
    assert zero_equity_price(0.60, 3) == pytest.approx(0.40)


def test_liquidation_barrier_adds_maintenance_buffer():
    assert liquidation_barrier(0.60, 3, 0.05) == pytest.approx(0.45)


def test_liquidation_triggers_below_barrier():
    assert liquidation_triggers(0.44, 0.45)
    assert not liquidation_triggers(0.45, 0.45)


def test_bad_debt_is_positive_when_debt_exceeds_recoveries():
    assert bad_debt_amount(debt=100, liquidation_proceeds=60, collateral_recovery=20) == 20
