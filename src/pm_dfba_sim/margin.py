from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LeveragedLongYes:
    p0: float
    quantity: float
    leverage: float

    @property
    def notional(self) -> float:
        return self.p0 * self.quantity

    @property
    def collateral(self) -> float:
        return self.notional / self.leverage

    @property
    def debt(self) -> float:
        return self.notional - self.collateral


def zero_equity_price(p0: float, leverage: float) -> float:
    if leverage <= 0:
        raise ValueError("leverage must be positive")
    return ((leverage - 1) * p0) / leverage


def liquidation_barrier(p0: float, leverage: float, maintenance_buffer: float) -> float:
    return float(np.clip(zero_equity_price(p0, leverage) + maintenance_buffer, 0.0, 1.0))


def liquidation_triggers(executable_exit_price: float, barrier: float) -> bool:
    return executable_exit_price < barrier


def bad_debt_amount(
    debt: float,
    liquidation_proceeds: float,
    collateral_recovery: float = 0.0,
) -> float:
    return max(0.0, debt - liquidation_proceeds - collateral_recovery)


def long_yes_bad_debt(position: LeveragedLongYes, liquidation_exit_price: float) -> float:
    proceeds = liquidation_exit_price * position.quantity
    return bad_debt_amount(position.debt, proceeds)


def liquidation_shortfall(quantity: float, barrier: float, liquidation_exit_price: float) -> float:
    return max(0.0, barrier - liquidation_exit_price) * quantity
