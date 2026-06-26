from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any


class VenueType(str, Enum):
    CLOB = "CLOB"
    FBA = "FBA"
    DFBA = "DFBA"
    PM_DFBA = "PM_DFBA"


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Outcome(str, Enum):
    YES = "YES"
    NO = "NO"


class FlowType(str, Enum):
    MAKER = "MAKER"
    TAKER = "TAKER"
    LIQUIDATION = "LIQUIDATION"
    BACKSTOP = "BACKSTOP"


@dataclass(frozen=True)
class Order:
    outcome: Outcome
    side: Side
    price: float
    quantity: float
    flow_type: FlowType

    def __post_init__(self) -> None:
        validate_probability(self.price, "price")
        if self.quantity < 0:
            raise ValueError("quantity must be non-negative")


@dataclass(frozen=True)
class NormalizedOrder:
    side_on_yes_axis: Side
    price: float
    quantity: float
    flow_type: FlowType

    def __post_init__(self) -> None:
        validate_probability(self.price, "price")
        if self.quantity < 0:
            raise ValueError("quantity must be non-negative")


@dataclass(frozen=True)
class NetExposure:
    riskless_sets: float
    net_yes_exposure: float


@dataclass(frozen=True)
class MarketConfig:
    seed: int
    n_trials: int
    initial_price: float
    quantity: float
    leverage_values: tuple[float, ...]
    maintenance_buffer: float
    bad_debt_tolerance: float
    jump_probability: float
    public_jump_probability: float
    terminal_jump_probability: float
    adverse_jump_probability: float
    jump_size_min: float
    jump_size_max: float
    maker_latency_mean_ms: float
    taker_latency_mean_ms: float
    batch_interval_ms: float
    volatility_call_threshold: float
    base_liquidation_depth: float
    clob_depth_multiplier: float
    fba_depth_multiplier: float
    dfba_depth_multiplier: float
    pm_dfba_depth_multiplier: float
    backstop_depth_multiplier: float
    clob_stale_loss_multiplier: float
    fba_stale_loss_multiplier: float
    dfba_stale_loss_multiplier: float
    pm_dfba_public_stale_loss_multiplier: float
    pm_dfba_private_stale_loss_multiplier: float
    clob_liquidation_slippage_multiplier: float
    fba_liquidation_slippage_multiplier: float
    dfba_liquidation_slippage_multiplier: float
    pm_dfba_liquidation_slippage_multiplier: float
    liquidation_collar_buffer: float = 0.05

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarketConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")

        normalized = dict(data)
        normalized["leverage_values"] = tuple(float(x) for x in data["leverage_values"])
        return cls(**normalized)


@dataclass(frozen=True)
class TrialResult:
    venue: VenueType
    leverage: float
    trial_id: int
    p0: float
    p_post: float
    jump_size: float
    public_jump: bool
    terminal_jump: bool
    stale_quote_loss: float
    public_stale_quote_loss: float
    liquidation_triggered: bool
    liquidation_exit_price: float
    liquidation_executed_quantity: float
    liquidation_unfilled_quantity: float
    liquidation_collar_breached: bool
    liquidation_used_backstop_depth: float
    liquidation_shortfall: float
    bad_debt: float
    maker_loss: float
    effective_liquidation_depth: float
    taker_delay_cost: float

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["venue"] = self.venue.value
        return row


def validate_probability(value: float, name: str) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def normalize_order(order: Order) -> NormalizedOrder:
    """Map YES and NO contracts onto one common YES-probability axis."""
    if order.outcome == Outcome.YES and order.side == Side.BUY:
        return NormalizedOrder(Side.BUY, order.price, order.quantity, order.flow_type)
    if order.outcome == Outcome.YES and order.side == Side.SELL:
        return NormalizedOrder(Side.SELL, order.price, order.quantity, order.flow_type)
    if order.outcome == Outcome.NO and order.side == Side.BUY:
        return NormalizedOrder(Side.SELL, 1 - order.price, order.quantity, order.flow_type)
    if order.outcome == Outcome.NO and order.side == Side.SELL:
        return NormalizedOrder(Side.BUY, 1 - order.price, order.quantity, order.flow_type)

    raise ValueError(f"unsupported order: {order}")


def yes_no_netting(yes_qty: float, no_qty: float) -> NetExposure:
    if yes_qty < 0 or no_qty < 0:
        raise ValueError("YES and NO quantities must be non-negative")
    return NetExposure(
        riskless_sets=min(yes_qty, no_qty),
        net_yes_exposure=yes_qty - no_qty,
    )
