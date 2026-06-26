from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pm_dfba_sim.probability import ProbabilityJump
from pm_dfba_sim.types import MarketConfig, NormalizedOrder, Side, VenueType


@dataclass(frozen=True)
class ClearingResult:
    clearing_price: float | None
    fill_quantity: float


@dataclass(frozen=True)
class LiquidationExecution:
    executable_price: float | None
    executed_quantity: float
    unfilled_quantity: float
    collar_breached: bool
    used_backstop_depth: float

    @property
    def proceeds(self) -> float:
        if self.executable_price is None:
            return 0.0
        return self.executable_price * self.executed_quantity


def uniform_batch_clear(orders: list[NormalizedOrder]) -> ClearingResult:
    buys = sorted(
        [order for order in orders if order.side_on_yes_axis == Side.BUY and order.quantity > 0],
        key=lambda order: order.price,
        reverse=True,
    )
    sells = sorted(
        [order for order in orders if order.side_on_yes_axis == Side.SELL and order.quantity > 0],
        key=lambda order: order.price,
    )

    if not buys or not sells or buys[0].price < sells[0].price:
        return ClearingResult(clearing_price=None, fill_quantity=0.0)

    buy_remaining = buys[0].quantity
    sell_remaining = sells[0].quantity
    buy_idx = 0
    sell_idx = 0
    fill_quantity = 0.0
    marginal_bid = buys[0].price
    marginal_ask = sells[0].price

    while buy_idx < len(buys) and sell_idx < len(sells):
        bid = buys[buy_idx]
        ask = sells[sell_idx]
        if bid.price < ask.price:
            break

        fill = min(buy_remaining, sell_remaining)
        fill_quantity += fill
        marginal_bid = bid.price
        marginal_ask = ask.price
        buy_remaining -= fill
        sell_remaining -= fill

        if buy_remaining == 0:
            buy_idx += 1
            if buy_idx < len(buys):
                buy_remaining = buys[buy_idx].quantity
        if sell_remaining == 0:
            sell_idx += 1
            if sell_idx < len(sells):
                sell_remaining = sells[sell_idx].quantity

    if fill_quantity == 0:
        return ClearingResult(clearing_price=None, fill_quantity=0.0)

    return ClearingResult(
        clearing_price=(marginal_bid + marginal_ask) / 2,
        fill_quantity=fill_quantity,
    )


def stale_quote_loss(
    venue: VenueType,
    event: ProbabilityJump,
    config: MarketConfig,
    maker_latency_ms: float,
    taker_latency_ms: float,
) -> float:
    if event.jump_size == 0 or event.terminal_jump:
        return 0.0

    displayed_depth = config.base_liquidation_depth * config.clob_depth_multiplier
    base_loss = event.jump_size * displayed_depth

    if venue == VenueType.CLOB:
        if taker_latency_ms >= maker_latency_ms:
            return 0.0
        return base_loss * config.clob_stale_loss_multiplier
    if venue == VenueType.FBA:
        return base_loss * config.fba_stale_loss_multiplier
    if venue == VenueType.DFBA:
        return base_loss * config.dfba_stale_loss_multiplier
    if venue == VenueType.PM_DFBA:
        if event.public_jump and event.jump_size >= config.volatility_call_threshold:
            return base_loss * config.pm_dfba_public_stale_loss_multiplier
        return base_loss * config.pm_dfba_private_stale_loss_multiplier

    raise ValueError(f"unsupported venue: {venue}")


def effective_liquidation_depth(venue: VenueType, config: MarketConfig) -> float:
    return primary_liquidation_depth(venue, config) + backstop_liquidation_depth(venue, config)


def primary_liquidation_depth(venue: VenueType, config: MarketConfig) -> float:
    if venue == VenueType.CLOB:
        multiplier = config.clob_depth_multiplier
    elif venue == VenueType.FBA:
        multiplier = config.fba_depth_multiplier
    elif venue == VenueType.DFBA:
        multiplier = config.dfba_depth_multiplier
    elif venue == VenueType.PM_DFBA:
        multiplier = config.pm_dfba_depth_multiplier
    else:
        raise ValueError(f"unsupported venue: {venue}")

    return config.base_liquidation_depth * multiplier


def backstop_liquidation_depth(venue: VenueType, config: MarketConfig) -> float:
    if venue == VenueType.PM_DFBA:
        return config.base_liquidation_depth * config.backstop_depth_multiplier
    return 0.0


def liquidation_slippage_multiplier(venue: VenueType, config: MarketConfig) -> float:
    if venue == VenueType.CLOB:
        return config.clob_liquidation_slippage_multiplier
    if venue == VenueType.FBA:
        return config.fba_liquidation_slippage_multiplier
    if venue == VenueType.DFBA:
        return config.dfba_liquidation_slippage_multiplier
    if venue == VenueType.PM_DFBA:
        return config.pm_dfba_liquidation_slippage_multiplier
    raise ValueError(f"unsupported venue: {venue}")


def sell_collar_breached(executable_price: float, collar_price: float) -> bool:
    """A sell collar rejects worse prices; it does not create a fill at the collar."""
    return executable_price < collar_price


def execute_liquidation(
    venue: VenueType,
    event: ProbabilityJump,
    config: MarketConfig,
) -> LiquidationExecution:
    if event.terminal_jump:
        return LiquidationExecution(
            executable_price=event.p_post,
            executed_quantity=config.quantity,
            unfilled_quantity=0.0,
            collar_breached=False,
            used_backstop_depth=0.0,
        )

    primary_depth = primary_liquidation_depth(venue, config)
    primary_quantity = min(config.quantity, primary_depth)
    primary_price = _liquidation_price_for_quantity(
        quantity=primary_quantity,
        depth=primary_depth,
        venue=venue,
        event=event,
        config=config,
    )

    if venue != VenueType.PM_DFBA:
        return LiquidationExecution(
            executable_price=primary_price if primary_quantity > 0 else None,
            executed_quantity=primary_quantity,
            unfilled_quantity=config.quantity - primary_quantity,
            collar_breached=False,
            used_backstop_depth=0.0,
        )

    collar_price = float(np.clip(event.p_post - config.liquidation_collar_buffer, 0.0, 1.0))
    primary_fill = _collared_primary_sell_quantity(
        requested_quantity=config.quantity,
        depth=primary_depth,
        venue=venue,
        event=event,
        config=config,
        collar_price=collar_price,
    )
    primary_fill_price = (
        _liquidation_price_for_quantity(primary_fill, primary_depth, venue, event, config)
        if primary_fill > 0
        else None
    )
    collar_breached = primary_fill < min(config.quantity, primary_depth)
    remaining = config.quantity - primary_fill
    backstop_fill = min(remaining, backstop_liquidation_depth(venue, config))
    total_executed = primary_fill + backstop_fill
    proceeds = ((primary_fill_price or 0.0) * primary_fill) + (collar_price * backstop_fill)

    return LiquidationExecution(
        executable_price=proceeds / total_executed if total_executed > 0 else None,
        executed_quantity=total_executed,
        unfilled_quantity=config.quantity - total_executed,
        collar_breached=collar_breached,
        used_backstop_depth=backstop_fill,
    )


def liquidation_exit_price(
    venue: VenueType,
    event: ProbabilityJump,
    config: MarketConfig,
) -> float:
    execution = execute_liquidation(venue, event, config)
    return execution.executable_price or 0.0


def _liquidation_price_for_quantity(
    quantity: float,
    depth: float,
    venue: VenueType,
    event: ProbabilityJump,
    config: MarketConfig,
) -> float:
    if quantity <= 0:
        return event.p_post

    depth_ratio = quantity / max(depth, 1e-9)
    slippage = event.jump_size * liquidation_slippage_multiplier(venue, config) * depth_ratio * 0.5
    return float(np.clip(event.p_post - slippage, 0.0, 1.0))


def _collared_primary_sell_quantity(
    requested_quantity: float,
    depth: float,
    venue: VenueType,
    event: ProbabilityJump,
    config: MarketConfig,
    collar_price: float,
) -> float:
    if requested_quantity <= 0 or depth <= 0:
        return 0.0

    full_quantity = min(requested_quantity, depth)
    full_price = _liquidation_price_for_quantity(full_quantity, depth, venue, event, config)
    if not sell_collar_breached(full_price, collar_price):
        return full_quantity

    slippage_unit = event.jump_size * liquidation_slippage_multiplier(venue, config) * 0.5
    if slippage_unit <= 0:
        return full_quantity if event.p_post >= collar_price else 0.0

    max_depth_ratio = max(0.0, (event.p_post - collar_price) / slippage_unit)
    return min(full_quantity, max_depth_ratio * depth)


def taker_delay_cost(venue: VenueType, event: ProbabilityJump, config: MarketConfig) -> float:
    if event.jump_size == 0 or event.terminal_jump:
        return 0.0
    if venue == VenueType.CLOB:
        delay_ms = 0.0
    else:
        delay_ms = config.batch_interval_ms
        if venue == VenueType.PM_DFBA and event.public_jump and event.jump_size >= config.volatility_call_threshold:
            delay_ms *= 1.5
    return event.jump_size * config.quantity * (delay_ms / 1000.0) * 0.01
