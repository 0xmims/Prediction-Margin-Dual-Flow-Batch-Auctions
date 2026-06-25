from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pm_dfba_sim.probability import ProbabilityJump
from pm_dfba_sim.types import MarketConfig, NormalizedOrder, Side, VenueType


@dataclass(frozen=True)
class ClearingResult:
    clearing_price: float | None
    fill_quantity: float


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
    if venue == VenueType.CLOB:
        multiplier = config.clob_depth_multiplier
    elif venue == VenueType.FBA:
        multiplier = config.fba_depth_multiplier
    elif venue == VenueType.DFBA:
        multiplier = config.dfba_depth_multiplier
    elif venue == VenueType.PM_DFBA:
        multiplier = config.pm_dfba_depth_multiplier + config.backstop_depth_multiplier
    else:
        raise ValueError(f"unsupported venue: {venue}")

    return config.base_liquidation_depth * multiplier


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


def apply_sell_collar(exit_price: float, collar_price: float) -> float:
    """A sell liquidation should not execute below its minimum collar."""
    return max(exit_price, collar_price)


def liquidation_exit_price(
    venue: VenueType,
    event: ProbabilityJump,
    config: MarketConfig,
) -> float:
    if event.terminal_jump:
        return event.p_post

    depth = effective_liquidation_depth(venue, config)
    depth_ratio = config.quantity / max(depth, 1e-9)
    slippage = event.jump_size * liquidation_slippage_multiplier(venue, config) * depth_ratio * 0.5
    exit_price = float(np.clip(event.p_post - slippage, 0.0, 1.0))

    if venue == VenueType.PM_DFBA:
        collar_price = float(np.clip(event.p_post - config.liquidation_collar_buffer, 0.0, 1.0))
        return apply_sell_collar(exit_price, collar_price)

    return exit_price


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
