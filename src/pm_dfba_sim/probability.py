from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pm_dfba_sim.types import MarketConfig


@dataclass(frozen=True)
class ProbabilityJump:
    p0: float
    p_post: float
    jump_size: float
    public_jump: bool
    private_jump: bool
    terminal_jump: bool
    adverse_jump: bool


def generate_probability_jump(
    config: MarketConfig,
    rng: np.random.Generator,
) -> ProbabilityJump:
    p0 = config.initial_price
    if rng.random() > config.jump_probability:
        return ProbabilityJump(
            p0=p0,
            p_post=p0,
            jump_size=0.0,
            public_jump=False,
            private_jump=False,
            terminal_jump=False,
            adverse_jump=False,
        )

    terminal_jump = rng.random() < config.terminal_jump_probability
    adverse_jump = rng.random() < config.adverse_jump_probability

    if terminal_jump:
        p_post = 0.0 if adverse_jump else 1.0
        return ProbabilityJump(
            p0=p0,
            p_post=p_post,
            jump_size=abs(p0 - p_post),
            public_jump=False,
            private_jump=False,
            terminal_jump=True,
            adverse_jump=adverse_jump,
        )

    public_jump = rng.random() < config.public_jump_probability
    jump_size = float(rng.uniform(config.jump_size_min, config.jump_size_max))
    signed_jump = -jump_size if adverse_jump else jump_size
    p_post = float(np.clip(p0 + signed_jump, 0.0, 1.0))

    return ProbabilityJump(
        p0=p0,
        p_post=p_post,
        jump_size=abs(p_post - p0),
        public_jump=public_jump,
        private_jump=not public_jump,
        terminal_jump=False,
        adverse_jump=adverse_jump,
    )
