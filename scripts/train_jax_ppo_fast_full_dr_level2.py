"""Train or evaluate Level2 PPO with fast pure-JAX rollout and full train DR."""

from __future__ import annotations

from typing import Any

import fire

from lsy_drone_racing.control import jax_ppo_fast_full_dr


def main(**kwargs: Any) -> Any:
    """Forward CLI flags to the full-DR fast JAX PPO trainer with Level2 selected."""
    kwargs.setdefault("level", "level2")
    return jax_ppo_fast_full_dr.main(**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
