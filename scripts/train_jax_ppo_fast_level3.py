"""Train or evaluate the Level3 direct racing PPO policy with fast pure-JAX rollout."""

from __future__ import annotations

from typing import Any

import fire

from lsy_drone_racing.control import jax_ppo_fast


def main(**kwargs: Any) -> Any:
    """Forward CLI flags to the fast JAX PPO trainer with level3 selected."""
    kwargs.setdefault("level", "level3")
    return jax_ppo_fast.main(**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
