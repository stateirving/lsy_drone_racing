"""Train or evaluate the Level2 direct racing PPO policy with JAX/Optax."""

from __future__ import annotations

from typing import Any

import fire

from lsy_drone_racing.control import jax_ppo


def main(**kwargs: Any) -> Any:
    """Forward CLI flags to the shared JAX PPO trainer with level2 selected."""
    kwargs.setdefault("level", "level2")
    return jax_ppo.main(**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
