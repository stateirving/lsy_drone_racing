"""Train or evaluate the Level3 direct racing PPO policy with fast pure-JAX rollout."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


def _configure_jax_cache() -> None:
    """Point XLA GPU autotune cache at a user-writable directory before JAX imports."""
    cache_dir = Path(os.environ.get("LSY_JAX_CACHE_DIR", f"/tmp/lsy_jax_cache_{os.getuid()}"))
    autotune_dir = cache_dir / "xla_gpu_per_fusion_autotune_cache_dir"
    autotune_dir.mkdir(parents=True, exist_ok=True)

    flag = "--xla_gpu_per_fusion_autotune_cache_dir"
    replacement = f"{flag}={autotune_dir}"
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if flag in xla_flags:
        xla_flags = re.sub(rf"{flag}(?:=|\s+)\S+", replacement, xla_flags)
    else:
        xla_flags = f"{xla_flags} {replacement}".strip()
    os.environ["XLA_FLAGS"] = xla_flags


_configure_jax_cache()

import fire

from lsy_drone_racing.control import jax_ppo_fast


def main(**kwargs: Any) -> Any:
    """Forward CLI flags to the fast JAX PPO trainer with level3 selected."""
    kwargs.setdefault("level", "level3")
    return jax_ppo_fast.main(**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
