"""Shared observation-layout metadata for the direct level2 PPO policy."""

from __future__ import annotations

from typing import Any

LEGACY_OBSERVATION_LAYOUT = "legacy_obstacle_top_xyz_v0"
OBSERVATION_LAYOUT = "obstacle_heading_xy_v1"


def make_checkpoint(model_state_dict: dict[str, Any]) -> dict[str, Any]:
    """Package model weights with the observation layout required at inference."""
    return {
        "model_state_dict": model_state_dict,
        "observation_layout": OBSERVATION_LAYOUT,
    }


def unpack_checkpoint(checkpoint: Any) -> tuple[dict[str, Any], str]:
    """Return model weights and layout, treating old raw state dicts as legacy."""
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint.get(
            "observation_layout", LEGACY_OBSERVATION_LAYOUT
        )
    return checkpoint, LEGACY_OBSERVATION_LAYOUT
