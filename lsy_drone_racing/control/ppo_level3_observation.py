"""Observation-layout metadata for the direct level3 PPO policy."""

from __future__ import annotations

from typing import Any

LEGACY_OBSERVATION_LAYOUT = "legacy_obstacle_top_xyz_v0"
WORLD_HISTORY_OBSERVATION_LAYOUT = "obstacle_heading_xy_v1"
LOCAL_HISTORY_WITH_PROGRESS_OBSERVATION_LAYOUT = "level3_obstacle_heading_xy_local_history_v2"
LOCAL_HISTORY_OBSERVATION_LAYOUT = "level3_obstacle_heading_xy_local_history_no_progress_v3"
ALL_GATES_OBSERVATION_LAYOUT = "level3_all_gates_onehot_known_local_history_v4"
LOCAL_OBSTACLE_OBSERVATION_LAYOUT = "level3_target_gate_nearest_gate_2obs_local_history_v5"
LOCAL_HISTORY_OBSERVATION_LAYOUT_ALIASES = (
    LOCAL_HISTORY_OBSERVATION_LAYOUT,
    LOCAL_HISTORY_WITH_PROGRESS_OBSERVATION_LAYOUT,
    ALL_GATES_OBSERVATION_LAYOUT,
    LOCAL_OBSTACLE_OBSERVATION_LAYOUT,
    "obstacle_heading_xy_local_history_v2",
)
ALL_GATES_OBSERVATION_LAYOUTS = (ALL_GATES_OBSERVATION_LAYOUT,)
LOCAL_OBSTACLE_OBSERVATION_LAYOUTS = (LOCAL_OBSTACLE_OBSERVATION_LAYOUT,)
TARGET_PROGRESS_OBSERVATION_LAYOUTS = (
    LEGACY_OBSERVATION_LAYOUT,
    WORLD_HISTORY_OBSERVATION_LAYOUT,
    LOCAL_HISTORY_WITH_PROGRESS_OBSERVATION_LAYOUT,
    "obstacle_heading_xy_local_history_v2",
)


def infer_hidden_dim(model_state_dict: dict[str, Any]) -> int:
    """Infer the shared actor/critic hidden width from PPO actor weights."""
    try:
        hidden_dim = int(model_state_dict["actor_mean.0.weight"].shape[0])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Cannot infer PPO hidden_dim from actor_mean.0.weight.") from exc
    if hidden_dim <= 0:
        raise ValueError(f"Invalid PPO hidden_dim inferred from checkpoint: {hidden_dim}.")
    return hidden_dim


def checkpoint_hidden_dim(checkpoint: Any, model_state_dict: dict[str, Any] | None = None) -> int:
    """Return checkpoint hidden width, validating metadata against the stored weights."""
    if model_state_dict is None:
        model_state_dict, _ = unpack_checkpoint(checkpoint)
    inferred_hidden_dim = infer_hidden_dim(model_state_dict)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        metadata_hidden_dim = checkpoint.get("hidden_dim")
        if metadata_hidden_dim is not None and int(metadata_hidden_dim) != inferred_hidden_dim:
            raise ValueError(
                f"Checkpoint hidden_dim metadata is {metadata_hidden_dim}, "
                f"but actor weights use {inferred_hidden_dim}."
            )
    return inferred_hidden_dim


def make_checkpoint(
    model_state_dict: dict[str, Any],
    hidden_dim: int | None = None,
    observation_layout: str = LOCAL_OBSTACLE_OBSERVATION_LAYOUT,
) -> dict[str, Any]:
    """Package level3 model weights with observation-layout and network-width metadata."""
    inferred_hidden_dim = infer_hidden_dim(model_state_dict)
    if hidden_dim is not None and hidden_dim != inferred_hidden_dim:
        raise ValueError(
            f"Requested hidden_dim={hidden_dim}, but actor weights use {inferred_hidden_dim}."
        )
    return {
        "model_state_dict": model_state_dict,
        "observation_layout": observation_layout,
        "hidden_dim": inferred_hidden_dim,
    }


def unpack_checkpoint(checkpoint: Any) -> tuple[dict[str, Any], str]:
    """Return model weights and layout, treating old raw state dicts as legacy."""
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint.get(
            "observation_layout", LEGACY_OBSERVATION_LAYOUT
        )
    return checkpoint, LEGACY_OBSERVATION_LAYOUT
