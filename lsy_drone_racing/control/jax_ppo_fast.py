"""Pure-JAX PPO rollout/training path for direct drone racing.

This module keeps the PPO update code from :mod:`jax_ppo`, but replaces the
Gym/Python rollout loop with a Brax-style state that calls the low-level
``env._step`` inside ``jax.lax.scan``.  The fast path mirrors the reference JAX
architecture in ``/home/aojili/lsy_drone_racing`` and is intended for the large
vectorized Level2 validation and Level3 training runs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, NamedTuple

import fire
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from lsy_drone_racing.control.jax_ppo import (
    CONTROL_DIR,
    JaxPPOArgs,
    _optimizer,
    _resolve_device,
    actor_critic_apply,
    build_update_fn,
    compute_advantage_batch,
    gaussian_entropy,
    gaussian_logprob,
    init_actor_critic_params,
    level2_fast2_args,
    load_jax_checkpoint,
    reward_coefs,
    save_jax_checkpoint,
    set_seeds,
    setup_wandb,
)
from lsy_drone_racing.control.ppo_level2_observation import (
    OBSERVATION_LAYOUT as LEVEL2_OBSERVATION_LAYOUT,
)
from lsy_drone_racing.control.ppo_level3_observation import (
    LOCAL_OBSTACLE_OBSERVATION_LAYOUT as LEVEL3_OBSERVATION_LAYOUT,
)
from lsy_drone_racing.utils import load_config

ROOT = Path(__file__).parents[2]
GATE_CORNERS_LOCAL = jnp.array(
    [[0.0, -0.2, 0.2], [0.0, 0.2, 0.2], [0.0, 0.2, -0.2], [0.0, -0.2, -0.2]], dtype=jnp.float32
)
ROLLOUT_KEYS = {"obs", "actions", "logprobs", "rewards", "dones", "values"}
REWARD_COMPONENT_KEYS = (
    "progress",
    "gate_axis_progress",
    "gate_stage_progress",
    "gate_front",
    "gate_back",
    "near_gate",
    "gate_bonus",
    "finish_bonus",
    "missed_gate",
    "wrong_side",
    "crash",
    "action",
    "cmd_tilt",
    "smooth",
    "tilt",
    "tilt_excess",
    "obstacle",
    "obstacle_clearance",
    "timeout",
    "time",
)
RACE_METRIC_KEYS = (
    "gate_distance",
    "passed_gate_rate",
    "finished_rate",
    "crashed_rate",
    "done_rate",
    "target_gate",
    "gate_axis_x",
    "gate_centerline_dist",
    "gate_plane_dist",
    "gate_plane_cross_rate",
    "missed_gate_rate",
    "gate_stage",
    "gate_front_hit_rate",
    "gate_pass_hit_rate",
    "gate_back_hit_rate",
    "wrong_side_gate_rate",
    "timeout_rate",
    "obstacle_distance",
    "obstacle_clearance_progress",
    "tilt_angle_deg",
    "cmd_tilt_deg",
)


class FastState(NamedTuple):
    """JAX-scan-compatible replacement for Brax State."""

    pipeline_state: Any
    obs: Array
    reward: Array
    done: Array
    metrics: dict[str, Array]
    info: dict[str, Any]


def observation_layout(level: str) -> str:
    """Return the checkpoint metadata layout for a level."""
    if level == "level2":
        return LEVEL2_OBSERVATION_LAYOUT
    if level == "level3":
        return LEVEL3_OBSERVATION_LAYOUT
    raise ValueError(f"Unsupported level {level!r}.")


def observation_kind(level: str) -> str:
    """Return the fast observation implementation key for a level."""
    if level == "level2":
        return "level2_progress"
    if level == "level3":
        return "level3_local"
    raise ValueError(f"Unsupported level {level!r}.")


def history_dim_for_kind(kind: str) -> int:
    """Return per-row history width for a fast observation kind."""
    if kind == "level2_progress":
        return 13
    if kind == "level3_local":
        return 7
    raise ValueError(f"Unsupported observation kind {kind!r}.")


def obs_dim_for_kind(kind: str, *, n_gates: int, n_obstacles: int, n_history: int) -> int:
    """Compute flattened observation size without instantiating Python wrappers."""
    if kind == "level2_progress":
        return 1 + 3 + 3 + 9 + 1 + 12 + 12 + 12 + 4 * n_obstacles + n_gates + 4 + (n_history * 13)
    if kind == "level3_local":
        n_local_obstacles = min(2, n_obstacles)
        return 1 + 3 + 3 + 9 + 12 + 1 + 12 + 1 + 4 * n_local_obstacles + 4 + (n_history * 7)
    raise ValueError(f"Unsupported observation kind {kind!r}.")


def scalar_mean(value: Any) -> float:
    """Convert a JAX/NumPy value to a host scalar mean."""
    return float(np.asarray(value).mean())


def block_until_ready(tree: Any) -> Any:
    """Block on arrays in a pytree while leaving static leaves alone."""
    return jax.tree_util.tree_map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        tree,
    )


def drop_drone_dim(tree: dict[str, Any]) -> dict[str, Array]:
    """Drop the singleton drone axis returned by low-level vector env steps."""
    return {
        key: value[:, 0]
        if hasattr(value, "ndim") and value.ndim >= 2 and value.shape[1] == 1
        else value
        for key, value in tree.items()
    }


def device_put_tree(tree: Any, device: jax.Device) -> Any:
    """Move a raw observation/state tree to a target JAX device."""
    return jax.tree_util.tree_map(lambda value: jax.device_put(jnp.asarray(value), device), tree)


def scale_action_jax(action_norm: Array, action_low: Array, action_high: Array) -> Array:
    """Map normalized PPO actions to simulator attitude/thrust commands."""
    clipped = jnp.clip(action_norm, -1.0, 1.0)
    return clipped * ((action_high - action_low) / 2.0) + ((action_high + action_low) / 2.0)


def quat_to_rotmat(quat: Array) -> Array:
    """Convert xyzw quaternions to body-to-world rotation matrices."""
    x, y, z, w = jnp.moveaxis(quat, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    row0 = jnp.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], axis=-1)
    row1 = jnp.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], axis=-1)
    row2 = jnp.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


def gate_corners_body(
    observations: dict[str, Array], gate_idx: Array, pos: Array, rot_t: Array
) -> Array:
    """Return selected gate corners in body frame, flattened per env."""
    n_gates = observations["gates_pos"].shape[1]
    gate_idx = jnp.mod(gate_idx, n_gates)
    batch_idx = jnp.arange(pos.shape[0])
    gate_pos = observations["gates_pos"][batch_idx, gate_idx]
    gate_quat = observations["gates_quat"][batch_idx, gate_idx]
    gate_rot = quat_to_rotmat(gate_quat)
    corners_world = gate_pos[:, None, :] + jnp.einsum("nij,kj->nki", gate_rot, GATE_CORNERS_LOCAL)
    corners_body = jnp.einsum("nij,nkj->nki", rot_t, corners_world - pos[:, None, :])
    return corners_body.reshape(pos.shape[0], -1)


def obstacle_heading_xy_by_obstacle(
    observations: dict[str, Array], pos: Array, rot: Array
) -> Array:
    """Return [forward, left, XY distance, detected] features for each obstacle."""
    relative_xy = observations["obstacles_pos"][..., :2] - pos[:, None, :2]
    heading_forward = rot[:, :2, 0]
    heading_forward = heading_forward / jnp.maximum(
        jnp.linalg.norm(heading_forward, axis=-1, keepdims=True), 1e-6
    )
    heading_left = jnp.stack([-heading_forward[:, 1], heading_forward[:, 0]], axis=-1)
    relative_forward = jnp.einsum("nki,ni->nk", relative_xy, heading_forward)
    relative_left = jnp.einsum("nki,ni->nk", relative_xy, heading_left)
    distance_xy = jnp.linalg.norm(relative_xy, axis=-1)
    detected = observations["obstacles_visited"].astype(jnp.float32)
    return jnp.stack([relative_forward, relative_left, distance_xy, detected], axis=-1)


def basic_history_level2(observations: dict[str, Array]) -> Array:
    """History row used by the Level2 progress layout."""
    return jnp.concatenate(
        [observations["pos"], observations["quat"], observations["vel"], observations["ang_vel"]],
        axis=-1,
    )


def basic_history_level3(observations: dict[str, Array]) -> Array:
    """History row used by the compact Level3 local layout."""
    rot = quat_to_rotmat(observations["quat"])
    rot_t = jnp.swapaxes(rot, -1, -2)
    vel_body = jnp.einsum("nij,nj->ni", rot_t, observations["vel"])
    return jnp.concatenate(
        [observations["pos"][:, 2:3], vel_body, observations["ang_vel"]], axis=-1
    )


def flatten_obs_level2(observations: dict[str, Array], history: Array, last_action: Array) -> Array:
    """Flatten raw observations with the direct Level2 PPO layout."""
    pos = observations["pos"]
    rot = quat_to_rotmat(observations["quat"])
    rot_t = jnp.swapaxes(rot, -1, -2)
    vel_body = jnp.einsum("nij,nj->ni", rot_t, observations["vel"])
    n_gates = observations["gates_pos"].shape[1]
    active_target_gate = jnp.where(
        observations["target_gate"] < 0, n_gates - 1, observations["target_gate"]
    )
    prev_gate = jnp.maximum(active_target_gate - 1, 0)
    gate_prev = gate_corners_body(observations, prev_gate, pos, rot_t)
    gate_prev = jnp.where((active_target_gate > 0)[:, None], gate_prev, jnp.zeros_like(gate_prev))
    gate_current = gate_corners_body(observations, observations["target_gate"], pos, rot_t)
    next_gate = jnp.minimum(jnp.maximum(observations["target_gate"], 0) + 1, n_gates - 1)
    gate_next = gate_corners_body(observations, next_gate, pos, rot_t)
    obstacle_features = obstacle_heading_xy_by_obstacle(observations, pos, rot).reshape(
        pos.shape[0], -1
    )
    target_progress = (
        jnp.where(observations["target_gate"] < 0, n_gates, observations["target_gate"]) / n_gates
    )[:, None]
    return jnp.concatenate(
        [
            pos[:, 2:3],
            vel_body,
            observations["ang_vel"],
            rot.reshape(pos.shape[0], -1),
            target_progress.astype(jnp.float32),
            gate_prev,
            gate_current,
            gate_next,
            obstacle_features,
            observations["gates_visited"].astype(jnp.float32),
            last_action,
            history.reshape(pos.shape[0], -1),
        ],
        axis=-1,
    )


def nearest_other_gate_idx(
    observations: dict[str, Array], pos: Array, active_target_gate: Array
) -> Array:
    """Return the closest non-target gate index in XY."""
    n_gates = observations["gates_pos"].shape[1]
    gate_idx = jnp.arange(n_gates)[None, :]
    relative_xy = observations["gates_pos"][..., :2] - pos[:, None, :2]
    distance_xy = jnp.linalg.norm(relative_xy, axis=-1)
    masked_distance = jnp.where(gate_idx != active_target_gate[:, None], distance_xy, jnp.inf)
    return jnp.argmin(masked_distance, axis=-1)


def gate_known_flag(observations: dict[str, Array], gate_idx: Array) -> Array:
    """Return the selected gate known/visited flag."""
    batch_idx = jnp.arange(gate_idx.shape[0])
    return observations["gates_visited"][batch_idx, gate_idx][:, None].astype(jnp.float32)


def flatten_obs_level3(observations: dict[str, Array], history: Array, last_action: Array) -> Array:
    """Flatten raw observations with the compact local-obstacle Level3 layout."""
    pos = observations["pos"]
    rot = quat_to_rotmat(observations["quat"])
    rot_t = jnp.swapaxes(rot, -1, -2)
    vel_body = jnp.einsum("nij,nj->ni", rot_t, observations["vel"])
    n_gates = observations["gates_pos"].shape[1]
    active_target_gate = jnp.where(
        observations["target_gate"] < 0, n_gates - 1, observations["target_gate"]
    )
    target_gate_corners = gate_corners_body(observations, active_target_gate, pos, rot_t)
    target_gate_known = gate_known_flag(observations, active_target_gate)
    nearest_gate = nearest_other_gate_idx(observations, pos, active_target_gate)
    nearest_gate_corners = gate_corners_body(observations, nearest_gate, pos, rot_t)
    nearest_gate_known = gate_known_flag(observations, nearest_gate)
    obstacle_features = obstacle_heading_xy_by_obstacle(observations, pos, rot)
    nearest_obstacle_idx = jnp.argsort(obstacle_features[..., 2], axis=-1)[:, :2]
    nearest_obstacles = jnp.take_along_axis(
        obstacle_features, nearest_obstacle_idx[..., None], axis=1
    )
    return jnp.concatenate(
        [
            pos[:, 2:3],
            vel_body,
            observations["ang_vel"],
            rot.reshape(pos.shape[0], -1),
            target_gate_corners,
            target_gate_known,
            nearest_gate_corners,
            nearest_gate_known,
            nearest_obstacles.reshape(pos.shape[0], -1),
            last_action,
            history.reshape(pos.shape[0], -1),
        ],
        axis=-1,
    )


def flatten_observation(
    kind: str, observations: dict[str, Array], history: Array, last_action: Array
) -> Array:
    """Flatten observations for the selected level layout."""
    if kind == "level2_progress":
        return flatten_obs_level2(observations, history, last_action)
    if kind == "level3_local":
        return flatten_obs_level3(observations, history, last_action)
    raise ValueError(f"Unsupported observation kind {kind!r}.")


def basic_history(kind: str, observations: dict[str, Array]) -> Array:
    """Build a history row for the selected level layout."""
    if kind == "level2_progress":
        return basic_history_level2(observations)
    if kind == "level3_local":
        return basic_history_level3(observations)
    raise ValueError(f"Unsupported observation kind {kind!r}.")


def gate_distance(observations: dict[str, Array]) -> Array:
    """Distance from drone to active target gate center."""
    target_gate = observations["target_gate"]
    gates_pos = observations["gates_pos"]
    gate_idx = jnp.mod(target_gate, gates_pos.shape[1])
    batch_idx = jnp.arange(gates_pos.shape[0])
    target_pos = gates_pos[batch_idx, gate_idx]
    return jnp.linalg.norm(target_pos - observations["pos"], axis=-1)


def gate_frame_pos_for_gate(observations: dict[str, Array], gate_idx: Array) -> Array:
    """Drone position expressed in the selected gate frame."""
    gates_pos = observations["gates_pos"]
    gate_idx = jnp.mod(gate_idx, gates_pos.shape[1])
    batch_idx = jnp.arange(gates_pos.shape[0])
    gate_pos = gates_pos[batch_idx, gate_idx]
    gate_quat = observations["gates_quat"][batch_idx, gate_idx]
    gate_rot_t = jnp.swapaxes(quat_to_rotmat(gate_quat), -1, -2)
    return jnp.einsum("nij,nj->ni", gate_rot_t, observations["pos"] - gate_pos)


def gate_frame_pos(observations: dict[str, Array]) -> Array:
    """Drone position expressed in the active target gate frame."""
    return gate_frame_pos_for_gate(observations, observations["target_gate"])


def gate_stage_distance(
    observations: dict[str, Array], stage: Array, target_gate: Array, gate_stage_offset: float
) -> Array:
    """Distance to front/center/back staged gate targets."""
    gate_local = gate_frame_pos_for_gate(observations, target_gate)
    zero = jnp.zeros_like(gate_local[:, 0])
    stage_x = jnp.where(
        stage == 0, -gate_stage_offset, jnp.where(stage == 1, 0.0, gate_stage_offset)
    )
    stage_target = jnp.stack([stage_x, zero, zero], axis=-1)
    return jnp.linalg.norm(gate_local - stage_target, axis=-1)


def closest_obstacle_distance(observations: dict[str, Array]) -> Array:
    """Closest obstacle XY distance."""
    dxy = observations["obstacles_pos"][..., :2] - observations["pos"][:, None, :2]
    return jnp.min(jnp.linalg.norm(dxy, axis=-1), axis=-1)


def tilt(quat: Array) -> Array:
    """Return simple body-z tilt cost."""
    body_z_world_z = jnp.clip(quat_to_rotmat(quat)[..., 2, 2], -1.0, 1.0)
    return 1.0 - body_z_world_z


def tilt_angle(quat: Array) -> Array:
    """Return body-z tilt angle."""
    body_z_world_z = jnp.clip(quat_to_rotmat(quat)[..., 2, 2], -1.0, 1.0)
    return jnp.arccos(body_z_world_z)


def action_tilt(actions: Array, action_low: Array, action_high: Array) -> Array:
    """Return commanded roll/pitch tilt angle from normalized actions."""
    scaled_actions = scale_action_jax(actions, action_low, action_high)
    body_z_world_z = jnp.clip(
        jnp.cos(scaled_actions[..., 0]) * jnp.cos(scaled_actions[..., 1]), -1.0, 1.0
    )
    return jnp.arccos(body_z_world_z)


def initial_reward_state(observations: dict[str, Array], coefs: dict[str, Any]) -> dict[str, Array]:
    """Initialize pure-JAX Level2RaceReward state from reset observations."""
    num_envs = observations["pos"].shape[0]
    gate_local = gate_frame_pos(observations)
    gate_stage = jnp.zeros((num_envs,), dtype=jnp.int32)
    gate_stage_offset = float(coefs.get("gate_stage_offset", 0.35))
    return {
        "prev_gate_dist": gate_distance(observations),
        "prev_gate_x": gate_local[:, 0],
        "prev_gate_local": gate_local,
        "prev_target_gate": observations["target_gate"].astype(jnp.int32),
        "gate_stage": gate_stage,
        "prev_stage_dist": gate_stage_distance(
            observations, gate_stage, observations["target_gate"], gate_stage_offset
        ),
        "back_gate_active": jnp.zeros((num_envs,), dtype=bool),
        "back_gate_idx": jnp.zeros((num_envs,), dtype=jnp.int32),
        "prev_back_gate_local": jnp.zeros((num_envs, 3), dtype=jnp.float32),
        "prev_obstacle_dist": closest_obstacle_distance(observations),
    }


def reward_components(
    observations: dict[str, Array],
    actions: Array,
    terminated: Array,
    truncated: Array,
    last_action: Array,
    state: dict[str, Array],
    coefs: dict[str, Any],
    action_low: Array,
    action_high: Array,
) -> tuple[Array, dict[str, Array], dict[str, Array], dict[str, Array]]:
    """Pure function equivalent of the existing Level2RaceReward wrapper."""
    gate_stage_offset = float(coefs.get("gate_stage_offset", 0.35))
    gate_stage_radius = float(coefs.get("gate_stage_radius", 0.24))
    gate_dist = gate_distance(observations)
    target_gate = observations["target_gate"]
    finished = target_gate < 0
    passed_gate = (target_gate != state["prev_target_gate"]) & (state["prev_target_gate"] >= 0)
    target_changed = target_gate != state["prev_target_gate"]
    crashed = terminated & ~finished
    timed_out = truncated & ~finished

    raw_progress = state["prev_gate_dist"] - gate_dist
    progress = jnp.where(passed_gate | finished, 0.0, jnp.clip(raw_progress, -0.25, 0.25))
    near_gate = jnp.exp(-gate_dist)
    gate_local = gate_frame_pos(observations)
    gate_x = gate_local[:, 0]
    centerline_dist = jnp.linalg.norm(gate_local[:, 1:3], axis=-1)
    centerline_weight = jnp.exp(-10.0 * centerline_dist**2)
    same_gate = (target_gate == state["prev_target_gate"]) & (state["prev_target_gate"] >= 0)
    raw_axis_progress = jnp.clip(gate_x - state["prev_gate_x"], -0.1, 0.1)
    crossed_gate_plane = same_gate & (state["prev_gate_x"] < 0.0) & (gate_x > 0.0) & ~finished
    gate_dx = jnp.maximum(gate_x - state["prev_gate_x"], 1e-6)
    plane_alpha = jnp.clip(-state["prev_gate_x"] / gate_dx, 0.0, 1.0)
    plane_yz = state["prev_gate_local"][:, 1:3] + plane_alpha[:, None] * (
        gate_local[:, 1:3] - state["prev_gate_local"][:, 1:3]
    )
    gate_plane_dist = jnp.linalg.norm(plane_yz, axis=-1)
    missed_gate = crossed_gate_plane & (gate_plane_dist > 0.25)
    prev_gate_local = gate_frame_pos_for_gate(observations, state["prev_target_gate"])
    back_dx = jnp.maximum(prev_gate_local[:, 0] - state["prev_gate_x"], 1e-6)
    back_alpha = jnp.clip((gate_stage_offset - state["prev_gate_x"]) / back_dx, 0.0, 1.0)
    back_yz = state["prev_gate_local"][:, 1:3] + back_alpha[:, None] * (
        prev_gate_local[:, 1:3] - state["prev_gate_local"][:, 1:3]
    )
    back_plane_dist = jnp.linalg.norm(back_yz, axis=-1)
    tracked_gate_local = gate_frame_pos_for_gate(observations, state["back_gate_idx"])
    tracked_back_dx = jnp.maximum(
        tracked_gate_local[:, 0] - state["prev_back_gate_local"][:, 0], 1e-6
    )
    tracked_back_alpha = jnp.clip(
        (gate_stage_offset - state["prev_back_gate_local"][:, 0]) / tracked_back_dx, 0.0, 1.0
    )
    tracked_back_yz = state["prev_back_gate_local"][:, 1:3] + tracked_back_alpha[:, None] * (
        tracked_gate_local[:, 1:3] - state["prev_back_gate_local"][:, 1:3]
    )
    tracked_back_plane_dist = jnp.linalg.norm(tracked_back_yz, axis=-1)
    gate_axis_progress = jnp.where(
        same_gate & ~finished, raw_axis_progress * centerline_weight, 0.0
    )
    stage_dist = gate_stage_distance(
        observations, state["gate_stage"], target_gate, gate_stage_offset
    )
    gate_stage_progress = jnp.where(
        same_gate & ~finished, jnp.clip(state["prev_stage_dist"] - stage_dist, -0.2, 0.2), 0.0
    )
    front_hit = (
        same_gate
        & (state["gate_stage"] == 0)
        & (state["prev_gate_x"] < -gate_stage_offset)
        & (gate_x >= -gate_stage_offset)
        & (centerline_dist < gate_stage_radius)
    )
    gate_pass_hit = passed_gate & (state["gate_stage"] == 1)
    back_hit_on_pass = (
        gate_pass_hit
        & (state["prev_gate_x"] < gate_stage_offset)
        & (prev_gate_local[:, 0] >= gate_stage_offset)
        & (back_plane_dist < gate_stage_radius)
    )
    back_hit_tracked = (
        state["back_gate_active"]
        & (state["prev_back_gate_local"][:, 0] < gate_stage_offset)
        & (tracked_gate_local[:, 0] >= gate_stage_offset)
        & (tracked_back_plane_dist < gate_stage_radius)
    )
    back_hit = back_hit_on_pass | back_hit_tracked
    start_back_tracking = gate_pass_hit & ~back_hit_on_pass & ~finished
    keep_back_tracking = state["back_gate_active"] & ~back_hit_tracked & ~finished
    new_back_gate_active = start_back_tracking | keep_back_tracking
    new_back_gate_idx = jnp.where(
        start_back_tracking, state["prev_target_gate"], state["back_gate_idx"]
    )
    new_back_gate_local = gate_frame_pos_for_gate(observations, new_back_gate_idx)
    new_prev_back_gate_local = jnp.where(
        new_back_gate_active[:, None], new_back_gate_local, jnp.zeros_like(new_back_gate_local)
    )
    wrong_side_gate = (
        same_gate
        & (gate_x > gate_stage_offset)
        & (centerline_dist > gate_stage_radius)
        & ~passed_gate
    )
    stage_after_front = jnp.where(front_hit, 1, state["gate_stage"])
    stage_after_pass = jnp.where(gate_pass_hit, 2, stage_after_front)
    new_gate_stage = jnp.where(target_changed | finished, 0, stage_after_pass)

    action_diff = actions - last_action
    cmd_tilt = action_tilt(actions, action_low, action_high)
    cmd_tilt_penalty = (cmd_tilt / (jnp.pi / 2.0)) ** 2
    act_penalty = actions[..., 2] ** 2 + actions[..., -1] ** 2
    smooth_penalty = (
        float(coefs.get("d_act_xy_coef", 0.05)) * jnp.sum(action_diff[..., :3] ** 2, axis=-1)
        + float(coefs.get("d_act_th_coef", 0.02)) * action_diff[..., -1] ** 2
    )
    tilt_value = tilt(observations["quat"])
    tilt_angle_value = tilt_angle(observations["quat"])
    tilt_limit_rad = float(np.deg2rad(float(coefs.get("tilt_limit_deg", 35.0))))
    tilt_excess = jnp.maximum(0.0, tilt_angle_value - tilt_limit_rad) ** 2
    obstacle_dist = closest_obstacle_distance(observations)
    obstacle_margin = float(coefs.get("obstacle_margin", 0.35))
    obstacle_penalty = jnp.maximum(0.0, obstacle_margin - obstacle_dist) ** 2
    obstacle_clearance_progress = jnp.clip(obstacle_dist - state["prev_obstacle_dist"], -0.1, 0.1)
    clearance_active = (
        jnp.minimum(obstacle_dist, state["prev_obstacle_dist"]) < 1.5 * obstacle_margin
    ) & ~(terminated | truncated)
    obstacle_clearance_progress = jnp.where(clearance_active, obstacle_clearance_progress, 0.0)

    components = {
        "progress": float(coefs.get("progress_coef", 10.0)) * progress,
        "gate_axis_progress": float(coefs.get("gate_axis_coef", 8.0)) * gate_axis_progress,
        "gate_stage_progress": float(coefs.get("gate_stage_coef", 5.0)) * gate_stage_progress,
        "gate_front": float(coefs.get("gate_front_bonus", 4.0)) * front_hit.astype(jnp.float32),
        "gate_back": float(coefs.get("gate_back_bonus", 4.0)) * back_hit.astype(jnp.float32),
        "near_gate": float(coefs.get("near_gate_coef", 0.0)) * near_gate,
        "gate_bonus": float(coefs.get("gate_bonus", 30.0)) * passed_gate.astype(jnp.float32),
        "finish_bonus": float(coefs.get("finish_bonus", 80.0)) * finished.astype(jnp.float32),
        "missed_gate": -float(coefs.get("missed_gate_penalty", 8.0))
        * missed_gate.astype(jnp.float32),
        "wrong_side": -float(coefs.get("wrong_side_penalty", 6.0))
        * wrong_side_gate.astype(jnp.float32),
        "crash": -float(coefs.get("crash_penalty", 50.0)) * crashed.astype(jnp.float32),
        "action": -float(coefs.get("act_coef", 0.005)) * act_penalty,
        "cmd_tilt": -float(coefs.get("cmd_tilt_coef", 1.0)) * cmd_tilt_penalty,
        "smooth": -smooth_penalty,
        "tilt": -float(coefs.get("rpy_coef", 1.0)) * tilt_value,
        "tilt_excess": -float(coefs.get("tilt_excess_coef", 10.0)) * tilt_excess,
        "obstacle": -float(coefs.get("obstacle_coef", 1.5)) * obstacle_penalty,
        "obstacle_clearance": float(coefs.get("obstacle_clearance_coef", 0.0))
        * obstacle_clearance_progress,
        "timeout": -float(coefs.get("timeout_penalty", 0.0)) * timed_out.astype(jnp.float32),
        "time": -float(coefs.get("time_penalty", 0.05)) * jnp.ones_like(gate_dist),
    }
    reward = sum(components.values())
    metrics = {
        "gate_distance": gate_dist,
        "passed_gate_rate": passed_gate.astype(jnp.float32),
        "finished_rate": finished.astype(jnp.float32),
        "crashed_rate": crashed.astype(jnp.float32),
        "done_rate": (terminated | truncated).astype(jnp.float32),
        "target_gate": jnp.maximum(target_gate, 0).astype(jnp.float32),
        "gate_axis_x": gate_x,
        "gate_centerline_dist": centerline_dist,
        "gate_plane_dist": gate_plane_dist,
        "gate_plane_cross_rate": crossed_gate_plane.astype(jnp.float32),
        "missed_gate_rate": missed_gate.astype(jnp.float32),
        "gate_stage": state["gate_stage"].astype(jnp.float32),
        "gate_front_hit_rate": front_hit.astype(jnp.float32),
        "gate_pass_hit_rate": gate_pass_hit.astype(jnp.float32),
        "gate_back_hit_rate": back_hit.astype(jnp.float32),
        "wrong_side_gate_rate": wrong_side_gate.astype(jnp.float32),
        "timeout_rate": timed_out.astype(jnp.float32),
        "obstacle_distance": obstacle_dist,
        "obstacle_clearance_progress": obstacle_clearance_progress,
        "tilt_angle_deg": jnp.rad2deg(tilt_angle_value),
        "cmd_tilt_deg": jnp.rad2deg(cmd_tilt),
    }
    new_state = {
        "prev_gate_dist": gate_distance(observations),
        "prev_gate_x": gate_frame_pos(observations)[:, 0],
        "prev_gate_local": gate_frame_pos(observations),
        "prev_target_gate": target_gate.astype(jnp.int32),
        "gate_stage": new_gate_stage.astype(jnp.int32),
        "prev_stage_dist": gate_stage_distance(
            observations, new_gate_stage, target_gate, gate_stage_offset
        ),
        "back_gate_active": new_back_gate_active,
        "back_gate_idx": new_back_gate_idx.astype(jnp.int32),
        "prev_back_gate_local": new_prev_back_gate_local,
        "prev_obstacle_dist": closest_obstacle_distance(observations),
    }
    return reward.astype(jnp.float32), components, metrics, new_state


def make_fast_base_env(
    *, level: str, config: str, num_envs: int, jax_device: str
) -> tuple[Any, Any, Array, Array]:
    """Create an unwrapped vector env suitable for low-level pure-JAX stepping."""
    del level
    cfg = load_config(ROOT / "config" / config)
    cfg.sim.render = False
    disturbances = cfg.env.get("disturbances")
    if disturbances is not None:
        disturbances = dict(disturbances)
        disturbances.pop("thrust", None)
    if cfg.env.control_mode != "attitude":
        raise ValueError("Fast JAX PPO currently expects env.control_mode = 'attitude'.")
    env = gym.make_vec(
        cfg.env.id,
        num_envs=num_envs,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode,
        track=cfg.env.track,
        disturbances=disturbances,
        randomizations=cfg.env.get("randomizations"),
        seed=cfg.env.seed,
        device=jax_device,
    )
    action_low = jnp.asarray(env.single_action_space.low, dtype=jnp.float32)
    action_high = jnp.asarray(env.single_action_space.high, dtype=jnp.float32)
    return env, cfg, action_low, action_high


def make_initial_state(
    env: Any, raw_obs: dict[str, Array], args: JaxPPOArgs, action_low: Array, action_high: Array
) -> FastState:
    """Create the Brax-style state used by fast rollouts."""
    del action_low, action_high
    kind = observation_kind(args.level)
    history_row = basic_history(kind, raw_obs)
    history = jnp.repeat(history_row[:, None, :], int(args.n_obs), axis=1)
    last_action = jnp.zeros((raw_obs["pos"].shape[0], 4), dtype=jnp.float32)
    obs = flatten_observation(kind, raw_obs, history, last_action)
    zero_metric = jnp.array(0.0, dtype=jnp.float32)
    return FastState(
        pipeline_state=env.data,
        obs=obs.astype(jnp.float32),
        reward=jnp.zeros((raw_obs["pos"].shape[0],), dtype=jnp.float32),
        done=jnp.zeros((raw_obs["pos"].shape[0],), dtype=jnp.float32),
        metrics={"reward_mean": zero_metric, "done_mean": zero_metric},
        info={
            "raw_obs": raw_obs,
            "history": history,
            "last_action_norm": last_action,
            "reward_state": initial_reward_state(raw_obs, reward_coefs(args)),
        },
    )


def build_fast_env_step(
    step_fn: Any, action_low: Array, action_high: Array, args: JaxPPOArgs
) -> Any:
    """Build one pure-JAX racing env step."""
    kind = observation_kind(args.level)
    coefs = reward_coefs(args)

    def env_step(
        state: FastState, action_norm: Array
    ) -> tuple[FastState, dict[str, Array]]:
        sim_action = scale_action_jax(action_norm, action_low, action_high)
        next_data, (raw_obs_full, _sparse_reward, terminated_full, truncated_full, _info) = step_fn(
            state.pipeline_state, sim_action
        )
        raw_obs = drop_drone_dim(raw_obs_full)
        terminated = terminated_full[:, 0]
        truncated = truncated_full[:, 0]
        done = terminated | truncated
        reward, components, race_metrics, reward_state = reward_components(
            raw_obs,
            action_norm,
            terminated,
            truncated,
            state.info["last_action_norm"],
            state.info["reward_state"],
            coefs,
            action_low,
            action_high,
        )
        obs = flatten_observation(kind, raw_obs, state.info["history"], action_norm)
        history_row = basic_history(kind, raw_obs)
        history = jnp.concatenate(
            [state.info["history"][:, 1:, :], history_row[:, None, :]], axis=1
        )
        next_state = state._replace(
            pipeline_state=next_data,
            obs=obs.astype(jnp.float32),
            reward=reward,
            done=done.astype(jnp.float32),
            metrics={
                "reward_mean": jnp.mean(reward),
                "done_mean": jnp.mean(done.astype(jnp.float32)),
            },
            info={
                "raw_obs": raw_obs,
                "history": history,
                "last_action_norm": action_norm,
                "reward_state": reward_state,
            },
        )
        metrics = {
            "reward_mean": jnp.mean(reward),
            "done_mean": jnp.mean(done.astype(jnp.float32)),
            "obs_abs_mean": jnp.mean(jnp.abs(obs)),
            "action_abs_mean": jnp.mean(jnp.abs(action_norm)),
        }
        metrics |= {f"reward_{name}": jnp.mean(value) for name, value in components.items()}
        metrics |= {f"race_{name}": jnp.mean(value) for name, value in race_metrics.items()}
        metrics |= {
            "eval_finished": race_metrics["finished_rate"],
            "eval_crashed": race_metrics["crashed_rate"],
            "eval_timeout": race_metrics["timeout_rate"],
        }
        return next_state, metrics

    return env_step


def build_fast_rollout_fn(env_step: Any, *, num_steps: int) -> Any:
    """Build a stochastic PPO rollout compiled as one JAX scan."""

    def rollout_step(
        carry: tuple[FastState, Array], params: dict[str, Any]
    ) -> tuple[tuple[FastState, Array], dict[str, Array]]:
        state, key = carry
        key, action_key = jax.random.split(key)
        obs = state.obs
        mean, log_std, value = actor_critic_apply(params, obs)
        action = mean + jnp.exp(log_std) * jax.random.normal(
            action_key, mean.shape, dtype=jnp.float32
        )
        logprob = gaussian_logprob(action, mean, log_std)
        entropy = gaussian_entropy(mean, log_std)
        next_state, metrics = env_step(state, action)
        transition = {
            "obs": obs,
            "actions": action.astype(jnp.float32),
            "logprobs": logprob.astype(jnp.float32),
            "values": value.astype(jnp.float32),
            "rewards": next_state.reward,
            "dones": next_state.done,
            "entropy": jnp.mean(entropy),
        }
        transition |= metrics
        return (next_state, key), transition

    @jax.jit
    def rollout(
        state: FastState, params: dict[str, Any], key: Array
    ) -> tuple[FastState, Array, dict[str, Array]]:
        (next_state, next_key), transitions = jax.lax.scan(
            lambda carry, _: rollout_step(carry, params), (state, key), None, length=num_steps
        )
        return next_state, next_key, transitions

    return rollout


def build_fast_eval_fn(env_step: Any, *, max_steps: int) -> Any:
    """Build deterministic batched evaluation as one JAX scan."""

    def eval_step(
        carry: tuple[FastState, Array, Array, Array, Array, Array, Array, Array],
        _unused: None,
    ) -> tuple[
        tuple[FastState, Array, Array, Array, Array, Array, Array, Array], dict[str, Array]
    ]:
        state, params, done_seen, success_seen, crash_seen, timeout_seen, rewards, steps = carry
        mean, _log_std, _value = actor_critic_apply(params, state.obs)
        next_state, metrics = env_step(state, mean)
        active = ~done_seen
        done_now = next_state.done > 0.5
        success_now = metrics["eval_finished"] > 0.5
        crash_now = metrics["eval_crashed"] > 0.5
        timeout_now = metrics["eval_timeout"] > 0.5
        rewards = rewards + jnp.where(active, next_state.reward, 0.0)
        steps = steps + active.astype(jnp.int32)
        success_seen = success_seen | (active & success_now)
        crash_seen = crash_seen | (active & crash_now)
        timeout_seen = timeout_seen | (active & timeout_now)
        done_seen = done_seen | (active & done_now)
        return (
            next_state,
            params,
            done_seen,
            success_seen,
            crash_seen,
            timeout_seen,
            rewards,
            steps,
        ), {
            "done_mean": jnp.mean(done_seen.astype(jnp.float32)),
            "success_mean": jnp.mean(success_seen.astype(jnp.float32)),
        }

    @jax.jit
    def eval_rollout(
        state: FastState, params: dict[str, Any]
    ) -> tuple[FastState, dict[str, Array], dict[str, Array]]:
        num_envs = state.obs.shape[0]
        init = (
            state,
            params,
            jnp.zeros((num_envs,), dtype=bool),
            jnp.zeros((num_envs,), dtype=bool),
            jnp.zeros((num_envs,), dtype=bool),
            jnp.zeros((num_envs,), dtype=bool),
            jnp.zeros((num_envs,), dtype=jnp.float32),
            jnp.zeros((num_envs,), dtype=jnp.int32),
        )
        carry, metrics = jax.lax.scan(eval_step, init, None, length=max_steps)
        next_state, _params, done_seen, success_seen, crash_seen, timeout_seen, rewards, steps = (
            carry
        )
        results = {
            "done": done_seen,
            "success": success_seen,
            "crashed": crash_seen,
            "timeout": timeout_seen & ~success_seen,
            "reward": rewards,
            "steps": steps,
        }
        return next_state, results, metrics

    return eval_rollout


def rollout_metric_means(transitions: dict[str, Array]) -> dict[str, float]:
    """Summarize non-storage rollout metrics from a transition pytree."""
    means: dict[str, float] = {}
    for name, value in transitions.items():
        if name in ROLLOUT_KEYS:
            continue
        prefix = "rollout"
        key = name
        if name.startswith("reward_"):
            prefix = "reward_components"
            key = name.removeprefix("reward_")
        elif name.startswith("race_"):
            prefix = "race"
            key = name.removeprefix("race_")
        means[f"{prefix}/{key}"] = scalar_mean(value)
    return means


def train_ppo_fast(
    args: JaxPPOArgs,
    model_path: Path | None,
    *,
    wandb_enabled: bool = False,
    checkpoint_dir: Path | str | None = None,
    checkpoint_interval: int = 0,
    resume_from: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Train PPO with a pure-JAX scanned rollout."""
    set_seeds(args.seed)
    device = _resolve_device(args.jax_device)
    layout = observation_layout(args.level)
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_interval > 0 and checkpoint_dir is None:
        checkpoint_dir = (
            model_path.parent if model_path is not None else CONTROL_DIR / "checkpoints"
        )
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = (
        model_path.stem.removesuffix("_final")
        if model_path is not None
        else f"jax_fast_{args.level}"
    )
    next_checkpoint_step = checkpoint_interval if checkpoint_interval > 0 else None
    run = setup_wandb(args) if wandb_enabled else None

    print(
        {
            "backend": "pure_jax_brax_state_ppo",
            "device": str(device),
            "level": args.level,
            "config": args.config,
            "num_iterations": args.num_iterations,
            "batch_size": args.batch_size,
            "observation_layout": layout,
            "note": (
                "train-only Python wrappers are bypassed; "
                "env randomizations/disturbances remain active"
            ),
        }
    )
    history: list[dict[str, Any]] = []
    with jax.default_device(device):
        env, _cfg, action_low, action_high = make_fast_base_env(
            level=args.level, config=args.config, num_envs=args.num_envs, jax_device=args.jax_device
        )
        try:
            raw_obs_np, _info = env.reset(seed=args.seed)
            raw_obs = device_put_tree(raw_obs_np, device)
            kind = observation_kind(args.level)
            n_gates = int(raw_obs["gates_pos"].shape[1])
            n_obstacles = int(raw_obs["obstacles_pos"].shape[1])
            obs_dim = obs_dim_for_kind(
                kind, n_gates=n_gates, n_obstacles=n_obstacles, n_history=int(args.n_obs)
            )
            action_dim = int(np.prod(env.single_action_space.shape))
            optimizer = _optimizer(args)
            rng_key = jax.random.PRNGKey(args.seed)
            rng_key, init_key, update_key = jax.random.split(rng_key, 3)
            global_step = 0
            if resume_from is not None:
                params, opt_state, global_step, restored_key, payload = load_jax_checkpoint(
                    Path(resume_from), device
                )
                if payload["metadata"].get("observation_layout") != layout:
                    raise ValueError(
                        "Checkpoint layout "
                        f"{payload['metadata'].get('observation_layout')!r} != {layout!r}."
                    )
                if restored_key is not None:
                    rng_key = restored_key
                if opt_state is None:
                    opt_state = optimizer.init(params)
                print(f"resumed fast JAX checkpoint from {resume_from} at step={global_step}")
                if checkpoint_interval > 0:
                    next_checkpoint_step = (
                        (global_step // checkpoint_interval) + 1
                    ) * checkpoint_interval
            else:
                params = init_actor_critic_params(
                    init_key,
                    obs_dim=obs_dim,
                    hidden_dim=args.hidden_dim,
                    action_dim=action_dim,
                    args=args,
                )
                opt_state = optimizer.init(params)
            ppo_update = build_update_fn(optimizer, args)
            state = make_initial_state(env, raw_obs, args, action_low, action_high)
            env_step = build_fast_env_step(env._step, action_low, action_high, args)  # noqa: SLF001
            rollout = build_fast_rollout_fn(env_step, num_steps=args.num_steps)
            start_time = time.time()
            for iteration in range(1, args.num_iterations + 1):
                iter_start = time.time()
                state, rng_key, transitions = rollout(state, params, rng_key)
                batch, rollout_summary = compute_advantage_batch(
                    params, state.obs, transitions, float(args.gamma), float(args.gae_lambda)
                )
                update_key, step_update_key = jax.random.split(update_key)
                params, opt_state, update_key, train_metrics = ppo_update(
                    params, opt_state, batch, step_update_key
                )
                block_until_ready((state, transitions, rollout_summary, train_metrics))
                global_step += args.batch_size
                elapsed = time.time() - iter_start
                sps = int(args.batch_size / max(elapsed, 1e-9))
                metrics: dict[str, Any] = {
                    "global_step": int(global_step),
                    "charts/SPS": sps,
                    "charts/update_elapsed_s": elapsed,
                    "train/iteration": iteration,
                }
                metrics |= {
                    f"losses/{key}": scalar_mean(value) for key, value in train_metrics.items()
                }
                metrics |= {
                    f"rollout/{key}": scalar_mean(value) for key, value in rollout_summary.items()
                }
                metrics |= rollout_metric_means(transitions)
                history.append(metrics)
                if run is not None:
                    import wandb

                    wandb.log(metrics, step=global_step)
                if iteration % args.log_interval == 0 or iteration == args.num_iterations:
                    print(metrics)
                while next_checkpoint_step is not None and global_step >= next_checkpoint_step:
                    assert checkpoint_dir is not None
                    checkpoint_path = (
                        checkpoint_dir / f"{checkpoint_stem}_step_{next_checkpoint_step:09d}.pkl"
                    )
                    save_jax_checkpoint(
                        checkpoint_path,
                        args=args,
                        params=params,
                        opt_state=opt_state,
                        global_step=global_step,
                        metrics=metrics,
                        observation_layout=layout,
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        rng_key=rng_key,
                    )
                    print(f"checkpoint saved to {checkpoint_path} at global_step={global_step}")
                    next_checkpoint_step += checkpoint_interval
            final_metrics = (history[-1] if history else {"global_step": int(global_step)}) | {
                "train/total_elapsed_s": time.time() - start_time
            }
            if model_path is not None:
                save_jax_checkpoint(
                    model_path,
                    args=args,
                    params=params,
                    opt_state=opt_state,
                    global_step=global_step,
                    metrics=final_metrics,
                    observation_layout=layout,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    rng_key=rng_key,
                )
                print(f"model saved to {model_path}")
        finally:
            env.close()
            if run is not None:
                import wandb

                wandb.finish()
    return history


def evaluate_ppo_fast(
    args: JaxPPOArgs,
    *,
    n_eval: int,
    model_path: Path,
    seed_start: int | None = None,
    max_steps: int = 1500,
) -> dict[str, Any]:
    """Evaluate a fast JAX checkpoint in a batched deterministic rollout."""
    device = _resolve_device(args.jax_device)
    params, _opt_state, global_step, _rng_key, payload = load_jax_checkpoint(model_path, device)
    layout = observation_layout(args.level)
    metadata = payload["metadata"]
    if metadata.get("level") != args.level:
        raise ValueError(f"Checkpoint level {metadata.get('level')!r} != requested {args.level!r}.")
    if metadata.get("observation_layout") != layout:
        raise ValueError(f"Checkpoint layout {metadata.get('observation_layout')!r} != {layout!r}.")
    seed_start = args.seed if seed_start is None else seed_start
    with jax.default_device(device):
        env, _cfg, action_low, action_high = make_fast_base_env(
            level=args.level, config=args.config, num_envs=n_eval, jax_device=args.jax_device
        )
        try:
            raw_obs_np, _info = env.reset(seed=seed_start)
            raw_obs = device_put_tree(raw_obs_np, device)
            state = make_initial_state(env, raw_obs, args, action_low, action_high)
            env_step = build_fast_env_step(env._step, action_low, action_high, args)  # noqa: SLF001
            eval_rollout = build_fast_eval_fn(env_step, max_steps=max_steps)
            state, results, _metrics = eval_rollout(state, params)
            block_until_ready((state, results))
        finally:
            env.close()
    results_host = jax.device_get(results)
    rows = [
        {
            "episode": index + 1,
            "seed": seed_start + index,
            "success": bool(results_host["success"][index]),
            "crashed": bool(results_host["crashed"][index]),
            "timeout": bool(results_host["timeout"][index]),
            "reward": float(results_host["reward"][index]),
            "steps": int(results_host["steps"][index]),
        }
        for index in range(n_eval)
    ]
    successes = [row for row in rows if row["success"]]
    summary = {
        "model_path": str(model_path),
        "global_step": int(global_step),
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])) if rows else float("nan"),
        "crash_rate": float(np.mean([row["crashed"] for row in rows])) if rows else float("nan"),
        "timeout_rate": float(np.mean([row["timeout"] for row in rows])) if rows else float("nan"),
        "mean_reward": float(np.mean([row["reward"] for row in rows])) if rows else float("nan"),
        "mean_steps": float(np.mean([row["steps"] for row in rows])) if rows else float("nan"),
        "mean_success_steps": (
            float(np.mean([row["steps"] for row in successes])) if successes else float("nan")
        ),
        "episodes_detail": rows,
    }
    print(summary)
    return summary


def benchmark_fast_rollout(args: JaxPPOArgs, *, repeat: int = 5, warmup: int = 1) -> dict[str, Any]:
    """Benchmark scanned rollout speed for the fast path."""
    device = _resolve_device(args.jax_device)
    with jax.default_device(device):
        env, _cfg, action_low, action_high = make_fast_base_env(
            level=args.level, config=args.config, num_envs=args.num_envs, jax_device=args.jax_device
        )
        try:
            raw_obs_np, _info = env.reset(seed=args.seed)
            raw_obs = device_put_tree(raw_obs_np, device)
            kind = observation_kind(args.level)
            obs_dim = obs_dim_for_kind(
                kind,
                n_gates=int(raw_obs["gates_pos"].shape[1]),
                n_obstacles=int(raw_obs["obstacles_pos"].shape[1]),
                n_history=int(args.n_obs),
            )
            action_dim = int(np.prod(env.single_action_space.shape))
            key, init_key = jax.random.split(jax.random.PRNGKey(args.seed))
            params = init_actor_critic_params(
                init_key,
                obs_dim=obs_dim,
                hidden_dim=args.hidden_dim,
                action_dim=action_dim,
                args=args,
            )
            state = make_initial_state(env, raw_obs, args, action_low, action_high)
            env_step = build_fast_env_step(env._step, action_low, action_high, args)  # noqa: SLF001
            rollout = build_fast_rollout_fn(env_step, num_steps=args.num_steps)
            compile_start = time.perf_counter()
            state, key, transitions = rollout(state, params, key)
            block_until_ready((state, transitions))
            compile_elapsed = time.perf_counter() - compile_start
            for _ in range(warmup):
                state, key, transitions = rollout(state, params, key)
                block_until_ready((state, transitions))
            timed = []
            for _ in range(repeat):
                start = time.perf_counter()
                state, key, transitions = rollout(state, params, key)
                block_until_ready((state, transitions))
                timed.append(time.perf_counter() - start)
        finally:
            env.close()
    steps = int(args.num_envs) * int(args.num_steps)
    mean_elapsed = float(np.mean(timed))
    summary = {
        "backend": "pure_jax_brax_state_ppo_rollout",
        "level": args.level,
        "config": args.config,
        "device": str(device),
        "num_envs": int(args.num_envs),
        "num_steps": int(args.num_steps),
        "steps_per_rollout": steps,
        "compile_elapsed_s": compile_elapsed,
        "mean_elapsed_s": mean_elapsed,
        "median_elapsed_s": float(np.median(timed)),
        "mean_steps_per_second": steps / mean_elapsed,
        "timed_elapsed_s": timed,
    }
    print(summary)
    return summary


def _default_config_for_level(level: str) -> str:
    return "level3.toml" if level == "level3" else "level2_dr.toml"


def _default_model_for_level(level: str) -> Path:
    run_name = f"jax_fast_ppo_{level}"
    return CONTROL_DIR / "checkpoints" / run_name / f"{run_name}_final.pkl"


def _resolve_model_path(model_name: str | Path | None, level: str) -> Path:
    if model_name is None:
        return _default_model_for_level(level)
    path = Path(model_name)
    if not path.is_absolute():
        path = CONTROL_DIR / path
    return path


def main(
    level: str = "level3",
    config: str | None = None,
    wandb_enabled: bool = False,
    train: bool = True,
    eval: int = 0,
    benchmark: bool = False,
    benchmark_repeat: int = 5,
    benchmark_warmup: int = 1,
    model_name: str | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = 0,
    resume_from: str | None = None,
    seed_start: int | None = None,
    **overrides: Any,
) -> Any:
    """CLI entry point for fast JAX PPO training/evaluation."""
    args = JaxPPOArgs.create(
        level=level,
        config=config if config is not None else _default_config_for_level(level),
        **overrides,
    )
    model_path = _resolve_model_path(model_name, level)
    results: dict[str, Any] = {}
    if benchmark:
        results["benchmark"] = benchmark_fast_rollout(
            args, repeat=int(benchmark_repeat), warmup=int(benchmark_warmup)
        )
    if train:
        results["history"] = train_ppo_fast(
            args,
            model_path,
            wandb_enabled=wandb_enabled,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            resume_from=resume_from,
        )
    if eval:
        results["eval"] = evaluate_ppo_fast(
            args, n_eval=int(eval), model_path=model_path, seed_start=seed_start
        )
    return results or None


def level2_validation_args(**overrides: Any) -> JaxPPOArgs:
    """Return the known-good Level2 PPO preset for fast validation."""
    return level2_fast2_args(**overrides)


if __name__ == "__main__":
    fire.Fire(main)
