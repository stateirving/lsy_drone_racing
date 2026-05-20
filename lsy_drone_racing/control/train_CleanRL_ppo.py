"""CleanRL-style PPO training directly on the level2 drone racing task."""

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
import gymnasium as gym
import jax
import jax.numpy as jp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from gymnasium.vector import VectorEnv, VectorObservationWrapper, VectorRewardWrapper, VectorWrapper
from gymnasium.vector.utils import batch_space
from gymnasium.wrappers.vector.jax_to_torch import JaxToTorch
from jax import Array
from torch import Tensor
from torch.distributions.normal import Normal

import wandb
from lsy_drone_racing.utils import load_config


# region Arguments
@dataclass
class Args:
    """Class to store configurations."""

    config: str = "level2.toml"
    """race configuration file from config/"""
    seed: int = 42
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    jax_device: str = "gpu"
    """environment device"""
    wandb_project_name: str = "ADR-PPO-Racing"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    # Algorithm specific arguments
    total_timesteps: int = 2_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 256
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 5
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.26
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.02
    """coefficient of the entropy"""
    vf_coef: float = 0.7
    """coefficient of the value function"""
    max_grad_norm: float = 1.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.03
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # Wrapper settings
    n_obs: int = 2
    rpy_coef: float = 0.03
    d_act_th_coef: float = 0.02
    d_act_xy_coef: float = 0.05
    act_coef: float = 0.005
    progress_coef: float = 10.0
    gate_axis_coef: float = 8.0
    gate_stage_coef: float = 5.0
    gate_front_bonus: float = 4.0
    gate_plane_bonus: float = 8.0
    gate_back_bonus: float = 4.0
    gate_stage_offset: float = 0.35
    gate_stage_radius: float = 0.24
    wrong_side_penalty: float = 6.0
    near_gate_coef: float = 0.0
    gate_bonus: float = 30.0
    finish_bonus: float = 80.0
    missed_gate_penalty: float = 8.0
    crash_penalty: float = 50.0
    obstacle_coef: float = 1.5
    obstacle_margin: float = 0.35
    time_penalty: float = 0.05
    debug_obs: bool = False
    debug_reward_every: int = 0
    """reward coefficients for training"""

    @staticmethod
    def create(**kwargs: Any) -> "Args":
        """Create arguments class."""
        args = Args(**kwargs)
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size
        return args


# region Wrappers
class NormalizeVectorActions(VectorWrapper):
    """Normalize vector env actions to [-1, 1] and scale them to the sim action space."""

    def __init__(self, env: VectorEnv):
        """Initialize action scaling from the wrapped action space."""
        super().__init__(env)
        self.action_sim_low = np.asarray(env.single_action_space.low, dtype=np.float32)
        self.action_sim_high = np.asarray(env.single_action_space.high, dtype=np.float32)
        self._scale = jp.asarray((self.action_sim_high - self.action_sim_low) / 2.0)
        self._mean = jp.asarray((self.action_sim_high + self.action_sim_low) / 2.0)
        self.single_action_space = spaces.Box(
            low=-np.ones_like(self.action_sim_low),
            high=np.ones_like(self.action_sim_high),
            dtype=np.float32,
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def step(self, actions: Array) -> tuple[dict, Array, Array, Array, dict]:
        """Scale normalized actions before stepping the wrapped environment."""
        return self.env.step(self._scale_actions(actions, self._scale, self._mean))

    @staticmethod
    @jax.jit
    def _scale_actions(actions: Array, scale: Array, mean: Array) -> Array:
        return jp.clip(actions, -1.0, 1.0) * scale + mean


class Level2RaceReward(VectorRewardWrapper):
    """Dense reward for direct level2 gate racing.

    The base race environment only exposes a sparse completion signal, so PPO needs a shaped reward
    while debugging. This wrapper keeps the reward components small and prints them on demand.
    """

    def __init__(
        self,
        env: VectorEnv,
        *,
        progress_coef: float = 10.0,
        near_gate_coef: float = 0.0,
        gate_bonus: float = 30.0,
        finish_bonus: float = 80.0,
        crash_penalty: float = 50.0,
        rpy_coef: float = 0.03,
        act_coef: float = 0.005,
        d_act_th_coef: float = 0.02,
        d_act_xy_coef: float = 0.05,
        gate_axis_coef: float = 8.0,
        gate_stage_coef: float = 5.0,
        gate_front_bonus: float = 4.0,
        gate_plane_bonus: float = 8.0,
        gate_back_bonus: float = 4.0,
        gate_stage_offset: float = 0.35,
        gate_stage_radius: float = 0.24,
        wrong_side_penalty: float = 6.0,
        missed_gate_penalty: float = 8.0,
        obstacle_coef: float = 1.5,
        obstacle_margin: float = 0.35,
        time_penalty: float = 0.05,
        debug_every: int = 0,
    ):
        """Initialize reward shaping."""
        super().__init__(env)
        self.progress_coef = progress_coef
        self.near_gate_coef = near_gate_coef
        self.gate_bonus = gate_bonus
        self.finish_bonus = finish_bonus
        self.crash_penalty = crash_penalty
        self.rpy_coef = rpy_coef
        self.act_coef = act_coef
        self.d_act_th_coef = d_act_th_coef
        self.d_act_xy_coef = d_act_xy_coef
        self.gate_axis_coef = gate_axis_coef
        self.gate_stage_coef = gate_stage_coef
        self.gate_front_bonus = gate_front_bonus
        self.gate_plane_bonus = gate_plane_bonus
        self.gate_back_bonus = gate_back_bonus
        self.gate_stage_offset = gate_stage_offset
        self.gate_stage_radius = gate_stage_radius
        self.wrong_side_penalty = wrong_side_penalty
        self.missed_gate_penalty = missed_gate_penalty
        self.obstacle_coef = obstacle_coef
        self.obstacle_margin = obstacle_margin
        self.time_penalty = time_penalty
        self.debug_every = debug_every
        self._debug_step = 0
        self._last_action = jp.zeros((self.num_envs, 4), dtype=jp.float32)
        self._prev_gate_dist = jp.zeros((self.num_envs,), dtype=jp.float32)
        self._prev_gate_x = jp.zeros((self.num_envs,), dtype=jp.float32)
        self._prev_gate_local = jp.zeros((self.num_envs, 3), dtype=jp.float32)
        self._prev_target_gate = jp.zeros((self.num_envs,), dtype=jp.int32)
        self._gate_stage = jp.zeros((self.num_envs,), dtype=jp.int32)
        self._prev_stage_dist = jp.zeros((self.num_envs,), dtype=jp.float32)
        self._back_gate_active = jp.zeros((self.num_envs,), dtype=bool)
        self._back_gate_idx = jp.zeros((self.num_envs,), dtype=jp.int32)
        self._prev_back_gate_local = jp.zeros((self.num_envs, 3), dtype=jp.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset wrapper state from raw race observations."""
        observations, infos = self.env.reset(seed=seed, options=options)
        self._last_action = jp.zeros((self.num_envs, 4), dtype=jp.float32)
        self._prev_gate_dist = self._gate_distance(observations)
        self._prev_gate_local = self._gate_frame_pos(observations)
        self._prev_gate_x = self._prev_gate_local[:, 0]
        self._prev_target_gate = observations["target_gate"]
        self._gate_stage = jp.zeros((self.num_envs,), dtype=jp.int32)
        self._prev_stage_dist = self._gate_stage_distance(
            observations, self._gate_stage, observations["target_gate"]
        )
        self._back_gate_active = jp.zeros((self.num_envs,), dtype=bool)
        self._back_gate_idx = jp.zeros((self.num_envs,), dtype=jp.int32)
        self._prev_back_gate_local = jp.zeros((self.num_envs, 3), dtype=jp.float32)
        return observations, infos

    def step(self, actions: Array) -> tuple[dict, Array, Array, Array, dict]:
        """Apply action and replace sparse environment reward with shaped reward."""
        observations, _rewards, terminations, truncations, infos = self.env.step(actions)
        (
            reward,
            components,
            metrics,
            new_gate_stage,
            new_back_gate_active,
            new_back_gate_idx,
            new_prev_back_gate_local,
        ) = self._reward_components(observations, actions, terminations, truncations)
        self._prev_gate_dist = self._gate_distance(observations)
        self._prev_gate_local = self._gate_frame_pos(observations)
        self._prev_gate_x = self._prev_gate_local[:, 0]
        self._prev_target_gate = observations["target_gate"]
        self._gate_stage = new_gate_stage
        self._prev_stage_dist = self._gate_stage_distance(
            observations, self._gate_stage, observations["target_gate"]
        )
        self._back_gate_active = new_back_gate_active
        self._back_gate_idx = new_back_gate_idx
        self._prev_back_gate_local = new_prev_back_gate_local
        self._last_action = jp.asarray(actions)

        infos = dict(infos)
        infos.update({f"reward_{k}": v for k, v in components.items()})
        infos.update({f"race_{k}": v for k, v in metrics.items()})
        self._maybe_print_debug(components, reward, terminations, truncations, observations)
        return observations, reward, terminations, truncations, infos

    def _reward_components(
        self,
        observations: dict[str, Array],
        actions: Array,
        terminations: Array,
        truncations: Array,
    ) -> tuple[Array, dict[str, Array], dict[str, Array], Array, Array, Array, Array]:
        gate_dist = self._gate_distance(observations)
        target_gate = observations["target_gate"]
        finished = target_gate < 0
        passed_gate = (target_gate != self._prev_target_gate) & (self._prev_target_gate >= 0)
        target_changed = target_gate != self._prev_target_gate
        crashed = terminations & ~finished

        raw_progress = self._prev_gate_dist - gate_dist
        progress = jp.where(passed_gate | finished, 0.0, jp.clip(raw_progress, -0.25, 0.25))
        near_gate = jp.exp(-gate_dist)
        gate_local = self._gate_frame_pos(observations)
        gate_x = gate_local[:, 0]
        centerline_dist = jp.linalg.norm(gate_local[:, 1:3], axis=-1)
        centerline_weight = jp.exp(-10.0 * centerline_dist**2)
        same_gate = (target_gate == self._prev_target_gate) & (self._prev_target_gate >= 0)
        raw_axis_progress = jp.clip(gate_x - self._prev_gate_x, -0.1, 0.1)
        crossed_gate_plane = same_gate & (self._prev_gate_x < 0.0) & (gate_x > 0.0) & ~finished
        gate_dx = jp.maximum(gate_x - self._prev_gate_x, 1e-6)
        plane_alpha = jp.clip(-self._prev_gate_x / gate_dx, 0.0, 1.0)
        plane_yz = self._prev_gate_local[:, 1:3] + plane_alpha[:, None] * (
            gate_local[:, 1:3] - self._prev_gate_local[:, 1:3]
        )
        gate_plane_dist = jp.linalg.norm(plane_yz, axis=-1)
        missed_gate = crossed_gate_plane & (gate_plane_dist > 0.25)
        prev_gate_local = self._gate_frame_pos_for_gate(observations, self._prev_target_gate)
        back_dx = jp.maximum(prev_gate_local[:, 0] - self._prev_gate_x, 1e-6)
        back_alpha = jp.clip((self.gate_stage_offset - self._prev_gate_x) / back_dx, 0.0, 1.0)
        back_yz = self._prev_gate_local[:, 1:3] + back_alpha[:, None] * (
            prev_gate_local[:, 1:3] - self._prev_gate_local[:, 1:3]
        )
        back_plane_dist = jp.linalg.norm(back_yz, axis=-1)
        tracked_gate_local = self._gate_frame_pos_for_gate(observations, self._back_gate_idx)
        tracked_back_dx = jp.maximum(
            tracked_gate_local[:, 0] - self._prev_back_gate_local[:, 0], 1e-6
        )
        tracked_back_alpha = jp.clip(
            (self.gate_stage_offset - self._prev_back_gate_local[:, 0]) / tracked_back_dx,
            0.0,
            1.0,
        )
        tracked_back_yz = self._prev_back_gate_local[:, 1:3] + tracked_back_alpha[:, None] * (
            tracked_gate_local[:, 1:3] - self._prev_back_gate_local[:, 1:3]
        )
        tracked_back_plane_dist = jp.linalg.norm(tracked_back_yz, axis=-1)
        gate_axis_progress = jp.where(
            same_gate & ~finished,
            raw_axis_progress * centerline_weight,
            0.0,
        )
        stage_dist = self._gate_stage_distance(observations, self._gate_stage, target_gate)
        gate_stage_progress = jp.where(
            same_gate & ~finished,
            jp.clip(self._prev_stage_dist - stage_dist, -0.2, 0.2),
            0.0,
        )
        front_hit = (
            same_gate
            & (self._gate_stage == 0)
            & (self._prev_gate_x < -self.gate_stage_offset)
            & (gate_x >= -self.gate_stage_offset)
            & (centerline_dist < self.gate_stage_radius)
        )
        gate_pass_hit = passed_gate & (self._gate_stage == 1)
        back_hit_on_pass = (
            gate_pass_hit
            & (self._prev_gate_x < self.gate_stage_offset)
            & (prev_gate_local[:, 0] >= self.gate_stage_offset)
            & (back_plane_dist < self.gate_stage_radius)
        )
        back_hit_tracked = (
            self._back_gate_active
            & (self._prev_back_gate_local[:, 0] < self.gate_stage_offset)
            & (tracked_gate_local[:, 0] >= self.gate_stage_offset)
            & (tracked_back_plane_dist < self.gate_stage_radius)
        )
        back_hit = back_hit_on_pass | back_hit_tracked
        start_back_tracking = gate_pass_hit & ~back_hit_on_pass & ~finished
        keep_back_tracking = self._back_gate_active & ~back_hit_tracked & ~finished
        new_back_gate_active = start_back_tracking | keep_back_tracking
        new_back_gate_idx = jp.where(
            start_back_tracking, self._prev_target_gate, self._back_gate_idx
        )
        new_back_gate_local = self._gate_frame_pos_for_gate(observations, new_back_gate_idx)
        new_prev_back_gate_local = jp.where(
            new_back_gate_active[:, None],
            new_back_gate_local,
            jp.zeros_like(new_back_gate_local),
        )
        wrong_side_gate = (
            same_gate
            & (gate_x > self.gate_stage_offset)
            & (centerline_dist > self.gate_stage_radius)
            & ~passed_gate
        )
        stage_after_front = jp.where(front_hit, 1, self._gate_stage)
        stage_after_pass = jp.where(gate_pass_hit, 2, stage_after_front)
        new_gate_stage = jp.where(target_changed | finished, 0, stage_after_pass)

        action_diff = actions - self._last_action
        act_penalty = jp.sum(actions[..., :3] ** 2, axis=-1) + actions[..., -1] ** 2
        smooth_penalty = (
            self.d_act_xy_coef * jp.sum(action_diff[..., :3] ** 2, axis=-1)
            + self.d_act_th_coef * action_diff[..., -1] ** 2
        )
        tilt = self._tilt(observations["quat"])
        obstacle_penalty = self._obstacle_penalty(observations)

        components = {
            "progress": self.progress_coef * progress,
            "gate_axis_progress": self.gate_axis_coef * gate_axis_progress,
            "gate_stage_progress": self.gate_stage_coef * gate_stage_progress,
            "gate_front": self.gate_front_bonus * front_hit.astype(jp.float32),
            "gate_plane": self.gate_plane_bonus * passed_gate.astype(jp.float32),
            "gate_back": self.gate_back_bonus * back_hit.astype(jp.float32),
            "near_gate": self.near_gate_coef * near_gate,
            "gate_bonus": self.gate_bonus * passed_gate.astype(jp.float32),
            "finish_bonus": self.finish_bonus * finished.astype(jp.float32),
            "missed_gate": -self.missed_gate_penalty * missed_gate.astype(jp.float32),
            "wrong_side": -self.wrong_side_penalty * wrong_side_gate.astype(jp.float32),
            "crash": -self.crash_penalty * crashed.astype(jp.float32),
            "action": -self.act_coef * act_penalty,
            "smooth": -smooth_penalty,
            "tilt": -self.rpy_coef * tilt,
            "obstacle": -self.obstacle_coef * obstacle_penalty,
            "time": -self.time_penalty * jp.ones_like(gate_dist),
        }
        reward = sum(components.values())
        metrics = {
            "gate_distance": gate_dist,
            "passed_gate_rate": passed_gate.astype(jp.float32),
            "finished_rate": finished.astype(jp.float32),
            "crashed_rate": crashed.astype(jp.float32),
            "done_rate": (terminations | truncations).astype(jp.float32),
            "target_gate": jp.maximum(target_gate, 0).astype(jp.float32),
            "gate_axis_x": gate_x,
            "gate_centerline_dist": centerline_dist,
            "gate_plane_dist": gate_plane_dist,
            "gate_plane_cross_rate": crossed_gate_plane.astype(jp.float32),
            "missed_gate_rate": missed_gate.astype(jp.float32),
            "gate_stage": self._gate_stage.astype(jp.float32),
            "gate_front_hit_rate": front_hit.astype(jp.float32),
            "gate_pass_hit_rate": gate_pass_hit.astype(jp.float32),
            "gate_back_hit_rate": back_hit.astype(jp.float32),
            "wrong_side_gate_rate": wrong_side_gate.astype(jp.float32),
        }
        return (
            reward,
            components,
            metrics,
            new_gate_stage,
            new_back_gate_active,
            new_back_gate_idx,
            new_prev_back_gate_local,
        )

    def _gate_distance(self, observations: dict[str, Array]) -> Array:
        target_gate = observations["target_gate"]
        gates_pos = observations["gates_pos"]
        n_gates = gates_pos.shape[1]
        gate_idx = jp.mod(target_gate, n_gates)
        batch_idx = jp.arange(gates_pos.shape[0])
        target_pos = gates_pos[batch_idx, gate_idx]
        return jp.linalg.norm(target_pos - observations["pos"], axis=-1)

    def _gate_frame_pos(self, observations: dict[str, Array]) -> Array:
        return self._gate_frame_pos_for_gate(observations, observations["target_gate"])

    def _gate_frame_pos_for_gate(self, observations: dict[str, Array], gate_idx: Array) -> Array:
        gates_pos = observations["gates_pos"]
        n_gates = gates_pos.shape[1]
        gate_idx = jp.mod(gate_idx, n_gates)
        batch_idx = jp.arange(gates_pos.shape[0])
        gate_pos = gates_pos[batch_idx, gate_idx]
        gate_quat = observations["gates_quat"][batch_idx, gate_idx]
        gate_rot = RaceObservation.quat_to_rotmat(gate_quat)
        gate_rot_t = jp.swapaxes(gate_rot, -1, -2)
        return jp.einsum("nij,nj->ni", gate_rot_t, observations["pos"] - gate_pos)

    def _gate_stage_distance(
        self, observations: dict[str, Array], stage: Array, target_gate: Array
    ) -> Array:
        gate_local = self._gate_frame_pos_for_gate(observations, target_gate)
        zero = jp.zeros_like(gate_local[:, 0])
        stage_x = jp.where(
            stage == 0,
            -self.gate_stage_offset,
            jp.where(stage == 1, 0.0, self.gate_stage_offset),
        )
        stage_target = jp.stack([stage_x, zero, zero], axis=-1)
        return jp.linalg.norm(gate_local - stage_target, axis=-1)

    @staticmethod
    def _tilt(quat: Array) -> Array:
        rot = RaceObservation.quat_to_rotmat(quat)
        body_z_world_z = jp.clip(rot[..., 2, 2], -1.0, 1.0)
        return 1.0 - body_z_world_z

    def _obstacle_penalty(self, observations: dict[str, Array]) -> Array:
        dxy = observations["obstacles_pos"][..., :2] - observations["pos"][:, None, :2]
        closest_xy = jp.min(jp.linalg.norm(dxy, axis=-1), axis=-1)
        return jp.maximum(0.0, self.obstacle_margin - closest_xy) ** 2

    def _maybe_print_debug(
        self,
        components: dict[str, Array],
        reward: Array,
        terminations: Array,
        truncations: Array,
        observations: dict[str, Array],
    ) -> None:
        if self.debug_every <= 0:
            return
        self._debug_step += 1
        if self._debug_step % self.debug_every != 0:
            return
        means = {
            name: float(np.asarray(jp.mean(value)))
            for name, value in {"reward": reward, **components}.items()
        }
        done_count = int(np.asarray(jp.sum(terminations | truncations)))
        target_gate = np.asarray(observations["target_gate"])
        print(
            "[reward-debug] "
            f"step={self._debug_step} done={done_count}/{self.num_envs} "
            f"target_gate={np.bincount(np.maximum(target_gate, 0), minlength=4).tolist()} "
            + " ".join(f"{key}={value:.3f}" for key, value in means.items())
        )


class RaceObservation(VectorObservationWrapper):
    """Flatten level2 race observations into a PPO-friendly vector."""

    GATE_CORNERS_LOCAL = jp.array(
        [
            [0.0, -0.2, 0.2],
            [0.0, 0.2, 0.2],
            [0.0, 0.2, -0.2],
            [0.0, -0.2, -0.2],
        ],
        dtype=jp.float32,
    )
    HISTORY_DIM = 13

    def __init__(self, env: VectorEnv, n_history: int = 2, debug_obs: bool = False):
        """Initialize observation vector layout."""
        super().__init__(env)
        self.n_history = n_history
        self.debug_obs = debug_obs
        self._printed_obs_debug = False
        raw_space = env.single_observation_space
        self.n_gates = raw_space["gates_pos"].shape[0]
        self.n_obstacles = raw_space["obstacles_pos"].shape[0]
        self.action_dim = env.single_action_space.shape[0]
        self.layout = self._build_layout()
        self.obs_dim = self.layout[-1][1].stop  # 91dim
        self.single_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self._history = jp.zeros(
            (self.num_envs, self.n_history, self.HISTORY_DIM), dtype=jp.float32
        )
        self._last_action = jp.zeros((self.num_envs, self.action_dim), dtype=jp.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[Array, dict]:
        """Reset observation state."""
        observations, infos = self.env.reset(seed=seed, options=options)
        basic = self._basic_history(observations)
        self._history = jp.repeat(basic[:, None, :], self.n_history, axis=1)
        self._last_action = jp.zeros((self.num_envs, self.action_dim), dtype=jp.float32)
        return self.observations(observations), infos

    def step(self, actions: Array) -> tuple[Array, Array, Array, Array, dict]:
        """Update the last action before returning the next observation vector."""
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self._last_action = jp.asarray(actions)
        return self.observations(observations), rewards, terminations, truncations, infos

    def observations(self, observations: dict) -> Array:
        """Convert raw dict observations to a flat vector."""
        flat = self._flatten_observations(observations, self._history, self._last_action)
        self._maybe_print_debug(flat)
        if self.n_history > 0:
            basic = self._basic_history(observations)
            self._history = jp.concatenate([self._history[:, 1:, :], basic[:, None, :]], axis=1)
        return flat

    def _build_layout(self) -> list[tuple[str, slice]]:
        layout = []
        start = 0

        def add(name: str, size: int) -> None:
            nonlocal start
            layout.append((name, slice(start, start + size)))
            start += size

        add("pos_z", 1)
        add("vel_body", 3)
        add("ang_vel", 3)
        add("rotmat", 9)
        add("target_progress", 1)
        add("gate_current_corners_body", 12)
        add("gate_next_corners_body", 12)
        add("obstacles_body", 3 * self.n_obstacles)
        add("gates_visited", self.n_gates)
        add("obstacles_visited", self.n_obstacles)
        add("last_action", self.action_dim)
        add("history", self.n_history * self.HISTORY_DIM)
        return layout

    def _flatten_observations(
        self, observations: dict[str, Array], history: Array, last_action: Array
    ) -> Array:
        pos = observations["pos"]
        quat = observations["quat"]
        vel = observations["vel"]
        ang_vel = observations["ang_vel"]
        rot = self.quat_to_rotmat(quat)
        rot_t = jp.swapaxes(rot, -1, -2)
        vel_body = jp.einsum("nij,nj->ni", rot_t, vel)
        gate_current = self._gate_corners_body(
            observations, observations["target_gate"], pos, rot_t
        )
        next_gate = jp.minimum(jp.maximum(observations["target_gate"], 0) + 1, self.n_gates - 1)
        gate_next = self._gate_corners_body(observations, next_gate, pos, rot_t)
        obstacles_body = jp.einsum(
            "nij,nkj->nki", rot_t, observations["obstacles_pos"] - pos[:, None, :]
        ).reshape(pos.shape[0], -1)
        target_progress = (
            jp.where(observations["target_gate"] < 0, self.n_gates, observations["target_gate"])
            / self.n_gates
        )[:, None]
        parts = [
            pos[:, 2:3],
            vel_body,
            ang_vel,
            rot.reshape(pos.shape[0], -1),
            target_progress.astype(jp.float32),
            gate_current,
            gate_next,
            obstacles_body,
            observations["gates_visited"].astype(jp.float32),
            observations["obstacles_visited"].astype(jp.float32),
            last_action,
            history.reshape(pos.shape[0], -1),
        ]
        return jp.concatenate(parts, axis=-1)

    def _gate_corners_body(
        self, observations: dict[str, Array], gate_idx: Array, pos: Array, rot_t: Array
    ) -> Array:
        gate_idx = jp.mod(gate_idx, self.n_gates)
        batch_idx = jp.arange(pos.shape[0])
        gate_pos = observations["gates_pos"][batch_idx, gate_idx]
        gate_quat = observations["gates_quat"][batch_idx, gate_idx]
        gate_rot = self.quat_to_rotmat(gate_quat)
        corners_world = gate_pos[:, None, :] + jp.einsum(
            "nij,kj->nki", gate_rot, self.GATE_CORNERS_LOCAL
        )
        corners_body = jp.einsum("nij,nkj->nki", rot_t, corners_world - pos[:, None, :])
        return corners_body.reshape(pos.shape[0], -1)

    @staticmethod
    def _basic_history(observations: dict[str, Array]) -> Array:
        return jp.concatenate(
            [
                observations["pos"],
                observations["quat"],
                observations["vel"],
                observations["ang_vel"],
            ],
            axis=-1,
        )

    @staticmethod
    def quat_to_rotmat(quat: Array) -> Array:
        """Convert xyzw quaternions to body-to-world rotation matrices."""
        x, y, z, w = jp.moveaxis(quat, -1, 0)
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        row0 = jp.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], axis=-1)
        row1 = jp.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], axis=-1)
        row2 = jp.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], axis=-1)
        return jp.stack([row0, row1, row2], axis=-2)

    def _maybe_print_debug(self, flat: Array) -> None:
        if not self.debug_obs or self._printed_obs_debug:
            return
        arr = np.asarray(flat)
        print(f"[obs-debug] obs_dim={self.obs_dim} num_envs={self.num_envs}")
        for name, slc in self.layout:
            chunk = arr[:, slc]
            print(
                f"[obs-debug] {name:<26} slice={slc.start:03d}:{slc.stop:03d} "
                f"mean={chunk.mean(): .3f} std={chunk.std(): .3f} "
                f"min={chunk.min(): .3f} max={chunk.max(): .3f}"
            )
        if not np.isfinite(arr).all():
            raise ValueError("Non-finite value found in flattened race observation")
        self._printed_obs_debug = True


def set_seeds(seed: int):
    """Seed everything."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


REWARD_COMPONENT_KEYS = (
    "progress",
    "gate_axis_progress",
    "gate_stage_progress",
    "gate_front",
    "gate_plane",
    "gate_back",
    "near_gate",
    "gate_bonus",
    "finish_bonus",
    "missed_gate",
    "wrong_side",
    "crash",
    "action",
    "smooth",
    "tilt",
    "obstacle",
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
)


def mean_scalar(value: Any) -> float:
    """Convert tensor-like values to a Python mean scalar for logging."""
    if isinstance(value, torch.Tensor):
        return value.float().mean().item()
    return float(np.asarray(value).mean())


def setup_wandb(args: Args) -> None:
    """Start or configure a W&B run with explicit metric names."""
    if wandb.run is None:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args))
    else:
        wandb.config.update(vars(args), allow_val_change=True)

    wandb.define_metric("global_step")
    for metric_pattern in (
        "charts/*",
        "losses/*",
        "reward_components/*",
        "race/*",
        "train/*",
    ):
        wandb.define_metric(metric_pattern, step_metric="global_step")


# region MakeEnvs
def make_envs(
    config: str = "level2.toml",
    num_envs: int = None,
    jax_device: str = "cpu",
    torch_device: torch.device = torch.device("cpu"),
    coefs: dict | None = None,
    debug_obs: bool = False,
    debug_reward_every: int = 0,
) -> VectorEnv:
    """Make direct level2 racing environments for PPO."""
    coefs = {} if coefs is None else coefs
    cfg = load_config(Path(__file__).parents[2] / "config" / config)
    cfg.sim.render = False
    if cfg.env.control_mode != "attitude":
        raise ValueError("Direct level2 PPO currently expects env.control_mode = 'attitude'.")
    env = gym.make_vec(
        cfg.env.id,
        num_envs=num_envs,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode,
        track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=cfg.env.seed,
        device=jax_device,
    )

    env = NormalizeVectorActions(env)
    env = Level2RaceReward(
        env,
        progress_coef=coefs.get("progress_coef", 10.0),
        near_gate_coef=coefs.get("near_gate_coef", 0.0),
        gate_bonus=coefs.get("gate_bonus", 30.0),
        finish_bonus=coefs.get("finish_bonus", 80.0),
        crash_penalty=coefs.get("crash_penalty", 50.0),
        rpy_coef=coefs.get("rpy_coef", 0.03),
        act_coef=coefs.get("act_coef", 0.005),
        d_act_th_coef=coefs.get("d_act_th_coef", 0.02),
        d_act_xy_coef=coefs.get("d_act_xy_coef", 0.05),
        gate_axis_coef=coefs.get("gate_axis_coef", 8.0),
        gate_stage_coef=coefs.get("gate_stage_coef", 5.0),
        gate_front_bonus=coefs.get("gate_front_bonus", 4.0),
        gate_plane_bonus=coefs.get("gate_plane_bonus", 8.0),
        gate_back_bonus=coefs.get("gate_back_bonus", 4.0),
        gate_stage_offset=coefs.get("gate_stage_offset", 0.35),
        gate_stage_radius=coefs.get("gate_stage_radius", 0.24),
        wrong_side_penalty=coefs.get("wrong_side_penalty", 6.0),
        missed_gate_penalty=coefs.get("missed_gate_penalty", 8.0),
        obstacle_coef=coefs.get("obstacle_coef", 1.5),
        obstacle_margin=coefs.get("obstacle_margin", 0.35),
        time_penalty=coefs.get("time_penalty", 0.05),
        debug_every=debug_reward_every,
    )
    env = RaceObservation(env, n_history=coefs.get("n_obs", 0), debug_obs=debug_obs)
    env = JaxToTorch(env, torch_device)
    return env


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize layer."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# region Agent
class Agent(nn.Module):
    """RL Agent."""

    def __init__(self, obs_shape: tuple, action_shape: tuple):
        """Init network structures."""
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(torch.tensor(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(torch.tensor(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, torch.tensor(action_shape).prod()), std=0.01),
            nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(
            torch.tensor([[-1.2, -1.2, -2.0, -0.7]], dtype=torch.float32)
        )

    def get_value(self, x: Tensor) -> Tensor:
        """Value estimation."""
        return self.critic(x)

    def get_action_and_value(
        self, x: Tensor, action: Tensor | None = None, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Action output."""
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        # During learning the agent explores the environment by sampling actions from a Normal
        # distribution. The standard deviation is a learnable parameter that should decrease during
        # training as the agent gets more confident in its actions.
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample() if not deterministic else action_mean
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


# region Train
def train_ppo(
    args: Args, model_path: Path, device: torch.device, jax_device: str, wandb_enabled: bool = False
) -> None:
    """Train.

    An implementation of PPO from cleanrl, see https://docs.cleanrl.dev/.
    """
    # train setup
    if wandb_enabled:
        setup_wandb(args)
    train_start_time = time.time()
    set_seeds(args.seed)  # TRY NOT TO MODIFY: seeding
    print("Training on device:", device, "| Environment device:", jax_device)

    # env setup
    r_coefs = {
        "n_obs": args.n_obs,
        "rpy_coef": args.rpy_coef,
        "d_act_xy_coef": args.d_act_xy_coef,
        "d_act_th_coef": args.d_act_th_coef,
        "act_coef": args.act_coef,
        "progress_coef": args.progress_coef,
        "gate_axis_coef": args.gate_axis_coef,
        "gate_stage_coef": args.gate_stage_coef,
        "gate_front_bonus": args.gate_front_bonus,
        "gate_plane_bonus": args.gate_plane_bonus,
        "gate_back_bonus": args.gate_back_bonus,
        "gate_stage_offset": args.gate_stage_offset,
        "gate_stage_radius": args.gate_stage_radius,
        "wrong_side_penalty": args.wrong_side_penalty,
        "near_gate_coef": args.near_gate_coef,
        "gate_bonus": args.gate_bonus,
        "finish_bonus": args.finish_bonus,
        "missed_gate_penalty": args.missed_gate_penalty,
        "crash_penalty": args.crash_penalty,
        "obstacle_coef": args.obstacle_coef,
        "obstacle_margin": args.obstacle_margin,
        "time_penalty": args.time_penalty,
    }
    envs = make_envs(
        config=args.config,
        num_envs=args.num_envs,
        jax_device=jax_device,
        torch_device=device,
        coefs=r_coefs,
        debug_obs=args.debug_obs,
        debug_reward_every=args.debug_reward_every,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    agent = Agent(envs.single_observation_space.shape, envs.single_action_space.shape).to(device)
    optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(
        device
    )
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(
        device
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    sum_rewards = torch.zeros((args.num_envs)).to(device)
    sum_rewards_hist = []

    for iteration in range(1, args.num_iterations + 1):
        start_time = time.time()
        reward_component_sums = dict.fromkeys(REWARD_COMPONENT_KEYS, 0.0)
        race_metric_sums = dict.fromkeys(RACE_METRIC_KEYS, 0.0)
        reward_component_batches = 0

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            # envs.render()
            rewards[step] = reward
            sum_rewards[next_done.bool()] = 0
            sum_rewards += reward
            next_done = terminations | truncations

            if wandb_enabled:
                reward_component_batches += 1
                for key in REWARD_COMPONENT_KEYS:
                    if (value := infos.get(f"reward_{key}")) is not None:
                        reward_component_sums[key] += mean_scalar(value)
                for key in RACE_METRIC_KEYS:
                    if (value := infos.get(f"race_{key}")) is not None:
                        race_metric_sums[key] += mean_scalar(value)

            if wandb_enabled and next_done.any():
                for r in sum_rewards[next_done.bool()]:
                    wandb.log(
                        {"global_step": global_step, "train/reward": r.item()},
                        step=global_step,
                    )
                    sum_rewards_hist.append(r.item())

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if wandb_enabled:
            reward_component_logs = {}
            race_metric_logs = {}
            if reward_component_batches > 0:
                reward_component_logs = {
                    f"reward_components/{key}": value / reward_component_batches
                    for key, value in reward_component_sums.items()
                }
                race_metric_logs = {
                    f"race/{key}": value / reward_component_batches
                    for key, value in race_metric_sums.items()
                }
                if args.gate_bonus:
                    reward_component_logs["reward_components/gate_bonus_rate"] = (
                        reward_component_logs["reward_components/gate_bonus"] / args.gate_bonus
                    )
            wandb.log(
                {
                    "global_step": global_step,
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    "charts/SPS": int(global_step / (time.time() - start_time)),
                    **reward_component_logs,
                    **race_metric_logs,
                },
                step=global_step,
            )
        end_time = time.time()
        print(f"Iter {iteration}/{args.num_iterations} took {end_time - start_time:.2f} seconds")
    train_end_time = time.time()
    print(f"Training for {global_step} steps took {train_end_time - train_start_time:.2f} seconds.")
    if model_path is not None:
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
    envs.close()

    return sum_rewards_hist


# region Evaluate
def evaluate_ppo(args: Args, n_eval: int, model_path: Path) -> tuple[float, float]:
    """Evaluate."""
    set_seeds(args.seed)
    device = torch.device("cpu")
    r_coefs = {
        "n_obs": args.n_obs,
        "rpy_coef": args.rpy_coef,
        "d_act_xy_coef": args.d_act_xy_coef,
        "d_act_th_coef": args.d_act_th_coef,
        "act_coef": args.act_coef,
        "progress_coef": args.progress_coef,
        "gate_axis_coef": args.gate_axis_coef,
        "gate_stage_coef": args.gate_stage_coef,
        "gate_front_bonus": args.gate_front_bonus,
        "gate_plane_bonus": args.gate_plane_bonus,
        "gate_back_bonus": args.gate_back_bonus,
        "gate_stage_offset": args.gate_stage_offset,
        "gate_stage_radius": args.gate_stage_radius,
        "wrong_side_penalty": args.wrong_side_penalty,
        "near_gate_coef": args.near_gate_coef,
        "gate_bonus": args.gate_bonus,
        "finish_bonus": args.finish_bonus,
        "missed_gate_penalty": args.missed_gate_penalty,
        "crash_penalty": args.crash_penalty,
        "obstacle_coef": args.obstacle_coef,
        "obstacle_margin": args.obstacle_margin,
        "time_penalty": args.time_penalty,
    }
    eval_env = make_envs(config=args.config, num_envs=1, coefs=r_coefs)
    agent = Agent(eval_env.single_observation_space.shape, eval_env.single_action_space.shape).to(
        device
    )
    agent.load_state_dict(torch.load(model_path, map_location=device))
    with torch.no_grad():
        episode_rewards = []
        episode_lengths = []
        ep_seed = args.seed
        # Evaluate the policy
        for episode in range(n_eval):
            obs, _ = eval_env.reset(seed=(ep_seed := ep_seed + 1))
            done = torch.zeros(1, dtype=bool, device=device)
            episode_reward = 0
            steps = 0
            while not done.any():
                act, _, _, _ = agent.get_action_and_value(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(act)
                eval_env.render()
                done = terminated | truncated
                episode_reward += reward[0].item()
                steps += 1
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {steps}")

        print(
            f"Average Reward = {np.mean(episode_rewards):.2f}, Length = {np.mean(episode_lengths)}"
        )
        eval_env.close()

        return episode_rewards, episode_lengths


def debug_rollout(args: Args, n_steps: int, device: torch.device, jax_device: str) -> None:
    """Run zero-action rollouts to inspect observation layout and reward components."""
    r_coefs = {
        "n_obs": args.n_obs,
        "rpy_coef": args.rpy_coef,
        "d_act_xy_coef": args.d_act_xy_coef,
        "d_act_th_coef": args.d_act_th_coef,
        "act_coef": args.act_coef,
        "progress_coef": args.progress_coef,
        "gate_axis_coef": args.gate_axis_coef,
        "gate_stage_coef": args.gate_stage_coef,
        "gate_front_bonus": args.gate_front_bonus,
        "gate_plane_bonus": args.gate_plane_bonus,
        "gate_back_bonus": args.gate_back_bonus,
        "gate_stage_offset": args.gate_stage_offset,
        "gate_stage_radius": args.gate_stage_radius,
        "wrong_side_penalty": args.wrong_side_penalty,
        "near_gate_coef": args.near_gate_coef,
        "gate_bonus": args.gate_bonus,
        "finish_bonus": args.finish_bonus,
        "missed_gate_penalty": args.missed_gate_penalty,
        "crash_penalty": args.crash_penalty,
        "obstacle_coef": args.obstacle_coef,
        "obstacle_margin": args.obstacle_margin,
        "time_penalty": args.time_penalty,
    }
    envs = make_envs(
        config=args.config,
        num_envs=args.num_envs,
        jax_device=jax_device,
        torch_device=device,
        coefs=r_coefs,
        debug_obs=True,
        debug_reward_every=1,
    )
    obs, _ = envs.reset(seed=args.seed)
    print(
        f"[debug-rollout] obs_shape={tuple(obs.shape)} "
        f"action_shape={envs.single_action_space.shape}"
    )
    action = torch.zeros((args.num_envs,) + envs.single_action_space.shape, device=device)
    for step in range(n_steps):
        obs, reward, terminated, truncated, _info = envs.step(action)
        done = terminated | truncated
        print(
            f"[debug-rollout] step={step + 1}/{n_steps} "
            f"reward_mean={reward.float().mean().item():.3f} "
            f"reward_min={reward.float().min().item():.3f} "
            f"reward_max={reward.float().max().item():.3f} "
            f"done={int(done.sum().item())}/{args.num_envs}"
        )
    envs.close()


# region Main
def main(
    config: str = "level2.toml",
    wandb_enabled: bool = False,
    train: bool = True,
    eval: int = 0,
    debug_steps: int = 0,
    total_timesteps: int = 2_000_000,
    num_envs: int = 256,
    num_steps: int = 32,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    update_epochs: int = 5,
    num_minibatches: int = 8,
    ent_coef: float = 0.02,
    target_kl: float = 0.03,
    cuda: bool = True,
    jax_device: str = "gpu",
    model_name: str = "ppo_level2_racing.ckpt",
    n_obs: int = 2,
    progress_coef: float = 10.0,
    gate_axis_coef: float = 8.0,
    gate_stage_coef: float = 5.0,
    gate_front_bonus: float = 4.0,
    gate_plane_bonus: float = 8.0,
    gate_back_bonus: float = 4.0,
    gate_stage_offset: float = 0.35,
    gate_stage_radius: float = 0.24,
    wrong_side_penalty: float = 6.0,
    near_gate_coef: float = 0.0,
    gate_bonus: float = 30.0,
    finish_bonus: float = 80.0,
    missed_gate_penalty: float = 8.0,
    crash_penalty: float = 50.0,
    obstacle_coef: float = 1.5,
    obstacle_margin: float = 0.35,
    time_penalty: float = 0.05,
    act_coef: float = 0.005,
    d_act_th_coef: float = 0.02,
    d_act_xy_coef: float = 0.05,
    rpy_coef: float = 0.03,
    debug_obs: bool = False,
    debug_reward_every: int = 0,
):
    """Main."""
    args = Args.create(
        config=config,
        total_timesteps=total_timesteps,
        num_envs=num_envs,
        num_steps=num_steps,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        update_epochs=update_epochs,
        num_minibatches=num_minibatches,
        ent_coef=ent_coef,
        target_kl=target_kl,
        cuda=cuda,
        jax_device=jax_device,
        n_obs=n_obs,
        progress_coef=progress_coef,
        gate_axis_coef=gate_axis_coef,
        gate_stage_coef=gate_stage_coef,
        gate_front_bonus=gate_front_bonus,
        gate_plane_bonus=gate_plane_bonus,
        gate_back_bonus=gate_back_bonus,
        gate_stage_offset=gate_stage_offset,
        gate_stage_radius=gate_stage_radius,
        wrong_side_penalty=wrong_side_penalty,
        near_gate_coef=near_gate_coef,
        gate_bonus=gate_bonus,
        finish_bonus=finish_bonus,
        missed_gate_penalty=missed_gate_penalty,
        crash_penalty=crash_penalty,
        obstacle_coef=obstacle_coef,
        obstacle_margin=obstacle_margin,
        time_penalty=time_penalty,
        act_coef=act_coef,
        d_act_th_coef=d_act_th_coef,
        d_act_xy_coef=d_act_xy_coef,
        rpy_coef=rpy_coef,
        debug_obs=debug_obs,
        debug_reward_every=debug_reward_every,
    )
    model_path = Path(__file__).parent / model_name
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if debug_steps > 0:
        debug_rollout(args, debug_steps, device, args.jax_device)

    if train:  # use "--train False" to skip training
        train_ppo(args, model_path, device, args.jax_device, wandb_enabled)

    if eval > 0:  # use "--eval <N>" to perform N evaluation episodes
        episode_rewards, episode_lengths = evaluate_ppo(args, eval, model_path)
        if wandb_enabled and train:
            wandb.log(
                {
                    "eval/mean_rewards": np.mean(episode_rewards),
                    "eval/mean_steps": np.mean(episode_lengths),
                }
            )
            wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
