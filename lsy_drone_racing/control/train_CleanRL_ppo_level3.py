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
from lsy_drone_racing.control.ppo_level3_observation import (
    LOCAL_OBSTACLE_OBSERVATION_LAYOUT,
    checkpoint_hidden_dim,
    make_checkpoint,
    unpack_checkpoint,
)
from lsy_drone_racing.utils import load_config


LEVEL3_OBSERVATION_LAYOUT = LOCAL_OBSTACLE_OBSERVATION_LAYOUT


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
    hidden_dim: int = 128
    """shared width of the two actor and critic hidden layers"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # Wrapper settings
    n_obs: int = 2
    rpy_coef: float = 1.0
    tilt_limit_deg: float = 35.0
    tilt_excess_coef: float = 10.0
    cmd_tilt_coef: float = 1.0
    d_act_th_coef: float = 0.02
    d_act_xy_coef: float = 0.05
    act_coef: float = 0.005
    progress_coef: float = 10.0
    gate_axis_coef: float = 8.0
    gate_stage_coef: float = 5.0
    gate_front_bonus: float = 4.0
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
    obstacle_clearance_coef: float = 0.0
    timeout_penalty: float = 0.0
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


class ThrustScaleBatterySag(VectorWrapper):
    """Apply train-only thrust calibration and battery sag to physical attitude actions."""

    def __init__(
        self,
        env: VectorEnv,
        config: dict | None,
        *,
        env_freq: int,
        seed: int | None = None,
    ):
        """Initialize per-episode thrust scale and sag state."""
        super().__init__(env)
        config = {} if config is None else config
        self.scale_min = float(config.get("scale_min", 1.0))
        self.scale_max = float(config.get("scale_max", 1.0))
        self.battery_sag_min = float(config.get("battery_sag_min", 0.0))
        self.battery_sag_max = float(config.get("battery_sag_max", 0.0))
        horizon_s = float(config.get("battery_sag_horizon_s", 10.0))
        if self.scale_min <= 0.0 or self.scale_max <= 0.0:
            raise ValueError("thrust scale_min/scale_max must be positive.")
        if self.scale_min > self.scale_max:
            raise ValueError("thrust scale_min must be <= scale_max.")
        if self.battery_sag_min < 0.0 or self.battery_sag_max < 0.0:
            raise ValueError("battery sag bounds must be non-negative.")
        if self.battery_sag_min > self.battery_sag_max:
            raise ValueError("battery_sag_min must be <= battery_sag_max.")
        if horizon_s <= 0.0:
            raise ValueError("battery_sag_horizon_s must be positive.")

        self.single_action_space = env.single_action_space
        self.action_space = env.action_space
        self._state_shape = (self.num_envs,) + tuple(self.single_action_space.shape[:-1])
        self._mask_shape = (self.num_envs,) + (1,) * (len(self._state_shape) - 1)
        self._sag_horizon_steps = float(env_freq) * horizon_s
        rng_seed = random.randrange(2**31) if seed is None or seed == -1 else int(seed)
        self._rng_key = jax.random.PRNGKey(rng_seed)
        self._steps = jp.zeros((self.num_envs,), dtype=jp.int32)
        self._thrust_scale = jp.ones(self._state_shape, dtype=jp.float32)
        self._battery_sag = jp.zeros(self._state_shape, dtype=jp.float32)
        self._sample_episode_params(jp.ones((self.num_envs,), dtype=bool))

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset wrapper state and sample per-episode thrust parameters."""
        observations, infos = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self._rng_key = jax.random.PRNGKey(int(seed))
        self._steps = jp.zeros((self.num_envs,), dtype=jp.int32)
        self._sample_episode_params(jp.ones((self.num_envs,), dtype=bool))
        return observations, infos

    def step(self, actions: Array) -> tuple[dict, Array, Array, Array, dict]:
        """Scale thrust before stepping the wrapped simulation environment."""
        observations, rewards, terminations, truncations, infos = self.env.step(
            self._apply_thrust_sag(actions)
        )
        done = jp.asarray(terminations) | jp.asarray(truncations)
        if done.ndim > 1:
            done = jp.any(done, axis=tuple(range(1, done.ndim)))
        self._steps = jp.where(done, 0, self._steps + 1)
        self._sample_episode_params(done)
        return observations, rewards, terminations, truncations, infos

    def _sample_episode_params(self, mask: Array) -> None:
        """Sample fixed thrust parameters for newly reset vector slots."""
        self._rng_key, scale_key, sag_key = jax.random.split(self._rng_key, 3)
        thrust_scale = jax.random.uniform(
            scale_key,
            self._state_shape,
            dtype=jp.float32,
            minval=self.scale_min,
            maxval=self.scale_max,
        )
        battery_sag = jax.random.uniform(
            sag_key,
            self._state_shape,
            dtype=jp.float32,
            minval=self.battery_sag_min,
            maxval=self.battery_sag_max,
        )
        mask = jp.asarray(mask, dtype=bool).reshape(self._mask_shape)
        self._thrust_scale = jp.where(mask, thrust_scale, self._thrust_scale)
        self._battery_sag = jp.where(mask, battery_sag, self._battery_sag)

    def _apply_thrust_sag(self, actions: Array) -> Array:
        """Apply thrust scale minus linear battery sag to the thrust command."""
        actions = jp.asarray(actions)
        progress = jp.clip(self._steps.astype(jp.float32) / self._sag_horizon_steps, 0.0, 1.0)
        progress = progress.reshape(self._mask_shape)
        thrust_scale = jp.maximum(self._thrust_scale - self._battery_sag * progress, 0.0)
        return actions.at[..., -1].set(actions[..., -1] * thrust_scale)


class ActionLatencyResponseLag(VectorWrapper):
    """Apply train-only command delay and first-order attitude/thrust response lag."""

    def __init__(
        self,
        env: VectorEnv,
        latency_config: dict | None,
        response_config: dict | None,
        *,
        env_freq: int,
        seed: int | None = None,
    ):
        """Initialize per-episode action delay and response parameters."""
        super().__init__(env)
        latency_config = {} if latency_config is None else latency_config
        response_config = {} if response_config is None else response_config
        self.delay_min_steps = int(latency_config.get("delay_min_steps", 0))
        self.delay_max_steps = int(latency_config.get("delay_max_steps", 0))
        self.rp_tau_min_s = float(
            response_config.get("rp_tau_min_s", response_config.get("rpy_tau_min_s", 0.0))
        )
        self.rp_tau_max_s = float(
            response_config.get("rp_tau_max_s", response_config.get("rpy_tau_max_s", 0.0))
        )
        self.yaw_tau_min_s = float(response_config.get("yaw_tau_min_s", self.rp_tau_min_s))
        self.yaw_tau_max_s = float(response_config.get("yaw_tau_max_s", self.rp_tau_max_s))
        self.thrust_tau_min_s = float(response_config.get("thrust_tau_min_s", 0.0))
        self.thrust_tau_max_s = float(response_config.get("thrust_tau_max_s", 0.0))
        if self.delay_min_steps < 0 or self.delay_max_steps < 0:
            raise ValueError("action delay bounds must be non-negative.")
        if self.delay_min_steps > self.delay_max_steps:
            raise ValueError("action delay_min_steps must be <= delay_max_steps.")
        for name, min_s, max_s in (
            ("rp_tau", self.rp_tau_min_s, self.rp_tau_max_s),
            ("yaw_tau", self.yaw_tau_min_s, self.yaw_tau_max_s),
            ("thrust_tau", self.thrust_tau_min_s, self.thrust_tau_max_s),
        ):
            if min_s < 0.0 or max_s < 0.0:
                raise ValueError(f"{name} bounds must be non-negative.")
            if min_s > max_s:
                raise ValueError(f"{name}_min_s must be <= {name}_max_s.")

        self.single_action_space = env.single_action_space
        self.action_space = env.action_space
        if self.single_action_space.shape != (4,):
            raise ValueError("ActionLatencyResponseLag expects attitude action shape (4,).")
        self._dt = 1.0 / float(env_freq)
        self._action_shape = (self.num_envs,) + tuple(self.single_action_space.shape)
        self._delay_mask_shape = (self.num_envs,) + (1,) * len(self.single_action_space.shape)
        action_low = jp.asarray(self.single_action_space.low, dtype=jp.float32)
        action_high = jp.asarray(self.single_action_space.high, dtype=jp.float32)
        neutral = (action_low + action_high) / 2.0
        self._neutral_action = jp.broadcast_to(neutral, self._action_shape)
        rng_seed = random.randrange(2**31) if seed is None or seed == -1 else int(seed) + 10_003
        self._rng_key = jax.random.PRNGKey(rng_seed)
        self._delay_steps = jp.zeros((self.num_envs,), dtype=jp.int32)
        self._rp_tau = jp.zeros((self.num_envs,), dtype=jp.float32)
        self._yaw_tau = jp.zeros((self.num_envs,), dtype=jp.float32)
        self._thrust_tau = jp.zeros((self.num_envs,), dtype=jp.float32)
        self._applied_action = self._neutral_action
        self._action_buffer = self._initial_action_buffer()
        self._sample_episode_params(jp.ones((self.num_envs,), dtype=bool))

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset wrapper state and sample per-episode response parameters."""
        observations, infos = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self._rng_key = jax.random.PRNGKey(int(seed) + 10_003)
        self._applied_action = self._neutral_action
        self._action_buffer = self._initial_action_buffer()
        self._sample_episode_params(jp.ones((self.num_envs,), dtype=bool))
        return observations, infos

    def step(self, actions: Array) -> tuple[dict, Array, Array, Array, dict]:
        """Delay and low-pass-filter physical actions before stepping the simulator."""
        actions = jp.asarray(actions)
        self._action_buffer = jp.concatenate(
            [actions[:, None, ...], self._action_buffer[:, :-1, ...]], axis=1
        )
        delayed_action = self._action_buffer[jp.arange(self.num_envs), self._delay_steps]
        applied_action = self._apply_response_lag(delayed_action)
        observations, rewards, terminations, truncations, infos = self.env.step(applied_action)
        self._applied_action = applied_action
        done = self._done_mask(terminations, truncations)
        self._reset_episode_state(done)
        return observations, rewards, terminations, truncations, infos

    def _initial_action_buffer(self) -> Array:
        """Create a delay buffer initialized with neutral hover-like commands."""
        return jp.repeat(self._neutral_action[:, None, ...], self.delay_max_steps + 1, axis=1)

    def _sample_episode_params(self, mask: Array) -> None:
        """Sample fixed command delay and response lags for newly reset vector slots."""
        self._rng_key, delay_key, rp_key, yaw_key, thrust_key = jax.random.split(
            self._rng_key, 5
        )
        delay_steps = jax.random.randint(
            delay_key,
            (self.num_envs,),
            minval=self.delay_min_steps,
            maxval=self.delay_max_steps + 1,
            dtype=jp.int32,
        )
        rp_tau = jax.random.uniform(
            rp_key,
            (self.num_envs,),
            dtype=jp.float32,
            minval=self.rp_tau_min_s,
            maxval=self.rp_tau_max_s,
        )
        yaw_tau = jax.random.uniform(
            yaw_key,
            (self.num_envs,),
            dtype=jp.float32,
            minval=self.yaw_tau_min_s,
            maxval=self.yaw_tau_max_s,
        )
        thrust_tau = jax.random.uniform(
            thrust_key,
            (self.num_envs,),
            dtype=jp.float32,
            minval=self.thrust_tau_min_s,
            maxval=self.thrust_tau_max_s,
        )
        mask = jp.asarray(mask, dtype=bool)
        self._delay_steps = jp.where(mask, delay_steps, self._delay_steps)
        self._rp_tau = jp.where(mask, rp_tau, self._rp_tau)
        self._yaw_tau = jp.where(mask, yaw_tau, self._yaw_tau)
        self._thrust_tau = jp.where(mask, thrust_tau, self._thrust_tau)

    def _apply_response_lag(self, delayed_action: Array) -> Array:
        """Apply per-axis first-order response lag to delayed actions."""
        rp_alpha = self._tau_to_alpha(self._rp_tau)
        yaw_alpha = self._tau_to_alpha(self._yaw_tau)
        thrust_alpha = self._tau_to_alpha(self._thrust_tau)
        alpha = jp.stack([rp_alpha, rp_alpha, yaw_alpha, thrust_alpha], axis=-1)
        applied = self._applied_action + alpha * (delayed_action - self._applied_action)
        return jp.clip(applied, self.single_action_space.low, self.single_action_space.high)

    def _reset_episode_state(self, done: Array) -> None:
        """Reset action memory and sample new parameters for completed vector slots."""
        if done.ndim > 1:
            done = jp.any(done, axis=tuple(range(1, done.ndim)))
        mask = done.reshape(self._delay_mask_shape)
        neutral_buffer = self._initial_action_buffer()
        self._applied_action = jp.where(mask, self._neutral_action, self._applied_action)
        buffer_mask = done.reshape((self.num_envs,) + (1,) * (self._action_buffer.ndim - 1))
        self._action_buffer = jp.where(buffer_mask, neutral_buffer, self._action_buffer)
        self._sample_episode_params(done)

    def _tau_to_alpha(self, tau_s: Array) -> Array:
        """Convert first-order time constants to discrete-time filter coefficients."""
        return jp.where(tau_s <= 0.0, 1.0, 1.0 - jp.exp(-self._dt / tau_s))

    @staticmethod
    def _done_mask(terminations: Array, truncations: Array) -> Array:
        done = jp.asarray(terminations) | jp.asarray(truncations)
        if done.ndim > 1:
            done = jp.any(done, axis=tuple(range(1, done.ndim)))
        return done


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
        rpy_coef: float = 1.0,
        tilt_limit_deg: float = 35.0,
        tilt_excess_coef: float = 10.0,
        cmd_tilt_coef: float = 1.0,
        act_coef: float = 0.005,
        d_act_th_coef: float = 0.02,
        d_act_xy_coef: float = 0.05,
        gate_axis_coef: float = 8.0,
        gate_stage_coef: float = 5.0,
        gate_front_bonus: float = 4.0,
        gate_back_bonus: float = 4.0,
        gate_stage_offset: float = 0.35,
        gate_stage_radius: float = 0.24,
        wrong_side_penalty: float = 6.0,
        missed_gate_penalty: float = 8.0,
        obstacle_coef: float = 1.5,
        obstacle_margin: float = 0.35,
        obstacle_clearance_coef: float = 0.0,
        timeout_penalty: float = 0.0,
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
        self.tilt_limit_rad = float(np.deg2rad(tilt_limit_deg))
        self.tilt_excess_coef = tilt_excess_coef
        self.cmd_tilt_coef = cmd_tilt_coef
        self.act_coef = act_coef
        self.d_act_th_coef = d_act_th_coef
        self.d_act_xy_coef = d_act_xy_coef
        self.gate_axis_coef = gate_axis_coef
        self.gate_stage_coef = gate_stage_coef
        self.gate_front_bonus = gate_front_bonus
        self.gate_back_bonus = gate_back_bonus
        self.gate_stage_offset = gate_stage_offset
        self.gate_stage_radius = gate_stage_radius
        self.wrong_side_penalty = wrong_side_penalty
        self.missed_gate_penalty = missed_gate_penalty
        self.obstacle_coef = obstacle_coef
        self.obstacle_margin = obstacle_margin
        self.obstacle_clearance_coef = obstacle_clearance_coef
        self.timeout_penalty = timeout_penalty
        self.time_penalty = time_penalty
        self.debug_every = debug_every
        action_sim_low = np.asarray(getattr(env, "action_sim_low"), dtype=np.float32)
        action_sim_high = np.asarray(getattr(env, "action_sim_high"), dtype=np.float32)
        self._action_scale = jp.asarray((action_sim_high - action_sim_low) / 2.0)
        self._action_mean = jp.asarray((action_sim_high + action_sim_low) / 2.0)
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
        self._prev_obstacle_dist = jp.zeros((self.num_envs,), dtype=jp.float32)

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
        self._prev_obstacle_dist = self._closest_obstacle_distance(observations)
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
        self._prev_obstacle_dist = self._closest_obstacle_distance(observations)
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
        timed_out = truncations & ~finished

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
        cmd_tilt = self._action_tilt(actions)
        cmd_tilt_penalty = (cmd_tilt / (jp.pi / 2.0)) ** 2
        act_penalty = actions[..., 2] ** 2 + actions[..., -1] ** 2
        smooth_penalty = (
            self.d_act_xy_coef * jp.sum(action_diff[..., :3] ** 2, axis=-1)
            + self.d_act_th_coef * action_diff[..., -1] ** 2
        )
        tilt = self._tilt(observations["quat"])
        tilt_angle = self._tilt_angle(observations["quat"])
        tilt_excess = jp.maximum(0.0, tilt_angle - self.tilt_limit_rad) ** 2
        obstacle_dist = self._closest_obstacle_distance(observations)
        obstacle_penalty = jp.maximum(0.0, self.obstacle_margin - obstacle_dist) ** 2
        obstacle_clearance_progress = jp.clip(
            obstacle_dist - self._prev_obstacle_dist, -0.1, 0.1
        )
        clearance_active = (
            jp.minimum(obstacle_dist, self._prev_obstacle_dist) < 1.5 * self.obstacle_margin
        ) & ~(terminations | truncations)
        obstacle_clearance_progress = jp.where(
            clearance_active, obstacle_clearance_progress, 0.0
        )

        components = {
            "progress": self.progress_coef * progress,
            "gate_axis_progress": self.gate_axis_coef * gate_axis_progress,
            "gate_stage_progress": self.gate_stage_coef * gate_stage_progress,
            "gate_front": self.gate_front_bonus * front_hit.astype(jp.float32),
            "gate_back": self.gate_back_bonus * back_hit.astype(jp.float32),
            "near_gate": self.near_gate_coef * near_gate,
            "gate_bonus": self.gate_bonus * passed_gate.astype(jp.float32),
            "finish_bonus": self.finish_bonus * finished.astype(jp.float32),
            "missed_gate": -self.missed_gate_penalty * missed_gate.astype(jp.float32),
            "wrong_side": -self.wrong_side_penalty * wrong_side_gate.astype(jp.float32),
            "crash": -self.crash_penalty * crashed.astype(jp.float32),
            "action": -self.act_coef * act_penalty,
            "cmd_tilt": -self.cmd_tilt_coef * cmd_tilt_penalty,
            "smooth": -smooth_penalty,
            "tilt": -self.rpy_coef * tilt,
            "tilt_excess": -self.tilt_excess_coef * tilt_excess,
            "obstacle": -self.obstacle_coef * obstacle_penalty,
            "obstacle_clearance": self.obstacle_clearance_coef * obstacle_clearance_progress,
            "timeout": -self.timeout_penalty * timed_out.astype(jp.float32),
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
            "timeout_rate": timed_out.astype(jp.float32),
            "obstacle_distance": obstacle_dist,
            "obstacle_clearance_progress": obstacle_clearance_progress,
            "tilt_angle_deg": jp.rad2deg(tilt_angle),
            "cmd_tilt_deg": jp.rad2deg(cmd_tilt),
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

    @staticmethod
    def _tilt_angle(quat: Array) -> Array:
        rot = RaceObservation.quat_to_rotmat(quat)
        body_z_world_z = jp.clip(rot[..., 2, 2], -1.0, 1.0)
        return jp.arccos(body_z_world_z)

    def _action_tilt(self, actions: Array) -> Array:
        scaled_actions = jp.clip(actions, -1.0, 1.0) * self._action_scale + self._action_mean
        roll_cmd = scaled_actions[..., 0]
        pitch_cmd = scaled_actions[..., 1]
        body_z_world_z = jp.clip(jp.cos(roll_cmd) * jp.cos(pitch_cmd), -1.0, 1.0)
        return jp.arccos(body_z_world_z)

    @staticmethod
    def _closest_obstacle_distance(observations: dict[str, Array]) -> Array:
        dxy = observations["obstacles_pos"][..., :2] - observations["pos"][:, None, :2]
        return jp.min(jp.linalg.norm(dxy, axis=-1), axis=-1)

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


class ObservationLatencyNoise(VectorObservationWrapper):
    """Apply train-only observation delay and sensor/object measurement noise."""

    def __init__(
        self,
        env: VectorEnv,
        latency_config: dict | None,
        noise_config: dict | None,
        *,
        seed: int | None = None,
    ):
        """Initialize observation delay buffers and noise parameters."""
        super().__init__(env)
        latency_config = {} if latency_config is None else latency_config
        noise_config = {} if noise_config is None else noise_config
        self.delay_min_steps = int(latency_config.get("delay_min_steps", 0))
        self.delay_max_steps = int(latency_config.get("delay_max_steps", 0))
        if self.delay_min_steps < 0 or self.delay_max_steps < 0:
            raise ValueError("observation delay bounds must be non-negative.")
        if self.delay_min_steps > self.delay_max_steps:
            raise ValueError("observation delay_min_steps must be <= delay_max_steps.")
        self.pos_std_m = float(noise_config.get("pos_std_m", 0.0))
        self.vel_std_mps = float(noise_config.get("vel_std_mps", 0.0))
        self.ang_vel_std_radps = float(noise_config.get("ang_vel_std_radps", 0.0))
        self.quat_rpy_std_rad = float(noise_config.get("quat_rpy_std_rad", 0.0))
        self.gate_pos_std_m = float(noise_config.get("gate_pos_std_m", 0.0))
        self.gate_rpy_std_rad = float(noise_config.get("gate_rpy_std_rad", 0.0))
        self.obstacle_pos_std_m = float(noise_config.get("obstacle_pos_std_m", 0.0))
        for name in (
            "pos_std_m",
            "vel_std_mps",
            "ang_vel_std_radps",
            "quat_rpy_std_rad",
            "gate_pos_std_m",
            "gate_rpy_std_rad",
            "obstacle_pos_std_m",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")

        self.single_observation_space = env.single_observation_space
        self.observation_space = env.observation_space
        rng_seed = random.randrange(2**31) if seed is None or seed == -1 else int(seed) + 20_003
        self._rng_key = jax.random.PRNGKey(rng_seed)
        self._delay_steps = jp.zeros((self.num_envs,), dtype=jp.int32)
        self._obs_buffer: dict[str, Array] | None = None

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset observation delay buffers and sample per-episode delay."""
        observations, infos = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self._rng_key = jax.random.PRNGKey(int(seed) + 20_003)
        observations = {key: jp.asarray(value) for key, value in observations.items()}
        self._obs_buffer = self._initial_obs_buffer(observations)
        self._sample_episode_params(jp.ones((self.num_envs,), dtype=bool))
        return self._add_noise(self._select_delayed_observations()), infos

    def step(self, actions: Array) -> tuple[dict, Array, Array, Array, dict]:
        """Return delayed and noisy observations after stepping the wrapped environment."""
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        observations = {key: jp.asarray(value) for key, value in observations.items()}
        if self._obs_buffer is None:
            self._obs_buffer = self._initial_obs_buffer(observations)
        else:
            self._obs_buffer = {
                key: jp.concatenate(
                    [value[:, None, ...], self._obs_buffer[key][:, :-1, ...]], axis=1
                )
                for key, value in observations.items()
            }
        delayed_observations = self._select_delayed_observations()
        noisy_observations = self._add_noise(delayed_observations)
        done = self._done_mask(terminations, truncations)
        self._reset_episode_state(done, observations)
        return noisy_observations, rewards, terminations, truncations, infos

    def _initial_obs_buffer(self, observations: dict[str, Array]) -> dict[str, Array]:
        """Create delay buffers initialized with the current observations."""
        return {
            key: jp.repeat(value[:, None, ...], self.delay_max_steps + 1, axis=1)
            for key, value in observations.items()
        }

    def _select_delayed_observations(self) -> dict[str, Array]:
        """Select per-env delayed observations from the delay buffers."""
        assert self._obs_buffer is not None
        env_idx = jp.arange(self.num_envs)
        return {
            key: value[env_idx, self._delay_steps] for key, value in self._obs_buffer.items()
        }

    def _sample_episode_params(self, mask: Array) -> None:
        """Sample fixed observation delay for newly reset vector slots."""
        self._rng_key, delay_key = jax.random.split(self._rng_key)
        delay_steps = jax.random.randint(
            delay_key,
            (self.num_envs,),
            minval=self.delay_min_steps,
            maxval=self.delay_max_steps + 1,
            dtype=jp.int32,
        )
        mask = jp.asarray(mask, dtype=bool)
        self._delay_steps = jp.where(mask, delay_steps, self._delay_steps)

    def _reset_episode_state(self, done: Array, observations: dict[str, Array]) -> None:
        """Reset delay buffers for completed vector slots."""
        if done.ndim > 1:
            done = jp.any(done, axis=tuple(range(1, done.ndim)))
        if self._obs_buffer is None:
            return
        fresh_buffer = self._initial_obs_buffer(observations)
        reset_mask = done[:, None]
        self._obs_buffer = {
            key: jp.where(
                reset_mask.reshape((self.num_envs, 1) + (1,) * (value.ndim - 2)),
                fresh_buffer[key],
                value,
            )
            for key, value in self._obs_buffer.items()
        }
        self._sample_episode_params(done)

    def _add_noise(self, observations: dict[str, Array]) -> dict[str, Array]:
        """Add zero-mean measurement noise to continuous observation fields."""
        noisy = dict(observations)
        for key, std in (
            ("pos", self.pos_std_m),
            ("vel", self.vel_std_mps),
            ("ang_vel", self.ang_vel_std_radps),
            ("gates_pos", self.gate_pos_std_m),
            ("obstacles_pos", self.obstacle_pos_std_m),
        ):
            if std > 0.0 and key in noisy:
                self._rng_key, subkey = jax.random.split(self._rng_key)
                noisy[key] = noisy[key] + std * jax.random.normal(
                    subkey, noisy[key].shape, dtype=noisy[key].dtype
                )
        if self.quat_rpy_std_rad > 0.0 and "quat" in noisy:
            self._rng_key, subkey = jax.random.split(self._rng_key)
            noisy["quat"] = self._noise_quat(noisy["quat"], subkey, self.quat_rpy_std_rad)
        if self.gate_rpy_std_rad > 0.0 and "gates_quat" in noisy:
            self._rng_key, subkey = jax.random.split(self._rng_key)
            noisy["gates_quat"] = self._noise_quat(
                noisy["gates_quat"], subkey, self.gate_rpy_std_rad
            )
        return noisy

    @staticmethod
    def _noise_quat(quat: Array, key: Array, std_rad: float) -> Array:
        """Apply small-angle quaternion noise to xyzw quaternions."""
        delta = std_rad * jax.random.normal(key, quat.shape[:-1] + (3,), dtype=quat.dtype)
        delta_quat = jp.concatenate(
            [0.5 * delta, jp.ones(delta.shape[:-1] + (1,), dtype=quat.dtype)], axis=-1
        )
        delta_quat = delta_quat / jp.linalg.norm(delta_quat, axis=-1, keepdims=True)
        noisy_quat = ObservationLatencyNoise._quat_multiply(delta_quat, quat)
        return noisy_quat / jp.linalg.norm(noisy_quat, axis=-1, keepdims=True)

    @staticmethod
    def _quat_multiply(q1: Array, q2: Array) -> Array:
        """Multiply xyzw quaternions."""
        x1, y1, z1, w1 = jp.moveaxis(q1, -1, 0)
        x2, y2, z2, w2 = jp.moveaxis(q2, -1, 0)
        return jp.stack(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            axis=-1,
        )

    @staticmethod
    def _done_mask(terminations: Array, truncations: Array) -> Array:
        done = jp.asarray(terminations) | jp.asarray(truncations)
        if done.ndim > 1:
            done = jp.any(done, axis=tuple(range(1, done.ndim)))
        return done


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
    HISTORY_DIM = 7
    N_LOCAL_OBSTACLES = 2

    def __init__(self, env: VectorEnv, n_history: int = 2, debug_obs: bool = False):
        """Initialize observation vector layout."""
        super().__init__(env)
        self.n_history = n_history
        self.debug_obs = debug_obs
        self._printed_obs_debug = False
        raw_space = env.single_observation_space
        self.n_gates = raw_space["gates_pos"].shape[0]
        self.n_obstacles = raw_space["obstacles_pos"].shape[0]
        self.n_local_obstacles = min(self.N_LOCAL_OBSTACLES, self.n_obstacles)
        self.action_dim = env.single_action_space.shape[0]
        self.layout = self._build_layout()
        self.obs_dim = self.layout[-1][1].stop
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
        add("target_gate_corners_body", 12)
        add("target_gate_known", 1)
        add("nearest_other_gate_corners_body", 12)
        add("nearest_other_gate_known", 1)
        add("nearest_obstacles_heading_xy", 4 * self.n_local_obstacles)
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
        active_target_gate = jp.where(
            observations["target_gate"] < 0,
            self.n_gates - 1,
            observations["target_gate"],
        )
        target_gate_corners = self._gate_corners_body(observations, active_target_gate, pos, rot_t)
        target_gate_known = self._gate_known_flag(observations, active_target_gate)
        nearest_other_gate = self._nearest_other_gate_idx(observations, pos, active_target_gate)
        nearest_other_gate_corners = self._gate_corners_body(
            observations, nearest_other_gate, pos, rot_t
        )
        nearest_other_gate_known = self._gate_known_flag(observations, nearest_other_gate)
        nearest_obstacles_heading_xy = self._nearest_obstacles_heading_xy(observations, pos, rot)
        parts = [
            pos[:, 2:3],
            vel_body,
            ang_vel,
            rot.reshape(pos.shape[0], -1),
            target_gate_corners,
            target_gate_known,
            nearest_other_gate_corners,
            nearest_other_gate_known,
            nearest_obstacles_heading_xy,
            last_action,
            history.reshape(pos.shape[0], -1),
        ]
        return jp.concatenate(parts, axis=-1)

    @staticmethod
    def _obstacle_heading_xy_by_obstacle(
        observations: dict[str, Array], pos: Array, rot: Array
    ) -> Array:
        """Return [forward, left, XY distance, detected] per obstacle."""
        relative_xy = observations["obstacles_pos"][..., :2] - pos[:, None, :2]
        heading_forward = rot[:, :2, 0]
        heading_forward /= jp.maximum(jp.linalg.norm(heading_forward, axis=-1, keepdims=True), 1e-6)
        heading_left = jp.stack([-heading_forward[:, 1], heading_forward[:, 0]], axis=-1)
        relative_forward = jp.einsum("nki,ni->nk", relative_xy, heading_forward)
        relative_left = jp.einsum("nki,ni->nk", relative_xy, heading_left)
        distance_xy = jp.linalg.norm(relative_xy, axis=-1)
        detected = observations["obstacles_visited"].astype(jp.float32)
        return jp.stack([relative_forward, relative_left, distance_xy, detected], axis=-1)

    def _nearest_obstacles_heading_xy(
        self, observations: dict[str, Array], pos: Array, rot: Array
    ) -> Array:
        """Return nearest obstacle features sorted by XY distance."""
        features = self._obstacle_heading_xy_by_obstacle(observations, pos, rot)
        nearest_idx = jp.argsort(features[..., 2], axis=-1)[:, : self.n_local_obstacles]
        nearest = jp.take_along_axis(features, nearest_idx[..., None], axis=1)
        return nearest.reshape(pos.shape[0], -1)

    @staticmethod
    def _obstacle_heading_xy(observations: dict[str, Array], pos: Array, rot: Array) -> Array:
        """Return fixed-order [forward, left, XY distance, detected] obstacle features."""
        features = RaceObservation._obstacle_heading_xy_by_obstacle(observations, pos, rot)
        return features.reshape(pos.shape[0], -1)

    def _nearest_other_gate_idx(
        self, observations: dict[str, Array], pos: Array, active_target_gate: Array
    ) -> Array:
        """Return the closest non-target gate index in XY."""
        gate_idx = jp.arange(self.n_gates)[None, :]
        relative_xy = observations["gates_pos"][..., :2] - pos[:, None, :2]
        distance_xy = jp.linalg.norm(relative_xy, axis=-1)
        masked_distance = jp.where(gate_idx != active_target_gate[:, None], distance_xy, jp.inf)
        return jp.argmin(masked_distance, axis=-1)

    def _gate_known_flag(self, observations: dict[str, Array], gate_idx: Array) -> Array:
        """Return the selected gate known/visited flag."""
        batch_idx = jp.arange(gate_idx.shape[0])
        return observations["gates_visited"][batch_idx, gate_idx][:, None].astype(jp.float32)

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

    def _all_gate_corners_body(
        self, observations: dict[str, Array], pos: Array, rot_t: Array
    ) -> Array:
        gate_rot = self.quat_to_rotmat(observations["gates_quat"])
        corners_world = observations["gates_pos"][:, :, None, :] + jp.einsum(
            "ngij,kj->ngki", gate_rot, self.GATE_CORNERS_LOCAL
        )
        corners_body = jp.einsum("nij,ngkj->ngki", rot_t, corners_world - pos[:, None, None, :])
        return corners_body.reshape(pos.shape[0], -1)

    @staticmethod
    def _basic_history(observations: dict[str, Array]) -> Array:
        quat = observations["quat"]
        vel = observations["vel"]
        rot = RaceObservation.quat_to_rotmat(quat)
        rot_t = jp.swapaxes(rot, -1, -2)
        vel_body = jp.einsum("nij,nj->ni", rot_t, vel)
        return jp.concatenate(
            [
                observations["pos"][:, 2:3],
                vel_body,
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
    wandb.config.update({"observation_layout": LEVEL3_OBSERVATION_LAYOUT}, allow_val_change=True)

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
    disturbances = cfg.env.get("disturbances")
    legacy_thrust_disturbance = None
    if disturbances is not None:
        disturbances = dict(disturbances)
        legacy_thrust_disturbance = disturbances.pop("thrust", None)
    train_cfg = cfg.get("train", {})
    thrust_disturbance = train_cfg.get("thrust") if train_cfg is not None else None
    action_latency = train_cfg.get("action_latency") if train_cfg is not None else None
    command_response = train_cfg.get("command_response") if train_cfg is not None else None
    observation_latency = train_cfg.get("observation_latency") if train_cfg is not None else None
    observation_noise = train_cfg.get("observation_noise") if train_cfg is not None else None
    if thrust_disturbance is None:
        thrust_disturbance = legacy_thrust_disturbance
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
        disturbances=disturbances,
        randomizations=cfg.env.get("randomizations"),
        seed=cfg.env.seed,
        device=jax_device,
    )
    if thrust_disturbance is not None:
        env = ThrustScaleBatterySag(
            env,
            thrust_disturbance,
            env_freq=cfg.env.freq,
            seed=cfg.env.seed,
        )
    if action_latency is not None or command_response is not None:
        env = ActionLatencyResponseLag(
            env,
            action_latency,
            command_response,
            env_freq=cfg.env.freq,
            seed=cfg.env.seed,
        )

    env = NormalizeVectorActions(env)
    env = Level2RaceReward(
        env,
        progress_coef=coefs.get("progress_coef", 10.0),
        near_gate_coef=coefs.get("near_gate_coef", 0.0),
        gate_bonus=coefs.get("gate_bonus", 30.0),
        finish_bonus=coefs.get("finish_bonus", 80.0),
        crash_penalty=coefs.get("crash_penalty", 50.0),
        rpy_coef=coefs.get("rpy_coef", 1.0),
        tilt_limit_deg=coefs.get("tilt_limit_deg", 35.0),
        tilt_excess_coef=coefs.get("tilt_excess_coef", 10.0),
        cmd_tilt_coef=coefs.get("cmd_tilt_coef", 1.0),
        act_coef=coefs.get("act_coef", 0.005),
        d_act_th_coef=coefs.get("d_act_th_coef", 0.02),
        d_act_xy_coef=coefs.get("d_act_xy_coef", 0.05),
        gate_axis_coef=coefs.get("gate_axis_coef", 8.0),
        gate_stage_coef=coefs.get("gate_stage_coef", 5.0),
        gate_front_bonus=coefs.get("gate_front_bonus", 4.0),
        gate_back_bonus=coefs.get("gate_back_bonus", 4.0),
        gate_stage_offset=coefs.get("gate_stage_offset", 0.35),
        gate_stage_radius=coefs.get("gate_stage_radius", 0.24),
        wrong_side_penalty=coefs.get("wrong_side_penalty", 6.0),
        missed_gate_penalty=coefs.get("missed_gate_penalty", 8.0),
        obstacle_coef=coefs.get("obstacle_coef", 1.5),
        obstacle_margin=coefs.get("obstacle_margin", 0.35),
        obstacle_clearance_coef=coefs.get("obstacle_clearance_coef", 0.0),
        timeout_penalty=coefs.get("timeout_penalty", 0.0),
        time_penalty=coefs.get("time_penalty", 0.05),
        debug_every=debug_reward_every,
    )
    if observation_latency is not None or observation_noise is not None:
        env = ObservationLatencyNoise(
            env,
            observation_latency,
            observation_noise,
            seed=cfg.env.seed,
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

    def __init__(self, obs_shape: tuple, action_shape: tuple, hidden_dim: int = 128):
        """Init network structures."""
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        self.hidden_dim = hidden_dim
        self.critic = nn.Sequential(
            layer_init(nn.Linear(torch.tensor(obs_shape).prod(), hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(torch.tensor(obs_shape).prod(), hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, torch.tensor(action_shape).prod()), std=0.01),
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
    args: Args,
    model_path: Path | None,
    device: torch.device,
    jax_device: str,
    wandb_enabled: bool = False,
    checkpoint_dir: Path | str | None = None,
    checkpoint_interval: int = 0,
    initial_model_path: Path | str | None = None,
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

    initial_model_path = Path(initial_model_path) if initial_model_path is not None else None
    if wandb_enabled and initial_model_path is not None:
        wandb.config.update({"initial_model_path": str(initial_model_path)}, allow_val_change=True)
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_interval > 0:
        if checkpoint_dir is None:
            checkpoint_dir = model_path.parent if model_path is not None else Path("checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    next_checkpoint_step = checkpoint_interval if checkpoint_interval > 0 else None
    checkpoint_stem = (
        model_path.stem.removesuffix("_final") if model_path is not None else "ppo_checkpoint"
    )

    # env setup
    r_coefs = {
        "n_obs": args.n_obs,
        "rpy_coef": args.rpy_coef,
        "tilt_limit_deg": args.tilt_limit_deg,
        "tilt_excess_coef": args.tilt_excess_coef,
        "cmd_tilt_coef": args.cmd_tilt_coef,
        "d_act_xy_coef": args.d_act_xy_coef,
        "d_act_th_coef": args.d_act_th_coef,
        "act_coef": args.act_coef,
        "progress_coef": args.progress_coef,
        "gate_axis_coef": args.gate_axis_coef,
        "gate_stage_coef": args.gate_stage_coef,
        "gate_front_bonus": args.gate_front_bonus,
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
        "obstacle_clearance_coef": args.obstacle_clearance_coef,
        "timeout_penalty": args.timeout_penalty,
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

    agent = Agent(
        envs.single_observation_space.shape,
        envs.single_action_space.shape,
        hidden_dim=args.hidden_dim,
    ).to(device)
    if initial_model_path is not None:
        checkpoint = torch.load(initial_model_path, map_location=device)
        model_state_dict, observation_layout = unpack_checkpoint(checkpoint)
        if observation_layout != LEVEL3_OBSERVATION_LAYOUT:
            raise ValueError(
                f"Cannot initialize {LEVEL3_OBSERVATION_LAYOUT} training from checkpoint layout "
                f"{observation_layout}. Train from scratch or use a matching checkpoint."
            )
        initial_hidden_dim = checkpoint_hidden_dim(checkpoint, model_state_dict)
        if initial_hidden_dim != args.hidden_dim:
            raise ValueError(
                f"Cannot initialize hidden_dim={args.hidden_dim} training from a "
                f"hidden_dim={initial_hidden_dim} checkpoint. Use a matching hidden_dim."
            )
        agent.load_state_dict(model_state_dict)
        print(f"initialized agent weights from {initial_model_path}; optimizer starts fresh")
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
            total_reward = rewards.float().sum().item()
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
                    "train/total_reward": total_reward,
                    "charts/SPS": int(global_step / (time.time() - start_time)),
                    **reward_component_logs,
                    **race_metric_logs,
                },
                step=global_step,
            )
        end_time = time.time()
        print(f"Iter {iteration}/{args.num_iterations} took {end_time - start_time:.2f} seconds")
        while next_checkpoint_step is not None and global_step >= next_checkpoint_step:
            checkpoint_name = f"{checkpoint_stem}_step_{next_checkpoint_step:09d}.ckpt"
            checkpoint_path = checkpoint_dir / checkpoint_name
            torch.save(
                make_checkpoint(
                    agent.state_dict(),
                    hidden_dim=args.hidden_dim,
                    observation_layout=LEVEL3_OBSERVATION_LAYOUT,
                ),
                checkpoint_path,
            )
            print(f"checkpoint saved to {checkpoint_path} at global_step={global_step}")
            next_checkpoint_step += checkpoint_interval
    train_end_time = time.time()
    print(f"Training for {global_step} steps took {train_end_time - train_start_time:.2f} seconds.")
    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            make_checkpoint(
                agent.state_dict(),
                hidden_dim=args.hidden_dim,
                observation_layout=LEVEL3_OBSERVATION_LAYOUT,
            ),
            model_path,
        )
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
        "tilt_limit_deg": args.tilt_limit_deg,
        "tilt_excess_coef": args.tilt_excess_coef,
        "cmd_tilt_coef": args.cmd_tilt_coef,
        "d_act_xy_coef": args.d_act_xy_coef,
        "d_act_th_coef": args.d_act_th_coef,
        "act_coef": args.act_coef,
        "progress_coef": args.progress_coef,
        "gate_axis_coef": args.gate_axis_coef,
        "gate_stage_coef": args.gate_stage_coef,
        "gate_front_bonus": args.gate_front_bonus,
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
        "obstacle_clearance_coef": args.obstacle_clearance_coef,
        "timeout_penalty": args.timeout_penalty,
        "time_penalty": args.time_penalty,
    }
    eval_env = make_envs(config=args.config, num_envs=1, coefs=r_coefs)
    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict, observation_layout = unpack_checkpoint(checkpoint)
    if observation_layout != LEVEL3_OBSERVATION_LAYOUT:
        raise ValueError(
            f"Cannot evaluate checkpoint layout {observation_layout} "
            f"with {LEVEL3_OBSERVATION_LAYOUT} env."
        )
    hidden_dim = checkpoint_hidden_dim(checkpoint, model_state_dict)
    agent = Agent(
        eval_env.single_observation_space.shape,
        eval_env.single_action_space.shape,
        hidden_dim=hidden_dim,
    ).to(device)
    agent.load_state_dict(model_state_dict)
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
        "tilt_limit_deg": args.tilt_limit_deg,
        "tilt_excess_coef": args.tilt_excess_coef,
        "cmd_tilt_coef": args.cmd_tilt_coef,
        "d_act_xy_coef": args.d_act_xy_coef,
        "d_act_th_coef": args.d_act_th_coef,
        "act_coef": args.act_coef,
        "progress_coef": args.progress_coef,
        "gate_axis_coef": args.gate_axis_coef,
        "gate_stage_coef": args.gate_stage_coef,
        "gate_front_bonus": args.gate_front_bonus,
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
        "obstacle_clearance_coef": args.obstacle_clearance_coef,
        "timeout_penalty": args.timeout_penalty,
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
    hidden_dim: int = 128,
    cuda: bool = True,
    jax_device: str = "gpu",
    model_name: str = "ppo_level2_racing.ckpt",
    initial_model_name: str | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = 0,
    n_obs: int = 2,
    progress_coef: float = 10.0,
    gate_axis_coef: float = 8.0,
    gate_stage_coef: float = 5.0,
    gate_front_bonus: float = 4.0,
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
    obstacle_clearance_coef: float = 0.0,
    timeout_penalty: float = 0.0,
    time_penalty: float = 0.05,
    act_coef: float = 0.005,
    d_act_th_coef: float = 0.02,
    d_act_xy_coef: float = 0.05,
    rpy_coef: float = 1.0,
    tilt_limit_deg: float = 35.0,
    tilt_excess_coef: float = 10.0,
    cmd_tilt_coef: float = 1.0,
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
        hidden_dim=hidden_dim,
        cuda=cuda,
        jax_device=jax_device,
        n_obs=n_obs,
        progress_coef=progress_coef,
        gate_axis_coef=gate_axis_coef,
        gate_stage_coef=gate_stage_coef,
        gate_front_bonus=gate_front_bonus,
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
        obstacle_clearance_coef=obstacle_clearance_coef,
        timeout_penalty=timeout_penalty,
        time_penalty=time_penalty,
        act_coef=act_coef,
        d_act_th_coef=d_act_th_coef,
        d_act_xy_coef=d_act_xy_coef,
        rpy_coef=rpy_coef,
        tilt_limit_deg=tilt_limit_deg,
        tilt_excess_coef=tilt_excess_coef,
        cmd_tilt_coef=cmd_tilt_coef,
        debug_obs=debug_obs,
        debug_reward_every=debug_reward_every,
    )
    model_path = Path(__file__).parent / model_name
    initial_model_path = Path(__file__).parent / initial_model_name if initial_model_name else None
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if debug_steps > 0:
        debug_rollout(args, debug_steps, device, args.jax_device)

    if train:  # use "--train False" to skip training
        train_ppo(
            args,
            model_path,
            device,
            args.jax_device,
            wandb_enabled,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            initial_model_path=initial_model_path,
        )

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
