"""JAX PPO training for the direct level3 drone racing policy.

This keeps the reward and observation wrappers from ``train_CleanRL_ppo_level3`` so the task
definition stays identical, but removes the JaxToTorch bridge and runs the PPO learner in JAX.
Checkpoints are exported in the existing PyTorch ``state_dict`` layout so
``ppo_level3_inference.py`` can load them unchanged.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import fire
import gymnasium as gym
import jax
import jax.numpy as jp
import numpy as np
from jax import Array

import wandb
from lsy_drone_racing.control.ppo_level3_observation import (
    LOCAL_OBSTACLE_OBSERVATION_LAYOUT,
    checkpoint_hidden_dim,
    make_checkpoint,
    unpack_checkpoint,
)
from lsy_drone_racing.control.train_CleanRL_ppo_level3 import (
    RACE_METRIC_KEYS,
    REWARD_COMPONENT_KEYS,
    ActionLatencyResponseLag,
    Level2RaceReward,
    NormalizeVectorActions,
    ObservationLatencyNoise,
    RaceObservation,
    ThrustScaleBatterySag,
)
from lsy_drone_racing.utils import load_config

LEVEL3_OBSERVATION_LAYOUT = LOCAL_OBSTACLE_OBSERVATION_LAYOUT
LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass
class Args:
    """Configuration for JAX PPO training."""

    config: str = "level2.toml"
    seed: int = 42
    cuda: bool = True
    jax_device: str = "gpu"
    wandb_project_name: str = "ADR-PPO-Racing"
    wandb_entity: str | None = None

    total_timesteps: int = 2_000_000
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    num_envs: int = 256
    num_steps: int = 32
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    update_epochs: int = 5
    norm_adv: bool = True
    clip_coef: float = 0.26
    clip_vloss: bool = True
    ent_coef: float = 0.02
    vf_coef: float = 0.7
    max_grad_norm: float = 1.5
    target_kl: float | None = 0.03
    hidden_dim: int = 128

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

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
    speed_limit_mps: float = 0.0
    speed_excess_coef: float = 0.0
    debug_obs: bool = False
    debug_reward_every: int = 0

    @staticmethod
    def create(**kwargs: Any) -> "Args":
        """Create the runtime configuration and derived batch sizes."""
        args = Args(**kwargs)
        args.batch_size = int(args.num_envs * args.num_steps)
        if args.batch_size <= 0:
            raise ValueError("num_envs * num_steps must be positive.")
        if args.num_minibatches <= 0:
            raise ValueError("num_minibatches must be positive.")
        if args.batch_size % args.num_minibatches != 0:
            raise ValueError("num_envs * num_steps must be divisible by num_minibatches.")
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size
        if args.num_iterations <= 0:
            raise ValueError("total_timesteps must be at least num_envs * num_steps.")
        return args


class AdamWState(NamedTuple):
    """Minimal AdamW optimizer state."""

    count: Array
    m: dict[str, Any]
    v: dict[str, Any]


def reward_coefs_from_args(args: Args) -> dict[str, float | int]:
    """Return the reward/observation wrapper coefficients used by both train and debug."""
    return {
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
        "speed_limit_mps": args.speed_limit_mps,
        "speed_excess_coef": args.speed_excess_coef,
    }


def make_envs(
    config: str = "level2.toml",
    num_envs: int | None = None,
    jax_device: str = "cpu",
    coefs: dict | None = None,
    debug_obs: bool = False,
    debug_reward_every: int = 0,
) -> gym.vector.VectorEnv:
    """Make direct level3 racing environments without JaxToTorch conversion."""
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
        raise ValueError("Direct level3 PPO currently expects env.control_mode = 'attitude'.")

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
            env, thrust_disturbance, env_freq=cfg.env.freq, seed=cfg.env.seed
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
        speed_limit_mps=coefs.get("speed_limit_mps", 0.0),
        speed_excess_coef=coefs.get("speed_excess_coef", 0.0),
        debug_every=debug_reward_every,
    )
    if observation_latency is not None or observation_noise is not None:
        env = ObservationLatencyNoise(
            env,
            observation_latency,
            observation_noise,
            seed=cfg.env.seed,
        )
    return RaceObservation(env, n_history=coefs.get("n_obs", 0), debug_obs=debug_obs)


def set_seeds(seed: int) -> None:
    """Seed Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)


def _init_linear(key: Array, in_dim: int, out_dim: int, scale: float) -> dict[str, Array]:
    initializer = jax.nn.initializers.orthogonal(scale=scale)
    return {
        "w": initializer(key, (in_dim, out_dim), jp.float32),
        "b": jp.zeros((out_dim,), dtype=jp.float32),
    }


def init_agent_params(key: Array, obs_dim: int, action_dim: int, hidden_dim: int) -> dict[str, Any]:
    """Initialize actor/critic params with the same topology as the PyTorch agent."""
    keys = jax.random.split(key, 6)
    return {
        "actor_mean": {
            "0": _init_linear(keys[0], obs_dim, hidden_dim, float(np.sqrt(2.0))),
            "2": _init_linear(keys[1], hidden_dim, hidden_dim, float(np.sqrt(2.0))),
            "4": _init_linear(keys[2], hidden_dim, action_dim, 0.01),
            "logstd": jp.array([[-1.2, -1.2, -2.0, -0.7]], dtype=jp.float32),
        },
        "critic": {
            "0": _init_linear(keys[3], obs_dim, hidden_dim, float(np.sqrt(2.0))),
            "2": _init_linear(keys[4], hidden_dim, hidden_dim, float(np.sqrt(2.0))),
            "4": _init_linear(keys[5], hidden_dim, 1, 1.0),
        },
    }


def _mlp(x: Array, layers: dict[str, dict[str, Array]], *, squash_output: bool) -> Array:
    x = jp.tanh(x @ layers["0"]["w"] + layers["0"]["b"])
    x = jp.tanh(x @ layers["2"]["w"] + layers["2"]["b"])
    x = x @ layers["4"]["w"] + layers["4"]["b"]
    return jp.tanh(x) if squash_output else x


def agent_apply(params: dict[str, Any], obs: Array) -> tuple[Array, Array]:
    """Return actor mean and critic value."""
    action_mean = _mlp(obs, params["actor_mean"], squash_output=True)
    value = _mlp(obs, params["critic"], squash_output=False)
    return action_mean, value


def normal_log_prob(action: Array, mean: Array, logstd: Array) -> Array:
    """Log probability of a diagonal Gaussian action."""
    var = jp.exp(2.0 * logstd)
    return (-0.5 * (((action - mean) ** 2) / var + 2.0 * logstd + LOG_2PI)).sum(axis=-1)


def normal_entropy(logstd: Array, action_shape: tuple[int, ...]) -> Array:
    """Entropy of a diagonal Gaussian broadcast to the action batch."""
    entropy = (0.5 + 0.5 * LOG_2PI + logstd).sum(axis=-1)
    return jp.broadcast_to(entropy, action_shape[:-1])


def get_action_and_value(
    params: dict[str, Any],
    obs: Array,
    key: Array | None = None,
    action: Array | None = None,
    deterministic: bool = False,
) -> tuple[Array, Array, Array, Array]:
    """Sample or score an action and return action, logprob, entropy, and value."""
    action_mean, value = agent_apply(params, obs)
    logstd = jp.broadcast_to(params["actor_mean"]["logstd"], action_mean.shape)
    if action is None:
        if deterministic:
            action = action_mean
        else:
            if key is None:
                raise ValueError("A JAX PRNG key is required for stochastic action sampling.")
            action = action_mean + jp.exp(logstd) * jax.random.normal(
                key, action_mean.shape, dtype=action_mean.dtype
            )
    logprob = normal_log_prob(action, action_mean, logstd)
    entropy = normal_entropy(logstd, action_mean.shape)
    return action, logprob, entropy, value


def adamw_init(params: dict[str, Any]) -> AdamWState:
    """Initialize AdamW state."""
    zeros = jax.tree_util.tree_map(jp.zeros_like, params)
    return AdamWState(count=jp.array(0, dtype=jp.int32), m=zeros, v=zeros)


def tree_global_norm(tree: Any) -> Array:
    """Return the global L2 norm of a pytree."""
    leaves = jax.tree_util.tree_leaves(tree)
    return jp.sqrt(sum(jp.sum(jp.square(leaf)) for leaf in leaves))


def adamw_update(
    params: dict[str, Any],
    grads: dict[str, Any],
    state: AdamWState,
    learning_rate: float,
    max_grad_norm: float,
    weight_decay: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-5,
) -> tuple[dict[str, Any], AdamWState, Array]:
    """Apply one clipped AdamW update."""
    grad_norm = tree_global_norm(grads)
    clip_scale = jp.minimum(1.0, max_grad_norm / (grad_norm + 1e-6))
    grads = jax.tree_util.tree_map(lambda g: g * clip_scale, grads)
    count = state.count + 1
    m = jax.tree_util.tree_map(lambda m_, g: beta1 * m_ + (1.0 - beta1) * g, state.m, grads)
    v = jax.tree_util.tree_map(lambda v_, g: beta2 * v_ + (1.0 - beta2) * (g * g), state.v, grads)
    m_hat = jax.tree_util.tree_map(lambda m_: m_ / (1.0 - beta1**count), m)
    v_hat = jax.tree_util.tree_map(lambda v_: v_ / (1.0 - beta2**count), v)
    updates = jax.tree_util.tree_map(
        lambda m_, v_, p: m_ / (jp.sqrt(v_) + eps) + weight_decay * p,
        m_hat,
        v_hat,
        params,
    )
    params = jax.tree_util.tree_map(lambda p, u: p - learning_rate * u, params, updates)
    return params, AdamWState(count=count, m=m, v=v), grad_norm


@jax.jit
def compute_gae(
    rewards: Array,
    dones: Array,
    values: Array,
    next_done: Array,
    next_value: Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[Array, Array]:
    """Compute GAE advantages and returns."""
    next_value = next_value.reshape(-1)
    next_done = next_done.astype(jp.float32).reshape(-1)

    def scan_step(
        carry: tuple[Array, Array, Array], inputs: tuple[Array, Array, Array]
    ) -> tuple[tuple[Array, Array, Array], Array]:
        lastgaelam, carry_next_value, carry_next_done = carry
        reward_t, done_t, value_t = inputs
        nextnonterminal = 1.0 - carry_next_done
        delta = reward_t + gamma * carry_next_value * nextnonterminal - value_t
        advantage_t = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        return (advantage_t, value_t, done_t.astype(jp.float32)), advantage_t

    init = (jp.zeros_like(next_value), next_value, next_done)
    _, advantages_rev = jax.lax.scan(
        scan_step,
        init,
        (rewards[::-1], dones[::-1], values[::-1]),
    )
    advantages = advantages_rev[::-1]
    return advantages, advantages + values


@jax.jit
def flatten_batch(
    obs: Array,
    actions: Array,
    logprobs: Array,
    advantages: Array,
    returns: Array,
    values: Array,
) -> dict[str, Array]:
    """Flatten rollout tensors from [T, N, ...] to [T*N, ...]."""
    return {
        "obs": obs.reshape((obs.shape[0] * obs.shape[1],) + obs.shape[2:]),
        "actions": actions.reshape((actions.shape[0] * actions.shape[1],) + actions.shape[2:]),
        "logprobs": logprobs.reshape(-1),
        "advantages": advantages.reshape(-1),
        "returns": returns.reshape(-1),
        "values": values.reshape(-1),
    }


@jax.jit
def ppo_minibatch_update(
    params: dict[str, Any],
    opt_state: AdamWState,
    mb_obs: Array,
    mb_actions: Array,
    mb_logprobs: Array,
    mb_advantages: Array,
    mb_returns: Array,
    mb_values: Array,
    learning_rate: float,
    weight_decay: float,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    norm_adv: bool,
    clip_vloss: bool,
) -> tuple[dict[str, Any], AdamWState, dict[str, Array]]:
    """Run one PPO minibatch update."""

    def loss_fn(train_params: dict[str, Any]) -> tuple[Array, dict[str, Array]]:
        _, newlogprob, entropy, newvalue = get_action_and_value(
            train_params, mb_obs, action=mb_actions
        )
        newvalue = newvalue.reshape(-1)
        logratio = newlogprob - mb_logprobs
        ratio = jp.exp(logratio)

        advantages = mb_advantages
        advantages = jp.where(
            norm_adv,
            (advantages - advantages.mean()) / (advantages.std() + 1e-8),
            advantages,
        )

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * jp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
        pg_loss = jp.maximum(pg_loss1, pg_loss2).mean()

        v_loss_unclipped = (newvalue - mb_returns) ** 2
        v_clipped = mb_values + jp.clip(newvalue - mb_values, -clip_coef, clip_coef)
        v_loss_clipped = (v_clipped - mb_returns) ** 2
        v_loss = jp.where(
            clip_vloss,
            0.5 * jp.maximum(v_loss_unclipped, v_loss_clipped).mean(),
            0.5 * v_loss_unclipped.mean(),
        )

        entropy_loss = entropy.mean()
        loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss
        approx_kl = ((ratio - 1.0) - logratio).mean()
        old_approx_kl = (-logratio).mean()
        clipfrac = (jp.abs(ratio - 1.0) > clip_coef).astype(jp.float32).mean()
        return loss, {
            "loss": loss,
            "policy_loss": pg_loss,
            "value_loss": v_loss,
            "entropy": entropy_loss,
            "old_approx_kl": old_approx_kl,
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
        }

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    params, opt_state, grad_norm = adamw_update(
        params,
        grads,
        opt_state,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        weight_decay=weight_decay,
    )
    metrics = dict(metrics)
    metrics["loss"] = loss
    metrics["grad_norm"] = grad_norm
    return params, opt_state, metrics


@jax.jit
def ppo_epoch_update(
    params: dict[str, Any],
    opt_state: AdamWState,
    batch: dict[str, Array],
    minibatch_indices: Array,
    learning_rate: float,
    weight_decay: float,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    norm_adv: bool,
    clip_vloss: bool,
) -> tuple[dict[str, Any], AdamWState, dict[str, Array], Array]:
    """Run one PPO epoch as a single compiled scan over minibatches."""

    def scan_step(
        carry: tuple[dict[str, Any], AdamWState], mb_inds: Array
    ) -> tuple[tuple[dict[str, Any], AdamWState], dict[str, Array]]:
        carry_params, carry_opt_state = carry
        carry_params, carry_opt_state, metrics = ppo_minibatch_update(
            carry_params,
            carry_opt_state,
            batch["obs"][mb_inds],
            batch["actions"][mb_inds],
            batch["logprobs"][mb_inds],
            batch["advantages"][mb_inds],
            batch["returns"][mb_inds],
            batch["values"][mb_inds],
            learning_rate,
            weight_decay,
            clip_coef,
            ent_coef,
            vf_coef,
            max_grad_norm,
            norm_adv,
            clip_vloss,
        )
        return (carry_params, carry_opt_state), metrics

    (params, opt_state), epoch_metrics = jax.lax.scan(
        scan_step,
        (params, opt_state),
        minibatch_indices,
    )
    latest_metrics = jax.tree_util.tree_map(lambda value: value[-1], epoch_metrics)
    return params, opt_state, latest_metrics, epoch_metrics["clipfrac"]


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def params_from_torch_state_dict(model_state_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert the existing PyTorch state_dict layout into JAX params."""
    return {
        "actor_mean": {
            "0": {
                "w": jp.asarray(_tensor_to_numpy(model_state_dict["actor_mean.0.weight"]).T),
                "b": jp.asarray(_tensor_to_numpy(model_state_dict["actor_mean.0.bias"])),
            },
            "2": {
                "w": jp.asarray(_tensor_to_numpy(model_state_dict["actor_mean.2.weight"]).T),
                "b": jp.asarray(_tensor_to_numpy(model_state_dict["actor_mean.2.bias"])),
            },
            "4": {
                "w": jp.asarray(_tensor_to_numpy(model_state_dict["actor_mean.4.weight"]).T),
                "b": jp.asarray(_tensor_to_numpy(model_state_dict["actor_mean.4.bias"])),
            },
            "logstd": jp.asarray(_tensor_to_numpy(model_state_dict["actor_logstd"])),
        },
        "critic": {
            "0": {
                "w": jp.asarray(_tensor_to_numpy(model_state_dict["critic.0.weight"]).T),
                "b": jp.asarray(_tensor_to_numpy(model_state_dict["critic.0.bias"])),
            },
            "2": {
                "w": jp.asarray(_tensor_to_numpy(model_state_dict["critic.2.weight"]).T),
                "b": jp.asarray(_tensor_to_numpy(model_state_dict["critic.2.bias"])),
            },
            "4": {
                "w": jp.asarray(_tensor_to_numpy(model_state_dict["critic.4.weight"]).T),
                "b": jp.asarray(_tensor_to_numpy(model_state_dict["critic.4.bias"])),
            },
        },
    }


def load_params_from_checkpoint(
    checkpoint_path: Path | str,
    *,
    expected_hidden_dim: int,
    device: jax.Device,
) -> dict[str, Any]:
    """Load a current PyTorch-compatible checkpoint into JAX params."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state_dict, observation_layout = unpack_checkpoint(checkpoint)
    if observation_layout != LEVEL3_OBSERVATION_LAYOUT:
        raise ValueError(
            f"Cannot initialize {LEVEL3_OBSERVATION_LAYOUT} training from checkpoint layout "
            f"{observation_layout}."
        )
    hidden_dim = checkpoint_hidden_dim(checkpoint, model_state_dict)
    if hidden_dim != expected_hidden_dim:
        raise ValueError(
            f"Cannot initialize hidden_dim={expected_hidden_dim} training from a "
            f"hidden_dim={hidden_dim} checkpoint."
        )
    return jax.device_put(params_from_torch_state_dict(model_state_dict), device)


def torch_state_dict_from_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert JAX params to the PyTorch state_dict layout used by inference."""
    import torch

    def tensor(array: Array) -> Any:
        return torch.tensor(np.asarray(jax.device_get(array)), dtype=torch.float32)

    return {
        "critic.0.weight": tensor(params["critic"]["0"]["w"].T),
        "critic.0.bias": tensor(params["critic"]["0"]["b"]),
        "critic.2.weight": tensor(params["critic"]["2"]["w"].T),
        "critic.2.bias": tensor(params["critic"]["2"]["b"]),
        "critic.4.weight": tensor(params["critic"]["4"]["w"].T),
        "critic.4.bias": tensor(params["critic"]["4"]["b"]),
        "actor_mean.0.weight": tensor(params["actor_mean"]["0"]["w"].T),
        "actor_mean.0.bias": tensor(params["actor_mean"]["0"]["b"]),
        "actor_mean.2.weight": tensor(params["actor_mean"]["2"]["w"].T),
        "actor_mean.2.bias": tensor(params["actor_mean"]["2"]["b"]),
        "actor_mean.4.weight": tensor(params["actor_mean"]["4"]["w"].T),
        "actor_mean.4.bias": tensor(params["actor_mean"]["4"]["b"]),
        "actor_logstd": tensor(params["actor_mean"]["logstd"]),
    }


def save_checkpoint(params: dict[str, Any], checkpoint_path: Path | str, hidden_dim: int) -> None:
    """Save a JAX-trained policy in the existing PyTorch inference format."""
    import torch

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model_state_dict = torch_state_dict_from_params(params)
    torch.save(
        make_checkpoint(
            model_state_dict,
            hidden_dim=hidden_dim,
            observation_layout=LEVEL3_OBSERVATION_LAYOUT,
        ),
        checkpoint_path,
    )


def mean_scalar(value: Any) -> float:
    """Convert array-like values to a Python mean scalar for logging."""
    return float(np.asarray(jax.device_get(value)).mean())


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


def train_ppo(
    args: Args,
    model_path: Path | None,
    jax_device: str,
    wandb_enabled: bool = False,
    checkpoint_dir: Path | str | None = None,
    checkpoint_interval: int = 0,
    initial_model_path: Path | str | None = None,
) -> list[float]:
    """Train PPO with JAX arrays and JAX optimizer updates."""
    if wandb_enabled:
        setup_wandb(args)
    set_seeds(args.seed)
    train_start_time = time.time()
    device = jax.devices(jax_device)[0]
    print("Training JAX PPO on device:", device)

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

    envs = make_envs(
        config=args.config,
        num_envs=args.num_envs,
        jax_device=jax_device,
        coefs=reward_coefs_from_args(args),
        debug_obs=args.debug_obs,
        debug_reward_every=args.debug_reward_every,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    if action_dim != 4:
        raise ValueError(f"Expected 4-D attitude action, got action_dim={action_dim}.")

    rng = jax.random.PRNGKey(args.seed)
    rng, init_key = jax.random.split(rng)
    params = init_agent_params(init_key, obs_dim, action_dim, args.hidden_dim)
    params = jax.device_put(params, device)
    if initial_model_path is not None:
        params = load_params_from_checkpoint(
            initial_model_path,
            expected_hidden_dim=args.hidden_dim,
            device=device,
        )
        print(f"initialized JAX params from {initial_model_path}; optimizer starts fresh")
    opt_state = adamw_init(params)

    obs = jp.zeros((args.num_steps, args.num_envs, obs_dim), dtype=jp.float32, device=device)
    actions = jp.zeros((args.num_steps, args.num_envs, action_dim), dtype=jp.float32, device=device)
    logprobs = jp.zeros((args.num_steps, args.num_envs), dtype=jp.float32, device=device)
    rewards = jp.zeros((args.num_steps, args.num_envs), dtype=jp.float32, device=device)
    dones = jp.zeros((args.num_steps, args.num_envs), dtype=bool, device=device)
    values = jp.zeros((args.num_steps, args.num_envs), dtype=jp.float32, device=device)

    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = jp.asarray(next_obs, dtype=jp.float32)
    next_done = jp.zeros((args.num_envs,), dtype=bool)
    sum_rewards = jp.zeros((args.num_envs,), dtype=jp.float32)
    sum_rewards_hist: list[float] = []

    for iteration in range(1, args.num_iterations + 1):
        iteration_start_time = time.time()
        zero_metric = jp.array(0.0, dtype=jp.float32)
        reward_component_sums = dict.fromkeys(REWARD_COMPONENT_KEYS, zero_metric)
        race_metric_sums = dict.fromkeys(RACE_METRIC_KEYS, zero_metric)
        reward_component_batches = 0
        episode_reward_rows = []
        episode_reward_steps = []

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
        else:
            lrnow = args.learning_rate

        rollout_start_time = time.time()
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs = obs.at[step].set(next_obs)
            dones = dones.at[step].set(next_done)

            rng, action_key = jax.random.split(rng)
            action, logprob, _, value = get_action_and_value(params, next_obs, action_key)
            actions = actions.at[step].set(action)
            logprobs = logprobs.at[step].set(logprob)
            values = values.at[step].set(value.reshape(-1))

            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_obs = jp.asarray(next_obs, dtype=jp.float32)
            reward = jp.asarray(reward, dtype=jp.float32)
            terminations = jp.asarray(terminations, dtype=bool)
            truncations = jp.asarray(truncations, dtype=bool)
            rewards = rewards.at[step].set(reward)

            sum_rewards = jp.where(next_done, 0.0, sum_rewards)
            sum_rewards = sum_rewards + reward
            next_done = terminations | truncations

            if wandb_enabled:
                reward_component_batches += 1
                for key in REWARD_COMPONENT_KEYS:
                    if (value := infos.get(f"reward_{key}")) is not None:
                        reward_component_sums[key] = reward_component_sums[key] + jp.mean(
                            jp.asarray(value, dtype=jp.float32)
                        )
                for key in RACE_METRIC_KEYS:
                    if (value := infos.get(f"race_{key}")) is not None:
                        race_metric_sums[key] = race_metric_sums[key] + jp.mean(
                            jp.asarray(value, dtype=jp.float32)
                        )

                episode_reward_rows.append(jp.where(next_done, sum_rewards, jp.nan))
                episode_reward_steps.append(global_step)
        rollout_seconds = time.time() - rollout_start_time

        _, _, _, next_value = get_action_and_value(params, next_obs, deterministic=True)
        advantages, returns = compute_gae(
            rewards,
            dones,
            values,
            next_done,
            next_value.reshape(-1),
            args.gamma,
            args.gae_lambda,
        )
        batch = flatten_batch(obs, actions, logprobs, advantages, returns, values)

        rng, update_key = jax.random.split(rng)
        latest_metrics: dict[str, Array] | None = None
        clipfracs: list[float] = []
        update_start_time = time.time()
        for _epoch in range(args.update_epochs):
            update_key, perm_key = jax.random.split(update_key)
            minibatch_indices = jax.random.permutation(perm_key, args.batch_size).reshape(
                args.num_minibatches,
                args.minibatch_size,
            )
            params, opt_state, latest_metrics, epoch_clipfracs = ppo_epoch_update(
                params,
                opt_state,
                batch,
                minibatch_indices,
                float(lrnow),
                args.weight_decay,
                args.clip_coef,
                args.ent_coef,
                args.vf_coef,
                args.max_grad_norm,
                args.norm_adv,
                args.clip_vloss,
            )
            clipfracs.extend(np.asarray(jax.device_get(epoch_clipfracs)).tolist())
            if (
                args.target_kl is not None
                and latest_metrics is not None
                and float(jax.device_get(latest_metrics["approx_kl"])) > args.target_kl
            ):
                break
        update_seconds = time.time() - update_start_time

        b_values = batch["values"]
        b_returns = batch["returns"]
        y_pred = np.asarray(jax.device_get(b_values))
        y_true = np.asarray(jax.device_get(b_returns))
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if wandb_enabled and latest_metrics is not None:
            if episode_reward_rows:
                episode_rewards = np.asarray(jax.device_get(jp.stack(episode_reward_rows)))
                for row, step_value in zip(episode_rewards, episode_reward_steps, strict=True):
                    for episode_reward in row[np.isfinite(row)]:
                        value = float(episode_reward)
                        wandb.log(
                            {"global_step": step_value, "train/reward": value},
                            step=step_value,
                        )
                        sum_rewards_hist.append(value)

            total_reward = float(jax.device_get(rewards.sum()))
            reward_component_logs = {}
            race_metric_logs = {}
            if reward_component_batches > 0:
                reward_component_logs = {
                    f"reward_components/{key}": float(
                        jax.device_get(value / reward_component_batches)
                    )
                    for key, value in reward_component_sums.items()
                }
                race_metric_logs = {
                    f"race/{key}": float(jax.device_get(value / reward_component_batches))
                    for key, value in race_metric_sums.items()
                }
                if args.gate_bonus:
                    reward_component_logs["reward_components/gate_bonus_rate"] = (
                        reward_component_logs["reward_components/gate_bonus"] / args.gate_bonus
                    )
            wandb.log(
                {
                    "global_step": global_step,
                    "charts/learning_rate": lrnow,
                    "charts/SPS": int(global_step / max(time.time() - train_start_time, 1e-6)),
                    "charts/rollout_seconds": rollout_seconds,
                    "charts/update_seconds": update_seconds,
                    "losses/value_loss": float(jax.device_get(latest_metrics["value_loss"])),
                    "losses/policy_loss": float(jax.device_get(latest_metrics["policy_loss"])),
                    "losses/entropy": float(jax.device_get(latest_metrics["entropy"])),
                    "losses/old_approx_kl": float(
                        jax.device_get(latest_metrics["old_approx_kl"])
                    ),
                    "losses/approx_kl": float(jax.device_get(latest_metrics["approx_kl"])),
                    "losses/clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
                    "losses/grad_norm": float(jax.device_get(latest_metrics["grad_norm"])),
                    "losses/explained_variance": explained_var,
                    "train/total_reward": total_reward,
                    **reward_component_logs,
                    **race_metric_logs,
                },
                step=global_step,
            )

        print(
            f"Iter {iteration}/{args.num_iterations} took "
            f"{time.time() - iteration_start_time:.2f} seconds "
            f"(rollout={rollout_seconds:.2f}s update={update_seconds:.2f}s)"
        )

        while next_checkpoint_step is not None and global_step >= next_checkpoint_step:
            checkpoint_path = (
                checkpoint_dir / f"{checkpoint_stem}_step_{next_checkpoint_step:09d}.ckpt"
            )
            save_checkpoint(params, checkpoint_path, args.hidden_dim)
            print(f"checkpoint saved to {checkpoint_path} at global_step={global_step}")
            next_checkpoint_step += checkpoint_interval

    print(f"Training for {global_step} steps took {time.time() - train_start_time:.2f} seconds.")
    if model_path is not None:
        save_checkpoint(params, model_path, args.hidden_dim)
        print(f"model saved to {model_path}")
    envs.close()
    return sum_rewards_hist


def evaluate_ppo(
    args: Args,
    n_eval: int,
    model_path: Path,
    *,
    render: bool = False,
) -> tuple[list[float], list[int]]:
    """Evaluate a JAX-trained PyTorch-compatible checkpoint."""
    set_seeds(args.seed)
    device = jax.devices(args.jax_device)[0]
    eval_env = make_envs(
        config=args.config,
        num_envs=1,
        jax_device=args.jax_device,
        coefs=reward_coefs_from_args(args),
    )
    import torch

    checkpoint = torch.load(model_path, map_location="cpu")
    model_state_dict, observation_layout = unpack_checkpoint(checkpoint)
    if observation_layout != LEVEL3_OBSERVATION_LAYOUT:
        raise ValueError(
            f"Cannot evaluate checkpoint layout {observation_layout} "
            f"with {LEVEL3_OBSERVATION_LAYOUT} env."
        )
    params = jax.device_put(params_from_torch_state_dict(model_state_dict), device)

    episode_rewards = []
    episode_lengths = []
    ep_seed = args.seed
    for episode in range(n_eval):
        obs, _ = eval_env.reset(seed=(ep_seed := ep_seed + 1))
        obs = jp.asarray(obs, dtype=jp.float32)
        done = jp.array([False])
        episode_reward = 0.0
        steps = 0
        while not bool(np.asarray(jp.any(done))):
            action, _, _, _ = get_action_and_value(params, obs, deterministic=True)
            obs, reward, terminated, truncated, _info = eval_env.step(action)
            obs = jp.asarray(obs, dtype=jp.float32)
            if render:
                eval_env.render()
            done = jp.asarray(terminated) | jp.asarray(truncated)
            episode_reward += float(np.asarray(reward)[0])
            steps += 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {steps}")

    print(f"Average Reward = {np.mean(episode_rewards):.2f}, Length = {np.mean(episode_lengths)}")
    eval_env.close()
    return episode_rewards, episode_lengths


def debug_rollout(args: Args, n_steps: int, jax_device: str) -> None:
    """Run zero-action rollouts to inspect observation layout and reward components."""
    envs = make_envs(
        config=args.config,
        num_envs=args.num_envs,
        jax_device=jax_device,
        coefs=reward_coefs_from_args(args),
        debug_obs=True,
        debug_reward_every=1,
    )
    obs, _ = envs.reset(seed=args.seed)
    print(
        f"[debug-rollout] obs_shape={tuple(obs.shape)} "
        f"action_shape={envs.single_action_space.shape}"
    )
    action = jp.zeros((args.num_envs,) + envs.single_action_space.shape, dtype=jp.float32)
    for step in range(n_steps):
        _obs, reward, terminated, truncated, _info = envs.step(action)
        done = jp.asarray(terminated) | jp.asarray(truncated)
        print(
            f"[debug-rollout] step={step + 1}/{n_steps} "
            f"reward_mean={float(np.asarray(jp.mean(reward))):.3f} "
            f"reward_min={float(np.asarray(jp.min(reward))):.3f} "
            f"reward_max={float(np.asarray(jp.max(reward))):.3f} "
            f"done={int(np.asarray(jp.sum(done)))}/{args.num_envs}"
        )
    envs.close()


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
    weight_decay: float = 0.01,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    update_epochs: int = 5,
    num_minibatches: int = 8,
    ent_coef: float = 0.02,
    target_kl: float | None = 0.03,
    hidden_dim: int = 128,
    cuda: bool = True,
    jax_device: str = "gpu",
    model_name: str = "ppo_level3_jax_racing.ckpt",
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
    speed_limit_mps: float = 0.0,
    speed_excess_coef: float = 0.0,
    act_coef: float = 0.005,
    d_act_th_coef: float = 0.02,
    d_act_xy_coef: float = 0.05,
    rpy_coef: float = 1.0,
    tilt_limit_deg: float = 35.0,
    tilt_excess_coef: float = 10.0,
    cmd_tilt_coef: float = 1.0,
    debug_obs: bool = False,
    debug_reward_every: int = 0,
) -> None:
    """CLI entrypoint."""
    args = Args.create(
        config=config,
        total_timesteps=total_timesteps,
        num_envs=num_envs,
        num_steps=num_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
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
        speed_limit_mps=speed_limit_mps,
        speed_excess_coef=speed_excess_coef,
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

    if debug_steps > 0:
        debug_rollout(args, debug_steps, args.jax_device)

    if train:
        train_ppo(
            args,
            model_path,
            args.jax_device,
            wandb_enabled,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            initial_model_path=initial_model_path,
        )

    if eval > 0:
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
