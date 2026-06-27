"""JAX/Optax PPO training for direct Level2 and Level3 racing policies.

The existing CleanRL trainers keep the simulator and reward wrappers in JAX but
move the policy update through PyTorch.  This module keeps the same environment
construction while replacing the actor, critic, rollout storage, GAE, PPO
update, checkpointing, and deterministic evaluation with JAX arrays and Optax.
"""

from __future__ import annotations

import os
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import fire
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from lsy_drone_racing.utils import load_config

if TYPE_CHECKING:
    from types import ModuleType

    from gymnasium.vector import VectorEnv

ROOT = Path(__file__).parents[2]
CONTROL_DIR = Path(__file__).parent
CHECKPOINT_FORMAT = "lsy_jax_optax_ppo_v1"
LOG_2PI = float(np.log(2.0 * np.pi))
LOG_2PI_E = float(np.log(2.0 * np.pi * np.e))

LEVEL2_FAST2_REFERENCE_PARAMS: dict[str, Any] = {
    "level": "level2",
    "config": "level2_dr.toml",
    "seed": 42,
    "total_timesteps": 200_000_000,
    "learning_rate": 3e-4,
    "num_envs": 1024,
    "num_steps": 32,
    "anneal_lr": True,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "num_minibatches": 8,
    "update_epochs": 5,
    "norm_adv": True,
    "clip_coef": 0.26,
    "clip_vloss": True,
    "ent_coef": 0.02,
    "vf_coef": 0.7,
    "max_grad_norm": 1.5,
    "weight_decay": 0.01,
    "target_kl": 0.03,
    "hidden_dim": 256,
    "n_obs": 2,
    "progress_coef": 0.0,
    "gate_stage_coef": 10.0,
    "gate_axis_coef": 12.0,
    "near_gate_coef": 0.0,
    "gate_bonus": 80.0,
    "gate_front_bonus": 4.0,
    "gate_back_bonus": 8.0,
    "finish_bonus": 160.0,
    "missed_gate_penalty": 0.0,
    "wrong_side_penalty": 6.0,
    "crash_penalty": 50.0,
    "obstacle_coef": 5.0,
    "obstacle_margin": 0.35,
    "obstacle_clearance_coef": 0.0,
    "timeout_penalty": 80.0,
    "time_penalty": 0.08,
    "act_coef": 0.02,
    "d_act_th_coef": 0.10,
    "d_act_xy_coef": 0.10,
    "cmd_tilt_coef": 1.0,
    "rpy_coef": 1.0,
    "tilt_limit_deg": 45.0,
    "tilt_excess_coef": 10.0,
}


@dataclass
class JaxPPOArgs:
    """Configuration for the JAX PPO trainer."""

    level: str = "level3"
    config: str = "level3.toml"
    seed: int = 42
    jax_device: str = "gpu"
    wandb_project_name: str = "ADR-PPO-Racing"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_run_id: str | None = None
    wandb_mode: str = "online"
    log_interval: int = 1

    total_timesteps: int = 2_000_000
    learning_rate: float = 3e-4
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
    weight_decay: float = 0.01
    target_kl: float | None = 0.03
    hidden_dim: int = 128
    initial_logstd_roll_pitch: float = -1.2
    initial_logstd_yaw: float = -2.0
    initial_logstd_thrust: float = -0.7

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
    debug_obs: bool = False
    debug_reward_every: int = 0

    @staticmethod
    def create(**kwargs: Any) -> "JaxPPOArgs":
        """Create args and derive PPO batch sizes."""
        args = JaxPPOArgs(**kwargs)
        if args.level not in {"level2", "level3"}:
            raise ValueError(f"level must be 'level2' or 'level3', got {args.level!r}.")
        if args.log_interval <= 0:
            raise ValueError("log_interval must be positive.")
        args.batch_size = int(args.num_envs * args.num_steps)
        if args.num_minibatches <= 0:
            raise ValueError("num_minibatches must be positive.")
        if args.batch_size % args.num_minibatches != 0:
            raise ValueError("num_envs * num_steps must be divisible by num_minibatches.")
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = max(1, int(args.total_timesteps) // args.batch_size)
        return args


def level2_fast2_args(**overrides: Any) -> JaxPPOArgs:
    """Return the Level2 Fast2 PPO parameter preset used for JAX validation."""
    params = dict(LEVEL2_FAST2_REFERENCE_PARAMS)
    params.update(overrides)
    return JaxPPOArgs.create(**params)


def _level_module(level: str) -> tuple[ModuleType, str]:
    """Load the existing environment wrapper module for a level."""
    try:
        if level == "level2":
            from lsy_drone_racing.control import train_CleanRL_ppo as module
            from lsy_drone_racing.control.ppo_level2_observation import OBSERVATION_LAYOUT

            return module, OBSERVATION_LAYOUT
        if level == "level3":
            from lsy_drone_racing.control import train_CleanRL_ppo_level3 as module

            return module, module.LEVEL3_OBSERVATION_LAYOUT
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "wandb"}:
            raise RuntimeError(
                "The JAX trainer reuses the existing race wrappers from the CleanRL files. "
                "Run it from an environment that can import those wrappers, e.g. "
                "`pixi run -e gpu python ...` or `pixi run -e tests python ...`."
            ) from exc
        raise
    raise ValueError(f"Unsupported level {level!r}.")


def set_seeds(seed: int) -> None:
    """Seed host-side random sources."""
    random.seed(seed)
    np.random.seed(seed)


def reward_coefs(args: JaxPPOArgs) -> dict[str, float | int]:
    """Extract reward and observation-wrapper coefficients from args."""
    keys = (
        "n_obs",
        "rpy_coef",
        "tilt_limit_deg",
        "tilt_excess_coef",
        "cmd_tilt_coef",
        "d_act_xy_coef",
        "d_act_th_coef",
        "act_coef",
        "progress_coef",
        "gate_axis_coef",
        "gate_stage_coef",
        "gate_front_bonus",
        "gate_back_bonus",
        "gate_stage_offset",
        "gate_stage_radius",
        "wrong_side_penalty",
        "near_gate_coef",
        "gate_bonus",
        "finish_bonus",
        "missed_gate_penalty",
        "crash_penalty",
        "obstacle_coef",
        "obstacle_margin",
        "obstacle_clearance_coef",
        "timeout_penalty",
        "time_penalty",
    )
    return {key: getattr(args, key) for key in keys}


def make_jax_envs(
    *,
    level: str,
    config: str,
    num_envs: int,
    jax_device: str,
    coefs: dict[str, Any] | None = None,
    debug_obs: bool = False,
    debug_reward_every: int = 0,
) -> VectorEnv:
    """Create direct racing environments that return JAX arrays."""
    module, _layout = _level_module(level)
    coefs = {} if coefs is None else coefs
    cfg = load_config(ROOT / "config" / config)
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
        raise ValueError("Direct JAX PPO currently expects env.control_mode = 'attitude'.")

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
        env = module.ThrustScaleBatterySag(
            env,
            thrust_disturbance,
            env_freq=cfg.env.freq,
            seed=cfg.env.seed,
        )
    if action_latency is not None or command_response is not None:
        env = module.ActionLatencyResponseLag(
            env,
            action_latency,
            command_response,
            env_freq=cfg.env.freq,
            seed=cfg.env.seed,
        )
    env = module.NormalizeVectorActions(env)
    env = module.Level2RaceReward(
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
        env = module.ObservationLatencyNoise(
            env,
            observation_latency,
            observation_noise,
            seed=cfg.env.seed,
        )
    env = module.RaceObservation(
        env,
        n_history=coefs.get("n_obs", 0),
        debug_obs=debug_obs,
    )
    return env


def _orthogonal(key: Array, shape: tuple[int, ...], scale: float) -> Array:
    """Return an orthogonal initializer in Linear's JAX layout, [in, out]."""
    init = jax.nn.initializers.orthogonal(scale=scale, column_axis=-1)
    return init(key, shape, jnp.float32)


def _initial_logstd(args: JaxPPOArgs, action_dim: int) -> Array:
    """Create CleanRL-compatible initial log standard deviations."""
    if action_dim == 4:
        return jnp.array(
            [
                args.initial_logstd_roll_pitch,
                args.initial_logstd_roll_pitch,
                args.initial_logstd_yaw,
                args.initial_logstd_thrust,
            ],
            dtype=jnp.float32,
        )
    return jnp.full((action_dim,), -1.0, dtype=jnp.float32)


def init_actor_critic_params(
    key: Array, *, obs_dim: int, hidden_dim: int, action_dim: int, args: JaxPPOArgs
) -> dict[str, Any]:
    """Initialize separate actor and critic MLPs matching the CleanRL topology."""
    keys = jax.random.split(key, 7)
    return {
        "actor": {
            "w1": _orthogonal(keys[0], (obs_dim, hidden_dim), np.sqrt(2.0)),
            "b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
            "w2": _orthogonal(keys[1], (hidden_dim, hidden_dim), np.sqrt(2.0)),
            "b2": jnp.zeros((hidden_dim,), dtype=jnp.float32),
            "w3": _orthogonal(keys[2], (hidden_dim, action_dim), 0.01),
            "b3": jnp.zeros((action_dim,), dtype=jnp.float32),
            "log_std": _initial_logstd(args, action_dim),
        },
        "critic": {
            "w1": _orthogonal(keys[3], (obs_dim, hidden_dim), np.sqrt(2.0)),
            "b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
            "w2": _orthogonal(keys[4], (hidden_dim, hidden_dim), np.sqrt(2.0)),
            "b2": jnp.zeros((hidden_dim,), dtype=jnp.float32),
            "w3": _orthogonal(keys[5], (hidden_dim, 1), 1.0),
            "b3": jnp.zeros((1,), dtype=jnp.float32),
        },
    }


def actor_apply(params: dict[str, Any], obs: Array) -> tuple[Array, Array]:
    """Return bounded action mean and log std."""
    hidden = jnp.tanh(obs @ params["actor"]["w1"] + params["actor"]["b1"])
    hidden = jnp.tanh(hidden @ params["actor"]["w2"] + params["actor"]["b2"])
    mean = jnp.tanh(hidden @ params["actor"]["w3"] + params["actor"]["b3"])
    return mean.astype(jnp.float32), params["actor"]["log_std"].astype(jnp.float32)


def critic_apply(params: dict[str, Any], obs: Array) -> Array:
    """Return scalar value estimates."""
    hidden = jnp.tanh(obs @ params["critic"]["w1"] + params["critic"]["b1"])
    hidden = jnp.tanh(hidden @ params["critic"]["w2"] + params["critic"]["b2"])
    value = hidden @ params["critic"]["w3"] + params["critic"]["b3"]
    return jnp.squeeze(value, axis=-1).astype(jnp.float32)


def actor_critic_apply(params: dict[str, Any], obs: Array) -> tuple[Array, Array, Array]:
    """Return action mean, action log std, and value."""
    mean, log_std = actor_apply(params, obs)
    return mean, log_std, critic_apply(params, obs)


def gaussian_logprob(action: Array, mean: Array, log_std: Array) -> Array:
    """Log probability under a diagonal Gaussian."""
    normalized = (action - mean) * jnp.exp(-log_std)
    return -0.5 * jnp.sum(jnp.square(normalized) + 2.0 * log_std + LOG_2PI, axis=-1)


def gaussian_entropy(mean: Array, log_std: Array) -> Array:
    """Entropy under a diagonal Gaussian, broadcast to batch shape."""
    entropy = 0.5 * jnp.sum(LOG_2PI_E + 2.0 * log_std)
    return jnp.broadcast_to(entropy, mean.shape[:-1])


@jax.jit
def policy_step(
    params: dict[str, Any], obs: Array, key: Array, deterministic: bool
) -> tuple[Array, Array, Array, Array]:
    """Sample or select an action and return action, logprob, entropy, value."""
    mean, log_std, value = actor_critic_apply(params, obs)
    action = jnp.where(
        deterministic,
        mean,
        mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape, dtype=jnp.float32),
    )
    logprob = gaussian_logprob(action, mean, log_std)
    entropy = gaussian_entropy(mean, log_std)
    return action.astype(jnp.float32), logprob, entropy, value


def compute_gae(
    rewards: Array,
    dones: Array,
    values: Array,
    next_value: Array,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Array, Array]:
    """Compute generalized advantage estimates."""

    def gae_step(
        carry: tuple[Array, Array], transition: tuple[Array, Array, Array]
    ) -> tuple[tuple[Array, Array], Array]:
        next_advantage, next_values = carry
        reward, done, value = transition
        next_nonterminal = 1.0 - done
        delta = reward + gamma * next_values * next_nonterminal - value
        advantage = delta + gamma * gae_lambda * next_nonterminal * next_advantage
        return (advantage, value), advantage

    (_carry, _value), reversed_advantages = jax.lax.scan(
        gae_step,
        (jnp.zeros_like(next_value), next_value),
        (rewards[::-1], dones[::-1], values[::-1]),
    )
    advantages = reversed_advantages[::-1]
    returns = advantages + values
    return advantages.astype(jnp.float32), returns.astype(jnp.float32)


def flatten_rollout(
    transitions: dict[str, Array], advantages: Array, returns: Array
) -> dict[str, Array]:
    """Flatten time/env rollout tensors into a PPO batch."""

    def flatten(value: Array) -> Array:
        return value.reshape((value.shape[0] * value.shape[1],) + value.shape[2:])

    return {
        "obs": flatten(transitions["obs"]),
        "actions": flatten(transitions["actions"]),
        "logprobs": flatten(transitions["logprobs"]),
        "advantages": advantages.reshape(-1),
        "returns": returns.reshape(-1),
        "values": flatten(transitions["values"]),
    }


@jax.jit
def compute_advantage_batch(
    params: dict[str, Any],
    final_obs: Array,
    transitions: dict[str, Array],
    gamma: float,
    gae_lambda: float,
) -> tuple[dict[str, Array], dict[str, Array]]:
    """Compute a flattened PPO batch and summary metrics."""
    next_value = critic_apply(params, final_obs)
    advantages, returns = compute_gae(
        transitions["rewards"],
        transitions["dones"],
        transitions["values"],
        next_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    batch = flatten_rollout(transitions, advantages, returns)
    summary = {
        "rollout_reward_mean": jnp.mean(transitions["rewards"]),
        "rollout_done_mean": jnp.mean(transitions["dones"]),
        "advantages_mean": jnp.mean(advantages),
        "advantages_std": jnp.std(advantages),
        "returns_mean": jnp.mean(returns),
        "values_mean": jnp.mean(transitions["values"]),
        "all_finite": (
            jnp.all(jnp.isfinite(transitions["obs"]))
            & jnp.all(jnp.isfinite(transitions["actions"]))
            & jnp.all(jnp.isfinite(transitions["rewards"]))
        ),
    }
    return batch, summary


def build_update_fn(optimizer: optax.GradientTransformation, args: JaxPPOArgs) -> Any:
    """Build a JIT-compiled clipped PPO update."""

    def loss_fn(params: dict[str, Any], batch: dict[str, Array]) -> tuple[Array, dict[str, Array]]:
        mean, log_std, new_values = actor_critic_apply(params, batch["obs"])
        new_logprobs = gaussian_logprob(batch["actions"], mean, log_std)
        entropy = gaussian_entropy(mean, log_std)
        logratio = new_logprobs - batch["logprobs"]
        ratio = jnp.exp(logratio)
        advantages = batch["advantages"]
        if args.norm_adv:
            advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * jnp.clip(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
        policy_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))
        if args.clip_vloss:
            v_loss_unclipped = jnp.square(new_values - batch["returns"])
            v_clipped = batch["values"] + jnp.clip(
                new_values - batch["values"], -args.clip_coef, args.clip_coef
            )
            v_loss_clipped = jnp.square(v_clipped - batch["returns"])
            value_loss = 0.5 * jnp.mean(jnp.maximum(v_loss_unclipped, v_loss_clipped))
        else:
            value_loss = 0.5 * jnp.mean(jnp.square(new_values - batch["returns"]))
        entropy_loss = jnp.mean(entropy)
        loss = policy_loss - args.ent_coef * entropy_loss + args.vf_coef * value_loss
        approx_kl = jnp.mean((ratio - 1.0) - logratio)
        old_approx_kl = jnp.mean(-logratio)
        clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > args.clip_coef).astype(jnp.float32))
        return loss, {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy_loss,
            "old_approx_kl": old_approx_kl,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }

    def minibatch_update(
        carry: tuple[dict[str, Any], optax.OptState], minibatch: dict[str, Array]
    ) -> tuple[tuple[dict[str, Any], optax.OptState], dict[str, Array]]:
        params, opt_state = carry
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, minibatch)
        grad_norm = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), metrics | {"loss": loss, "grad_norm": grad_norm}

    @jax.jit
    def ppo_update(
        params: dict[str, Any],
        opt_state: optax.OptState,
        batch: dict[str, Array],
        key: Array,
    ) -> tuple[dict[str, Any], optax.OptState, Array, dict[str, Array]]:
        def epoch_update(
            carry: tuple[dict[str, Any], optax.OptState, Array], _unused: None
        ) -> tuple[tuple[dict[str, Any], optax.OptState, Array], dict[str, Array]]:
            epoch_params, epoch_opt_state, epoch_key = carry
            epoch_key, perm_key = jax.random.split(epoch_key)
            permutation = jax.random.permutation(perm_key, args.batch_size)
            shuffled = jax.tree_util.tree_map(lambda value: value[permutation], batch)
            minibatches = jax.tree_util.tree_map(
                lambda value: value.reshape(
                    (args.num_minibatches, args.minibatch_size) + value.shape[1:]
                ),
                shuffled,
            )
            (epoch_params, epoch_opt_state), mb_metrics = jax.lax.scan(
                minibatch_update, (epoch_params, epoch_opt_state), minibatches
            )
            return (epoch_params, epoch_opt_state, epoch_key), jax.tree_util.tree_map(
                jnp.mean, mb_metrics
            )

        (params, opt_state, key), epoch_metrics = jax.lax.scan(
            epoch_update, (params, opt_state, key), None, length=args.update_epochs
        )
        return params, opt_state, key, jax.tree_util.tree_map(jnp.mean, epoch_metrics)

    return ppo_update


def collect_rollout(
    envs: VectorEnv,
    params: dict[str, Any],
    next_obs: Array,
    next_done: Array,
    key: Array,
    args: JaxPPOArgs,
) -> tuple[Array, Array, Array, dict[str, Array], dict[str, float]]:
    """Collect one PPO rollout using JAX policy arrays."""
    obs_rows: list[Array] = []
    action_rows: list[Array] = []
    logprob_rows: list[Array] = []
    reward_rows: list[Array] = []
    done_rows: list[Array] = []
    value_rows: list[Array] = []
    info_accumulator: dict[str, list[float]] = {}

    for _step in range(args.num_steps):
        obs_rows.append(next_obs)
        done_rows.append(next_done.astype(jnp.float32))
        key, action_key = jax.random.split(key)
        action, logprob, _entropy, value = policy_step(params, next_obs, action_key, False)
        action_rows.append(action)
        logprob_rows.append(logprob)
        value_rows.append(value)
        next_obs_np, reward, terminations, truncations, infos = envs.step(action)
        next_obs = jnp.asarray(next_obs_np, dtype=jnp.float32)
        reward_rows.append(jnp.asarray(reward, dtype=jnp.float32))
        next_done = jnp.asarray(terminations | truncations, dtype=jnp.float32)
        for name, value in infos.items():
            if not (name.startswith("reward_") or name.startswith("race_")):
                continue
            info_accumulator.setdefault(name, []).append(float(np.asarray(value).mean()))

    transitions = {
        "obs": jnp.stack(obs_rows),
        "actions": jnp.stack(action_rows),
        "logprobs": jnp.stack(logprob_rows),
        "rewards": jnp.stack(reward_rows),
        "dones": jnp.stack(done_rows),
        "values": jnp.stack(value_rows),
    }
    info_means = {
        name: float(np.mean(values)) for name, values in sorted(info_accumulator.items())
    }
    return next_obs, next_done, key, transitions, info_means


def setup_wandb(args: JaxPPOArgs) -> Any:
    """Initialize W&B only when requested."""
    import wandb

    init_kwargs = {
        "project": args.wandb_project_name,
        "entity": args.wandb_entity,
        "name": args.wandb_run_name,
        "id": args.wandb_run_id,
        "mode": args.wandb_mode,
        "config": asdict(args),
    }
    if args.wandb_run_id:
        init_kwargs["resume"] = "allow"
    run = wandb.init(**init_kwargs)
    wandb.define_metric("global_step")
    wandb.define_metric("losses/*", step_metric="global_step")
    wandb.define_metric("rollout/*", step_metric="global_step")
    wandb.define_metric("reward_components/*", step_metric="global_step")
    wandb.define_metric("race/*", step_metric="global_step")
    return run


def jsonable(value: Any) -> Any:
    """Convert arrays and paths to pickle/JSON friendly values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def save_jax_checkpoint(
    path: Path,
    *,
    args: JaxPPOArgs,
    params: dict[str, Any],
    opt_state: optax.OptState,
    global_step: int,
    metrics: dict[str, Any],
    observation_layout: str,
    obs_dim: int,
    action_dim: int,
    rng_key: Array,
) -> None:
    """Save a JAX PPO checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "global_step": int(global_step),
        "params": jax.device_get(params),
        "optimizer_state": jax.device_get(opt_state),
        "rng_key": jax.device_get(rng_key),
        "metadata": {
            "backend": "jax_optax_ppo",
            "level": args.level,
            "config": args.config,
            "observation_layout": observation_layout,
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "hidden_dim": int(args.hidden_dim),
            "num_envs": int(args.num_envs),
            "num_steps": int(args.num_steps),
            "args": jsonable(asdict(args)),
        },
        "metrics": jsonable(metrics),
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_jax_checkpoint(
    path: Path, device: jax.Device | None = None
) -> tuple[dict[str, Any], optax.OptState | None, int, Array | None, dict[str, Any]]:
    """Load a JAX PPO checkpoint."""
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format in {path}: {payload.get('format')!r}.")
    put = (
        (lambda value: jax.device_put(value, device))
        if device is not None
        else (lambda value: value)
    )
    params = jax.tree_util.tree_map(put, payload["params"])
    opt_state = (
        jax.tree_util.tree_map(put, payload["optimizer_state"])
        if "optimizer_state" in payload
        else None
    )
    rng_key = put(payload["rng_key"]) if "rng_key" in payload else None
    return params, opt_state, int(payload["global_step"]), rng_key, payload


def checkpoint_step_path(model_path: Path, step: int) -> Path:
    """Return a milestone checkpoint path."""
    stem = model_path.stem.removesuffix("_final")
    return model_path.with_name(f"{stem}_step_{int(step):09d}{model_path.suffix}")


def _resolve_device(jax_device: str) -> jax.Device:
    """Resolve a JAX device name with a clear error."""
    try:
        return jax.devices(jax_device)[0]
    except RuntimeError as exc:
        raise RuntimeError(f"No JAX device available for {jax_device!r}.") from exc


def _optimizer(args: JaxPPOArgs) -> optax.GradientTransformation:
    """Build the PPO optimizer."""
    if args.anneal_lr:
        total_gradient_steps = max(
            1, args.num_iterations * args.update_epochs * args.num_minibatches
        )
        learning_rate: float | optax.Schedule = optax.linear_schedule(
            init_value=float(args.learning_rate),
            end_value=0.0,
            transition_steps=total_gradient_steps,
        )
    else:
        learning_rate = float(args.learning_rate)
    return optax.chain(
        optax.clip_by_global_norm(float(args.max_grad_norm)),
        optax.adamw(
            learning_rate=learning_rate,
            eps=1e-5,
            weight_decay=float(args.weight_decay),
        ),
    )


def train_ppo(
    args: JaxPPOArgs,
    model_path: Path | None,
    *,
    wandb_enabled: bool = False,
    checkpoint_dir: Path | str | None = None,
    checkpoint_interval: int = 0,
    resume_from: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Train a JAX PPO policy."""
    set_seeds(args.seed)
    device = _resolve_device(args.jax_device)
    module, observation_layout = _level_module(args.level)
    run = setup_wandb(args) if wandb_enabled else None
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_interval > 0 and checkpoint_dir is None:
        checkpoint_dir = (
            model_path.parent if model_path is not None else CONTROL_DIR / "checkpoints"
        )
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = (
        model_path.stem.removesuffix("_final") if model_path is not None else f"jax_{args.level}"
    )
    next_checkpoint_step = checkpoint_interval if checkpoint_interval > 0 else None

    print(
        {
            "backend": "jax_optax_ppo",
            "device": str(device),
            "level": args.level,
            "config": args.config,
            "num_iterations": args.num_iterations,
            "batch_size": args.batch_size,
            "observation_layout": observation_layout,
        }
    )

    with jax.default_device(device):
        envs = make_jax_envs(
            level=args.level,
            config=args.config,
            num_envs=args.num_envs,
            jax_device=args.jax_device,
            coefs=reward_coefs(args),
            debug_obs=args.debug_obs,
            debug_reward_every=args.debug_reward_every,
        )
        if not isinstance(envs.single_action_space, gym.spaces.Box):
            raise ValueError("Only continuous action spaces are supported.")
        obs_shape = tuple(envs.single_observation_space.shape)
        action_shape = tuple(envs.single_action_space.shape)
        obs_dim = int(np.prod(obs_shape))
        action_dim = int(np.prod(action_shape))
        optimizer = _optimizer(args)
        rng_key = jax.random.PRNGKey(args.seed)
        rng_key, init_key, update_key = jax.random.split(rng_key, 3)
        global_step = 0
        if resume_from is not None:
            params, opt_state, global_step, restored_key, _payload = load_jax_checkpoint(
                Path(resume_from), device
            )
            if restored_key is not None:
                rng_key = restored_key
            if opt_state is None:
                opt_state = optimizer.init(params)
            print(f"resumed JAX PPO checkpoint from {resume_from} at step={global_step}")
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
        next_obs_np, _info = envs.reset(seed=args.seed)
        next_obs = jnp.asarray(next_obs_np, dtype=jnp.float32)
        next_done = jnp.zeros((args.num_envs,), dtype=jnp.float32)
        history: list[dict[str, Any]] = []
        start_time = time.time()
        try:
            for iteration in range(1, args.num_iterations + 1):
                iter_start = time.time()
                next_obs, next_done, rng_key, transitions, info_means = collect_rollout(
                    envs, params, next_obs, next_done, rng_key, args
                )
                rng_key, update_key = jax.random.split(update_key)
                batch, rollout_summary = compute_advantage_batch(
                    params, next_obs, transitions, float(args.gamma), float(args.gae_lambda)
                )
                params, opt_state, update_key, train_metrics = ppo_update(
                    params, opt_state, batch, update_key
                )
                jax.tree_util.tree_map(lambda value: value.block_until_ready(), train_metrics)
                jax.tree_util.tree_map(lambda value: value.block_until_ready(), rollout_summary)
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
                    f"losses/{key}": float(np.asarray(value))
                    for key, value in train_metrics.items()
                }
                metrics |= {
                    f"rollout/{key}": float(np.asarray(value))
                    for key, value in rollout_summary.items()
                }
                metrics |= {
                    (
                        f"reward_components/{key.removeprefix('reward_')}"
                        if key.startswith("reward_")
                        else f"race/{key.removeprefix('race_')}"
                    ): value
                    for key, value in info_means.items()
                }
                history.append(metrics)
                if run is not None:
                    import wandb

                    wandb.log(metrics, step=global_step)
                should_print = (
                    iteration % args.log_interval == 0 or iteration == args.num_iterations
                )
                if should_print:
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
                        observation_layout=observation_layout,
                        obs_dim=obs_dim,
                        action_dim=action_dim,
                        rng_key=rng_key,
                    )
                    print(f"checkpoint saved to {checkpoint_path} at global_step={global_step}")
                    next_checkpoint_step += checkpoint_interval
            total_elapsed = time.time() - start_time
            final_metrics = history[-1] if history else {"global_step": int(global_step)}
            final_metrics = dict(final_metrics) | {"train/total_elapsed_s": total_elapsed}
            if model_path is not None:
                save_jax_checkpoint(
                    model_path,
                    args=args,
                    params=params,
                    opt_state=opt_state,
                    global_step=global_step,
                    metrics=final_metrics,
                    observation_layout=observation_layout,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    rng_key=rng_key,
                )
                print(f"model saved to {model_path}")
        finally:
            envs.close()
            if run is not None:
                import wandb

                wandb.finish()
    return history


def evaluate_ppo(
    args: JaxPPOArgs,
    *,
    n_eval: int,
    model_path: Path,
    seed_start: int | None = None,
    max_steps: int = 1500,
) -> dict[str, Any]:
    """Evaluate a JAX checkpoint deterministically in the training wrapper."""
    device = _resolve_device(args.jax_device)
    params, _opt_state, global_step, _rng_key, payload = load_jax_checkpoint(model_path, device)
    metadata = payload["metadata"]
    if metadata.get("level") != args.level:
        raise ValueError(f"Checkpoint level {metadata.get('level')!r} != requested {args.level!r}.")
    module, observation_layout = _level_module(args.level)
    if metadata.get("observation_layout") != observation_layout:
        raise ValueError(
            f"Checkpoint layout {metadata.get('observation_layout')!r} != {observation_layout!r}."
        )
    seed_start = args.seed if seed_start is None else seed_start
    rows: list[dict[str, Any]] = []
    with jax.default_device(device):
        env = make_jax_envs(
            level=args.level,
            config=args.config,
            num_envs=1,
            jax_device=args.jax_device,
            coefs=reward_coefs(args),
        )
        try:
            for episode in range(n_eval):
                obs_np, _info = env.reset(seed=seed_start + episode)
                obs = jnp.asarray(obs_np, dtype=jnp.float32)
                episode_reward = 0.0
                success = False
                crashed = False
                timeout = False
                steps = 0
                while steps < max_steps:
                    action, _logprob, _entropy, _value = policy_step(
                        params, obs, jax.random.PRNGKey(0), True
                    )
                    obs_np, reward, terminated, truncated, info = env.step(action)
                    obs = jnp.asarray(obs_np, dtype=jnp.float32)
                    episode_reward += float(np.asarray(reward).reshape(-1)[0])
                    steps += 1
                    done = bool(np.asarray(terminated | truncated).reshape(-1)[0])
                    if "race_finished_rate" in info:
                        success = bool(np.asarray(info["race_finished_rate"]).reshape(-1)[0] > 0.5)
                    if "race_crashed_rate" in info:
                        crashed = bool(np.asarray(info["race_crashed_rate"]).reshape(-1)[0] > 0.5)
                    timeout = bool(np.asarray(truncated).reshape(-1)[0]) and not success
                    if done:
                        break
                rows.append(
                    {
                        "episode": episode + 1,
                        "seed": seed_start + episode,
                        "success": success,
                        "crashed": crashed,
                        "timeout": timeout,
                        "reward": episode_reward,
                        "steps": steps,
                    }
                )
        finally:
            env.close()
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


def debug_rollout(args: JaxPPOArgs, *, n_steps: int = 20) -> None:
    """Run zero-action rollout diagnostics without training."""
    envs = make_jax_envs(
        level=args.level,
        config=args.config,
        num_envs=args.num_envs,
        jax_device=args.jax_device,
        coefs=reward_coefs(args),
        debug_obs=True,
        debug_reward_every=args.debug_reward_every,
    )
    try:
        obs, _info = envs.reset(seed=args.seed)
        print(f"[debug-rollout] obs_shape={tuple(np.asarray(obs).shape)}")
        print(f"[debug-rollout] single_obs_space={envs.single_observation_space}")
        print(f"[debug-rollout] single_action_space={envs.single_action_space}")
        action = jnp.zeros((args.num_envs,) + envs.single_action_space.shape, dtype=jnp.float32)
        for step in range(n_steps):
            _obs, reward, terminated, truncated, _info = envs.step(action)
            done = terminated | truncated
            print(
                f"[debug-rollout] step={step + 1}/{n_steps} "
                f"reward_mean={float(np.asarray(reward).mean()):.3f} "
                f"done={int(np.asarray(done).sum())}/{args.num_envs}"
            )
    finally:
        envs.close()


def _default_config_for_level(level: str) -> str:
    return "level3.toml" if level == "level3" else "level2_dr.toml"


def _default_model_for_level(level: str) -> Path:
    run_name = f"jax_ppo_{level}"
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
    debug_steps: int = 0,
    model_name: str | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = 0,
    resume_from: str | None = None,
    seed_start: int | None = None,
    **overrides: Any,
) -> Any:
    """CLI entry point for JAX PPO training and evaluation."""
    args = JaxPPOArgs.create(
        level=level,
        config=config if config is not None else _default_config_for_level(level),
        **overrides,
    )
    model_path = _resolve_model_path(model_name, level)
    if debug_steps > 0:
        debug_rollout(args, n_steps=debug_steps)
    history = None
    if train:
        history = train_ppo(
            args,
            model_path,
            wandb_enabled=wandb_enabled,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            resume_from=resume_from,
        )
    if eval:
        return evaluate_ppo(
            args,
            n_eval=int(eval),
            model_path=model_path,
            seed_start=seed_start,
        )
    return history


if __name__ == "__main__":
    fire.Fire(main)
