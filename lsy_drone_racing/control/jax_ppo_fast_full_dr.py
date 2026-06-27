"""Pure-JAX fast PPO path with train-only DR replicated inside ``lax.scan``.

The existing :mod:`jax_ppo_fast` module keeps the simulator, reward shaping,
PPO update, checkpointing, and evaluation in JAX, but deliberately bypasses the
Python training wrappers to preserve speed.  This module keeps that fast path
unchanged and adds a second entry point that ports the train-only DR wrappers
from the CleanRL trainers into JAX state:

* thrust scale and battery sag
* action latency
* first-order command response lag
* observation latency
* observation noise, including small-angle quaternion noise

No existing files need to be edited to use this path.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, NamedTuple

import fire
import jax
import jax.numpy as jnp
from jax import Array

from lsy_drone_racing.control import jax_ppo_fast as fast
from lsy_drone_racing.control.jax_ppo import CONTROL_DIR, JaxPPOArgs, reward_coefs
from lsy_drone_racing.utils import load_config

ROOT = Path(__file__).parents[2]


class FullDRConfig(NamedTuple):
    """Static train-DR values captured by the JIT-compiled env step."""

    seed: int
    env_freq: float
    thrust_scale_min: float
    thrust_scale_max: float
    battery_sag_min: float
    battery_sag_max: float
    battery_sag_horizon_steps: float
    action_delay_min_steps: int
    action_delay_max_steps: int
    rp_tau_min_s: float
    rp_tau_max_s: float
    yaw_tau_min_s: float
    yaw_tau_max_s: float
    thrust_tau_min_s: float
    thrust_tau_max_s: float
    obs_delay_min_steps: int
    obs_delay_max_steps: int
    pos_std_m: float
    vel_std_mps: float
    ang_vel_std_radps: float
    quat_rpy_std_rad: float
    gate_pos_std_m: float
    gate_rpy_std_rad: float
    obstacle_pos_std_m: float


def _get(section: Any, key: str, default: Any = None) -> Any:
    """Read a config key from dict-like or attribute-like sections."""
    if section is None:
        return default
    if hasattr(section, "get"):
        return section.get(key, default)
    return getattr(section, key, default)


def _require_order(name: str, min_value: float, max_value: float) -> None:
    if min_value < 0.0 or max_value < 0.0:
        raise ValueError(f"{name} bounds must be non-negative.")
    if min_value > max_value:
        raise ValueError(f"{name}_min must be <= {name}_max.")


def load_full_dr_config(args: JaxPPOArgs) -> FullDRConfig:
    """Load ``train.*`` DR settings from the selected TOML config."""
    cfg = load_config(ROOT / "config" / args.config)
    env_cfg = _get(cfg, "env", {})
    train_cfg = _get(cfg, "train", {}) or {}

    disturbances = _get(env_cfg, "disturbances", None)
    legacy_thrust = _get(disturbances, "thrust", None) if disturbances is not None else None
    thrust_cfg = _get(train_cfg, "thrust", None) or legacy_thrust or {}
    action_latency_cfg = _get(train_cfg, "action_latency", {}) or {}
    command_response_cfg = _get(train_cfg, "command_response", {}) or {}
    observation_latency_cfg = _get(train_cfg, "observation_latency", {}) or {}
    observation_noise_cfg = _get(train_cfg, "observation_noise", {}) or {}

    env_freq = float(_get(env_cfg, "freq", 50))
    if env_freq <= 0.0:
        raise ValueError("env.freq must be positive for full-DR JAX training.")
    seed = int(_get(env_cfg, "seed", args.seed))
    if seed == -1:
        seed = int(args.seed)

    thrust_scale_min = float(_get(thrust_cfg, "scale_min", 1.0))
    thrust_scale_max = float(_get(thrust_cfg, "scale_max", 1.0))
    battery_sag_min = float(_get(thrust_cfg, "battery_sag_min", 0.0))
    battery_sag_max = float(_get(thrust_cfg, "battery_sag_max", 0.0))
    battery_sag_horizon_s = float(_get(thrust_cfg, "battery_sag_horizon_s", 10.0))
    if thrust_scale_min <= 0.0 or thrust_scale_max <= 0.0:
        raise ValueError("thrust scale_min/scale_max must be positive.")
    _require_order("thrust_scale", thrust_scale_min, thrust_scale_max)
    _require_order("battery_sag", battery_sag_min, battery_sag_max)
    if battery_sag_horizon_s <= 0.0:
        raise ValueError("battery_sag_horizon_s must be positive.")

    action_delay_min_steps = int(_get(action_latency_cfg, "delay_min_steps", 0))
    action_delay_max_steps = int(_get(action_latency_cfg, "delay_max_steps", 0))
    if action_delay_min_steps < 0 or action_delay_max_steps < 0:
        raise ValueError("action delay bounds must be non-negative.")
    if action_delay_min_steps > action_delay_max_steps:
        raise ValueError("action delay_min_steps must be <= delay_max_steps.")

    rp_tau_min_s = float(
        _get(command_response_cfg, "rp_tau_min_s", _get(command_response_cfg, "rpy_tau_min_s", 0.0))
    )
    rp_tau_max_s = float(
        _get(command_response_cfg, "rp_tau_max_s", _get(command_response_cfg, "rpy_tau_max_s", 0.0))
    )
    yaw_tau_min_s = float(_get(command_response_cfg, "yaw_tau_min_s", rp_tau_min_s))
    yaw_tau_max_s = float(_get(command_response_cfg, "yaw_tau_max_s", rp_tau_max_s))
    thrust_tau_min_s = float(_get(command_response_cfg, "thrust_tau_min_s", 0.0))
    thrust_tau_max_s = float(_get(command_response_cfg, "thrust_tau_max_s", 0.0))
    _require_order("rp_tau", rp_tau_min_s, rp_tau_max_s)
    _require_order("yaw_tau", yaw_tau_min_s, yaw_tau_max_s)
    _require_order("thrust_tau", thrust_tau_min_s, thrust_tau_max_s)

    obs_delay_min_steps = int(_get(observation_latency_cfg, "delay_min_steps", 0))
    obs_delay_max_steps = int(_get(observation_latency_cfg, "delay_max_steps", 0))
    if obs_delay_min_steps < 0 or obs_delay_max_steps < 0:
        raise ValueError("observation delay bounds must be non-negative.")
    if obs_delay_min_steps > obs_delay_max_steps:
        raise ValueError("observation delay_min_steps must be <= delay_max_steps.")

    noise_values = {
        "pos_std_m": float(_get(observation_noise_cfg, "pos_std_m", 0.0)),
        "vel_std_mps": float(_get(observation_noise_cfg, "vel_std_mps", 0.0)),
        "ang_vel_std_radps": float(_get(observation_noise_cfg, "ang_vel_std_radps", 0.0)),
        "quat_rpy_std_rad": float(_get(observation_noise_cfg, "quat_rpy_std_rad", 0.0)),
        "gate_pos_std_m": float(_get(observation_noise_cfg, "gate_pos_std_m", 0.0)),
        "gate_rpy_std_rad": float(_get(observation_noise_cfg, "gate_rpy_std_rad", 0.0)),
        "obstacle_pos_std_m": float(_get(observation_noise_cfg, "obstacle_pos_std_m", 0.0)),
    }
    for name, value in noise_values.items():
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative.")

    return FullDRConfig(
        seed=seed,
        env_freq=env_freq,
        thrust_scale_min=thrust_scale_min,
        thrust_scale_max=thrust_scale_max,
        battery_sag_min=battery_sag_min,
        battery_sag_max=battery_sag_max,
        battery_sag_horizon_steps=env_freq * battery_sag_horizon_s,
        action_delay_min_steps=action_delay_min_steps,
        action_delay_max_steps=action_delay_max_steps,
        rp_tau_min_s=rp_tau_min_s,
        rp_tau_max_s=rp_tau_max_s,
        yaw_tau_min_s=yaw_tau_min_s,
        yaw_tau_max_s=yaw_tau_max_s,
        thrust_tau_min_s=thrust_tau_min_s,
        thrust_tau_max_s=thrust_tau_max_s,
        obs_delay_min_steps=obs_delay_min_steps,
        obs_delay_max_steps=obs_delay_max_steps,
        **noise_values,
    )


def _uniform_or_full(key: Array, shape: tuple[int, ...], minval: float, maxval: float) -> Array:
    if minval == maxval:
        return jnp.full(shape, minval, dtype=jnp.float32)
    return jax.random.uniform(key, shape, dtype=jnp.float32, minval=minval, maxval=maxval)


def _randint_or_full(key: Array, shape: tuple[int, ...], minval: int, maxval: int) -> Array:
    if minval == maxval:
        return jnp.full(shape, minval, dtype=jnp.int32)
    return jax.random.randint(key, shape, minval=minval, maxval=maxval + 1, dtype=jnp.int32)


def _neutral_action(num_envs: int, action_low: Array, action_high: Array) -> Array:
    neutral = (action_low + action_high) / 2.0
    return jnp.broadcast_to(neutral, (num_envs,) + tuple(action_low.shape)).astype(jnp.float32)


def _initial_action_buffer(neutral_action: Array, cfg: FullDRConfig) -> Array:
    return jnp.repeat(neutral_action[:, None, :], cfg.action_delay_max_steps + 1, axis=1)


def _initial_obs_buffer(observations: dict[str, Array], cfg: FullDRConfig) -> dict[str, Array]:
    return {
        key: jnp.repeat(jnp.asarray(value)[:, None, ...], cfg.obs_delay_max_steps + 1, axis=1)
        for key, value in observations.items()
    }


def _sample_episode_params(
    dr_state: dict[str, Any], mask: Array, cfg: FullDRConfig
) -> dict[str, Any]:
    """Sample per-episode DR parameters for reset vector slots."""
    num_envs = int(dr_state["thrust_steps"].shape[0])
    key, scale_key, sag_key, delay_key, rp_key, yaw_key, thrust_key, obs_key = jax.random.split(
        dr_state["rng_key"], 8
    )
    mask = jnp.asarray(mask, dtype=bool)
    thrust_scale = _uniform_or_full(
        scale_key, (num_envs,), cfg.thrust_scale_min, cfg.thrust_scale_max
    )
    battery_sag = _uniform_or_full(
        sag_key, (num_envs,), cfg.battery_sag_min, cfg.battery_sag_max
    )
    action_delay_steps = _randint_or_full(
        delay_key, (num_envs,), cfg.action_delay_min_steps, cfg.action_delay_max_steps
    )
    rp_tau = _uniform_or_full(rp_key, (num_envs,), cfg.rp_tau_min_s, cfg.rp_tau_max_s)
    yaw_tau = _uniform_or_full(yaw_key, (num_envs,), cfg.yaw_tau_min_s, cfg.yaw_tau_max_s)
    thrust_tau = _uniform_or_full(
        thrust_key, (num_envs,), cfg.thrust_tau_min_s, cfg.thrust_tau_max_s
    )
    obs_delay_steps = _randint_or_full(
        obs_key, (num_envs,), cfg.obs_delay_min_steps, cfg.obs_delay_max_steps
    )
    return dr_state | {
        "rng_key": key,
        "thrust_scale": jnp.where(mask, thrust_scale, dr_state["thrust_scale"]),
        "battery_sag": jnp.where(mask, battery_sag, dr_state["battery_sag"]),
        "action_delay_steps": jnp.where(
            mask, action_delay_steps, dr_state["action_delay_steps"]
        ),
        "rp_tau": jnp.where(mask, rp_tau, dr_state["rp_tau"]),
        "yaw_tau": jnp.where(mask, yaw_tau, dr_state["yaw_tau"]),
        "thrust_tau": jnp.where(mask, thrust_tau, dr_state["thrust_tau"]),
        "obs_delay_steps": jnp.where(mask, obs_delay_steps, dr_state["obs_delay_steps"]),
    }


def _quat_multiply(q1: Array, q2: Array) -> Array:
    """Multiply xyzw quaternions."""
    x1, y1, z1, w1 = jnp.moveaxis(q1, -1, 0)
    x2, y2, z2, w2 = jnp.moveaxis(q2, -1, 0)
    return jnp.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def _noise_quat(quat: Array, key: Array, std_rad: float) -> Array:
    delta = std_rad * jax.random.normal(key, quat.shape[:-1] + (3,), dtype=quat.dtype)
    delta_quat = jnp.concatenate(
        [0.5 * delta, jnp.ones(delta.shape[:-1] + (1,), dtype=quat.dtype)], axis=-1
    )
    delta_quat = delta_quat / jnp.linalg.norm(delta_quat, axis=-1, keepdims=True)
    noisy_quat = _quat_multiply(delta_quat, quat)
    return noisy_quat / jnp.linalg.norm(noisy_quat, axis=-1, keepdims=True)


def _split_key(dr_state: dict[str, Any]) -> tuple[dict[str, Any], Array]:
    key, subkey = jax.random.split(dr_state["rng_key"])
    return dr_state | {"rng_key": key}, subkey


def _add_observation_noise(
    observations: dict[str, Array], dr_state: dict[str, Any], cfg: FullDRConfig
) -> tuple[dict[str, Array], dict[str, Any]]:
    noisy = dict(observations)
    for key, std in (
        ("pos", cfg.pos_std_m),
        ("vel", cfg.vel_std_mps),
        ("ang_vel", cfg.ang_vel_std_radps),
        ("gates_pos", cfg.gate_pos_std_m),
        ("obstacles_pos", cfg.obstacle_pos_std_m),
    ):
        if std > 0.0 and key in noisy:
            dr_state, subkey = _split_key(dr_state)
            noisy[key] = noisy[key] + std * jax.random.normal(
                subkey, noisy[key].shape, dtype=noisy[key].dtype
            )
    if cfg.quat_rpy_std_rad > 0.0 and "quat" in noisy:
        dr_state, subkey = _split_key(dr_state)
        noisy["quat"] = _noise_quat(noisy["quat"], subkey, cfg.quat_rpy_std_rad)
    if cfg.gate_rpy_std_rad > 0.0 and "gates_quat" in noisy:
        dr_state, subkey = _split_key(dr_state)
        noisy["gates_quat"] = _noise_quat(noisy["gates_quat"], subkey, cfg.gate_rpy_std_rad)
    return noisy, dr_state


def _select_delayed_observations(dr_state: dict[str, Any]) -> dict[str, Array]:
    env_index = jnp.arange(dr_state["obs_delay_steps"].shape[0])
    return {
        key: value[env_index, dr_state["obs_delay_steps"]]
        for key, value in dr_state["obs_buffer"].items()
    }


def _push_obs_buffer(
    obs_buffer: dict[str, Array], observations: dict[str, Array]
) -> dict[str, Array]:
    return {
        key: jnp.concatenate(
            [jnp.asarray(value)[:, None, ...], obs_buffer[key][:, :-1, ...]], axis=1
        )
        for key, value in observations.items()
    }


def _reset_done_obs_buffer(
    obs_buffer: dict[str, Array], observations: dict[str, Array], done: Array, cfg: FullDRConfig
) -> dict[str, Array]:
    fresh = _initial_obs_buffer(observations, cfg)
    return {
        key: jnp.where(
            done.reshape((done.shape[0],) + (1,) * (value.ndim - 1)),
            fresh[key],
            value,
        )
        for key, value in obs_buffer.items()
    }


def _tau_to_alpha(tau_s: Array, cfg: FullDRConfig) -> Array:
    dt = 1.0 / cfg.env_freq
    return jnp.where(tau_s <= 0.0, 1.0, 1.0 - jnp.exp(-dt / tau_s))


def _apply_action_dr(
    dr_state: dict[str, Any],
    sim_action: Array,
    action_low: Array,
    action_high: Array,
    cfg: FullDRConfig,
) -> tuple[dict[str, Any], Array]:
    action_buffer = jnp.concatenate(
        [sim_action[:, None, ...], dr_state["action_buffer"][:, :-1, ...]], axis=1
    )
    env_index = jnp.arange(sim_action.shape[0])
    delayed_action = action_buffer[env_index, dr_state["action_delay_steps"]]
    rp_alpha = _tau_to_alpha(dr_state["rp_tau"], cfg)
    yaw_alpha = _tau_to_alpha(dr_state["yaw_tau"], cfg)
    thrust_alpha = _tau_to_alpha(dr_state["thrust_tau"], cfg)
    alpha = jnp.stack([rp_alpha, rp_alpha, yaw_alpha, thrust_alpha], axis=-1)
    applied_action = dr_state["applied_action"] + alpha * (
        delayed_action - dr_state["applied_action"]
    )
    applied_action = jnp.clip(applied_action, action_low, action_high)

    progress = jnp.clip(
        dr_state["thrust_steps"].astype(jnp.float32) / cfg.battery_sag_horizon_steps,
        0.0,
        1.0,
    )
    thrust_scale = jnp.maximum(dr_state["thrust_scale"] - dr_state["battery_sag"] * progress, 0.0)
    applied_action = applied_action.at[..., -1].set(applied_action[..., -1] * thrust_scale)
    dr_state = dr_state | {"action_buffer": action_buffer, "applied_action": applied_action}
    return dr_state, applied_action


def _reset_done_action_state(
    dr_state: dict[str, Any], done: Array, action_low: Array, action_high: Array, cfg: FullDRConfig
) -> dict[str, Any]:
    num_envs = int(done.shape[0])
    neutral = _neutral_action(num_envs, action_low, action_high)
    neutral_buffer = _initial_action_buffer(neutral, cfg)
    return dr_state | {
        "thrust_steps": jnp.where(done, 0, dr_state["thrust_steps"] + 1),
        "applied_action": jnp.where(done[:, None], neutral, dr_state["applied_action"]),
        "action_buffer": jnp.where(done[:, None, None], neutral_buffer, dr_state["action_buffer"]),
    }


def _initial_dr_state(
    observations: dict[str, Array], args: JaxPPOArgs, action_low: Array, action_high: Array
) -> tuple[dict[str, Array], dict[str, Any], FullDRConfig]:
    cfg = load_full_dr_config(args)
    num_envs = int(observations["pos"].shape[0])
    neutral = _neutral_action(num_envs, action_low, action_high)
    zero_i = jnp.zeros((num_envs,), dtype=jnp.int32)
    zero_f = jnp.zeros((num_envs,), dtype=jnp.float32)
    dr_state: dict[str, Any] = {
        "rng_key": jax.random.PRNGKey(cfg.seed + 30_003),
        "thrust_steps": zero_i,
        "thrust_scale": jnp.ones((num_envs,), dtype=jnp.float32),
        "battery_sag": zero_f,
        "action_delay_steps": zero_i,
        "rp_tau": zero_f,
        "yaw_tau": zero_f,
        "thrust_tau": zero_f,
        "applied_action": neutral,
        "action_buffer": _initial_action_buffer(neutral, cfg),
        "obs_delay_steps": zero_i,
        "obs_buffer": _initial_obs_buffer(observations, cfg),
    }
    dr_state = _sample_episode_params(dr_state, jnp.ones((num_envs,), dtype=bool), cfg)
    policy_observations, dr_state = _add_observation_noise(
        _select_delayed_observations(dr_state), dr_state, cfg
    )
    return policy_observations, dr_state, cfg


def make_initial_state(
    env: Any, raw_obs: dict[str, Array], args: JaxPPOArgs, action_low: Array, action_high: Array
) -> fast.FastState:
    """Create a fast state whose policy observation includes full train DR."""
    policy_obs, dr_state, _cfg = _initial_dr_state(raw_obs, args, action_low, action_high)
    kind = fast.observation_kind(args.level)
    history_row = fast.basic_history(kind, policy_obs)
    history = jnp.repeat(history_row[:, None, :], int(args.n_obs), axis=1)
    last_action = jnp.zeros((policy_obs["pos"].shape[0], 4), dtype=jnp.float32)
    obs = fast.flatten_observation(kind, policy_obs, history, last_action)
    zero_metric = jnp.array(0.0, dtype=jnp.float32)
    return fast.FastState(
        pipeline_state=env.data,
        obs=obs.astype(jnp.float32),
        reward=jnp.zeros((policy_obs["pos"].shape[0],), dtype=jnp.float32),
        done=jnp.zeros((policy_obs["pos"].shape[0],), dtype=jnp.float32),
        metrics={"reward_mean": zero_metric, "done_mean": zero_metric},
        info={
            "raw_obs": raw_obs,
            "history": history,
            "last_action_norm": last_action,
            "reward_state": fast.initial_reward_state(raw_obs, reward_coefs(args)),
            "dr_state": dr_state,
        },
    )


def build_fast_env_step(
    step_fn: Any, action_low: Array, action_high: Array, args: JaxPPOArgs
) -> Any:
    """Build one pure-JAX racing step with train-only DR applied inside the scan."""
    cfg = load_full_dr_config(args)
    kind = fast.observation_kind(args.level)
    coefs = reward_coefs(args)

    def env_step(
        state: fast.FastState, action_norm: Array
    ) -> tuple[fast.FastState, dict[str, Array]]:
        sim_action = fast.scale_action_jax(action_norm, action_low, action_high)
        dr_state, applied_action = _apply_action_dr(
            state.info["dr_state"], sim_action, action_low, action_high, cfg
        )
        next_data, (raw_obs_full, _sparse_reward, terminated_full, truncated_full, _info) = step_fn(
            state.pipeline_state, applied_action
        )
        raw_obs = fast.drop_drone_dim(raw_obs_full)
        terminated = terminated_full[:, 0]
        truncated = truncated_full[:, 0]
        done = terminated | truncated
        reward, components, race_metrics, reward_state = fast.reward_components(
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

        dr_state = _reset_done_action_state(dr_state, done, action_low, action_high, cfg)
        obs_buffer = _push_obs_buffer(dr_state["obs_buffer"], raw_obs)
        dr_state = dr_state | {"obs_buffer": obs_buffer}
        policy_obs, dr_state = _add_observation_noise(
            _select_delayed_observations(dr_state), dr_state, cfg
        )
        dr_state = dr_state | {
            "obs_buffer": _reset_done_obs_buffer(dr_state["obs_buffer"], raw_obs, done, cfg)
        }
        dr_state = _sample_episode_params(dr_state, done, cfg)

        obs = fast.flatten_observation(kind, policy_obs, state.info["history"], action_norm)
        history_row = fast.basic_history(kind, policy_obs)
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
                "dr_state": dr_state,
            },
        )
        metrics = {
            "reward_mean": jnp.mean(reward),
            "done_mean": jnp.mean(done.astype(jnp.float32)),
            "obs_abs_mean": jnp.mean(jnp.abs(obs)),
            "action_abs_mean": jnp.mean(jnp.abs(action_norm)),
            "dr_thrust_scale_mean": jnp.mean(dr_state["thrust_scale"]),
            "dr_battery_sag_mean": jnp.mean(dr_state["battery_sag"]),
            "dr_action_delay_mean": jnp.mean(dr_state["action_delay_steps"].astype(jnp.float32)),
            "dr_obs_delay_mean": jnp.mean(dr_state["obs_delay_steps"].astype(jnp.float32)),
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


@contextmanager
def _patched_fast_dr() -> Iterator[None]:
    """Temporarily route the reusable fast PPO loop through the full-DR step."""
    original_make_initial_state = fast.make_initial_state
    original_build_fast_env_step = fast.build_fast_env_step
    fast.make_initial_state = make_initial_state
    fast.build_fast_env_step = build_fast_env_step
    try:
        yield
    finally:
        fast.make_initial_state = original_make_initial_state
        fast.build_fast_env_step = original_build_fast_env_step


def train_ppo_fast_full_dr(
    args: JaxPPOArgs,
    model_path: Path | None,
    *,
    wandb_enabled: bool = False,
    checkpoint_dir: Path | str | None = None,
    checkpoint_interval: int = 0,
    resume_from: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Train PPO with full train-only DR inside the pure-JAX fast path."""
    print(
        {
            "backend_extension": "pure_jax_full_train_dr",
            "config": args.config,
            "train_dr": "thrust/action_latency/command_response/observation_latency/noise",
        }
    )
    with _patched_fast_dr():
        return fast.train_ppo_fast(
            args,
            model_path,
            wandb_enabled=wandb_enabled,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            resume_from=resume_from,
        )


def evaluate_ppo_fast_full_dr(
    args: JaxPPOArgs,
    *,
    n_eval: int,
    model_path: Path,
    seed_start: int | None = None,
    max_steps: int = 1500,
) -> dict[str, Any]:
    """Evaluate a fast checkpoint with full train-only DR enabled."""
    eval_args = args if seed_start is None else replace(args, seed=int(seed_start))
    with _patched_fast_dr():
        summary = fast.evaluate_ppo_fast(
            eval_args,
            n_eval=n_eval,
            model_path=model_path,
            seed_start=seed_start,
            max_steps=max_steps,
        )
    summary["full_train_dr"] = True
    return summary


def benchmark_fast_rollout_full_dr(
    args: JaxPPOArgs, *, repeat: int = 5, warmup: int = 1
) -> dict[str, Any]:
    """Benchmark scanned rollout speed with full train-only DR enabled."""
    with _patched_fast_dr():
        summary = fast.benchmark_fast_rollout(args, repeat=repeat, warmup=warmup)
    summary["backend"] = "pure_jax_brax_state_full_train_dr_ppo_rollout"
    summary["full_train_dr"] = True
    return summary


def _default_config_for_level(level: str) -> str:
    return "level3_dr.toml" if level == "level3" else "level2_dr.toml"


def _default_model_for_level(level: str) -> Path:
    run_name = f"jax_fast_full_dr_ppo_{level}"
    return CONTROL_DIR / "checkpoints" / run_name / f"{run_name}_final.pkl"


def _resolve_model_path(model_name: str | Path | None, level: str) -> Path:
    if model_name is None:
        return _default_model_for_level(level)
    path = Path(model_name)
    if not path.is_absolute():
        path = CONTROL_DIR / path
    return path


def level2_validation_args(**overrides: Any) -> JaxPPOArgs:
    """Return the Level2 Fast2 preset for full-DR validation."""
    return fast.level2_validation_args(**overrides)


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
    """CLI entry point for fast JAX PPO with full train-only DR."""
    args = JaxPPOArgs.create(
        level=level,
        config=config if config is not None else _default_config_for_level(level),
        **overrides,
    )
    model_path = _resolve_model_path(model_name, level)
    results: dict[str, Any] = {}
    if benchmark:
        results["benchmark"] = benchmark_fast_rollout_full_dr(
            args, repeat=int(benchmark_repeat), warmup=int(benchmark_warmup)
        )
    if train:
        results["history"] = train_ppo_fast_full_dr(
            args,
            model_path,
            wandb_enabled=wandb_enabled,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            resume_from=resume_from,
        )
    if eval:
        results["eval"] = evaluate_ppo_fast_full_dr(
            args, n_eval=int(eval), model_path=model_path, seed_start=seed_start
        )
    return results or None


if __name__ == "__main__":
    fire.Fire(main)
