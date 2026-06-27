"""Fast Level3 checkpoint sweep using the pure-JAX rollout path."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import Array

from lsy_drone_racing.control import jax_ppo_fast
from lsy_drone_racing.control.jax_ppo import JaxPPOArgs, _resolve_device, load_jax_checkpoint
from lsy_drone_racing.control.ppo_level3_observation import (
    LOCAL_OBSTACLE_OBSERVATION_LAYOUT,
    checkpoint_hidden_dim,
    unpack_checkpoint,
)
from lsy_drone_racing.utils import load_config
from scripts.compare_level3_ppo_checkpoints import (
    SUMMARY_FIELDS,
    TRAINING_REWARD_COEFS,
    checkpoint_label,
    checkpoint_sort_key,
    checkpoint_step,
    percentile,
    safe_mean,
    safe_median,
    write_csv,
)

ROOT = Path(__file__).parents[1]
DEFAULT_CHECKPOINT_DIR = (
    ROOT
    / "lsy_drone_racing/control/checkpoints/level3_Curriculum_Xobstacle_speedlimit2_jax_fast"
)
DEFAULT_OUT_PREFIX = ROOT / "artifacts/level3_checkpoint_eval_fast"


def torch_params_from_checkpoint(path: Path, device: jax.Device) -> tuple[dict[str, Any], str, int]:
    """Load a PyTorch inference checkpoint into the fast JAX parameter layout."""
    checkpoint = torch.load(path, map_location="cpu")
    state_dict, observation_layout = unpack_checkpoint(checkpoint)
    if observation_layout != LOCAL_OBSTACLE_OBSERVATION_LAYOUT:
        raise ValueError(
            f"{path.name} uses {observation_layout}, expected {LOCAL_OBSTACLE_OBSERVATION_LAYOUT}."
        )
    hidden_dim = checkpoint_hidden_dim(checkpoint, state_dict)

    def array(key: str, *, transpose: bool = False, squeeze: bool = False) -> Array:
        value = state_dict[key].detach().cpu().numpy().astype(np.float32)
        if transpose:
            value = value.T
        if squeeze:
            value = value.reshape(-1)
        return jax.device_put(jnp.asarray(value), device)

    params = {
        "actor": {
            "w1": array("actor_mean.0.weight", transpose=True),
            "b1": array("actor_mean.0.bias"),
            "w2": array("actor_mean.2.weight", transpose=True),
            "b2": array("actor_mean.2.bias"),
            "w3": array("actor_mean.4.weight", transpose=True),
            "b3": array("actor_mean.4.bias"),
            "log_std": array("actor_logstd", squeeze=True),
        },
        "critic": {
            "w1": array("critic.0.weight", transpose=True),
            "b1": array("critic.0.bias"),
            "w2": array("critic.2.weight", transpose=True),
            "b2": array("critic.2.bias"),
            "w3": array("critic.4.weight", transpose=True),
            "b3": array("critic.4.bias"),
        },
    }
    return params, observation_layout, hidden_dim


def load_params(path: Path, device: jax.Device) -> tuple[dict[str, Any], str, int]:
    """Load either a fast JAX pickle checkpoint or a PyTorch inference checkpoint."""
    if path.suffix == ".pkl":
        params, _opt_state, _global_step, _rng_key, payload = load_jax_checkpoint(path, device)
        metadata = payload["metadata"]
        observation_layout = metadata.get("observation_layout")
        if observation_layout != LOCAL_OBSTACLE_OBSERVATION_LAYOUT:
            raise ValueError(
                f"{path.name} uses {observation_layout}, expected "
                f"{LOCAL_OBSTACLE_OBSERVATION_LAYOUT}."
            )
        return params, str(observation_layout), int(metadata["hidden_dim"])
    return torch_params_from_checkpoint(path, device)


def build_args(config: str, jax_device: str) -> JaxPPOArgs:
    """Create fast Level3 PPO args matching the checkpoint family."""
    return JaxPPOArgs.create(
        level="level3",
        config=config,
        seed=42,
        jax_device=jax_device,
        hidden_dim=512,
        **TRAINING_REWARD_COEFS,
    )


def build_detailed_eval_fn(
    env_step: Any,
    *,
    max_steps: int,
    env_freq: float,
    pos_limit_low: Array,
    pos_limit_high: Array,
    tilt_limit_deg: float,
    safety_tol_m: float,
) -> Any:
    """Build a compiled deterministic evaluator with per-env diagnostics."""

    def eval_step(
        carry: tuple[Any, ...], _unused: None
    ) -> tuple[tuple[Any, ...], dict[str, Array]]:
        (
            state,
            params,
            done_seen,
            success_seen,
            crash_seen,
            timeout_seen,
            rewards,
            steps,
            gates,
            missed_gates,
            wrong_side_events,
            speed_sum,
            speed_excess_sum,
            speed_excess_reward_sum,
            smooth_penalty_sum,
            max_speed,
            max_tilt,
            max_cmd_tilt,
            tilt_over_limit,
            cmd_tilt_over_limit,
            action_sat_count,
            thrust_sat_count,
            rpy_sat_count,
            action_value_count,
            thrust_value_count,
            rpy_value_count,
            pos_safety_violation_steps,
            pos_samples,
            worst_safety_margin,
        ) = carry
        active = ~done_seen
        previous_action = state.info["last_action_norm"]
        mean, _log_std, _value = jax_ppo_fast.actor_critic_apply(params, state.obs)
        action = jnp.clip(mean, -1.0, 1.0)
        action_delta = action - previous_action
        action_delta_l2 = jnp.linalg.norm(action_delta, axis=-1)
        rpy_delta_l2 = jnp.linalg.norm(action_delta[:, :3], axis=-1)
        thrust_delta_abs = jnp.abs(action_delta[:, 3])

        pos = state.info["raw_obs"]["pos"]
        margins = jnp.minimum(pos - pos_limit_low, pos_limit_high - pos)
        current_margin = jnp.min(margins, axis=-1)
        outside_safety = jnp.any(
            (pos < pos_limit_low - safety_tol_m) | (pos > pos_limit_high + safety_tol_m),
            axis=-1,
        )
        worst_safety_margin = jnp.where(
            active,
            jnp.minimum(worst_safety_margin, current_margin),
            worst_safety_margin,
        )
        pos_safety_violation_steps = pos_safety_violation_steps + jnp.where(
            active, outside_safety.astype(jnp.int32), 0
        )
        pos_samples = pos_samples + active.astype(jnp.int32)

        next_state, metrics = env_step(state, action)
        done_now = active & (next_state.done > 0.5)
        success_now = metrics["eval_finished"] > 0.5
        crash_now = metrics["eval_crashed"] > 0.5
        timeout_now = metrics["eval_timeout"] > 0.5
        rewards = rewards + jnp.where(active, next_state.reward, 0.0)
        steps = steps + active.astype(jnp.int32)
        gates = gates + jnp.where(active, metrics["race_passed_gate_rate"], 0.0)
        missed_gates = missed_gates + jnp.where(active, metrics["race_missed_gate_rate"], 0.0)
        wrong_side_events = wrong_side_events + jnp.where(
            active, metrics["race_wrong_side_gate_rate"], 0.0
        )
        speed = metrics["race_speed"]
        speed_excess = metrics["race_speed_excess"]
        speed_sum = speed_sum + jnp.where(active, speed, 0.0)
        speed_excess_sum = speed_excess_sum + jnp.where(active, speed_excess, 0.0)
        speed_excess_reward_sum = speed_excess_reward_sum + jnp.where(
            active, metrics["reward_speed_excess"], 0.0
        )
        smooth_penalty_sum = smooth_penalty_sum + jnp.where(
            active, -metrics["reward_smooth"], 0.0
        )
        max_speed = jnp.where(active, jnp.maximum(max_speed, speed), max_speed)
        tilt = metrics["race_tilt_angle_deg"]
        cmd_tilt = metrics["race_cmd_tilt_deg"]
        max_tilt = jnp.where(active, jnp.maximum(max_tilt, tilt), max_tilt)
        max_cmd_tilt = jnp.where(active, jnp.maximum(max_cmd_tilt, cmd_tilt), max_cmd_tilt)
        tilt_over_limit = tilt_over_limit + jnp.where(
            active, (tilt > tilt_limit_deg).astype(jnp.int32), 0
        )
        cmd_tilt_over_limit = cmd_tilt_over_limit + jnp.where(
            active, (cmd_tilt > tilt_limit_deg).astype(jnp.int32), 0
        )
        action_sat_count = action_sat_count + jnp.where(
            active, jnp.count_nonzero(jnp.abs(action) >= 0.95, axis=1), 0
        )
        thrust_sat_count = thrust_sat_count + jnp.where(
            active, (jnp.abs(action[:, 3]) >= 0.95).astype(jnp.int32), 0
        )
        rpy_sat_count = rpy_sat_count + jnp.where(
            active, jnp.count_nonzero(jnp.abs(action[:, :3]) >= 0.95, axis=1), 0
        )
        action_value_count = action_value_count + active.astype(jnp.int32) * action.shape[1]
        thrust_value_count = thrust_value_count + active.astype(jnp.int32)
        rpy_value_count = rpy_value_count + active.astype(jnp.int32) * 3
        success_seen = success_seen | (active & success_now)
        crash_seen = crash_seen | (active & crash_now)
        timeout_seen = timeout_seen | (active & timeout_now)
        done_seen = done_seen | done_now

        next_carry = (
            next_state,
            params,
            done_seen,
            success_seen,
            crash_seen,
            timeout_seen,
            rewards,
            steps,
            gates,
            missed_gates,
            wrong_side_events,
            speed_sum,
            speed_excess_sum,
            speed_excess_reward_sum,
            smooth_penalty_sum,
            max_speed,
            max_tilt,
            max_cmd_tilt,
            tilt_over_limit,
            cmd_tilt_over_limit,
            action_sat_count,
            thrust_sat_count,
            rpy_sat_count,
            action_value_count,
            thrust_value_count,
            rpy_value_count,
            pos_safety_violation_steps,
            pos_samples,
            worst_safety_margin,
        )
        traces = {
            "active": active,
            "action_delta_l2": action_delta_l2,
            "rpy_delta_l2": rpy_delta_l2,
            "thrust_delta_abs": thrust_delta_abs,
        }
        return next_carry, traces

    @jax.jit
    def eval_rollout(state: Any, params: dict[str, Any]) -> dict[str, Array]:
        num_envs = state.obs.shape[0]
        zeros_f = jnp.zeros((num_envs,), dtype=jnp.float32)
        zeros_i = jnp.zeros((num_envs,), dtype=jnp.int32)
        zeros_b = jnp.zeros((num_envs,), dtype=bool)
        init = (
            state,
            params,
            zeros_b,
            zeros_b,
            zeros_b,
            zeros_b,
            zeros_f,
            zeros_i,
            zeros_f,
            zeros_f,
            zeros_f,
            zeros_f,
            zeros_f,
            zeros_f,
            zeros_f,
            jnp.full((num_envs,), -jnp.inf, dtype=jnp.float32),
            jnp.full((num_envs,), -jnp.inf, dtype=jnp.float32),
            jnp.full((num_envs,), -jnp.inf, dtype=jnp.float32),
            zeros_i,
            zeros_i,
            zeros_i,
            zeros_i,
            zeros_i,
            zeros_i,
            zeros_i,
            zeros_i,
            zeros_i,
            zeros_i,
            jnp.full((num_envs,), jnp.inf, dtype=jnp.float32),
        )
        carry, traces = jax.lax.scan(eval_step, init, None, length=max_steps)
        (
            _state,
            _params,
            done_seen,
            success_seen,
            crash_seen,
            timeout_seen,
            rewards,
            steps,
            gates,
            missed_gates,
            wrong_side_events,
            speed_sum,
            speed_excess_sum,
            speed_excess_reward_sum,
            smooth_penalty_sum,
            max_speed,
            max_tilt,
            max_cmd_tilt,
            tilt_over_limit,
            cmd_tilt_over_limit,
            action_sat_count,
            thrust_sat_count,
            rpy_sat_count,
            action_value_count,
            thrust_value_count,
            rpy_value_count,
            pos_safety_violation_steps,
            pos_samples,
            worst_safety_margin,
        ) = carry
        denom = jnp.maximum(steps, 1).astype(jnp.float32)
        return {
            "done": done_seen,
            "success": success_seen,
            "crashed": crash_seen,
            "timeout": timeout_seen & ~success_seen,
            "reward": rewards,
            "steps": steps,
            "time_s": steps.astype(jnp.float32) / env_freq,
            "gates": gates,
            "missed_gates": missed_gates,
            "wrong_side_events": wrong_side_events,
            "mean_speed_mps": speed_sum / denom,
            "mean_speed_excess": speed_excess_sum / denom,
            "speed_excess_reward_per_step": speed_excess_reward_sum / denom,
            "max_speed_mps": max_speed,
            "smooth_penalty_per_step": smooth_penalty_sum / denom,
            "max_tilt_deg": max_tilt,
            "tilt_over_limit_frac": tilt_over_limit.astype(jnp.float32) / denom,
            "max_cmd_tilt_deg": max_cmd_tilt,
            "cmd_tilt_over_limit_frac": cmd_tilt_over_limit.astype(jnp.float32) / denom,
            "action_sat_frac": action_sat_count.astype(jnp.float32)
            / jnp.maximum(action_value_count, 1).astype(jnp.float32),
            "thrust_sat_frac": thrust_sat_count.astype(jnp.float32)
            / jnp.maximum(thrust_value_count, 1).astype(jnp.float32),
            "rpy_sat_frac": rpy_sat_count.astype(jnp.float32)
            / jnp.maximum(rpy_value_count, 1).astype(jnp.float32),
            "pos_safety_violation": pos_safety_violation_steps > 0,
            "pos_safety_step_frac": pos_safety_violation_steps.astype(jnp.float32)
            / jnp.maximum(pos_samples, 1).astype(jnp.float32),
            "worst_safety_margin_m": worst_safety_margin,
            "traces": traces,
        }

    return eval_rollout


def rows_from_results(
    results: dict[str, Any],
    checkpoint_path: Path,
    seed_start: int,
) -> list[dict[str, Any]]:
    """Convert JAX result arrays to per-seed CSV rows."""
    host = jax.device_get(results)
    traces = host.pop("traces")
    rows: list[dict[str, Any]] = []
    n_envs = int(host["steps"].shape[0])
    label, _ = checkpoint_label(checkpoint_path)
    checkpoint_file = str(checkpoint_path.relative_to(ROOT))
    for idx in range(n_envs):
        active = traces["active"][:, idx].astype(bool)
        action_delta_l2 = traces["action_delta_l2"][:, idx][active].astype(float).tolist()
        rpy_delta_l2 = traces["rpy_delta_l2"][:, idx][active].astype(float).tolist()
        thrust_delta_abs = traces["thrust_delta_abs"][:, idx][active].astype(float).tolist()
        rows.append(
            {
                "checkpoint": label,
                "checkpoint_file": checkpoint_file,
                "seed": seed_start + idx,
                "success": bool(host["success"][idx]),
                "crashed": bool(host["crashed"][idx]),
                "timeout": bool(host["timeout"][idx]),
                "steps": int(host["steps"][idx]),
                "time_s": float(host["time_s"][idx]),
                "gates": float(host["gates"][idx]),
                "reward": float(host["reward"][idx]),
                "mean_speed_mps": float(host["mean_speed_mps"][idx]),
                "mean_speed_excess": float(host["mean_speed_excess"][idx]),
                "speed_excess_reward_per_step": float(
                    host["speed_excess_reward_per_step"][idx]
                ),
                "max_speed_mps": float(host["max_speed_mps"][idx]),
                "smooth_penalty_per_step": float(host["smooth_penalty_per_step"][idx]),
                "mean_action_delta_l2": safe_mean(action_delta_l2),
                "p95_action_delta_l2": percentile(action_delta_l2, 95),
                "mean_rpy_delta_l2": safe_mean(rpy_delta_l2),
                "mean_thrust_delta_abs": safe_mean(thrust_delta_abs),
                "max_tilt_deg": float(host["max_tilt_deg"][idx]),
                "tilt_over_limit_frac": float(host["tilt_over_limit_frac"][idx]),
                "max_cmd_tilt_deg": float(host["max_cmd_tilt_deg"][idx]),
                "cmd_tilt_over_limit_frac": float(host["cmd_tilt_over_limit_frac"][idx]),
                "action_sat_frac": float(host["action_sat_frac"][idx]),
                "thrust_sat_frac": float(host["thrust_sat_frac"][idx]),
                "rpy_sat_frac": float(host["rpy_sat_frac"][idx]),
                "pos_safety_violation": bool(host["pos_safety_violation"][idx]),
                "pos_safety_step_frac": float(host["pos_safety_step_frac"][idx]),
                "worst_safety_margin_m": float(host["worst_safety_margin_m"][idx]),
                "missed_gates": float(host["missed_gates"][idx]),
                "wrong_side_events": float(host["wrong_side_events"][idx]),
            }
        )
    return rows


def summarize(
    checkpoint_path: Path,
    rows: list[dict[str, Any]],
    observation_layout: str,
    hidden_dim: int,
) -> dict[str, Any]:
    """Aggregate per-seed rows for one checkpoint."""
    label, step_m = checkpoint_label(checkpoint_path)
    success_rows = [row for row in rows if row["success"]]
    return {
        "checkpoint": label,
        "step_m": step_m,
        "episodes": len(rows),
        "success_count": sum(int(row["success"]) for row in rows),
        "success_rate": safe_mean([float(row["success"]) for row in rows]),
        "crash_count": sum(int(row["crashed"]) for row in rows),
        "crash_rate": safe_mean([float(row["crashed"]) for row in rows]),
        "timeout_count": sum(int(row["timeout"]) for row in rows),
        "timeout_rate": safe_mean([float(row["timeout"]) for row in rows]),
        "mean_gates": safe_mean([row["gates"] for row in rows]),
        "median_gates": safe_median([row["gates"] for row in rows]),
        "mean_time_s_success": safe_mean([row["time_s"] for row in success_rows]),
        "median_time_s_success": safe_median([row["time_s"] for row in success_rows]),
        "mean_time_s_all": safe_mean([row["time_s"] for row in rows]),
        "mean_reward": safe_mean([row["reward"] for row in rows]),
        "mean_speed_mps": safe_mean([row["mean_speed_mps"] for row in rows]),
        "mean_speed_excess": safe_mean([row["mean_speed_excess"] for row in rows]),
        "mean_speed_excess_reward_per_step": safe_mean(
            [row["speed_excess_reward_per_step"] for row in rows]
        ),
        "mean_max_speed_mps": safe_mean([row["max_speed_mps"] for row in rows]),
        "worst_max_speed_mps": max(row["max_speed_mps"] for row in rows),
        "mean_smooth_penalty_per_step": safe_mean(
            [row["smooth_penalty_per_step"] for row in rows]
        ),
        "mean_action_delta_l2": safe_mean([row["mean_action_delta_l2"] for row in rows]),
        "p95_action_delta_l2": safe_mean([row["p95_action_delta_l2"] for row in rows]),
        "mean_rpy_delta_l2": safe_mean([row["mean_rpy_delta_l2"] for row in rows]),
        "mean_thrust_delta_abs": safe_mean([row["mean_thrust_delta_abs"] for row in rows]),
        "mean_max_tilt_deg": safe_mean([row["max_tilt_deg"] for row in rows]),
        "worst_tilt_deg": max(row["max_tilt_deg"] for row in rows),
        "tilt_over_limit_frac": safe_mean([row["tilt_over_limit_frac"] for row in rows]),
        "mean_max_cmd_tilt_deg": safe_mean([row["max_cmd_tilt_deg"] for row in rows]),
        "worst_cmd_tilt_deg": max(row["max_cmd_tilt_deg"] for row in rows),
        "cmd_tilt_over_limit_frac": safe_mean(
            [row["cmd_tilt_over_limit_frac"] for row in rows]
        ),
        "action_sat_frac": safe_mean([row["action_sat_frac"] for row in rows]),
        "thrust_sat_frac": safe_mean([row["thrust_sat_frac"] for row in rows]),
        "rpy_sat_frac": safe_mean([row["rpy_sat_frac"] for row in rows]),
        "pos_safety_violation_rate": safe_mean(
            [float(row["pos_safety_violation"]) for row in rows]
        ),
        "pos_safety_step_frac": safe_mean([row["pos_safety_step_frac"] for row in rows]),
        "worst_safety_margin_m": min(row["worst_safety_margin_m"] for row in rows),
        "mean_missed_gates": safe_mean([row["missed_gates"] for row in rows]),
        "mean_wrong_side_events": safe_mean([row["wrong_side_events"] for row in rows]),
        "checkpoint_file": str(checkpoint_path.relative_to(ROOT)),
        "observation_layout": observation_layout,
        "hidden_dim": hidden_dim,
    }


def checkpoint_paths(checkpoint_dir: Path, suffix: str) -> list[Path]:
    """Return chronologically sorted checkpoint paths for a suffix."""
    pattern = f"*{suffix}"
    return sorted(checkpoint_dir.glob(pattern), key=checkpoint_sort_key)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint-suffix", choices=(".ckpt", ".pkl"), default=".ckpt")
    parser.add_argument("--config", default="level3_no_obstacles.toml")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--jax-device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--include-final", action="store_true")
    parser.add_argument("--tilt-limit-deg", type=float, default=40.0)
    parser.add_argument("--safety-tol-m", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    """Run fast deterministic checkpoint evaluation."""
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoints = checkpoint_paths(checkpoint_dir, args.checkpoint_suffix)
    if not args.include_final:
        checkpoints = [path for path in checkpoints if checkpoint_step(path) is not None]
    if args.min_step is not None:
        checkpoints = [
            path
            for path in checkpoints
            if checkpoint_step(path) is None or checkpoint_step(path) >= args.min_step
        ]
    if args.max_step is not None:
        checkpoints = [
            path
            for path in checkpoints
            if checkpoint_step(path) is None or checkpoint_step(path) <= args.max_step
        ]
    if not checkpoints:
        raise FileNotFoundError(
            f"No {args.checkpoint_suffix} checkpoints found in {checkpoint_dir}"
        )

    device = _resolve_device(args.jax_device)
    run_args = build_args(args.config, args.jax_device)
    config = load_config(ROOT / "config" / args.config)
    pos_limit_low = jax.device_put(
        jnp.asarray(config.env.track.safety_limits.pos_limit_low, dtype=jnp.float32), device
    )
    pos_limit_high = jax.device_put(
        jnp.asarray(config.env.track.safety_limits.pos_limit_high, dtype=jnp.float32), device
    )
    episode_fields = [
        "checkpoint",
        "checkpoint_file",
        "seed",
        "success",
        "crashed",
        "timeout",
        "steps",
        "time_s",
        "gates",
        "reward",
        "mean_speed_mps",
        "mean_speed_excess",
        "speed_excess_reward_per_step",
        "max_speed_mps",
        "smooth_penalty_per_step",
        "mean_action_delta_l2",
        "p95_action_delta_l2",
        "mean_rpy_delta_l2",
        "mean_thrust_delta_abs",
        "max_tilt_deg",
        "tilt_over_limit_frac",
        "max_cmd_tilt_deg",
        "cmd_tilt_over_limit_frac",
        "action_sat_frac",
        "thrust_sat_frac",
        "rpy_sat_frac",
        "pos_safety_violation",
        "pos_safety_step_frac",
        "worst_safety_margin_m",
        "missed_gates",
        "wrong_side_events",
    ]
    episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    start = time.time()

    with jax.default_device(device):
        env, _cfg, action_low, action_high = jax_ppo_fast.make_fast_base_env(
            level="level3",
            config=args.config,
            num_envs=args.num_seeds,
            jax_device=args.jax_device,
        )
        try:
            raw_obs_np, _info = env.reset(seed=args.seed_start)
            raw_obs = jax_ppo_fast.device_put_tree(raw_obs_np, device)
            state0 = jax_ppo_fast.make_initial_state(
                env, raw_obs, run_args, action_low, action_high
            )
            env_step = jax_ppo_fast.build_fast_env_step(
                env._step,  # noqa: SLF001
                action_low,
                action_high,
                run_args,
                per_env_metrics=True,
            )
            eval_rollout = build_detailed_eval_fn(
                env_step,
                max_steps=args.max_steps,
                env_freq=float(config.env.freq),
                pos_limit_low=pos_limit_low,
                pos_limit_high=pos_limit_high,
                tilt_limit_deg=float(args.tilt_limit_deg),
                safety_tol_m=float(args.safety_tol_m),
            )
            for checkpoint_path in checkpoints:
                print(
                    f"evaluating {checkpoint_path.name} on {args.num_seeds} seeds",
                    flush=True,
                )
                params, observation_layout, hidden_dim = load_params(checkpoint_path, device)
                results = eval_rollout(state0, params)
                jax_ppo_fast.block_until_ready(results)
                rows = rows_from_results(results, checkpoint_path, args.seed_start)
                summary = summarize(checkpoint_path, rows, observation_layout, hidden_dim)
                episode_rows.extend(rows)
                summary_rows.append(summary)
                print(
                    f"  success={summary['success_count']}/{summary['episodes']} "
                    f"({summary['success_rate']:.2%}) "
                    f"gates={summary['mean_gates']:.2f} "
                    f"time_success={summary['mean_time_s_success']:.2f}s "
                    f"crash={summary['crash_rate']:.2%} "
                    f"speed={summary['mean_speed_mps']:.2f}m/s",
                    flush=True,
                )
        finally:
            env.close()

    summary_path = args.out_prefix.with_name(args.out_prefix.name + "_summary.csv")
    episode_path = args.out_prefix.with_name(args.out_prefix.name + "_episodes.csv")
    write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    write_csv(episode_path, episode_fields, episode_rows)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {episode_path}", flush=True)
    print(f"elapsed_s={time.time() - start:.1f}", flush=True)


if __name__ == "__main__":
    main()
