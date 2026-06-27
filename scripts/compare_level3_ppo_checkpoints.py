"""Compare Level3 PPO checkpoints over fixed deterministic simulation seeds."""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lsy_drone_racing.control.ppo_level3_observation import (
    LOCAL_OBSTACLE_OBSERVATION_LAYOUT,
    checkpoint_hidden_dim,
    unpack_checkpoint,
)
from lsy_drone_racing.control.train_CleanRL_ppo_level3 import Agent, Args, make_envs
from lsy_drone_racing.utils import load_config

ROOT = Path(__file__).parents[1]
DEFAULT_CHECKPOINT_DIR = (
    ROOT
    / "lsy_drone_racing/control/checkpoints/level3_Curriculum_Xobstacle_speedlimit2_jax_fast"
)
DEFAULT_OUT_PREFIX = ROOT / "artifacts/level3_checkpoint_eval"
TRAINING_REWARD_COEFS = {
    "n_obs": 2,
    "progress_coef": 0.0,
    "gate_stage_coef": 15.0,
    "gate_axis_coef": 15.0,
    "near_gate_coef": 0.0,
    "gate_bonus": 120.0,
    "gate_back_bonus": 20.0,
    "finish_bonus": 160.0,
    "missed_gate_penalty": 0.0,
    "wrong_side_penalty": 6.0,
    "crash_penalty": 50.0,
    "obstacle_coef": 5.0,
    "obstacle_margin": 0.3,
    "obstacle_clearance_coef": 0.0,
    "timeout_penalty": 80.0,
    "time_penalty": 0.03,
    "speed_limit_mps": 2.0,
    "speed_excess_coef": 3.0,
    "act_coef": 0.03,
    "d_act_th_coef": 0.10,
    "d_act_xy_coef": 0.10,
    "cmd_tilt_coef": 1.0,
    "rpy_coef": 1.0,
    "tilt_limit_deg": 40.0,
    "tilt_excess_coef": 15.0,
}
SUMMARY_FIELDS = [
    "checkpoint",
    "step_m",
    "episodes",
    "success_count",
    "success_rate",
    "crash_count",
    "crash_rate",
    "timeout_count",
    "timeout_rate",
    "mean_gates",
    "median_gates",
    "mean_time_s_success",
    "median_time_s_success",
    "mean_time_s_all",
    "mean_reward",
    "mean_speed_mps",
    "mean_speed_excess",
    "mean_speed_excess_reward_per_step",
    "mean_max_speed_mps",
    "worst_max_speed_mps",
    "mean_smooth_penalty_per_step",
    "mean_action_delta_l2",
    "p95_action_delta_l2",
    "mean_rpy_delta_l2",
    "mean_thrust_delta_abs",
    "mean_max_tilt_deg",
    "worst_tilt_deg",
    "tilt_over_limit_frac",
    "mean_max_cmd_tilt_deg",
    "worst_cmd_tilt_deg",
    "cmd_tilt_over_limit_frac",
    "action_sat_frac",
    "thrust_sat_frac",
    "rpy_sat_frac",
    "pos_safety_violation_rate",
    "pos_safety_step_frac",
    "worst_safety_margin_m",
    "mean_missed_gates",
    "mean_wrong_side_events",
    "checkpoint_file",
    "observation_layout",
    "hidden_dim",
]


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    """Sort step checkpoints chronologically and put final checkpoints last."""
    match = re.search(r"_step_(\d+)\.ckpt$", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (10**18, path.name)


def checkpoint_step(path: Path) -> int | None:
    """Return the numeric checkpoint step, or None for non-step checkpoints."""
    match = re.search(r"_step_(\d+)\.ckpt$", path.name)
    if match:
        return int(match.group(1))
    return None


def checkpoint_label(path: Path) -> tuple[str, float | str]:
    """Return a display label and step in millions if present."""
    step = checkpoint_step(path)
    if step is not None:
        return (f"{step // 1_000_000}M", step / 1_000_000)
    return ("final", "final")


def scalar(value: Any) -> float:
    """Convert a tensor/array/scalar to float."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(np.asarray(value).reshape(-1)[0])


def array1(value: Any) -> np.ndarray:
    """Convert a tensor/array/scalar batch to a 1D numpy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


def safe_mean(values: list[float]) -> float:
    """Mean that returns NaN for empty lists."""
    return float(np.mean(values)) if values else float("nan")


def safe_median(values: list[float]) -> float:
    """Median that returns NaN for empty lists."""
    return float(np.median(values)) if values else float("nan")


def percentile(values: list[float], q: float) -> float:
    """Percentile that returns NaN for empty lists."""
    return float(np.percentile(values, q)) if values else float("nan")


def make_args(config: str) -> Args:
    """Create Level3 args with the reward coefficients used by this checkpoint family."""
    return Args.create(config=config, cuda=False, jax_device="cpu", **TRAINING_REWARD_COEFS)


def load_agent(env: Any, checkpoint_path: Path, device: torch.device) -> tuple[Agent, str, int]:
    """Load one PPO agent and validate the observation shape."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict, observation_layout = unpack_checkpoint(checkpoint)
    if observation_layout != LOCAL_OBSTACLE_OBSERVATION_LAYOUT:
        raise ValueError(
            f"{checkpoint_path.name} uses {observation_layout}, but this evaluator uses "
            f"{LOCAL_OBSTACLE_OBSERVATION_LAYOUT}."
        )
    obs_dim = int(model_state_dict["actor_mean.0.weight"].shape[1])
    expected_dim = int(np.prod(env.single_observation_space.shape))
    if obs_dim != expected_dim:
        raise ValueError(
            f"{checkpoint_path.name} expects obs_dim={obs_dim}, "
            f"but env observation dim is {expected_dim}."
        )
    hidden_dim = checkpoint_hidden_dim(checkpoint, model_state_dict)
    agent = Agent(
        env.single_observation_space.shape,
        env.single_action_space.shape,
        hidden_dim=hidden_dim,
    ).to(device)
    agent.load_state_dict(model_state_dict)
    agent.eval()
    return agent, observation_layout, hidden_dim


def unwrap_race_env(env: Any) -> Any:
    """Return the wrapped vector race env that owns sim data and the JAX step fn."""
    current = env
    while True:
        if hasattr(current, "data") and hasattr(current, "build_step_fn"):
            return current
        if not hasattr(current, "env"):
            raise TypeError(f"Could not find base race env under wrapper {type(env)!r}.")
        current = current.env


def run_vector_episodes(
    env: Any,
    base_env: Any,
    agent: Agent,
    seed_labels: list[int],
    reset_seed: int,
    env_freq: float,
    pos_limit_low: np.ndarray,
    pos_limit_high: np.ndarray,
    tilt_limit_deg: float,
    safety_tol_m: float,
    device: torch.device,
) -> list[dict[str, float | int | bool]]:
    """Run deterministic vector episodes and collect per-world metrics."""
    obs, _ = env.reset(seed=reset_seed)
    n_envs = len(seed_labels)
    action_shape = env.single_action_space.shape
    previous_action = np.zeros((n_envs, *action_shape), dtype=np.float64)
    done = np.zeros(n_envs, dtype=bool)
    finished = np.zeros(n_envs, dtype=bool)
    crashed = np.zeros(n_envs, dtype=bool)
    timeout = np.zeros(n_envs, dtype=bool)
    steps = np.zeros(n_envs, dtype=np.int32)
    episode_reward = np.zeros(n_envs, dtype=np.float64)
    passed_gates = np.zeros(n_envs, dtype=np.float64)
    missed_gates = np.zeros(n_envs, dtype=np.float64)
    wrong_side_events = np.zeros(n_envs, dtype=np.float64)
    speed_sum = np.zeros(n_envs, dtype=np.float64)
    speed_excess_sum = np.zeros(n_envs, dtype=np.float64)
    speed_excess_reward_sum = np.zeros(n_envs, dtype=np.float64)
    smooth_penalty = np.zeros(n_envs, dtype=np.float64)
    action_delta_l2: list[list[float]] = [[] for _ in range(n_envs)]
    rpy_delta_l2: list[list[float]] = [[] for _ in range(n_envs)]
    thrust_delta_abs: list[list[float]] = [[] for _ in range(n_envs)]
    max_speed = np.full(n_envs, -np.inf, dtype=np.float64)
    max_tilt = np.full(n_envs, -np.inf, dtype=np.float64)
    max_cmd_tilt = np.full(n_envs, -np.inf, dtype=np.float64)
    tilt_over_limit = np.zeros(n_envs, dtype=np.int32)
    cmd_tilt_over_limit = np.zeros(n_envs, dtype=np.int32)
    action_sat_count = np.zeros(n_envs, dtype=np.int32)
    thrust_sat_count = np.zeros(n_envs, dtype=np.int32)
    rpy_sat_count = np.zeros(n_envs, dtype=np.int32)
    action_value_count = np.zeros(n_envs, dtype=np.int32)
    thrust_value_count = np.zeros(n_envs, dtype=np.int32)
    rpy_value_count = np.zeros(n_envs, dtype=np.int32)
    pos_safety_violation_steps = np.zeros(n_envs, dtype=np.int32)
    pos_samples = np.zeros(n_envs, dtype=np.int32)
    worst_safety_margin = np.full(n_envs, np.inf, dtype=np.float64)

    while not np.all(done):
        active = ~done
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs, deterministic=True)
        if np.any(done):
            action = action.clone()
            action[torch.as_tensor(done, device=device)] = 0.0
        action_np = action.detach().cpu().numpy().astype(np.float64)
        action_clipped = np.clip(action_np, -1.0, 1.0)
        action_delta = action_clipped - previous_action

        active_indices = np.flatnonzero(active)
        for idx in active_indices:
            delta = action_delta[idx]
            action_delta_l2[idx].append(float(np.linalg.norm(delta)))
            rpy_delta_l2[idx].append(float(np.linalg.norm(delta[:3])))
            thrust_delta_abs[idx].append(float(abs(delta[3])))
        action_sat_count[active] += np.count_nonzero(
            np.abs(action_clipped[active]) >= 0.95, axis=1
        )
        thrust_sat_count[active] += (np.abs(action_clipped[active, 3]) >= 0.95).astype(np.int32)
        rpy_sat_count[active] += np.count_nonzero(
            np.abs(action_clipped[active, :3]) >= 0.95, axis=1
        )
        action_value_count[active] += action_clipped.shape[1]
        thrust_value_count[active] += 1
        rpy_value_count[active] += 3

        pos = np.asarray(base_env.data.sim_data.states.pos[:, 0], dtype=np.float64)
        margins = np.minimum(pos - pos_limit_low, pos_limit_high - pos)
        current_margin = np.min(margins, axis=1)
        worst_safety_margin[active] = np.minimum(
            worst_safety_margin[active], current_margin[active]
        )
        outside_safety = np.any(
            (pos < pos_limit_low - safety_tol_m) | (pos > pos_limit_high + safety_tol_m), axis=1
        )
        pos_safety_violation_steps[active] += outside_safety[active].astype(np.int32)
        pos_samples[active] += 1

        obs, reward, terminated, truncated, info = env.step(action)
        reward_np = array1(reward)
        terminated_np = array1(terminated).astype(bool)
        truncated_np = array1(truncated).astype(bool)
        steps[active] += 1
        episode_reward[active] += reward_np[active]
        passed_gates[active] += array1(info["race_passed_gate_rate"])[active]
        missed_gates[active] += array1(info["race_missed_gate_rate"])[active]
        wrong_side_events[active] += array1(info["race_wrong_side_gate_rate"])[active]
        smooth_penalty[active] += -array1(info["reward_smooth"])[active]
        speed = array1(info["race_speed"])
        speed_excess = array1(info["race_speed_excess"])
        speed_sum[active] += speed[active]
        speed_excess_sum[active] += speed_excess[active]
        speed_excess_reward_sum[active] += array1(info["reward_speed_excess"])[active]
        max_speed[active] = np.maximum(max_speed[active], speed[active])

        tilt = array1(info["race_tilt_angle_deg"])
        cmd_tilt = array1(info["race_cmd_tilt_deg"])
        max_tilt[active] = np.maximum(max_tilt[active], tilt[active])
        max_cmd_tilt[active] = np.maximum(max_cmd_tilt[active], cmd_tilt[active])
        tilt_over_limit[active] += (tilt[active] > tilt_limit_deg).astype(np.int32)
        cmd_tilt_over_limit[active] += (cmd_tilt[active] > tilt_limit_deg).astype(np.int32)

        done_now = active & (terminated_np | truncated_np)
        finished[done_now] = array1(info["race_finished_rate"])[done_now] > 0.5
        crashed[done_now] = array1(info["race_crashed_rate"])[done_now] > 0.5
        timeout[done_now] = truncated_np[done_now]
        done |= done_now
        previous_action[active] = action_clipped[active]

    rows = []
    for idx, seed in enumerate(seed_labels):
        rows.append(
            {
                "seed": seed,
                "success": bool(finished[idx]),
                "crashed": bool(crashed[idx]),
                "timeout": bool(timeout[idx]),
                "steps": int(steps[idx]),
                "time_s": float(steps[idx] / env_freq),
                "gates": float(passed_gates[idx]),
                "reward": float(episode_reward[idx]),
                "mean_speed_mps": float(speed_sum[idx] / steps[idx]),
                "mean_speed_excess": float(speed_excess_sum[idx] / steps[idx]),
                "speed_excess_reward_per_step": float(speed_excess_reward_sum[idx] / steps[idx]),
                "max_speed_mps": float(max_speed[idx]),
                "smooth_penalty_per_step": float(smooth_penalty[idx] / steps[idx]),
                "mean_action_delta_l2": safe_mean(action_delta_l2[idx]),
                "p95_action_delta_l2": percentile(action_delta_l2[idx], 95),
                "mean_rpy_delta_l2": safe_mean(rpy_delta_l2[idx]),
                "mean_thrust_delta_abs": safe_mean(thrust_delta_abs[idx]),
                "max_tilt_deg": float(max_tilt[idx]),
                "tilt_over_limit_frac": float(tilt_over_limit[idx] / steps[idx]),
                "max_cmd_tilt_deg": float(max_cmd_tilt[idx]),
                "cmd_tilt_over_limit_frac": float(cmd_tilt_over_limit[idx] / steps[idx]),
                "action_sat_frac": float(action_sat_count[idx] / action_value_count[idx]),
                "thrust_sat_frac": float(thrust_sat_count[idx] / thrust_value_count[idx]),
                "rpy_sat_frac": float(rpy_sat_count[idx] / rpy_value_count[idx]),
                "pos_safety_violation": bool(pos_safety_violation_steps[idx] > 0),
                "pos_safety_step_frac": float(pos_safety_violation_steps[idx] / pos_samples[idx]),
                "worst_safety_margin_m": float(worst_safety_margin[idx]),
                "missed_gates": float(missed_gates[idx]),
                "wrong_side_events": float(wrong_side_events[idx]),
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


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--config", default="level3_no_obstacles.toml")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument(
        "--min-step",
        type=int,
        default=None,
        help="Only evaluate step checkpoints with at least this many timesteps; final is kept.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Only evaluate step checkpoints with at most this many timesteps; final is kept.",
    )
    parser.add_argument("--tilt-limit-deg", type=float, default=40.0)
    parser.add_argument("--safety-tol-m", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    """Run the checkpoint comparison."""
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    config = load_config(ROOT / "config" / args.config)
    run_args = make_args(args.config)
    device = torch.device("cpu")
    env = make_envs(
        config=run_args.config,
        num_envs=args.num_seeds,
        jax_device="cpu",
        torch_device=device,
        coefs=TRAINING_REWARD_COEFS,
    )
    base_env = unwrap_race_env(env)
    base_env.settings = base_env.settings.replace(autoreset=False)
    base_env._step = base_env.build_step_fn()
    checkpoints = sorted(checkpoint_dir.glob("*.ckpt"), key=checkpoint_sort_key)
    if args.min_step is not None:
        checkpoints = [
            checkpoint_path
            for checkpoint_path in checkpoints
            if checkpoint_step(checkpoint_path) is None
            or checkpoint_step(checkpoint_path) >= args.min_step
        ]
    if args.max_step is not None:
        checkpoints = [
            checkpoint_path
            for checkpoint_path in checkpoints
            if checkpoint_step(checkpoint_path) is None
            or checkpoint_step(checkpoint_path) <= args.max_step
        ]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    pos_limit_low = np.asarray(config.env.track.safety_limits.pos_limit_low, dtype=np.float64)
    pos_limit_high = np.asarray(config.env.track.safety_limits.pos_limit_high, dtype=np.float64)
    episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    start = time.time()

    try:
        for checkpoint_path in checkpoints:
            label, _ = checkpoint_label(checkpoint_path)
            agent, observation_layout, hidden_dim = load_agent(env, checkpoint_path, device)
            print(f"evaluating {checkpoint_path.name} on {len(seeds)} seeds")
            checkpoint_rows = run_vector_episodes(
                env=env,
                base_env=base_env,
                agent=agent,
                seed_labels=seeds,
                reset_seed=args.seed_start,
                env_freq=float(config.env.freq),
                pos_limit_low=pos_limit_low,
                pos_limit_high=pos_limit_high,
                tilt_limit_deg=args.tilt_limit_deg,
                safety_tol_m=args.safety_tol_m,
                device=device,
            )
            checkpoint_file = str(checkpoint_path.relative_to(ROOT))
            for episode in checkpoint_rows:
                episode["checkpoint"] = label
                episode["checkpoint_file"] = checkpoint_file
                episode_rows.append(episode)
            summary = summarize(checkpoint_path, checkpoint_rows, observation_layout, hidden_dim)
            summary_rows.append(summary)
            print(
                f"  success={summary['success_count']}/{summary['episodes']} "
                f"({summary['success_rate']:.2%}) "
                f"gates={summary['mean_gates']:.2f} "
                f"time_success={summary['mean_time_s_success']:.2f}s "
                f"crash={summary['crash_rate']:.2%} "
                f"speed={summary['mean_speed_mps']:.2f}m/s"
            )
    finally:
        env.close()

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
    summary_path = args.out_prefix.with_name(args.out_prefix.name + "_summary.csv")
    episode_path = args.out_prefix.with_name(args.out_prefix.name + "_episodes.csv")
    write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    write_csv(episode_path, episode_fields, episode_rows)
    print(f"wrote {summary_path}")
    print(f"wrote {episode_path}")
    print(f"elapsed_s={time.time() - start:.1f}")


if __name__ == "__main__":
    main()
