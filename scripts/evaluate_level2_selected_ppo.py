"""Evaluate selected level2 PPO checkpoints through the inference controller.

This path supports both current and legacy observation-layout checkpoints because
``PPOLevel2Inference`` already contains the compatibility logic used for deployment.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.control import ppo_level2_inference
from lsy_drone_racing.control.ppo_level2_inference import N_HISTORY
from lsy_drone_racing.utils import load_config

ROOT = Path(__file__).parents[1]


def safe_mean(values: list[float]) -> float:
    """Return NaN for empty lists."""
    return float(np.mean(values)) if values else float("nan")


def label_for_checkpoint(path: Path) -> str:
    """Make a compact label that remains unique across checkpoint families."""
    parent = path.parent.name.removeprefix("ppo_level2_")
    stem = path.stem
    if "_step_" in stem:
        step = int(stem.rsplit("_step_", 1)[1]) // 1_000_000
        return f"{parent}:{step}M"
    return f"{parent}:final"


def reset_controller_state(controller: Any, obs: dict[str, Any]) -> None:
    """Reset inference-only recurrent state between episodes."""
    controller._history = np.repeat(  # noqa: SLF001
        controller._basic_history(obs)[None, :], N_HISTORY, axis=0  # noqa: SLF001
    )
    controller._last_action_norm = np.zeros(controller.action_dim, dtype=np.float32)  # noqa: SLF001
    controller._finished = False  # noqa: SLF001


def run_checkpoint(
    checkpoint: Path,
    *,
    config_name: str,
    seed_start: int,
    num_seeds: int,
    smooth_coef_rpy: float,
    smooth_coef_thrust: float,
    tilt_limit_deg: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run exact single-env episodes for one checkpoint."""
    config = load_config(ROOT / "config" / config_name)
    config.sim.render = False
    control_dir = Path(ppo_level2_inference.__file__).parent
    ppo_level2_inference.MODEL_NAME = str(checkpoint.resolve().relative_to(control_dir))
    env = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    env = JaxToNumpy(env)
    label = label_for_checkpoint(checkpoint)
    rows: list[dict[str, Any]] = []
    controller = None
    try:
        for seed in range(seed_start, seed_start + num_seeds):
            obs, info = env.reset(seed=seed)
            if controller is None:
                controller = ppo_level2_inference.PPOLevel2Inference(obs, info, config)
            else:
                reset_controller_state(controller, obs)

            steps = 0
            previous_action_norm = np.zeros(controller.action_dim, dtype=np.float64)
            smooth_sum = 0.0
            action_delta_l2: list[float] = []
            tilt_values: list[float] = []
            cmd_tilt_values: list[float] = []
            target_gate = int(np.asarray(obs["target_gate"]).item())
            finished = False
            crashed = False
            timeout = False

            while True:
                action = controller.compute_control(obs, info)
                roll_cmd, pitch_cmd = float(action[0]), float(action[1])
                cmd_body_z_world_z = np.clip(np.cos(roll_cmd) * np.cos(pitch_cmd), -1.0, 1.0)
                cmd_tilt_values.append(float(np.rad2deg(np.arccos(cmd_body_z_world_z))))
                action_norm = np.asarray(controller._last_action_norm, dtype=np.float64)  # noqa: SLF001
                delta = np.clip(action_norm, -1.0, 1.0) - previous_action_norm
                smooth_sum += (
                    smooth_coef_rpy * float(np.sum(delta[:3] ** 2))
                    + smooth_coef_thrust * float(delta[3] ** 2)
                )
                action_delta_l2.append(float(np.linalg.norm(delta)))
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                rot = controller.quat_to_rotmat(np.asarray(obs["quat"], dtype=np.float32))
                body_z_world_z = np.clip(float(rot[2, 2]), -1.0, 1.0)
                tilt_values.append(float(np.rad2deg(np.arccos(body_z_world_z))))
                target_gate = int(np.asarray(obs["target_gate"]).item())
                finished = target_gate < 0
                crashed = bool(terminated and not finished)
                timeout = bool(truncated and not finished)
                controller_finished = controller.step_callback(
                    action, obs, reward, terminated, truncated, info
                )
                if terminated or truncated or controller_finished:
                    break
                previous_action_norm = np.clip(action_norm, -1.0, 1.0)

            n_gates = int(np.asarray(obs["gates_pos"]).shape[0])
            rows.append(
                {
                    "checkpoint": label,
                    "checkpoint_file": str(checkpoint.relative_to(ROOT)),
                    "seed": seed,
                    "success": finished,
                    "crashed": crashed,
                    "timeout": timeout,
                    "steps": steps,
                    "time_s": steps / float(config.env.freq),
                    "gates": n_gates if finished else max(target_gate, 0),
                    "smooth_penalty_per_step": smooth_sum / steps,
                    "mean_action_delta_l2": safe_mean(action_delta_l2),
                    "max_tilt_deg": max(tilt_values),
                    "tilt_over_limit_frac": safe_mean(
                        [float(value > tilt_limit_deg) for value in tilt_values]
                    ),
                    "max_cmd_tilt_deg": max(cmd_tilt_values),
                    "cmd_tilt_over_limit_frac": safe_mean(
                        [float(value > tilt_limit_deg) for value in cmd_tilt_values]
                    ),
                }
            )
    finally:
        env.close()

    successes = [row for row in rows if row["success"]]
    summary = {
        "checkpoint": label,
        "checkpoint_file": str(checkpoint.relative_to(ROOT)),
        "episodes": len(rows),
        "success_rate": safe_mean([float(row["success"]) for row in rows]),
        "crash_rate": safe_mean([float(row["crashed"]) for row in rows]),
        "timeout_rate": safe_mean([float(row["timeout"]) for row in rows]),
        "mean_gates": safe_mean([float(row["gates"]) for row in rows]),
        "mean_time_s_success": safe_mean([float(row["time_s"]) for row in successes]),
        "mean_smooth_penalty_per_step": safe_mean(
            [float(row["smooth_penalty_per_step"]) for row in rows]
        ),
        "mean_action_delta_l2": safe_mean([float(row["mean_action_delta_l2"]) for row in rows]),
        "mean_max_tilt_deg": safe_mean([float(row["max_tilt_deg"]) for row in rows]),
        "worst_tilt_deg": max(float(row["max_tilt_deg"]) for row in rows),
        "tilt_over_limit_frac": safe_mean(
            [float(row["tilt_over_limit_frac"]) for row in rows]
        ),
        "mean_max_cmd_tilt_deg": safe_mean([float(row["max_cmd_tilt_deg"]) for row in rows]),
        "worst_cmd_tilt_deg": max(float(row["max_cmd_tilt_deg"]) for row in rows),
        "cmd_tilt_over_limit_frac": safe_mean(
            [float(row["cmd_tilt_over_limit_frac"]) for row in rows]
        ),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV when rows are present."""
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="level2_dr.toml")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--out-prefix", type=Path, default=ROOT / "evaluation_level2_selected")
    parser.add_argument("--smooth-coef-rpy", type=float, default=0.15)
    parser.add_argument("--smooth-coef-thrust", type=float, default=0.15)
    parser.add_argument("--tilt-limit-deg", type=float, default=30.0)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    """Evaluate selected checkpoints and write summary/episode CSV files."""
    args = parse_args()
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for checkpoint in args.checkpoints:
        checkpoint = checkpoint.resolve()
        print(f"evaluating {checkpoint}")
        rows, summary = run_checkpoint(
            checkpoint,
            config_name=args.config,
            seed_start=args.seed_start,
            num_seeds=args.num_seeds,
            smooth_coef_rpy=args.smooth_coef_rpy,
            smooth_coef_thrust=args.smooth_coef_thrust,
            tilt_limit_deg=args.tilt_limit_deg,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        print(
            f"  success={summary['success_rate']:.2%} "
            f"crash={summary['crash_rate']:.2%} "
            f"time_success={summary['mean_time_s_success']:.2f}s "
            f"smooth={summary['mean_smooth_penalty_per_step']:.4f} "
            f"max_tilt={summary['mean_max_tilt_deg']:.1f}deg "
            f"max_cmd_tilt={summary['mean_max_cmd_tilt_deg']:.1f}deg"
        )

    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_episodes.csv"), all_rows)
    write_csv(args.out_prefix.with_name(args.out_prefix.name + "_summary.csv"), summaries)


if __name__ == "__main__":
    main()
