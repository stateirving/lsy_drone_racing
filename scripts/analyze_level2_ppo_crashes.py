"""Locate crash hotspots for one deterministic level2 PPO checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import gymnasium
import numpy as np
import torch
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import ppo_level2_inference
from lsy_drone_racing.control.ppo_level2_observation import OBSERVATION_LAYOUT, unpack_checkpoint
from lsy_drone_racing.control.train_CleanRL_ppo import Agent, make_envs
from lsy_drone_racing.utils import load_config

ROOT = Path(__file__).parents[1]
DEFAULT_CHECKPOINT = (
    ROOT / "lsy_drone_racing/control/checkpoints/ppo_level2_cmdtilt1p5_160M/"
    "ppo_level2_cmdtilt1p5_160M_step_100000000.ckpt"
)
DEFAULT_OUT_PREFIX = ROOT / "evaluation_level2_100M_crash_hotspots"
N_HISTORY = 2

GATE_BOXES = {
    "top": (np.array([0.0, 0.0, 0.28]), np.array([0.01, 0.36, 0.08])),
    "bottom": (np.array([0.0, 0.0, -0.28]), np.array([0.01, 0.36, 0.08])),
    "left": (np.array([0.0, -0.28, 0.0]), np.array([0.01, 0.08, 0.36])),
    "right": (np.array([0.0, 0.28, 0.0]), np.array([0.01, 0.08, 0.36])),
    "stand": (np.array([0.0, 0.0, -0.86]), np.array([0.05, 0.05, 0.5])),
}


def array1(value: Any) -> np.ndarray:
    """Convert a vector tensor/array to a 1D numpy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


def base_env_from_wrappers(env: Any) -> Any:
    """Return the raw vector race environment beneath all wrappers."""
    while hasattr(env, "env"):
        env = env.env
    return env


def load_agent(env: Any, checkpoint_path: Path, device: torch.device) -> Agent:
    """Load the PPO checkpoint and validate its observation size."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict, observation_layout = unpack_checkpoint(checkpoint)
    if observation_layout != OBSERVATION_LAYOUT:
        raise ValueError(
            f"Vector mode uses {OBSERVATION_LAYOUT}, but checkpoint uses {observation_layout}. "
            "Use --mode single for legacy checkpoints."
        )
    obs_dim = int(model_state_dict["actor_mean.0.weight"].shape[1])
    expected_dim = int(np.prod(env.single_observation_space.shape))
    if obs_dim != expected_dim:
        raise ValueError(f"Checkpoint obs_dim={obs_dim}, environment obs_dim={expected_dim}.")
    agent = Agent(env.single_observation_space.shape, env.single_action_space.shape).to(device)
    agent.load_state_dict(model_state_dict)
    agent.eval()
    return agent


def point_box_distance(point: np.ndarray, center: np.ndarray, half_size: np.ndarray) -> float:
    """Return the distance from a point to an axis-aligned box."""
    return float(np.linalg.norm(np.maximum(np.abs(point - center) - half_size, 0.0)))


def classify_crash(
    pos: np.ndarray, gates_pos: np.ndarray, gates_quat: np.ndarray, obstacles_pos: np.ndarray
) -> dict[str, Any]:
    """Classify a last-valid crash position by its nearest collision geometry."""
    gate_local = R.from_quat(gates_quat).inv().apply(pos[None, :] - gates_pos)
    gate_candidates = []
    for gate_idx, local in enumerate(gate_local):
        for part, (center, half_size) in GATE_BOXES.items():
            gate_candidates.append((point_box_distance(local, center, half_size), gate_idx, part))
    gate_distance, nearest_gate, nearest_gate_part = min(gate_candidates)

    obstacle_candidates = []
    for obstacle_idx, obstacle_pos in enumerate(obstacles_pos):
        segment_z = np.clip(pos[2], obstacle_pos[2] - 1.6, obstacle_pos[2])
        segment_distance = np.linalg.norm(pos - np.array([*obstacle_pos[:2], segment_z]))
        obstacle_candidates.append((max(float(segment_distance) - 0.015, 0.0), obstacle_idx))
    obstacle_distance, nearest_obstacle = min(obstacle_candidates)

    nearest_distance = min(gate_distance, obstacle_distance)
    if nearest_distance > 0.25:
        likely_object = "bounds_or_ground"
    elif gate_distance <= obstacle_distance:
        likely_object = f"gate_{nearest_gate}_{nearest_gate_part}"
    else:
        likely_object = f"obstacle_{nearest_obstacle}"

    return {
        "likely_object": likely_object,
        "nearest_object_distance_m": nearest_distance,
        "nearest_gate": nearest_gate,
        "nearest_gate_part": nearest_gate_part,
        "nearest_gate_distance_m": gate_distance,
        "nearest_obstacle": nearest_obstacle,
        "nearest_obstacle_distance_m": obstacle_distance,
    }


def run_batch(
    env: Any,
    base_env: Any,
    agent: Agent,
    batch_seed: int,
    num_worlds: int,
    env_freq: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Run one reproducible vector batch and return one row per world."""
    obs, _ = env.reset(seed=batch_seed)
    gates_pos = np.asarray(base_env.data.gates_pos, dtype=np.float64).copy()
    gates_quat = np.asarray(base_env.data.gates_quat, dtype=np.float64).copy()
    obstacles_pos = np.asarray(base_env.data.obstacles_pos, dtype=np.float64).copy()
    n_gates = gates_pos.shape[1]

    done = np.zeros(num_worlds, dtype=bool)
    steps = np.zeros(num_worlds, dtype=np.int32)
    rows: list[dict[str, Any]] = []

    while not np.all(done):
        active = ~done
        pos_before = np.asarray(base_env.data.sim_data.states.pos[:, 0], dtype=np.float64).copy()
        vel_before = np.asarray(base_env.data.sim_data.states.vel[:, 0], dtype=np.float64).copy()
        target_before = np.asarray(base_env.data.target_gate[:, 0], dtype=np.int32).copy()

        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs, deterministic=True)
        if np.any(done):
            action = action.clone()
            action[torch.as_tensor(done, device=device)] = 0.0

        obs, _, terminated, truncated, info = env.step(action)
        terminated_np = array1(terminated).astype(bool)
        truncated_np = array1(truncated).astype(bool)
        steps[active] += 1
        done_now = active & (terminated_np | truncated_np)
        finished = array1(info["race_finished_rate"]) > 0.5
        crashed = array1(info["race_crashed_rate"]) > 0.5

        for world_idx in np.flatnonzero(done_now):
            target_gate = int(target_before[world_idx])
            row: dict[str, Any] = {
                "run_id": f"{batch_seed}:{world_idx}",
                "batch_seed": batch_seed,
                "world_index": int(world_idx),
                "success": bool(finished[world_idx]),
                "crashed": bool(crashed[world_idx]),
                "timeout": bool(truncated_np[world_idx]),
                "steps": int(steps[world_idx]),
                "time_s": float(steps[world_idx] / env_freq),
                "target_gate": target_gate,
                "gates_passed": n_gates if target_gate < 0 else target_gate,
                "last_x": float(pos_before[world_idx, 0]),
                "last_y": float(pos_before[world_idx, 1]),
                "last_z": float(pos_before[world_idx, 2]),
                "last_vx": float(vel_before[world_idx, 0]),
                "last_vy": float(vel_before[world_idx, 1]),
                "last_vz": float(vel_before[world_idx, 2]),
            }
            if crashed[world_idx] and target_gate >= 0:
                target_rot = R.from_quat(gates_quat[world_idx, target_gate])
                target_local = target_rot.inv().apply(
                    pos_before[world_idx] - gates_pos[world_idx, target_gate]
                )
                classification = classify_crash(
                    pos_before[world_idx],
                    gates_pos[world_idx],
                    gates_quat[world_idx],
                    obstacles_pos[world_idx],
                )
                row.update(
                    {
                        "target_gate_x": float(gates_pos[world_idx, target_gate, 0]),
                        "target_gate_y": float(gates_pos[world_idx, target_gate, 1]),
                        "target_gate_z": float(gates_pos[world_idx, target_gate, 2]),
                        "target_local_x": float(target_local[0]),
                        "target_local_y": float(target_local[1]),
                        "target_local_z": float(target_local[2]),
                        **classification,
                    }
                )
            rows.append(row)
        done |= done_now
    return rows


def run_single_seed_episodes(
    config: Any, checkpoint: Path, seed_start: int, num_seeds: int
) -> list[dict[str, Any]]:
    """Run exact single-environment episodes through PPOLevel2Inference."""
    control_dir = Path(ppo_level2_inference.__file__).parent
    ppo_level2_inference.MODEL_NAME = str(checkpoint.relative_to(control_dir))
    config.sim.render = False
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
    rows: list[dict[str, Any]] = []
    controller: ppo_level2_inference.PPOLevel2Inference | None = None

    try:
        for seed in range(seed_start, seed_start + num_seeds):
            obs, info = env.reset(seed=seed)
            if controller is None:
                controller = ppo_level2_inference.PPOLevel2Inference(obs, info, config)
            else:
                controller._history = np.repeat(  # noqa: SLF001
                    controller._basic_history(obs)[None, :],
                    N_HISTORY,
                    axis=0,  # noqa: SLF001
                )
                controller._last_action_norm = np.zeros(controller.action_dim, dtype=np.float32)  # noqa: SLF001
                controller._finished = False  # noqa: SLF001

            base_env = env.unwrapped
            gates_pos = np.asarray(base_env.data.gates_pos[0], dtype=np.float64).copy()
            gates_quat = np.asarray(base_env.data.gates_quat[0], dtype=np.float64).copy()
            obstacles_pos = np.asarray(base_env.data.obstacles_pos[0], dtype=np.float64).copy()
            n_gates = gates_pos.shape[0]
            steps = 0

            while True:
                last_pos = np.asarray(obs["pos"], dtype=np.float64).copy()
                last_vel = np.asarray(obs["vel"], dtype=np.float64).copy()
                target_gate = int(np.asarray(obs["target_gate"]).item())
                action = controller.compute_control(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                finished = int(np.asarray(obs["target_gate"]).item()) < 0
                crashed = bool(terminated and not finished)
                controller_finished = controller.step_callback(
                    action, obs, reward, terminated, truncated, info
                )
                if terminated or truncated or controller_finished:
                    break

            row: dict[str, Any] = {
                "run_id": str(seed),
                "seed": seed,
                "success": finished,
                "crashed": crashed,
                "timeout": bool(truncated),
                "steps": steps,
                "time_s": float(steps / config.env.freq),
                "target_gate": target_gate,
                "gates_passed": n_gates if target_gate < 0 else target_gate,
                "last_x": float(last_pos[0]),
                "last_y": float(last_pos[1]),
                "last_z": float(last_pos[2]),
                "last_vx": float(last_vel[0]),
                "last_vy": float(last_vel[1]),
                "last_vz": float(last_vel[2]),
            }
            if crashed and target_gate >= 0:
                target_rot = R.from_quat(gates_quat[target_gate])
                target_local = target_rot.inv().apply(last_pos - gates_pos[target_gate])
                row.update(
                    {
                        "target_gate_x": float(gates_pos[target_gate, 0]),
                        "target_gate_y": float(gates_pos[target_gate, 1]),
                        "target_gate_z": float(gates_pos[target_gate, 2]),
                        "target_local_x": float(target_local[0]),
                        "target_local_y": float(target_local[1]),
                        "target_local_z": float(target_local[2]),
                        **classify_crash(last_pos, gates_pos, gates_quat, obstacles_pos),
                    }
                )
            rows.append(row)
            print(
                f"seed={seed} success={finished} crash={crashed} "
                f"target_gate={target_gate} time_s={steps / config.env.freq:.2f}"
            )
    finally:
        env.close()
    return rows


def count_dict(values: list[Any]) -> dict[str, int]:
    """Return string-keyed counts sorted by descending frequency."""
    counts = Counter(str(value) for value in values)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def build_summary(
    rows: list[dict[str, Any]], checkpoint: Path, mode: str, num_worlds: int
) -> dict[str, Any]:
    """Aggregate episode and crash hotspot statistics."""
    crashes = [row for row in rows if row["crashed"]]
    successes = [row for row in rows if row["success"]]
    timeouts = [row for row in rows if row["timeout"]]
    xy_bins = Counter(
        (np.floor(row["last_x"] / 0.25) * 0.25, np.floor(row["last_y"] / 0.25) * 0.25)
        for row in crashes
    )
    by_target: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in crashes:
        by_target[int(row["target_gate"])].append(row)
    target_details = {}
    for target_gate, target_rows in sorted(by_target.items()):
        target_details[str(target_gate)] = {
            "count": len(target_rows),
            "mean_world_pos": [
                float(np.mean([row[f"last_{axis}"] for row in target_rows]))
                for axis in ("x", "y", "z")
            ],
            "mean_target_local_pos": [
                float(np.mean([row[f"target_local_{axis}"] for row in target_rows]))
                for axis in ("x", "y", "z")
            ],
            "nearest_gate_parts": count_dict([row["nearest_gate_part"] for row in target_rows]),
            "likely_objects": count_dict([row["likely_object"] for row in target_rows]),
        }
    total = len(rows)
    return {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "mode": mode,
        "episodes": total,
        "num_worlds_per_batch": num_worlds,
        "successes": len(successes),
        "crashes": len(crashes),
        "timeouts": len(timeouts),
        "success_rate": len(successes) / total,
        "crash_rate": len(crashes) / total,
        "timeout_rate": len(timeouts) / total,
        "mean_success_time_s": (
            float(np.mean([row["time_s"] for row in successes])) if successes else None
        ),
        "crashes_by_target_gate": count_dict([row["target_gate"] for row in crashes]),
        "crashes_by_likely_object": count_dict([row["likely_object"] for row in crashes]),
        "crashes_by_nearest_gate_part": count_dict([row["nearest_gate_part"] for row in crashes]),
        "top_crash_xy_bins_0.25m": [
            {"x_bin": float(x), "y_bin": float(y), "count": count}
            for (x, y), count in xy_bins.most_common(10)
        ],
        "target_gate_details": target_details,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV with the union of all row fields."""
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_hotspots(path: Path, rows: list[dict[str, Any]], config: Any) -> None:
    """Plot world-frame and target-gate-frame crash concentrations."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    crashes = [row for row in rows if row["crashed"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = np.asarray([row["target_gate"] for row in crashes])

    axes[0].scatter(
        [row["last_x"] for row in crashes],
        [row["last_y"] for row in crashes],
        c=colors,
        cmap="tab10",
        alpha=0.55,
        s=18,
    )
    for gate_idx, gate in enumerate(config.env.track.gates):
        axes[0].scatter(gate["pos"][0], gate["pos"][1], marker="s", color="black")
        axes[0].annotate(f"G{gate_idx}", (gate["pos"][0], gate["pos"][1]))
    for obstacle_idx, obstacle in enumerate(config.env.track.obstacles):
        axes[0].scatter(obstacle["pos"][0], obstacle["pos"][1], marker="x", color="red")
        axes[0].annotate(f"O{obstacle_idx}", (obstacle["pos"][0], obstacle["pos"][1]))
    axes[0].set(title="Crash endpoints, world XY", xlabel="x [m]", ylabel="y [m]")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(
        [row["target_local_y"] for row in crashes],
        [row["target_local_z"] for row in crashes],
        c=colors,
        cmap="tab10",
        alpha=0.55,
        s=18,
    )
    axes[1].add_patch(
        Rectangle((-0.36, -0.36), 0.72, 0.72, fill=False, edgecolor="black", linewidth=2)
    )
    axes[1].add_patch(
        Rectangle((-0.20, -0.20), 0.40, 0.40, fill=False, edgecolor="black", linestyle="--")
    )
    axes[1].set(
        title="Crash endpoints relative to target gate",
        xlabel="gate-local y [m]",
        ylabel="gate-local z [m]",
        xlim=(-0.8, 0.8),
        ylim=(-1.0, 0.8),
    )
    axes[1].set_aspect("equal")
    axes[1].grid(alpha=0.25)

    gate_counts = Counter(int(row["target_gate"]) for row in crashes)
    gate_indices = list(range(len(config.env.track.gates)))
    axes[2].bar(gate_indices, [gate_counts[idx] for idx in gate_indices])
    axes[2].set(
        title="Crashes by target gate", xlabel="target gate", ylabel="crashes", xticks=gate_indices
    )
    axes[2].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default="level2.toml")
    parser.add_argument("--mode", choices=("vector", "single"), default="vector")
    parser.add_argument("--num-worlds", type=int, default=256)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    """Run the crash hotspot analysis."""
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    config = load_config(ROOT / "config" / args.config)
    rows: list[dict[str, Any]] = []
    start = time.time()

    if args.mode == "single":
        rows = run_single_seed_episodes(config, checkpoint, args.seed_start, args.num_seeds)
        num_worlds = 1
    else:
        device = torch.device("cpu")
        env = make_envs(
            config=args.config,
            num_envs=args.num_worlds,
            jax_device="cpu",
            torch_device=device,
            coefs={"n_obs": N_HISTORY},
        )
        base_env = base_env_from_wrappers(env)
        base_env.settings = base_env.settings.replace(autoreset=False)
        base_env._step = base_env.build_step_fn()
        agent = load_agent(env, checkpoint, device)
        try:
            for batch_seed in range(args.seed_start, args.seed_start + args.num_batches):
                batch_rows = run_batch(
                    env=env,
                    base_env=base_env,
                    agent=agent,
                    batch_seed=batch_seed,
                    num_worlds=args.num_worlds,
                    env_freq=float(config.env.freq),
                    device=device,
                )
                rows.extend(batch_rows)
                batch_crashes = sum(row["crashed"] for row in batch_rows)
                batch_successes = sum(row["success"] for row in batch_rows)
                print(
                    f"batch_seed={batch_seed} success={batch_successes}/{args.num_worlds} "
                    f"crash={batch_crashes}/{args.num_worlds}"
                )
        finally:
            env.close()
        num_worlds = args.num_worlds

    summary = build_summary(rows, checkpoint, args.mode, num_worlds)
    csv_path = args.out_prefix.with_name(args.out_prefix.name + "_episodes.csv")
    json_path = args.out_prefix.with_name(args.out_prefix.name + "_summary.json")
    plot_path = args.out_prefix.with_name(args.out_prefix.name + "_hotspots.png")
    write_csv(csv_path, rows)
    with json_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    plot_hotspots(plot_path, rows, config)
    print(json.dumps(summary, indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {plot_path}")
    print(f"elapsed_s={time.time() - start:.1f}")


if __name__ == "__main__":
    main()
