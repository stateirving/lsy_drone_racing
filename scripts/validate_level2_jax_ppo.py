"""Train and evaluate the Level2 JAX PPO port with the known-good PPO preset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from lsy_drone_racing.control.jax_ppo import CONTROL_DIR, evaluate_ppo, level2_fast2_args, train_ppo

DEFAULT_RUN_NAME = "jax_level2_fast2_validation"
DEFAULT_CHECKPOINT_DIR = CONTROL_DIR / "checkpoints" / DEFAULT_RUN_NAME
DEFAULT_MODEL_PATH = DEFAULT_CHECKPOINT_DIR / f"{DEFAULT_RUN_NAME}_final.pkl"
DEFAULT_SUMMARY_PATH = DEFAULT_CHECKPOINT_DIR / f"{DEFAULT_RUN_NAME}_summary.json"


def parse_args() -> argparse.Namespace:
    """Parse validation CLI flags."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--checkpoint-interval", type=int, default=5_000_000)
    parser.add_argument("--jax-device", default="gpu")
    parser.add_argument("--wandb-enabled", action="store_true")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--wandb-project-name", default="ADR-PPO-Racing")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--wandb-run-id", default=DEFAULT_RUN_NAME)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed-start", type=int, default=1)
    parser.add_argument("--success-threshold", type=float, default=0.80)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    """Convert rows containing numpy scalar values to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    """Run the validation train/eval workflow."""
    args = parse_args()
    total_timesteps = 64 if args.smoke else None
    num_envs = 4 if args.smoke else None
    num_steps = 8 if args.smoke else None
    num_minibatches = 2 if args.smoke else None
    checkpoint_interval = 0 if args.smoke else int(args.checkpoint_interval)
    ppo_args = level2_fast2_args(
        jax_device=args.jax_device,
        wandb_mode=args.wandb_mode,
        wandb_project_name=args.wandb_project_name,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_run_id=args.wandb_run_id,
        log_interval=args.log_interval,
        **({"total_timesteps": total_timesteps} if total_timesteps is not None else {}),
        **({"num_envs": num_envs} if num_envs is not None else {}),
        **({"num_steps": num_steps} if num_steps is not None else {}),
        **({"num_minibatches": num_minibatches} if num_minibatches is not None else {}),
    )
    if not args.skip_train:
        train_ppo(
            ppo_args,
            args.model_path,
            wandb_enabled=args.wandb_enabled,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
        )
    summary = evaluate_ppo(
        ppo_args,
        n_eval=args.eval_episodes,
        model_path=args.model_path,
        seed_start=args.eval_seed_start,
    )
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n")
    if summary["success_rate"] < args.success_threshold:
        raise SystemExit(
            f"Level2 JAX PPO validation failed: success_rate={summary['success_rate']:.2%} "
            f"< threshold={args.success_threshold:.2%}."
        )
    print(
        f"Level2 JAX PPO validation passed: success_rate={summary['success_rate']:.2%} "
        f">= {args.success_threshold:.2%}"
    )


if __name__ == "__main__":
    main()
