# JAX PPO Migration Notes

This note records the additive JAX PPO migration path.  No existing PyTorch
training files are deleted or edited by this migration.

## Added Files

- `lsy_drone_racing/control/jax_ppo.py`
  - Shared JAX/Optax PPO implementation for Level2 and Level3.
  - Saves independent JAX checkpoints with format `lsy_jax_optax_ppo_v1`.
- `lsy_drone_racing/control/jax_ppo_fast.py`
  - Fast JAX PPO path that calls the low-level race env `env._step` inside
    `jax.lax.scan`, matching the accelerated reference architecture in
    `/home/aojili/lsy_drone_racing`.
  - Keeps the actor/critic, GAE, PPO update, checkpointing, and batched eval in
    JAX while avoiding the Python/Gym rollout loop.
- `scripts/train_jax_ppo_level2.py`
  - Thin CLI wrapper for the slower wrapper-based Level2 JAX PPO path.
- `scripts/train_jax_ppo_level3.py`
  - Thin CLI wrapper for the slower wrapper-based Level3 JAX PPO path.
- `scripts/validate_level2_jax_ppo.py`
  - Level2 Fast2-parameter validation entrypoint with a `success_rate >= 0.80`
    gate for the slower wrapper-based path.
- `scripts/train_jax_ppo_fast_level2.py`
  - Thin CLI wrapper for Level2 fast pure-JAX PPO training/evaluation.
- `scripts/train_jax_ppo_fast_level3.py`
  - Thin CLI wrapper for Level3 fast pure-JAX PPO training/evaluation.
- `scripts/validate_level2_jax_ppo_fast.py`
  - Level2 Fast2-parameter validation entrypoint for the fast path.  The latest
    accepted gate is `success_rate >= 0.79`.
- `notebooks/train_level3_jax_ppo.ipynb`
  - Notebook matching the existing Level3 PPO workflow, with the policy/update
    path switched to the fast JAX architecture.

## PyTorch to JAX Mapping

| Existing PyTorch path | New JAX path |
|---|---|
| `Args` in `train_CleanRL_ppo*.py` | `JaxPPOArgs` |
| `make_envs(...); JaxToTorch(...)` | `make_jax_envs(...)` without `JaxToTorch` |
| Python/Gym rollout loop | `jax_ppo_fast.FastState` plus `env._step` in `jax.lax.scan` |
| `Agent(nn.Module)` actor/critic | JAX parameter pytree plus `actor_apply`, `critic_apply` |
| `torch.distributions.Normal` | `gaussian_logprob`, `gaussian_entropy`, `policy_step` |
| Torch rollout tensors | JAX rollout arrays in `collect_rollout` |
| Torch GAE loop | `compute_gae` / `compute_advantage_batch` |
| `torch.optim.AdamW` | `optax.adamw` with gradient clipping |
| `torch.save(make_checkpoint(...))` | `save_jax_checkpoint(...)` pickle payload |
| `evaluate_ppo` deterministic Torch action | `evaluate_ppo` deterministic JAX action |

The slower `jax_ppo.py` trainer reuses the existing Python wrappers exactly, but
only reaches about 14k steps/s because rollout still crosses Gym/Python at each
step.  The fast `jax_ppo_fast.py` trainer ports action scaling, observation
flattening, and shaped reward state into pure JAX and calls `env._step` inside
`lax.scan`; this is the path to use for accelerated Level2 validation and Level3
training.

Current fast-path limitation: train-only Python wrappers from `[train.*]` in
`level2_dr.toml` such as thrust sag, command latency/response lag, and
observation latency/noise are bypassed.  Core env randomizations and env
disturbances remain active because they are passed into `gym.make_vec`.

## Level3 JAX Training

Smoke train:

```bash
pixi run -e gpu python scripts/train_jax_ppo_fast_level3.py \
  --config=level3.toml \
  --jax_device=cpu \
  --num_envs=2 \
  --num_steps=8 \
  --num_minibatches=2 \
  --total_timesteps=64 \
  --hidden_dim=16 \
  --model_name=/tmp/jax_ppo_level3_smoke.pkl \
  --train=True \
  --eval=1 \
  --wandb_enabled=False
```

Full Level3 run, matching the new notebook defaults:

```bash
pixi run -e gpu python scripts/train_jax_ppo_fast_level3.py \
  --config=level3_dr.toml \
  --jax_device=gpu \
  --num_envs=1024 \
  --num_steps=32 \
  --num_minibatches=8 \
  --update_epochs=5 \
  --total_timesteps=300000000 \
  --hidden_dim=256 \
  --learning_rate=0.0003 \
  --ent_coef=0.02 \
  --model_name=checkpoints/jax_level3_ppo/jax_level3_ppo_final.pkl \
  --checkpoint_dir=lsy_drone_racing/control/checkpoints/jax_level3_ppo \
  --checkpoint_interval=10000000 \
  --wandb_enabled=True
```

## Level2 Architecture Validation

Fast smoke check:

```bash
pixi run -e gpu python scripts/validate_level2_jax_ppo_fast.py \
  --smoke \
  --jax-device cpu \
  --eval-episodes 1 \
  --success-threshold 0.0
```

Full validation gate:

```bash
pixi run -e gpu python scripts/validate_level2_jax_ppo_fast.py \
  --eval-episodes 100 \
  --eval-seed-start 1 \
  --success-threshold 0.79
```

The full validation uses the Level2 Fast2 PPO preset from
`report/checkpoint_training_parameters.md`: `level2_dr.toml`, `num_envs=1024`,
`num_steps=32`, `hidden_dim=256`, `total_timesteps=200M`, and the Fast2 reward
coefficients.  The accepted fast-path validation result is:

- checkpoint:
  `lsy_drone_racing/control/checkpoints/jax_fast_level2_fast2_validation/jax_fast_level2_fast2_validation_final.pkl`
- summary:
  `lsy_drone_racing/control/checkpoints/jax_fast_level2_fast2_validation/jax_fast_level2_fast2_validation_summary.json`
- global step: `199,983,104`
- eval seeds: `1..100`
- success rate: `79%`
- crash rate: `21%`
- timeout rate: `0%`
- mean reward: `509.2178`
- mean steps: `220.13`

The user accepted the 79% result as sufficient validation.

## Speed Checks

Fast rollout benchmark:

```bash
pixi run -e gpu python scripts/train_jax_ppo_fast_level2.py \
  --train=False \
  --benchmark=True \
  --benchmark_repeat=5 \
  --benchmark_warmup=2 \
  --num_envs=2048 \
  --num_steps=32 \
  --num_minibatches=8 \
  --total_timesteps=65536 \
  --hidden_dim=256 \
  --jax_device=gpu
```

Observed result: `1,056,539` steps/s for Level2 rollout-only at 2048 envs x 32
steps.  The full PPO training loop after compilation ran around 500k steps/s at
1024 envs x 32 steps with 5 PPO epochs.

## Verified So Far

- `ruff check` passes for all newly added Python files.
- `py_compile` passes for all newly added Python files.
- `notebooks/train_level3_jax_ppo.ipynb` is valid JSON.
- Level3 fast JAX smoke training saves, loads, and evaluates a checkpoint.
- Level2 Fast2 fast validation smoke training saves, loads, evaluates, and writes a
  summary JSON.
- Level2 Fast2 fast validation completed at 79% deterministic success over 100
  seeds and was accepted by the user.
