"""Inference controller for the direct level2 CleanRL PPO policy."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from drone_models.core import load_params
from torch import Tensor
from torch.distributions.normal import Normal

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.ppo_level2_observation import (
    LEGACY_OBSERVATION_LAYOUT,
    OBSERVATION_LAYOUT,
    checkpoint_hidden_dim,
    unpack_checkpoint,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Most smooth without DR; lower real-deploy robustness than DR checkpoints.
# MODEL_NAME = "checkpoints/ppo_level2_cmdtilt1p5_160M/ppo_level2_cmdtilt1p5_160M_step_100000000.ckpt"

# Best current real-deploy baseline: DR(mass,inertia,dynamics,action), level2 seeds 1-100 success 85%.
# MODEL_NAME = "checkpoints/ppo_level2_DR_nn256/ppo_level2_DR_nn256_final.ckpt"

# DR(mass,inertia,dynamics,action,thrust scale,battery sag); nominal level2 success 71%.
# MODEL_NAME = "checkpoints/ppo_level2_DR_nn256_thrustsag/ppo_level2_DR_nn256_thrustsag_final.ckpt"

# DR with 20% thrust-to-weight style randomization; best checkpoint was 110M, fast but aggressive.
# MODEL_NAME = "checkpoints/ppo_level2_DR_nn256_th2w20per/ppo_level2_DR_nn256_th2w20per_step_110000000.ckpt"

# DR + thrust/sag + action latency/response lag + observation latency/noise.
# Best balanced new-DR real-deploy candidate: level2 seeds 1-100 success 80%,
# smoother/safer than onemoretime (actual tilt p95 ~=33.5 deg, worst ~=34.1 deg).
MODEL_NAME = "checkpoints/level2_DR_latencyobs_middlemanuever/level2_DR_latencyobs_middlemanuever_final.ckpt"

# Same reward as middlemanuever, trained longer; level2 success 81% but more aggressive
# (actual tilt p95 ~=39.7 deg, worst ~=41.8 deg). Keep as speed/success backup.
# MODEL_NAME = "checkpoints/level2_DR_latencyobs_middlemanuever_onemoretime/level2_DR_latencyobs_middlemanuever_onemoretime_final.ckpt"

N_HISTORY = 2
HISTORY_DIM = 13
GATE_CORNERS_LOCAL = np.array(
    [
        [0.0, -0.2, 0.2],
        [0.0, 0.2, 0.2],
        [0.0, 0.2, -0.2],
        [0.0, -0.2, -0.2],
    ],
    dtype=np.float32,
)


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize a linear layer exactly as in training."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    """Policy network matching train_CleanRL_ppo.Agent."""

    def __init__(
        self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...], hidden_dim: int = 128
    ):
        """Build actor and critic modules so the training checkpoint loads unchanged."""
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

    def get_action_and_value(
        self, x: Tensor, action: Tensor | None = None, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return action, log-probability, entropy, and value."""
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = action_mean if deterministic else probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class PPOLevel2Inference(Controller):
    """Run the trained direct level2 PPO policy as an attitude controller."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize policy, action scaling, and observation history."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("PPOLevel2Inference expects env.control_mode = 'attitude'.")

        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.thrust_min = float(drone_params["thrust_min"] * 4)
        self.thrust_max = float(drone_params["thrust_max"] * 4)
        self.action_low = np.array(
            [-np.pi / 2, -np.pi / 2, -np.pi / 2, self.thrust_min], dtype=np.float32
        )
        self.action_high = np.array(
            [np.pi / 2, np.pi / 2, np.pi / 2, self.thrust_max], dtype=np.float32
        )
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_mean = (self.action_high + self.action_low) / 2.0

        self.n_gates = int(np.asarray(obs["gates_pos"]).shape[0])
        self.n_obstacles = int(np.asarray(obs["obstacles_pos"]).shape[0])
        self.action_dim = 4
        current_obs_dim = (
            1
            + 3
            + 3
            + 9
            + 1
            + 12
            + 12
            + 12
            + 4 * self.n_obstacles
            + self.n_gates
            + self.action_dim
            + N_HISTORY * HISTORY_DIM
        )

        self.device = torch.device("cpu")
        model_path = Path(__file__).parent / MODEL_NAME
        if not model_path.exists():
            raise FileNotFoundError(f"PPO checkpoint not found: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        model_state_dict, self.observation_layout = unpack_checkpoint(checkpoint)
        self.hidden_dim = checkpoint_hidden_dim(checkpoint, model_state_dict)
        if self.observation_layout not in (LEGACY_OBSERVATION_LAYOUT, OBSERVATION_LAYOUT):
            raise ValueError(f"Unsupported PPO observation layout: {self.observation_layout}")
        self.obs_dim = int(model_state_dict["actor_mean.0.weight"].shape[1])
        self._include_prev_gate = self.obs_dim == current_obs_dim
        if self.obs_dim not in (current_obs_dim, current_obs_dim - 12):
            raise ValueError(
                f"Unsupported PPO checkpoint input size {self.obs_dim}; "
                f"expected {current_obs_dim} or {current_obs_dim - 12}."
            )
        self.agent = PPOAgent((self.obs_dim,), (self.action_dim,), hidden_dim=self.hidden_dim).to(
            self.device
        )
        self.agent.load_state_dict(model_state_dict)
        self.agent.eval()

        self._history = np.repeat(self._basic_history(obs)[None, :], N_HISTORY, axis=0)
        self._last_action_norm = np.zeros(self.action_dim, dtype=np.float32)
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute attitude command [roll, pitch, yaw, thrust]."""
        target_gate = int(np.asarray(obs["target_gate"]).item())
        if target_gate < 0:
            self._finished = True

        obs_rl = self._obs_rl(obs)
        obs_tensor = torch.tensor(obs_rl, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_norm, _, _, _ = self.agent.get_action_and_value(obs_tensor, deterministic=True)

        action_norm_np = action_norm.squeeze(0).cpu().numpy().astype(np.float32)
        if not np.isfinite(action_norm_np).all():
            action_norm_np = np.zeros(self.action_dim, dtype=np.float32)
        self._last_action_norm = action_norm_np
        return self._scale_action(action_norm_np).astype(np.float32)

    def _obs_rl(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.float32]:
        """Build the same flat observation vector as RaceObservation in training."""
        pos = np.asarray(obs["pos"], dtype=np.float32)
        quat = np.asarray(obs["quat"], dtype=np.float32)
        vel = np.asarray(obs["vel"], dtype=np.float32)
        ang_vel = np.asarray(obs["ang_vel"], dtype=np.float32)
        target_gate = int(np.asarray(obs["target_gate"]).item())

        rot = self.quat_to_rotmat(quat)
        rot_t = rot.T
        vel_body = rot_t @ vel
        active_target_gate = self.n_gates - 1 if target_gate < 0 else target_gate
        prev_gate = max(active_target_gate - 1, 0)
        gate_prev = self._gate_corners_body(obs, prev_gate, pos, rot_t)
        if active_target_gate <= 0:
            gate_prev = np.zeros_like(gate_prev)
        gate_current = self._gate_corners_body(obs, target_gate, pos, rot_t)
        next_gate = min(max(target_gate, 0) + 1, self.n_gates - 1)
        gate_next = self._gate_corners_body(obs, next_gate, pos, rot_t)
        if self.observation_layout == OBSERVATION_LAYOUT:
            obstacle_features = self._obstacle_heading_xy(obs, pos, rot)
        else:
            obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=np.float32)
            obstacle_features = ((obstacles_pos - pos[None, :]) @ rot_t.T).reshape(-1)
        target_progress = np.array(
            [self.n_gates if target_gate < 0 else target_gate], dtype=np.float32
        ) / self.n_gates

        obs_parts = [
            pos[2:3],
            vel_body,
            ang_vel,
            rot.reshape(-1),
            target_progress,
        ]
        if self._include_prev_gate:
            obs_parts.append(gate_prev)
        obs_parts.extend(
            [
                gate_current,
                gate_next,
                obstacle_features,
                np.asarray(obs["gates_visited"], dtype=np.float32),
            ]
        )
        if self.observation_layout == LEGACY_OBSERVATION_LAYOUT:
            obs_parts.append(np.asarray(obs["obstacles_visited"], dtype=np.float32))
        obs_parts.extend([self._last_action_norm, self._history.reshape(-1)])
        flat = np.concatenate(obs_parts).astype(np.float32)

        self._history = np.concatenate([self._history[1:], self._basic_history(obs)[None, :]])
        return flat

    def _gate_corners_body(
        self,
        obs: dict[str, NDArray[np.floating]],
        gate_idx: int,
        pos: NDArray[np.floating],
        rot_t: NDArray[np.floating],
    ) -> NDArray[np.float32]:
        """Return the current gate corners relative to the drone body frame."""
        gate_idx = gate_idx % self.n_gates
        gate_pos = np.asarray(obs["gates_pos"], dtype=np.float32)[gate_idx]
        gate_quat = np.asarray(obs["gates_quat"], dtype=np.float32)[gate_idx]
        gate_rot = self.quat_to_rotmat(gate_quat)
        corners_world = gate_pos[None, :] + GATE_CORNERS_LOCAL @ gate_rot.T
        return ((corners_world - pos[None, :]) @ rot_t.T).reshape(-1).astype(np.float32)

    @staticmethod
    def _obstacle_heading_xy(
        obs: dict[str, NDArray[np.floating]],
        pos: NDArray[np.floating],
        rot: NDArray[np.floating],
    ) -> NDArray[np.float32]:
        """Return fixed-order [forward, left, XY distance, detected] obstacle features."""
        obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=np.float32)
        relative_xy = obstacles_pos[:, :2] - pos[None, :2]
        heading_forward = np.array(rot[:2, 0], dtype=np.float32, copy=True)
        heading_forward /= max(float(np.linalg.norm(heading_forward)), 1e-6)
        heading_left = np.array([-heading_forward[1], heading_forward[0]], dtype=np.float32)
        relative_forward = relative_xy @ heading_forward
        relative_left = relative_xy @ heading_left
        distance_xy = np.linalg.norm(relative_xy, axis=-1)
        detected = np.asarray(obs["obstacles_visited"], dtype=np.float32)
        features = np.stack([relative_forward, relative_left, distance_xy, detected], axis=-1)
        return features.reshape(-1).astype(np.float32)

    @staticmethod
    def _basic_history(obs: dict[str, NDArray[np.floating]]) -> NDArray[np.float32]:
        """Return [pos, quat, vel, ang_vel]."""
        return np.concatenate(
            [
                np.asarray(obs["pos"], dtype=np.float32),
                np.asarray(obs["quat"], dtype=np.float32),
                np.asarray(obs["vel"], dtype=np.float32),
                np.asarray(obs["ang_vel"], dtype=np.float32),
            ]
        ).astype(np.float32)

    @staticmethod
    def quat_to_rotmat(quat: NDArray[np.floating]) -> NDArray[np.float32]:
        """Convert xyzw quaternion to a body-to-world rotation matrix."""
        x, y, z, w = np.asarray(quat, dtype=np.float32)
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float32,
        )

    def _scale_action(self, action_norm: NDArray[np.floating]) -> NDArray[np.float32]:
        """Scale normalized policy actions to the attitude action space."""
        action_norm = np.clip(np.asarray(action_norm, dtype=np.float32), -1.0, 1.0)
        return action_norm * self.action_scale + self.action_mean

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Signal completion once the race is finished."""
        if int(np.asarray(obs["target_gate"]).item()) < 0:
            self._finished = True
        return self._finished

    def episode_callback(self):
        """Reset completion state after an episode."""
        self._finished = False
