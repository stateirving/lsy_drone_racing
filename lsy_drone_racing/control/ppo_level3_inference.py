"""Inference controller for the direct level3 CleanRL PPO policy."""

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
from lsy_drone_racing.control.ppo_level3_observation import (
    ALL_GATES_OBSERVATION_LAYOUTS,
    LEGACY_OBSERVATION_LAYOUT,
    LOCAL_OBSTACLE_FRAME_OBSERVATION_LAYOUT,
    LOCAL_OBSTACLE_OBSERVATION_LAYOUTS,
    LOCAL_HISTORY_OBSERVATION_LAYOUT_ALIASES,
    SPHERICAL_GATE_OBS_COUNT_BY_LAYOUT,
    SPHERICAL_GATE_OBSTACLE_POLAR_OBSERVATION_LAYOUTS,
    TARGET_PROGRESS_OBSERVATION_LAYOUTS,
    WORLD_HISTORY_OBSERVATION_LAYOUT,
    checkpoint_hidden_dim,
    unpack_checkpoint,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Historical no-obstacle warm start; keep only as a pre-obstacle reference.
# MODEL_NAME = "checkpoints/level3_Curriculum_Xobstacle_speedlimit1.5/level3_Curriculum_Xobstacle_speedlimit1.5_final.ckpt"

# 150140_dataxxl_step_1210M: level3.toml seeds 1-100, Torch/vector-env,
# 62/100 success, mean gates 3.01, success time 11.28 s, 38% crash,
# 0% timeout, 27% pos-safety violation. Best slow-eval dataxxl candidate.
# MODEL_NAME = "checkpoints/level3_Curriculum_Vobstacle_speedlimit1.5_150140_dataxxl/level3_Curriculum_Vobstacle_speedlimit1.5_150140_dataxxl_step_1210000000.ckpt"

# 150140_dataxxl_step_1290M: current level3.toml seeds 1-50, fast JAX,
# 38/50 success, mean gates 3.42, success time 11.99 s, 24% crash,
# 24% pos-safety violation. Active until newtrack finalists get slow-eval checked.
MODEL_NAME = "checkpoints/level3_Curriculum_Vobstacle_speedlimit1.5_150140_dataxxl/level3_Curriculum_Vobstacle_speedlimit1.5_150140_dataxxl_step_1290000000.ckpt"

# level3_Curriculum_Vobstacle_newtrack, current level3.toml seeds 1-50, fast JAX:
# 520M: 40/50 success, clean 30/50, mean gates 3.58, success time 11.26 s,
# 20% crash, 28% pos-safety violation. Best raw-success newtrack candidate.
# MODEL_NAME = "checkpoints/level3_Curriculum_Vobstacle_newtrack/level3_Curriculum_Vobstacle_newtrack_step_520000000.ckpt"
# 700M: 40/50 success, clean 28/50, mean gates 3.54, success time 11.50 s,
# 20% crash, 34% pos-safety violation. Tied raw-success backup.
# MODEL_NAME = "checkpoints/level3_Curriculum_Vobstacle_newtrack/level3_Curriculum_Vobstacle_newtrack_step_700000000.ckpt"
# 620M: 39/50 success, clean 32/50, mean gates 3.56, success time 11.02 s,
# 22% crash, 22% pos-safety violation. Stronger clean/safety backup.
# MODEL_NAME = "checkpoints/level3_Curriculum_Vobstacle_newtrack/level3_Curriculum_Vobstacle_newtrack_step_620000000.ckpt"
# final: 39/50 success, clean 28/50, mean gates 3.56, success time 11.10 s,
# 22% crash, 26% pos-safety violation. Good, but not the best checkpoint.
# MODEL_NAME = "checkpoints/level3_Curriculum_Vobstacle_newtrack/level3_Curriculum_Vobstacle_newtrack_final.ckpt"


# MODEL_NAME = "checkpoints/finetune_level2_dr/finetune_level2_dr_final.ckpt"

N_HISTORY = 2
WORLD_HISTORY_DIM = 13
LOCAL_HISTORY_DIM = 7
N_LOCAL_OBSTACLES = 2
HISTORY_DIM_BY_LAYOUT = {
    LEGACY_OBSERVATION_LAYOUT: WORLD_HISTORY_DIM,
    WORLD_HISTORY_OBSERVATION_LAYOUT: WORLD_HISTORY_DIM,
    **{layout: LOCAL_HISTORY_DIM for layout in LOCAL_HISTORY_OBSERVATION_LAYOUT_ALIASES},
}
GATE_OPENING_CORNERS_LOCAL = np.array(
    [
        [0.0, -0.2, 0.2],
        [0.0, 0.2, 0.2],
        [0.0, 0.2, -0.2],
        [0.0, -0.2, -0.2],
    ],
    dtype=np.float32,
)
GATE_FRAME_CORNERS_LOCAL = np.array(
    [
        [0.0, -0.36, 0.36],
        [0.0, 0.36, 0.36],
        [0.0, 0.36, -0.36],
        [0.0, -0.36, -0.36],
    ],
    dtype=np.float32,
)
GATE_CORNERS_LOCAL = GATE_OPENING_CORNERS_LOCAL


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
    """Run the trained direct level3 PPO policy as an attitude controller."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize policy, action scaling, and observation history."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("PPOLevel2Inference expects env.control_mode = 'attitude'.")
        train_cfg = config.get("train", {})
        observation_cfg = train_cfg.get("observation", {}) if train_cfg is not None else {}
        self._zero_obstacle_obs = bool(observation_cfg.get("zero_obstacle_obs", False))

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
        self.n_local_obstacles = min(N_LOCAL_OBSTACLES, self.n_obstacles)
        base_obs_dim = 1 + 3 + 3 + 9

        self.device = torch.device("cpu")
        model_path = Path(__file__).parent / MODEL_NAME
        if not model_path.exists():
            raise FileNotFoundError(f"PPO checkpoint not found: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        model_state_dict, self.observation_layout = unpack_checkpoint(checkpoint)
        self.hidden_dim = checkpoint_hidden_dim(checkpoint, model_state_dict)
        if self.observation_layout not in HISTORY_DIM_BY_LAYOUT:
            raise ValueError(f"Unsupported PPO observation layout: {self.observation_layout}")
        self.history_dim = HISTORY_DIM_BY_LAYOUT[self.observation_layout]
        self._include_target_progress = (
            self.observation_layout in TARGET_PROGRESS_OBSERVATION_LAYOUTS
        )
        self._use_all_gates = self.observation_layout in ALL_GATES_OBSERVATION_LAYOUTS
        self._use_local_obstacles = (
            self.observation_layout in LOCAL_OBSTACLE_OBSERVATION_LAYOUTS
        )
        self._use_nearest_other_gate_frame = (
            self.observation_layout == LOCAL_OBSTACLE_FRAME_OBSERVATION_LAYOUT
        )
        self._use_spherical_gate_obstacle_polar = (
            self.observation_layout in SPHERICAL_GATE_OBSTACLE_POLAR_OBSERVATION_LAYOUTS
        )
        self.n_spherical_gates = SPHERICAL_GATE_OBS_COUNT_BY_LAYOUT.get(
            self.observation_layout, 0
        )
        if self._use_local_obstacles or self._use_spherical_gate_obstacle_polar:
            self.n_local_obstacles = N_LOCAL_OBSTACLES
        if self._use_all_gates:
            current_obs_dim = (
                base_obs_dim
                + 12 * self.n_gates
                + self.n_gates
                + self.n_gates
                + 4 * self.n_obstacles
                + self.action_dim
                + N_HISTORY * self.history_dim
            )
        elif self._use_spherical_gate_obstacle_polar:
            current_obs_dim = (
                1
                + 3
                + 3
                + 3
                + 4 * self.n_spherical_gates
                + 2 * self.n_local_obstacles
                + self.action_dim
                + N_HISTORY * self.history_dim
            )
        elif self._use_local_obstacles:
            current_obs_dim = (
                base_obs_dim
                + 12
                + 1
                + 12
                + 1
                + 4 * self.n_local_obstacles
                + self.action_dim
                + N_HISTORY * self.history_dim
            )
        else:
            current_obs_dim = (
                base_obs_dim
                + 36
                + int(self._include_target_progress)
                + 4 * self.n_obstacles
                + self.n_gates
                + self.action_dim
                + N_HISTORY * self.history_dim
            )
        self.obs_dim = int(model_state_dict["actor_mean.0.weight"].shape[1])
        self._include_prev_gate = (
            not self._use_all_gates
            and not self._use_local_obstacles
            and not self._use_spherical_gate_obstacle_polar
            and self.obs_dim == current_obs_dim
        )
        supported_obs_dims = (current_obs_dim,) if (
            self._use_all_gates
            or self._use_local_obstacles
            or self._use_spherical_gate_obstacle_polar
        ) else (
            current_obs_dim,
            current_obs_dim - 12,
        )
        if self.obs_dim not in supported_obs_dims:
            raise ValueError(
                f"Unsupported PPO checkpoint input size {self.obs_dim}; "
                f"expected {supported_obs_dims} "
                f"for layout {self.observation_layout}."
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
        if self._use_spherical_gate_obstacle_polar:
            gate_spherical = self._gate_spherical_target_next_nearest_observation(
                obs, active_target_gate, pos, rot_t
            )
            flat = np.concatenate(
                [
                    pos[2:3],
                    vel_body,
                    ang_vel,
                    rot_t[:, 2],
                    gate_spherical,
                    self._nearest_obstacles_polar_xy(obs, pos, rot),
                    self._last_action_norm,
                    self._history.reshape(-1),
                ]
            ).astype(np.float32)
            self._history = np.concatenate(
                [self._history[1:], self._basic_history(obs)[None, :]]
            )
            return flat

        prev_gate = max(active_target_gate - 1, 0)
        gate_prev = self._gate_corners_body(obs, prev_gate, pos, rot_t)
        if active_target_gate <= 0:
            gate_prev = np.zeros_like(gate_prev)
        gate_current = self._gate_corners_body(obs, target_gate, pos, rot_t)
        next_gate = min(max(target_gate, 0) + 1, self.n_gates - 1)
        gate_next = self._gate_corners_body(obs, next_gate, pos, rot_t)
        if self.observation_layout != LEGACY_OBSERVATION_LAYOUT:
            obstacle_features = self._obstacle_heading_xy(obs, pos, rot)
        else:
            obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=np.float32)
            obstacle_features = ((obstacles_pos - pos[None, :]) @ rot_t.T).reshape(-1)
        obs_parts = [
            pos[2:3],
            vel_body,
            ang_vel,
            rot.reshape(-1),
        ]
        if self._use_all_gates:
            target_gate_onehot = np.zeros(self.n_gates, dtype=np.float32)
            if 0 <= target_gate < self.n_gates:
                target_gate_onehot[target_gate] = 1.0
            obs_parts.extend(
                [
                    self._all_gate_corners_body(obs, pos, rot_t),
                    target_gate_onehot,
                    np.asarray(obs["gates_visited"], dtype=np.float32),
                    obstacle_features,
                    self._last_action_norm,
                    self._history.reshape(-1),
                ]
            )
            flat = np.concatenate(obs_parts).astype(np.float32)
            self._history = np.concatenate(
                [self._history[1:], self._basic_history(obs)[None, :]]
            )
            return flat

        if self._use_local_obstacles:
            nearest_other_gate = self._nearest_other_gate_idx(obs, pos, active_target_gate)
            nearest_corners_local = (
                GATE_FRAME_CORNERS_LOCAL
                if self._use_nearest_other_gate_frame
                else GATE_OPENING_CORNERS_LOCAL
            )
            obs_parts.extend(
                [
                    gate_current,
                    self._gate_known_flag(obs, active_target_gate),
                    self._gate_corners_body(
                        obs, nearest_other_gate, pos, rot_t, nearest_corners_local
                    ),
                    self._gate_known_flag(obs, nearest_other_gate),
                    self._nearest_obstacles_heading_xy(obs, pos, rot),
                    self._last_action_norm,
                    self._history.reshape(-1),
                ]
            )
            flat = np.concatenate(obs_parts).astype(np.float32)
            self._history = np.concatenate(
                [self._history[1:], self._basic_history(obs)[None, :]]
            )
            return flat

        if self._include_target_progress:
            target_progress = np.array(
                [self.n_gates if target_gate < 0 else target_gate], dtype=np.float32
            ) / self.n_gates
            obs_parts.append(target_progress)
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
        corners_local: NDArray[np.floating] | None = None,
    ) -> NDArray[np.float32]:
        """Return the current gate corners relative to the drone body frame."""
        corners_local = GATE_OPENING_CORNERS_LOCAL if corners_local is None else corners_local
        gate_idx = gate_idx % self.n_gates
        gate_pos = np.asarray(obs["gates_pos"], dtype=np.float32)[gate_idx]
        gate_quat = np.asarray(obs["gates_quat"], dtype=np.float32)[gate_idx]
        gate_rot = self.quat_to_rotmat(gate_quat)
        corners_world = (
            gate_pos[None, :]
            + np.asarray(corners_local, dtype=np.float32) @ gate_rot.T
        )
        return ((corners_world - pos[None, :]) @ rot_t.T).reshape(-1).astype(np.float32)

    def _all_gate_corners_body(
        self,
        obs: dict[str, NDArray[np.floating]],
        pos: NDArray[np.floating],
        rot_t: NDArray[np.floating],
    ) -> NDArray[np.float32]:
        gates_pos = np.asarray(obs["gates_pos"], dtype=np.float32)
        gates_quat = np.asarray(obs["gates_quat"], dtype=np.float32)
        gate_rot = np.stack([self.quat_to_rotmat(quat) for quat in gates_quat], axis=0)
        corners_world = gates_pos[:, None, :] + np.einsum(
            "gij,kj->gki", gate_rot, GATE_CORNERS_LOCAL
        )
        corners_body = np.einsum("ij,gkj->gki", rot_t, corners_world - pos[None, None, :])
        return corners_body.reshape(-1).astype(np.float32)

    def _nearest_other_gate_idx(
        self,
        obs: dict[str, NDArray[np.floating]],
        pos: NDArray[np.floating],
        active_target_gate: int,
    ) -> int:
        """Return the closest non-target gate index in XY."""
        gates_pos = np.asarray(obs["gates_pos"], dtype=np.float32)
        distance_xy = np.linalg.norm(gates_pos[:, :2] - pos[None, :2], axis=-1)
        distance_xy[int(active_target_gate) % self.n_gates] = np.inf
        return int(np.argmin(distance_xy))

    def _gate_known_flag(
        self, obs: dict[str, NDArray[np.floating]], gate_idx: int
    ) -> NDArray[np.float32]:
        """Return the selected gate known/visited flag."""
        gates_visited = np.asarray(obs["gates_visited"], dtype=np.float32)
        return np.array([gates_visited[int(gate_idx) % self.n_gates]], dtype=np.float32)

    def _select_gate(
        self, obs: dict[str, NDArray[np.floating]], gate_idx: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return selected gate position and quaternion."""
        gate_idx = int(gate_idx) % self.n_gates
        gate_pos = np.asarray(obs["gates_pos"], dtype=np.float32)[gate_idx]
        gate_quat = np.asarray(obs["gates_quat"], dtype=np.float32)[gate_idx]
        return gate_pos, gate_quat

    def _gate_spherical_observation(
        self,
        obs: dict[str, NDArray[np.floating]],
        gate_idx: int,
        reference_pos: NDArray[np.floating],
        reference_rot_t: NDArray[np.floating],
    ) -> NDArray[np.float32]:
        """Return [r, theta, phi, alpha] for a gate from a reference frame."""
        gate_pos, gate_quat = self._select_gate(obs, gate_idx)
        gate_rot = self.quat_to_rotmat(gate_quat)
        relative_world = gate_pos - np.asarray(reference_pos, dtype=np.float32)
        relative_ref = np.asarray(reference_rot_t, dtype=np.float32) @ relative_world
        radius = float(np.linalg.norm(relative_ref))
        xy_radius = float(np.linalg.norm(relative_ref[:2]))
        theta = float(np.arctan2(relative_ref[1], relative_ref[0]))
        phi = float(np.arctan2(relative_ref[2], xy_radius))
        direction_world = relative_world / max(radius, 1e-6)
        gate_normal_world = gate_rot[:, 0]
        normal_alignment = float(np.dot(gate_normal_world, direction_world))
        alpha = float(np.arccos(np.clip(normal_alignment, -1.0, 1.0)))
        return np.array([radius, theta, phi, alpha], dtype=np.float32)

    def _gate_spherical_target_next_nearest_observation(
        self,
        obs: dict[str, NDArray[np.floating]],
        active_target_gate: int,
        reference_pos: NDArray[np.floating],
        reference_rot_t: NDArray[np.floating],
    ) -> NDArray[np.float32]:
        """Return current target, next target, and nearest non-current gate features."""
        target_gate_spherical = self._gate_spherical_observation(
            obs, active_target_gate, reference_pos, reference_rot_t
        )
        next_gate = min(active_target_gate + 1, self.n_gates - 1)
        target_gate_pos, target_gate_quat = self._select_gate(obs, active_target_gate)
        target_gate_rot_t = self.quat_to_rotmat(target_gate_quat).T
        next_gate_spherical = self._gate_spherical_observation(
            obs, next_gate, target_gate_pos, target_gate_rot_t
        )
        if active_target_gate >= self.n_gates - 1:
            next_gate_spherical = np.zeros_like(next_gate_spherical)
        if self.n_spherical_gates == 2:
            return np.concatenate([target_gate_spherical, next_gate_spherical]).astype(
                np.float32
            )
        nearest_gate_idx = self._nearest_gate_indices_excluding_target(
            obs, reference_pos, active_target_gate
        )
        nearest_gate1_spherical = self._gate_spherical_observation(
            obs, int(nearest_gate_idx[0]), reference_pos, reference_rot_t
        )
        nearest_gate2_spherical = self._gate_spherical_observation(
            obs, int(nearest_gate_idx[1]), reference_pos, reference_rot_t
        )
        return np.concatenate(
            [
                target_gate_spherical,
                next_gate_spherical,
                nearest_gate1_spherical,
                nearest_gate2_spherical,
            ]
        ).astype(np.float32)

    def _nearest_gate_indices_excluding_target(
        self,
        obs: dict[str, NDArray[np.floating]],
        pos: NDArray[np.floating],
        active_target_gate: int,
    ) -> NDArray[np.int64]:
        """Return the two nearest gate indices after masking the current target."""
        gates_pos = np.asarray(obs["gates_pos"], dtype=np.float32)
        distance = np.linalg.norm(
            gates_pos - np.asarray(pos, dtype=np.float32)[None, :], axis=-1
        )
        distance[int(active_target_gate) % self.n_gates] = np.inf
        return np.argsort(distance)[:2]

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

    def _nearest_obstacles_heading_xy(
        self,
        obs: dict[str, NDArray[np.floating]],
        pos: NDArray[np.floating],
        rot: NDArray[np.floating],
    ) -> NDArray[np.float32]:
        """Return nearest obstacle [forward, left, XY distance, detected] features."""
        obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=np.float32)
        if self._zero_obstacle_obs or obstacles_pos.shape[0] == 0:
            return np.zeros(4 * self.n_local_obstacles, dtype=np.float32)
        relative_xy = obstacles_pos[:, :2] - pos[None, :2]
        heading_forward = np.array(rot[:2, 0], dtype=np.float32, copy=True)
        heading_forward /= max(float(np.linalg.norm(heading_forward)), 1e-6)
        heading_left = np.array([-heading_forward[1], heading_forward[0]], dtype=np.float32)
        relative_forward = relative_xy @ heading_forward
        relative_left = relative_xy @ heading_left
        distance_xy = np.linalg.norm(relative_xy, axis=-1)
        detected = np.asarray(obs["obstacles_visited"], dtype=np.float32)
        features = np.stack([relative_forward, relative_left, distance_xy, detected], axis=-1)
        n_take = min(self.n_local_obstacles, obstacles_pos.shape[0])
        order = np.argsort(distance_xy)[:n_take]
        nearest = features[order]
        if n_take < self.n_local_obstacles:
            padding = np.zeros((self.n_local_obstacles - n_take, 4), dtype=np.float32)
            nearest = np.concatenate([nearest, padding], axis=0)
        return nearest.reshape(-1).astype(np.float32)

    def _nearest_obstacles_polar_xy(
        self,
        obs: dict[str, NDArray[np.floating]],
        pos: NDArray[np.floating],
        rot: NDArray[np.floating],
    ) -> NDArray[np.float32]:
        """Return nearest vertical obstacle [XY distance, body-frame azimuth] features."""
        obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=np.float32)
        if self._zero_obstacle_obs or obstacles_pos.shape[0] == 0:
            return np.zeros(2 * self.n_local_obstacles, dtype=np.float32)
        relative_xy = obstacles_pos[:, :2] - pos[None, :2]
        heading_forward = np.array(rot[:2, 0], dtype=np.float32, copy=True)
        heading_forward /= max(float(np.linalg.norm(heading_forward)), 1e-6)
        heading_left = np.array([-heading_forward[1], heading_forward[0]], dtype=np.float32)
        relative_forward = relative_xy @ heading_forward
        relative_left = relative_xy @ heading_left
        distance_xy = np.linalg.norm(relative_xy, axis=-1)
        theta = np.arctan2(relative_left, relative_forward)
        features = np.stack([distance_xy, theta], axis=-1)
        n_take = min(self.n_local_obstacles, obstacles_pos.shape[0])
        order = np.argsort(distance_xy)[:n_take]
        nearest = features[order]
        if n_take < self.n_local_obstacles:
            padding = np.zeros((self.n_local_obstacles - n_take, 2), dtype=np.float32)
            nearest = np.concatenate([nearest, padding], axis=0)
        return nearest.reshape(-1).astype(np.float32)

    def _basic_history(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.float32]:
        """Return the short state history expected by the loaded checkpoint layout."""
        if self.history_dim == LOCAL_HISTORY_DIM:
            pos = np.asarray(obs["pos"], dtype=np.float32)
            quat = np.asarray(obs["quat"], dtype=np.float32)
            vel = np.asarray(obs["vel"], dtype=np.float32)
            ang_vel = np.asarray(obs["ang_vel"], dtype=np.float32)
            rot = self.quat_to_rotmat(quat)
            vel_body = rot.T @ vel
            return np.concatenate([pos[2:3], vel_body, ang_vel]).astype(np.float32)

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
