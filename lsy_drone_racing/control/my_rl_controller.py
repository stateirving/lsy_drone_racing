"""PPO Racing Controller - Inference controller for sim.py

Adapts trained PPO checkpoint to single-environment inference interface.
Key point: Observation processing logic must be completely consistent with RacingObservationWrapper during training.

Usage:
    python scripts/sim.py --config level0_no_obst.toml --controller ppo_racing_controller.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation
from drone_controllers.mellinger.params import ForceTorqueParams

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ============================================================================
# Network Definition (completely consistent with ppo_racing.py)
# ============================================================================

def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Orthogonal initialization of network layer."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """PPO Agent Network (completely consistent with training)."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        
        # Actor network
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
            nn.Tanh(),
        )
        
        # Action standard deviation
        if action_dim == 4:
            init_logstd = torch.tensor([[-1.0, -1.0, -1.0, -0.5]])
        self.actor_logstd = nn.Parameter(init_logstd)
    
    def get_action_and_value(
        self, 
        x: torch.Tensor, 
        action: torch.Tensor | None = None, 
        deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        probs = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = action_mean if deterministic else probs.sample()
        
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


# ============================================================================
# Observation Processing (single-environment version, consistent with RacingObservationWrapper logic)
# ============================================================================

class ObservationProcessor:
    """Single-environment observation processor, replicates RacingObservationWrapper logic.
    
    Key timing:
    - During training, history is updated at the start of step using the previous step's obs
    - prev_action is updated at the end of step
    - Here we need to precisely replicate this timing
    """
    
    # 4 corner offsets in gate local coordinate system
    GATE_CORNERS_LOCAL = np.array([
        [0.0, -0.2,  0.2],  # Top left
        [0.0,  0.2,  0.2],  # Top right
        [0.0,  0.2, -0.2],  # Bottom right
        [0.0, -0.2, -0.2],  # Bottom left
    ], dtype=np.float32)
    
    # Dimension constants
    BASE_OBS_DIM = 56
    HISTORY_STATE_DIM = 16  # pos_z(1) + rot_mat(9) + vel(3) + ang_vel(3)
    
    def __init__(
        self,
        n_gates: int = 4,
        n_obstacles: int = 4,
        stage: int = 1,
        n_history: int = 2,
    ):
        self.n_gates = n_gates
        self.n_obstacles = n_obstacles
        self.stage = stage
        self.n_history = n_history
        
        self.OBS_DIM = self.BASE_OBS_DIM + n_history * self.HISTORY_STATE_DIM
        
        # Internal state
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._history_buffer = np.zeros((n_history, self.HISTORY_STATE_DIM), dtype=np.float32)
        self._initialized = False
        
        # Cache for delayed history update
        # During training: history is updated at the start of step using previous step's obs
        # So we need to cache the obs from compute_control and update history on the next compute_control
        self._pending_history_obs = None
    
    def reset(self, obs: dict):
        """Reset internal state."""
        self._prev_action = np.zeros(4, dtype=np.float32)
        
        # Fill history buffer with initial state
        init_state = self._extract_basic_state(obs)
        for i in range(self.n_history):
            self._history_buffer[i] = init_state
        
        # Reset cache
        self._pending_history_obs = None
        self._initialized = True
    
    def process(self, obs: dict) -> NDArray:
        """Convert observation dictionary to vector (single-environment version).
        
        Timing explanation:
        - First check if there's pending history to update (obs from last compute_control)
        - Then generate observation using current history and prev_action
        - Finally cache current obs for next history update
        """
        if not self._initialized:
            self.reset(obs)
        
        # First update history with cached obs (if available)
        # This replicates the training logic of updating history at step start using cache
        if self._pending_history_obs is not None:
            self._update_history_buffer(self._pending_history_obs)
        
        # 1. Extract raw data
        pos = np.array(obs["pos"])                    # (3,)
        pos_z = pos[2:3]                              # (1,)
        vel = np.array(obs["vel"])                    # (3,)
        ang_vel = np.array(obs["ang_vel"])            # (3,)
        quat = np.array(obs["quat"])                  # (4,) [x, y, z, w]
        target_gate = int(obs["target_gate"])         # scalar
        gates_pos = np.array(obs["gates_pos"])        # (n_gates, 3)
        gates_quat = np.array(obs["gates_quat"])      # (n_gates, 4)
        obstacles_pos = np.array(obs["obstacles_pos"])  # (n_obstacles, 3)
        
        # 2. Compute drone attitude
        rot_matrix = Rotation.from_quat(quat).as_matrix()  # (3, 3)
        rot_matrix_flat = rot_matrix.flatten()             # (9,)
        vel_body = self._world_to_body(vel, rot_matrix)    # (3,)
        
        # 3. Compute gate corners (body coordinates)
        gate1_idx = np.clip(target_gate, 0, self.n_gates - 1)
        gate2_idx = np.clip(target_gate + 1, 0, self.n_gates - 1)
        
        if target_gate == -1:  # Finished
            gate1_idx = self.n_gates - 1
            gate2_idx = self.n_gates - 1
        
        gate1_corners = self._compute_gate_corners_body(
            gates_pos[gate1_idx], gates_quat[gate1_idx], pos, rot_matrix
        )
        gate2_corners = self._compute_gate_corners_body(
            gates_pos[gate2_idx], gates_quat[gate2_idx], pos, rot_matrix
        )
        
        # 4. Compute obstacle positions (body coordinates)
        obstacles_body = self._compute_obstacles_body(obstacles_pos, pos, rot_matrix)
        
        # 5. Concatenate observation vector
        obs_parts = [
            pos_z,              # 1D
            vel_body,           # 3D
            ang_vel,            # 3D
            rot_matrix_flat,    # 9D
            gate1_corners,      # 12D
            gate2_corners,      # 12D
            self._prev_action,  # 4D
            obstacles_body,     # 12D
        ]
        
        # 6. Add historical states
        if self.n_history > 0:
            history_flat = self._history_buffer.flatten()
            obs_parts.append(history_flat)
        
        obs_vector = np.concatenate(obs_parts)
        
        # 7. Curriculum learning masking
        obs_vector = self._apply_stage_masking(obs_vector)
        
        # Cache current obs for next process call to update history
        self._pending_history_obs = obs
        
        return obs_vector.astype(np.float32)
    
    def update_prev_action(self, action_raw: NDArray):
        """Update prev_action (called in step_callback)."""
        self._prev_action = action_raw.copy()
    
    def _update_history_buffer(self, obs: dict):
        """Update history buffer."""
        current_state = self._extract_basic_state(obs)
        self._history_buffer = np.concatenate([
            self._history_buffer[1:],
            current_state[np.newaxis, :]
        ], axis=0)
    
    def _extract_basic_state(self, obs: dict) -> NDArray:
        """Extract basic state (pos_z + rot_mat + vel + ang_vel)."""
        pos = np.array(obs["pos"])
        quat = np.array(obs["quat"])
        vel = np.array(obs["vel"])
        ang_vel = np.array(obs["ang_vel"])
        
        pos_z = pos[2:3]  # (1,)
        rot_matrix = Rotation.from_quat(quat).as_matrix()
        rot_flat = rot_matrix.flatten()  # (9,)
        
        return np.concatenate([pos_z, rot_flat, vel, ang_vel])  # (16,)
    
    def _world_to_body(self, vec_world: NDArray, rot_matrix: NDArray) -> NDArray:
        """World coordinates -> Body coordinates."""
        return rot_matrix.T @ vec_world
    
    def _compute_gate_corners_body(
        self,
        gate_pos: NDArray,
        gate_quat: NDArray,
        drone_pos: NDArray,
        drone_rot: NDArray,
    ) -> NDArray:
        """Compute positions of gate's 4 corners in body coordinate system."""
        gate_rot = Rotation.from_quat(gate_quat).as_matrix()
        
        # Corner positions in world coordinates
        corners_world = (gate_rot @ self.GATE_CORNERS_LOCAL.T).T + gate_pos
        
        # Positions relative to drone
        corners_rel = corners_world - drone_pos
        
        # Convert to body coordinates
        corners_body = (drone_rot.T @ corners_rel.T).T
        
        return corners_body.flatten()  # (12,)
    
    def _compute_obstacles_body(
        self,
        obstacles_pos: NDArray,
        drone_pos: NDArray,
        drone_rot: NDArray,
    ) -> NDArray:
        """Compute relative positions of obstacles in body coordinate system."""
        if obstacles_pos.size == 0 or obstacles_pos.shape[0] == 0:
            return np.full(12, 10.0, dtype=np.float32)
        
        # Use the closest point on obstacle axis to the drone
        obs_x = obstacles_pos[:, 0]
        obs_y = obstacles_pos[:, 1]
        obs_z_top = obstacles_pos[:, 2]
        
        drone_z = drone_pos[2]
        effective_obs_z = np.minimum(obs_z_top, drone_z)
        
        effective_obs_pos = np.stack([obs_x, obs_y, effective_obs_z], axis=-1)
        
        # Compute relative position
        rel_world = effective_obs_pos - drone_pos
        
        # Rotate to body coordinates
        rel_body = (drone_rot.T @ rel_world.T).T
        
        # Sort by distance and take closest 4
        dists = np.linalg.norm(rel_body, axis=1)
        sorted_idx = np.argsort(dists)
        
        n_keep = min(len(sorted_idx), 4)
        result = np.full(12, 10.0, dtype=np.float32)
        result[:n_keep * 3] = rel_body[sorted_idx[:n_keep]].flatten()
        
        return result
    
    def _apply_stage_masking(self, obs_vector: NDArray) -> NDArray:
        """Apply observation masking based on stage."""
        obs_vector = obs_vector.copy()
        
        if self.stage == 0:
            # Stage 0: Mask gate2 and obstacles
            obs_vector[28:40] = obs_vector[16:28]
            obs_vector[44:56] = 10.0
        elif self.stage == 1:
            # Stage 1: Only mask obstacles
            obs_vector[44:56] = 10.0
        # Stage 2: All enabled
        
        return obs_vector


# ============================================================================
# Controller Implementation
# ============================================================================

class PPORacingController(Controller):
    """PPO Racing Controller - Uses trained PPO network for control."""
    
    def __init__(self, obs: dict[str, NDArray], info: dict, config: dict):
        super().__init__(obs, info, config)
        
        # ---------- Configuration Parameters ----------
        self.n_gates = 4
        self.n_obstacles = 4
        self.n_history = 2
        self.hidden_dim = 256
        self.stage = 2
        
        # Calculate observation dimension
        self.obs_dim = 56 + self.n_history * 16  # 88
        self.action_dim = 4
        
        # ---------- Action Scaling Parameters ----------
        params = ForceTorqueParams.load(config.sim.drone_model)
        self.thrust_min = 0.2
        self.thrust_max = 0.8
        
        # action_sim_low = [-pi/2, -pi/2, -pi/2, thrust_min]
        # action_sim_high = [pi/2, pi/2, pi/2, thrust_max]
        self.action_low = np.array(
            [-np.pi/2, -np.pi/2, -np.pi/2, self.thrust_min], dtype=np.float32
        )
        self.action_high = np.array(
            [np.pi/2, np.pi/2, np.pi/2, self.thrust_max], dtype=np.float32
        )
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_mean = (self.action_high + self.action_low) / 2.0
        
        print(f"[PPORacingController] obs_dim: {self.obs_dim}, action_dim: {self.action_dim}")
        print(f"[PPORacingController] action_low: {self.action_low}")
        print(f"[PPORacingController] action_high: {self.action_high}")
        
        # ---------- Observation Processor ----------
        self.obs_processor = ObservationProcessor(
            n_gates=self.n_gates,
            n_obstacles=self.n_obstacles,
            stage=self.stage,
            n_history=self.n_history,
        )
        
        # ---------- Load Network ----------
        self.device = torch.device("cpu")
        self.agent = Agent(self.obs_dim, self.action_dim, self.hidden_dim).to(self.device)
        
        root_dir = Path(__file__).resolve().parent.parent
        model_path = root_dir / "rl_training" / "checkpoints" / "ppo_racing.ckpt"
        
        print(f"[PPORacingController] Loading model from: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.agent.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.agent.load_state_dict(checkpoint)
            print("[PPORacingController] Model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        
        self.agent.eval()
        
        # ---------- Initialize Observation Processor ----------
        self.obs_processor.reset(obs)
        
        self._finished = False
    
    def compute_control(
        self, obs: dict[str, NDArray], info: dict | None = None
    ) -> NDArray:
        """Compute control command."""
        # Check if finished
        if obs["target_gate"] == -1:
            self._finished = True
        
        # 1. Process observation (internally handles history update timing)
        obs_vector = self.obs_processor.process(obs)
        obs_tensor = torch.tensor(obs_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 2. Network inference
        with torch.no_grad():
            action_raw, _, _, _ = self.agent.get_action_and_value(obs_tensor, deterministic=True)
            action_raw = action_raw.squeeze(0).cpu().numpy()
        
        # 3. Action scaling: [-1, 1] -> [low, high]
        action = self._scale_action(action_raw)
        
        # 4. Cache raw action for step_callback to update prev_action
        self._cached_action_raw = action_raw
        
        return action.astype(np.float32)
    
    def _scale_action(self, action_raw: NDArray) -> NDArray:
        """Scale network output [-1, 1] to actual action range."""
        action_clipped = np.clip(action_raw, -1.0, 1.0)
        return action_clipped * self.action_scale + self.action_mean
    
    def step_callback(
        self,
        action: NDArray,
        obs: dict[str, NDArray],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Per-step callback - update prev_action.
        
        Note: History update happens in next compute_control (replicates training timing).
        """
        # Update prev_action (using cached raw action)
        if hasattr(self, '_cached_action_raw'):
            # The observation wrapper used during training stored previous actions in
            # the *scaled* action range (after NormalizeActions). We must replicate
            # that during inference — pass the scaled action, not the raw network output.
            self.obs_processor.update_prev_action(self._scale_action(self._cached_action_raw))
        
        return self._finished
    
    def episode_callback(self):
        """Episode end callback."""
        pass
    
    def episode_reset(self):
        """Reset episode state."""
        self._finished = False
        self.obs_processor._initialized = False
        self.obs_processor._pending_history_obs = None
        if hasattr(self, '_cached_action_raw'):
            del self._cached_action_raw