"""
Observation Space Design (Updated Version):
    - Position Z (world coordinate): 1D  <-- Modified here
    - Linear Velocity (body coordinate): 3D  
    - Angular Velocity: 3D
    - Rotation Matrix (flattened): 9D
    - Next Gate 4 corners (body coordinate): 12D
    - Next Next Gate 4 corners (body coordinate): 12D
    - Previous Action: 4D
    - 4 obstacle positions (body coordinate): 12D
    - Historical states (n_history frames): n_history * 16D (Pos_Z + Rot + Vel + AngVel)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorEnv, VectorObservationWrapper
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from numpy.typing import NDArray


class RacingObservationWrapper(VectorObservationWrapper):
    """Wrapper that converts raw observations to vectors.
    
    Core functions:
    1. World coordinate → Body coordinate transformation (gates, obstacles, velocity)
    2. Gate center point → 4 corner calculation
    3. Observation masking for curriculum learning
    4. Maintain prev_action state
    5. Historical state stacking
    """
    
    # 4 corner offsets in gate local coordinate system (gate faces X-axis, YZ plane is gate frame)
    GATE_CORNERS_LOCAL = np.array([
        [0.0, -0.2,  0.2],  # Top left
        [0.0,  0.2,  0.2],  # Top right
        [0.0,  0.2, -0.2],  # Bottom right
        [0.0, -0.2, -0.2],  # Bottom left
    ], dtype=np.float32)
    
    # Observation dimension constants (base part)
    BASE_OBS_DIM = 56
    SELF_DIM = 16        # pos(1) + vel(3) + ang_vel(3) + rot_mat(9)
    GATES_DIM = 24       # 2 gates × 4 corners × 3 coordinates
    ACTION_DIM = 4       # prev_action
    OBSTACLES_DIM = 12   # 4 obstacles × 3 coordinates
    HISTORY_STATE_DIM = 16  # pos(1) + quat(4) + vel(3) + ang_vel(3)
    
    def __init__(
        self, 
        env: VectorEnv,
        n_gates: int = 4,
        n_obstacles: int = 4,
        stage: int = 0,
        n_history: int = 2,  # New: historical frame count
    ):
        """Initialize Wrapper.
        
        Args:
            env: Underlying vectorized environment (VecDroneRaceEnv)
            n_gates: Number of track gates
            n_obstacles: Number of obstacles
            stage: Curriculum stage (0=mask obstacles and next next gate, 1=mask obstacles, 2=all enabled)
            n_history: Historical state frame count (0 means no history)
        """
        super().__init__(env)
        
        self.n_gates = n_gates
        self.n_obstacles = n_obstacles
        self.stage = stage
        self.n_history = n_history
        
        # Compute total observation dimension
        self.OBS_DIM = self.BASE_OBS_DIM + self.n_history * self.HISTORY_STATE_DIM
        
        # Internal state: previous action
        self._prev_action = np.zeros((self.num_envs, self.ACTION_DIM), dtype=np.float32)
        
        # Internal state: historical state buffer (num_envs, n_history, 13)
        if self.n_history > 0:
            self._history_buffer = np.zeros(
                (self.num_envs, self.n_history, self.HISTORY_STATE_DIM), 
                dtype=np.float32
            )
        
        # Define new observation space
        self.single_observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.OBS_DIM,), 
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_envs, self.OBS_DIM),
            dtype=np.float32
        )
    
    def reset(self, **kwargs):
        """Reset environment, clear prev_action and history buffer."""
        obs, info = self.env.reset(**kwargs)
        
        # Reset internal state
        self._prev_action = np.zeros((self.num_envs, self.ACTION_DIM), dtype=np.float32)
        
        # Reset history buffer, fill with initial state
        if self.n_history > 0:
            init_state = self._extract_basic_state(obs)  # (num_envs, 13)
            for i in range(self.n_history):
                self._history_buffer[:, i, :] = init_state
        
        # Transform observation
        transformed_obs = self.observations(obs)
        
        return transformed_obs, info
    
    def step(self, action):
        current_obs_dict = self._get_current_obs_dict()
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        done_mask = terminated | truncated
        if np.any(done_mask):
            for idx in np.where(done_mask)[0]:
                quat = obs["quat"][idx]
                rot = Rotation.from_quat(quat).as_matrix()
        # Conditional history update
        if self.n_history > 0 and current_obs_dict is not None:
            if np.any(done_mask):
                # For autoreset environments: directly fill with new state, don't roll old state
                init_state = self._extract_basic_state(obs)
                # Manually zero velocity part (last 6 dimensions of 16)
                init_state[done_mask, 10:13] = 0.0  # hist_vel
                init_state[done_mask, 13:16] = 0.0  # hist_ang_vel
                for i in range(self.n_history):
                    self._history_buffer[done_mask, i, :] = init_state[done_mask]
                
                # For normal environments: normal rolling
                active_mask = ~done_mask
                if np.any(active_mask):
                    # Only update history for normal environments
                    old_state = self._extract_basic_state(current_obs_dict)
                    self._history_buffer[active_mask] = np.concatenate([
                        self._history_buffer[active_mask, 1:, :],
                        old_state[active_mask, np.newaxis, :]
                    ], axis=1)
            else:
                # All environments normal: use original logic
                self._update_history_buffer(current_obs_dict)
        
        # Reset prev_action
        if np.any(done_mask):
            self._prev_action[done_mask] = 0.0
        
        transformed_obs = self.observations(obs)
        if np.any(done_mask):
            transformed_obs[done_mask, 1:4] = 0.0   # vel_body
            transformed_obs[done_mask, 4:7] = 0.0   # ang_vel

        self._prev_action = np.array(action, dtype=np.float32).reshape(self.num_envs, -1)
        self._cached_obs = obs
        
        return transformed_obs, reward, terminated, truncated, info    

    def _get_current_obs_dict(self):
        """Get cached observation dictionary."""
        return getattr(self, '_cached_obs', None)
    
    def _extract_basic_state(self, obs: dict) -> NDArray:
        """Extract basic state from observation dictionary (pos, quat, vel, ang_vel).
        
        Args:
            obs: Observation dictionary
            
        Returns:
            (num_envs, 16) basic state vector
        """
        pos = np.array(obs["pos"])          # (num_envs, 3)
        quat = np.array(obs["quat"])        # (num_envs, 4)
        vel = np.array(obs["vel"])          # (num_envs, 3)
        ang_vel = np.array(obs["ang_vel"])  # (num_envs, 3)
        
        
        pos_z = pos[:, 2:3]  # Only take z coordinate (num_envs, 1)
        rot_matrices = Rotation.from_quat(quat).as_matrix() # (num_envs, 3, 3)
        rot_flat = rot_matrices.reshape(self.num_envs, 9)   # (num_envs, 9)
        # 1 + 9 + 3 + 3 = 16
        return np.concatenate([pos_z, rot_flat, vel, ang_vel], axis=1)  # (num_envs, 16)
    
    def _update_history_buffer(self, obs: dict):
        """Update historical state buffer.
        
        Add current state to buffer, remove oldest state.
        
        Args:
            obs: Current observation dictionary
        """
        current_state = self._extract_basic_state(obs)  # (num_envs, 13)
        
        # Roll buffer: discard oldest, add newest
        # [:, 1:, :] take frame 1 to last, then concatenate new frame
        self._history_buffer = np.concatenate([
            self._history_buffer[:, 1:, :],
            current_state[:, np.newaxis, :]
        ], axis=1)
    
    def observations(self, obs: dict) -> NDArray:
        """Convert raw observation dictionary to vector.
        
        Args:
            obs: Observation dictionary returned by VecDroneRaceEnv
            
        Returns:
            (num_envs, OBS_DIM) observation vector
        """
        num_envs = self.num_envs
        
        # ========== 1. Extract Raw Data ==========
        pos = np.array(obs["pos"])              # (num_envs, 3) world coordinate
        pos_z = pos[:, 2:3]            # (num_envs, 1) altitude
        vel = np.array(obs["vel"])              # (num_envs, 3) world coordinate
        ang_vel = np.array(obs["ang_vel"])      # (num_envs, 3)
        quat = np.array(obs["quat"])            # (num_envs, 4) [x, y, z, w] scipy order
        target_gate = np.array(obs["target_gate"])  # (num_envs,)
        gates_pos = np.array(obs["gates_pos"])      # (num_envs, n_gates, 3)
        gates_quat = np.array(obs["gates_quat"])    # (num_envs, n_gates, 4)
        obstacles_pos = np.array(obs["obstacles_pos"])  # (num_envs, n_obstacles, 3)
        
        # ========== 2. Compute Drone Attitude Related Quantities ==========
        rot_matrices = self._quat_to_rotation_matrix(quat)  # (num_envs, 3, 3)
        rot_matrices_flat = rot_matrices.reshape(num_envs, 9)
        vel_body = self._world_to_body_batch(vel, rot_matrices)  # (num_envs, 3)
        
        # ========== 3. Compute Gate Corners (Body Coordinate) ==========
        gate1_idx = np.clip(target_gate, 0, self.n_gates - 1)
        gate2_idx = np.clip(target_gate + 1, 0, self.n_gates - 1)
        
        finished_mask = (target_gate == -1)
        gate1_idx = np.where(finished_mask, self.n_gates - 1, gate1_idx)
        gate2_idx = np.where(finished_mask, self.n_gates - 1, gate2_idx)
        
        batch_idx = np.arange(num_envs)
        gate1_pos = gates_pos[batch_idx, gate1_idx]
        gate1_quat = gates_quat[batch_idx, gate1_idx]
        gate2_pos = gates_pos[batch_idx, gate2_idx]
        gate2_quat = gates_quat[batch_idx, gate2_idx]
        
        gate1_corners_body = self._compute_gate_corners_body(
            gate1_pos, gate1_quat, pos, rot_matrices
        )
        gate2_corners_body = self._compute_gate_corners_body(
            gate2_pos, gate2_quat, pos, rot_matrices
        )
        
        # ========== 4. Compute Obstacle Positions (Body Coordinate) ==========
        obstacles_body = self._compute_obstacles_body(
            obstacles_pos, pos, rot_matrices
        )
        
        # ========== 5. Concatenate Base Observation Vector ==========
        obs_parts = [
            pos_z,                  # 1D  - world coordinate z
            vel_body,               # 3D  - body coordinate velocity
            ang_vel,                # 3D  - angular velocity
            rot_matrices_flat,      # 9D  - rotation matrix
            gate1_corners_body,     # 12D - current gate corners
            gate2_corners_body,     # 12D - next gate corners
            self._prev_action,      # 4D  - previous action
            obstacles_body,         # 12D - obstacle positions
        ]
        
        # ========== 6. Add Historical States ==========
        if self.n_history > 0:
            # Flatten history buffer: (num_envs, n_history, 16) -> (num_envs, n_history * 16)
            history_flat = self._history_buffer.reshape(num_envs, -1)
            obs_parts.append(history_flat)
        
        obs_vector = np.concatenate(obs_parts, axis=1)
        
        # ========== 7. Curriculum Learning Masking ==========
        obs_vector = self._apply_stage_masking(obs_vector)
        
        return obs_vector.astype(np.float32)
    
    def set_stage(self, stage: int):
        """Set curriculum learning stage."""
        self.stage = stage
        print(f"[RacingObservationWrapper] Switching to Stage {stage}")
    
    # ========== Helper Functions ==========
    
    def _quat_to_rotation_matrix(self, quat: NDArray) -> NDArray:
        """Quaternion to rotation matrix (batch)."""
        rotations = Rotation.from_quat(quat)
        return rotations.as_matrix()
    
    def _world_to_body_batch(self, vec_world: NDArray, rot_matrices: NDArray) -> NDArray:
        """Batch convert world coordinate vectors to body coordinate."""
        return np.einsum('nij,nj->ni', rot_matrices.transpose(0, 2, 1), vec_world)
    
    def _compute_gate_corners_body(
        self,
        gate_pos: NDArray,
        gate_quat: NDArray,
        drone_pos: NDArray,
        drone_rot: NDArray,
    ) -> NDArray:
        """Compute positions of gate's 4 corners in body coordinate system."""
        num_envs = gate_pos.shape[0]
        gate_rot = Rotation.from_quat(gate_quat).as_matrix()
        corners_world = np.einsum('nij,kj->nki', gate_rot, self.GATE_CORNERS_LOCAL)
        corners_world = corners_world + gate_pos[:, np.newaxis, :]
        corners_rel_world = corners_world - drone_pos[:, np.newaxis, :]
        drone_rot_inv = drone_rot.transpose(0, 2, 1)
        corners_body = np.einsum('nij,nkj->nki', drone_rot_inv, corners_rel_world)
        return corners_body.reshape(num_envs, -1)
    
    def _compute_obstacles_body(
        self,
        obstacles_pos: NDArray,
        drone_pos: NDArray,
        drone_rot: NDArray,
    ) -> NDArray:
        """Compute relative positions of obstacles in body coordinate system.
        
        Improved strategy: 
        Don't use obstacle vertices directly, but use the closest point on obstacle axis to the drone.
        This prevents collisions when drone flies at low altitude, because looking at vertices makes it think 'obstacle is high up'.
        """
        num_envs = drone_pos.shape[0]
        
        if obstacles_pos.size == 0 or obstacles_pos.shape[1] == 0:
            return np.full((num_envs, 12), 10.0, dtype=np.float32)
        
        # ========== 1. Construct 'Effective' Obstacle Coordinates ==========
        # Decompose obstacle coordinates (N, n_obs, 3)
        obs_x = obstacles_pos[:, :, 0]
        obs_y = obstacles_pos[:, :, 1]
        obs_z_top = obstacles_pos[:, :, 2] # This is the pole's top height
        
        # Get drone height (N, 1) - expand dimension for broadcasting
        drone_z = drone_pos[:, 2:3]
        
        # Compute effective height: min(pole top height, drone height)
        # Assume pole stands on ground (base=0).
        # If drone is below pole, treat obstacle as same height (Z difference is 0)
        # If drone is above pole, treat obstacle as below (Z difference is negative)
        effective_obs_z = np.minimum(obs_z_top, drone_z)
        
        # Restack to (N, n_obs, 3)
        # Now effective_obs_pos is the closest point on pole axis to drone
        effective_obs_pos = np.stack([obs_x, obs_y, effective_obs_z], axis=-1)
        
        # ========== 2. Compute Relative Position and Rotate ==========
        # Compute relative vector in world coordinate system
        rel_world = effective_obs_pos - drone_pos[:, np.newaxis, :]
        
        # Rotate to body coordinate system
        # drone_rot is (N, 3, 3) rotation matrix
        drone_rot_inv = drone_rot.transpose(0, 2, 1)
        
        # Einsum: batch matrix multiplication
        # nij: env i, row j (inv matrix)
        # nkj: env i, obs k, row j (vector)
        # -> nki: env i, obs k, row i (result)
        rel_body = np.einsum('nij,nkj->nki', drone_rot_inv, rel_world)
        
        # ========== 3. Sort and Take Closest n ==========
        # Compute distance (for sorting)
        dists = np.linalg.norm(rel_body, axis=2)
        
        # Sort and take top 4
        # argsort is ascending by default, so top ones are closest
        sorted_idx = np.argsort(dists, axis=1)
        
        # Get closest 4 obstacles (handle if less than 4)
        n_obs_available = rel_body.shape[1]
        n_keep = min(n_obs_available, 4) # Assume observation space is fixed at 4
        
        # Create result container (default fill 10.0 means very far)
        result = np.full((num_envs, 12), 10.0, dtype=np.float32)
        
        # This gather operation in numpy requires fancy indexing
        # Create batch indices: [[0,0,0,0], [1,1,1,1], ...]
        batch_indices = np.arange(num_envs)[:, None]
        keep_indices = sorted_idx[:, :n_keep]
        
        nearest_rel_body = rel_body[batch_indices, keep_indices] # (num_envs, n_keep, 3)
        
        # Fill in result
        result[:, :n_keep * 3] = nearest_rel_body.reshape(num_envs, -1)
        
        return result
    
    def _apply_stage_masking(self, obs_vector: NDArray) -> NDArray:
        """Apply masking to observation based on curriculum stage.
        
        Observation vector layout (updated):
            [0:1]   - pos_z (1D)
            [1:4]   - vel_body
            [4:7]   - ang_vel
            [7:16]  - rot_matrix
            [16:28] - gate1_corners
            [28:40] - gate2_corners  
            [40:44] - prev_action
            [44:56] - obstacles      
            [56:88] - history
        """
        obs_vector = obs_vector.copy()
        
        if self.stage == 0:
            # Stage 0: Mask next next gate + mask obstacles
            # Replace gate2 (28:40) with gate1 (16:28), simulate only current gate visible
            obs_vector[:, 28:40] = obs_vector[:, 16:28]
            # Mask obstacles
            obs_vector[:, 44:56] = 10.0
            
        if self.stage == 1:
            # Stage 1: Only mask obstacles
            obs_vector[:, 44:56] = 10.0
        elif self.stage == 2:
            # Stage 2: All enabled
            pass
        
        return obs_vector