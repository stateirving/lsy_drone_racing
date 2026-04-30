"""Reward Shaping Wrapper - Racing Specialized Version

Optimized for fixed tracks:
1. Use differential reward (Potential-based) instead of absolute distance reward to prevent hovering exploitation.
2. Add time penalty to encourage fast completion.
3. Simplify attitude penalty to allow aggressive maneuvers.
4. Gradually increase gate passing reward
Reward = (d_t-1 - d_t) * C_prog + R_gate + R_finish - P_time - P_crash - P_smooth
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gymnasium.vector import VectorEnv, VectorWrapper
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from numpy.typing import NDArray


class RacingRewardWrapper(VectorWrapper):
    """Racing-specialized reward wrapper."""
    
    # Normal vector in gate local coordinate system (gate faces X-axis)
    GATE_NORMAL_LOCAL = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    def __init__(
        self,
        env: VectorEnv,
        n_gates: int = 4,
        stage: int = 0, # For interface compatibility, can actually be ignored
        # ========== Key Coefficient Adjustments ==========
        coef_progress: float = 20.0,   # Differential reward coefficient, usually needs to be large (because difference is small)
        coef_gate: float = 10.0,       # Single gate passing reward
        coef_finish: float = 50.0,     # Track completion bonus
        coef_time: float = 0.05,       # Time penalty (deduct points per step)
        coef_collision: float = 10.0,  # Collision penalty (crash means restart, penalty should be large)
        coef_smooth: float = 0.1,      # Action smoothness
        coef_spin: float = 0.1,        # Angular velocity penalty (prevent oscillation)
        coef_align: float = 0.5,       # Alignment reward (auxiliary guidance)
        coef_angle: float = 0.02,
    ):
        super().__init__(env)
        
        self.n_gates = n_gates
        
        self.coefs = {
            "progress": coef_progress,
            "gate": coef_gate,
            "finish": coef_finish,
            "time": coef_time,
            "collision": coef_collision,
            "smooth": coef_smooth,
            "spin": coef_spin,
            "align": coef_align,
            "angle": coef_angle,
        }
        
        # Internal state
        self._last_dist_to_gate = np.zeros(self.num_envs, dtype=np.float32)
        self._last_action = np.zeros((self.num_envs, 4), dtype=np.float32)
        self._last_target_gate = np.zeros(self.num_envs, dtype=np.int32)

        # New: Cumulative reward tracking
        self._ep_rewards = {
            "progress": np.zeros(self.num_envs, dtype=np.float32),
            "gate": np.zeros(self.num_envs, dtype=np.float32),
            "finish": np.zeros(self.num_envs, dtype=np.float32),
            "align": np.zeros(self.num_envs, dtype=np.float32),
            "time": np.zeros(self.num_envs, dtype=np.float32),
            "ground": np.zeros(self.num_envs, dtype=np.float32),
            "collision": np.zeros(self.num_envs, dtype=np.float32),
            "smooth": np.zeros(self.num_envs, dtype=np.float32),
            "spin": np.zeros(self.num_envs, dtype=np.float32),
            "angle": np.zeros(self.num_envs, dtype=np.float32),
            "total": np.zeros(self.num_envs, dtype=np.float32),
        }
        # Episode statistics completed during rollout
        self._finished_episodes = {k: [] for k in self._ep_rewards.keys()}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Initialize distance
        self._last_dist_to_gate = self._compute_dist_to_gate(obs)
        
        # Reset state
        self._last_action = np.zeros((self.num_envs, 4), dtype=np.float32)
        self._last_target_gate = np.array(obs["target_gate"], dtype=np.int32)
        
        return obs, info
    
    def step(self, action):
        # Convert action format
        action_array = np.array(action, dtype=np.float32).reshape(self.num_envs, -1)
        
        # Environment step
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # Compute new reward
        reward = self._compute_reward(obs, action_array, terminated, truncated)
        
        # Update state
        self._update_state(obs, action_array)
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(
        self, 
        obs: dict, 
        action: NDArray,
        terminated: NDArray,
        truncated: NDArray
    ) -> NDArray:
        target_gate = np.array(obs["target_gate"])
        pos = np.array(obs["pos"])
        
        # 1. Compute distance
        curr_dist = self._compute_dist_to_gate(obs)
        
        # ========== Core: Differential Progress Reward (Potential-based) ==========
        # (previous distance - current distance)
        # positive = approaching target; negative = moving away from target
        dist_diff = self._last_dist_to_gate - curr_dist

        # Handle gate transition:
        # If gate passed, target changed, distance will suddenly increase. We don't compute differential reward in this step to avoid huge negative penalty.
        # Also don't compute when finished (target=-1)
        gate_changed = (target_gate != self._last_target_gate)
        finished = (target_gate == -1)

        r_progress = np.where(gate_changed | finished, 0.0, dist_diff)
        max_velocity_threshold = 1.5  # meters per second
        max_distance_per_step = max_velocity_threshold * 0.02  # Assuming 50 Hz update rate
        # Clip progress based on max velocity threshold (2.5 m/s)
        # At 50 Hz (0.02s per step): max_distance = 2.5 m/s * 0.02s = 0.05 m
        max_progress_per_step = max_distance_per_step  # meters
        r_progress = np.clip(r_progress, -max_progress_per_step, max_progress_per_step)

        r_progress = r_progress * self.coefs["progress"]
        
        # ========== 2. Gate Passing and Completion Reward (Modified Version) ==========
        # Detect if just passed gate (target_gate increase means gate passed)
        passed_gate = (target_gate > self._last_target_gate) & (self._last_target_gate >= 0)

        # Compute incremental reward: (gate index + 1) * 20
        # Passing gate 0 (index 0) rewards 20, gate 1 rewards 40... and so on
        incremental_reward = (self._last_target_gate.astype(np.float32) + 1) * self.coefs["gate"]
        r_gate = np.where(passed_gate, incremental_reward, 0.0)

        # Completion reward (if it's the last gate)
        just_finished = (target_gate == -1) & (self._last_target_gate != -1)
        # If gate 4 (index 3) is the last one, completion reward can be set to 100 or higher
        r_finish = np.where(just_finished, self.coefs["finish"], 0.0)
        
        # ========== 3. Time Penalty (Step Penalty) ==========
        # As long as not finished, deduct points every step to force fast flight
        p_time = np.where(finished, 0.0, self.coefs["time"])
        
        # ========== 4. Takeoff Guidance (Altitude) ==========
        # If rubbing on ground (z < 0.1), give extra negative score to force it to fly
        z = pos[:, 2]
        on_ground = (z < 0.1) & ~finished
        p_ground = np.where(on_ground, 0.1, 0.0)
        
        # ========== 5. Alignment Reward (Auxiliary) ==========
        # Only compute when velocity is large enough, guide it to aim at gate
        r_align = self._compute_align_reward(obs) * self.coefs["align"]
        
        # ========== 6. Penalty Terms ==========
        # Collision (terminated but not due to completion or timeout)
        # Note: VecDroneRaceEnv also sets terminated=True when step limit expires, need to exclude
        is_crash = terminated & ~truncated & ~finished
        p_collision = np.where(is_crash, self.coefs["collision"], 0.0)
        
        # Action smoothness (anti-jitter)
        action_diff = action - self._last_action
        p_smooth = self.coefs["smooth"] * np.sum(action_diff ** 2, axis=1)
        
        # Angular velocity (anti-oscillation)
        ang_vel = np.array(obs["ang_vel"])
        p_spin = self.coefs["spin"] * np.sum(ang_vel ** 2, axis=1)
        
        # Attitude penalty (light): Only penalize excessive Roll/Pitch to prevent flipping, but allow banking in turns
        quat = np.array(obs["quat"])
        rpy = Rotation.from_quat(quat).as_euler("xyz")
        p_angle = self.coefs["angle"] * np.linalg.norm(rpy[:, :2], axis=-1) # Coefficient is very small
        
        # ========== Total ==========
        reward = (
            r_progress 
            + r_gate 
            + r_finish 
            + r_align
            - p_time 
            - p_ground
            - p_collision 
            - p_smooth 
            - p_spin
            - p_angle
        )
        self._ep_rewards["progress"] += r_progress
        self._ep_rewards["gate"] += r_gate
        self._ep_rewards["finish"] += r_finish
        self._ep_rewards["align"] += r_align
        self._ep_rewards["time"] -= p_time
        self._ep_rewards["ground"] -= p_ground
        self._ep_rewards["collision"] -= p_collision
        self._ep_rewards["smooth"] -= p_smooth
        self._ep_rewards["spin"] -= p_spin
        self._ep_rewards["angle"] -= p_angle
        self._ep_rewards["total"] += reward
        
        # When episode ends, record and reset
        done_mask = terminated | truncated
        if np.any(done_mask):
            for key in self._ep_rewards:
                # Record cumulative values of finished episodes
                self._finished_episodes[key].extend(self._ep_rewards[key][done_mask].tolist())
                # Reset these environments
                self._ep_rewards[key][done_mask] = 0.0
        return reward.astype(np.float32)

    def _compute_dist_to_gate(self, obs: dict) -> NDArray:
        pos = np.array(obs["pos"])
        gates_pos = np.array(obs["gates_pos"])
        target_gate = np.array(obs["target_gate"])
        
        safe_idx = np.clip(target_gate, 0, self.n_gates - 1)
        batch_idx = np.arange(self.num_envs)
        
        # Get current target gate position
        current_gate_pos = gates_pos[batch_idx, safe_idx]
        
        # Compute distance
        dist = np.linalg.norm(pos - current_gate_pos, axis=1)
        return dist

    def _compute_align_reward(self, obs: dict) -> NDArray:
        """
        Simplified alignment reward: Only encourage velocity alignment with gate normal vector
        
        Let the policy learn how to pass through gates itself, rather than hand-crafting paths.
        """
        # 1. Extract data
        vel = np.array(obs["vel"])                      # (N, 3)
        gates_quat = np.array(obs["gates_quat"])        # (N, n_gates, 4)
        target_gate_idx = np.array(obs["target_gate"])  # (N,)
        
        # Handle completion state
        valid_mask = (target_gate_idx != -1)
        safe_idx = np.clip(target_gate_idx, 0, self.n_gates - 1)
        
        # 2. Get target gate's normal vector
        batch_indices = np.arange(self.num_envs)
        curr_gate_quat = gates_quat[batch_indices, safe_idx]  # (N, 4)
        
        gate_rot = Rotation.from_quat(curr_gate_quat)
        gate_normal = gate_rot.apply(np.array([1.0, 0.0, 0.0]))  # (N, 3)
        
        # 3. Compute alignment between velocity and gate normal vector
        # Reward = v · n (projection of velocity in correct direction)
        align_reward = np.sum(vel * gate_normal, axis=1)
        
        # 4. Optional: scaling factor
        align_reward *= 0.5
        
        # 5. No reward after completion
        align_reward = np.where(valid_mask, align_reward, 0.0)
        
        return align_reward

    def _update_state(self, obs: dict, action: NDArray):
        self._last_dist_to_gate = self._compute_dist_to_gate(obs)
        self._last_action = action.copy()
        self._last_target_gate = np.array(obs["target_gate"], dtype=np.int32)
        
    def set_stage(self, stage: int):
        pass # Fixed scenario doesn't need stage adjustment