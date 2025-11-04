"""Sequential Dynamic Waypoint Controller fully using obs for DroneRacing-v0 with gate orientation based extension."""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
from lsy_drone_racing.control.controller import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ObsBasedWaypointController(Controller):
    """Follow gates sequentially using only obs, with gate orientation based extension."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        
        
        # control parameters
        self._tolerance = 0.14     # waypoint reach tolerance
        self._gate_radius = 0.7    # gate fuzzy area radius
        self._step_length = 0.2    # waypoint interpolation step length
        self._extension_steps = 2   # gate exit extension steps
        self._use_gate_orientation = True  # use gate orientation for extension

        # state variables
        self._current_wp_idx = 0
        self._finished = False
        self._extension_doing = False  # current gate extension has been generated
        self.gate_quat = []
        self.old_gate_idx = -1

    def _quat_to_forward(self, quat: NDArray[np.floating]) -> NDArray[np.floating]:
        """Convert quaternion to forward vector (horizontal only)."""
        # quaternion format: [x, y, z, w]
        x, y, z, w = quat
        
        # Calculate forward vector (assuming forward is positive X-axis)
        forward_x = 1 - 2*(y**2 + z**2)
        forward_y = 2*(x*y + z*w)
        forward_z = 2*(x*z - y*w)
        
        forward = np.array([forward_x, forward_y, forward_z])
        
        # Keep only horizontal direction and normalize
        forward[2] = 0  # remove vertical component
        norm = np.linalg.norm(forward)
        if norm > 0:
            forward = forward / norm
        else:
            forward = np.array([1.0, 0.0, 0.0])  # default forward
        
        return forward

    def _linear_interpolation(self, start: NDArray[np.floating], end: NDArray[np.floating]) -> NDArray[np.floating]:
        """Generate a list of linear interpolation waypoints from start to end."""
        vec = end - start
        dist = np.linalg.norm(vec)
        if dist == 0:
            return np.array([start])
        direction = vec / dist
        num_steps = max(int(dist / self._step_length), 1)
        waypoints = np.array([start + direction * (i * self._step_length) for i in range(1, num_steps + 1)])
        waypoints = np.vstack([waypoints, end])
        return waypoints

    def compute_control(self, obs: dict, info: dict = None) -> NDArray[np.float32]:
        if not self._extension_doing:
            gate_idx = obs["target_gate"] 
            gate_pos = obs["gates_pos"][gate_idx]
            if gate_idx == 2:
                gate_pos = gate_pos + np.array([0.0, 0.0, -0.2])
            self.gate_quat = obs["gates_quat"][gate_idx]
            # print(f"Current Target Gate Index: {gate_idx}")
            
            if self.old_gate_idx != gate_idx and self.old_gate_idx != -1:
                # print(f"Generating extension for gate {self.old_gate_idx}")
                self._extension_doing = True
                target_pos = obs["gates_pos"][gate_idx - 1]
            else:
                current_pos = obs["pos"].copy()
                target_pos = self._linear_interpolation(current_pos, gate_pos)[0]
                self.old_gate_idx = gate_idx
        else:
            old_gate_pos = obs["gates_pos"][self.old_gate_idx]
            old_gate_quat = obs["gates_quat"][self.old_gate_idx]
            extension_direction = self._quat_to_forward(old_gate_quat)
            extension_target_pos = old_gate_pos + extension_direction * self._step_length * self._extension_steps
            if self.old_gate_idx == 0:
                    extension_target_pos = extension_target_pos + extension_direction * self._step_length * 5
            if self.old_gate_idx == 1:
                    extension_target_pos = extension_target_pos + np.array([0.0, 0.0, 0])
            if self.old_gate_idx == 2:
                    extension_target_pos = extension_target_pos + extension_direction * self._step_length * 3
            target_pos = self._linear_interpolation(obs["pos"].copy(), extension_target_pos)[0]
            # print(f"Extending towards {extension_target_pos}, current pos {obs['pos']}")
                
            if np.linalg.norm(obs["pos"] - extension_target_pos) < self._tolerance:
                self._extension_doing = False
                self.old_gate_idx = obs["target_gate"] 

        # Generate control action (position control)
        action = np.zeros(13, dtype=np.float32)
        action[0:3] = target_pos
        
        return action

    def step_callback(self, action: NDArray[np.float32], obs: dict, reward: float, 
                     terminated: bool, truncated: bool, info: dict) -> bool:
        """Callback after each environment step."""
        return self._finished

    def episode_callback(self):
        """Reset controller state for new episode."""
        self._current_wp_idx = 0
        self._finished = False
        self._extension_done = False
        # Waypoints will be regenerated in the first compute_control call