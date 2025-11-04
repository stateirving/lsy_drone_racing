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
        self._step_length = 0.5    # waypoint interpolation step length
        self._extension_steps = 3  # gate exit extension steps
        self._use_gate_orientation = True  # use gate orientation for extension

        # state variables
        self._current_wp_idx = 0
        self._finished = False
        self._extension_done = False  # current gate extension has been generated
        self._waypoints = self._generate_waypoints(obs)

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
        return waypoints

    def _generate_waypoints(self, obs: dict) -> NDArray[np.floating]:
        """Generate initial waypoints based on obs."""
        start_pos = obs["pos"].copy()
        if "target_gate" in obs and "gates_pos" in obs:
            gate_idx = obs["target_gate"]
            gate_pos = obs["gates_pos"][gate_idx]
            return self._linear_interpolation(start_pos, gate_pos)
        else:
            return np.array([start_pos])

    def compute_control(self, obs: dict, info: dict = None) -> NDArray[np.float32]:
        """Compute control action based on current observation."""
        if self._finished:
            return np.zeros(13, dtype=np.float32)

        # Check if all gates are completed
        if "target_gate" not in obs or obs["target_gate"] >= len(obs["gates_pos"]):
            self._finished = True
            return np.zeros(13, dtype=np.float32)

        # Get current gate information
        if "target_gate" in obs and "gates_pos" in obs and "gates_quat" in obs:
            gate_idx = obs["target_gate"]
            print(f"Current Target Gate Index: {gate_idx}")
            gate_pos = obs["gates_pos"][gate_idx]
            gate_quat = obs["gates_quat"][gate_idx]
        else:
            self._finished = True
            return np.zeros(13, dtype=np.float32)

        # Safety check for waypoints array
        if len(self._waypoints) == 0:
            self._finished = True
            return np.zeros(13, dtype=np.float32)

        # If entering gate fuzzy area, update waypoints and reset extension flag
        dist_to_gate = np.linalg.norm(gate_pos - obs["pos"])
        if dist_to_gate < self._gate_radius and not self._extension_done:
            self._waypoints = self._linear_interpolation(obs["pos"], gate_pos)
            self._current_wp_idx = 0
            self._extension_done = False

        # Get current target waypoint
        target_pos = self._waypoints[self._current_wp_idx]
        pos_error = target_pos - obs["pos"]
        
        print(f"Current WP idx: {self._current_wp_idx}/{len(self._waypoints)-1}, "
              f"Target Pos: {target_pos}, Pos Error norm: {np.linalg.norm(pos_error):.3f}")

        # Check if waypoint is reached
        if np.linalg.norm(pos_error) < self._tolerance:
            self._current_wp_idx += 1

            # If all waypoints are reached, generate extension if needed
            if self._current_wp_idx >= len(self._waypoints):
                if not self._extension_done:
                    last_wp = self._waypoints[-1]
                    
                    if self._use_gate_orientation:
                        # Use gate orientation for extension direction
                        gate_forward = -self._quat_to_forward(gate_quat)
                        print(f"Using gate forward direction: {gate_forward}")
                    else:
                        # Fallback: use flight direction (safely)
                        if len(self._waypoints) >= 2:
                            prev_wp = self._waypoints[-2]
                            direction = last_wp - prev_wp
                            norm = np.linalg.norm(direction)
                            if norm > 0:
                                gate_forward = direction / norm
                            else:
                                gate_forward = np.array([1.0, 0.0, 0.0])
                        else:
                            gate_forward = np.array([1.0, 0.0, 0.0])
                    
                    # Generate extension waypoints
                    extension = np.array([
                        [last_wp[0] + gate_forward[0] * self._step_length * i,
                         last_wp[1] + gate_forward[1] * self._step_length * i,
                         last_wp[2] + gate_forward[2] * self._step_length * i]
                        for i in range(1, self._extension_steps + 1)
                    ])
                    print(f"Generating extension waypoints from {last_wp} in direction {gate_forward}")
                    
                    self._waypoints = np.vstack([self._waypoints, extension])
                    self._extension_done = True
                    self._current_wp_idx = len(self._waypoints) - len(extension)
                    print(f"Generated {len(extension)} extension waypoints")
                else:
                    # All waypoints including extensions are completed
                    # Move to next gate in the next control cycle
                    self._current_wp_idx = min(self._current_wp_idx, len(self._waypoints) - 1)

        # Ensure current waypoint index is valid
        self._current_wp_idx = min(self._current_wp_idx, len(self._waypoints) - 1)
        target_pos = self._waypoints[self._current_wp_idx]

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