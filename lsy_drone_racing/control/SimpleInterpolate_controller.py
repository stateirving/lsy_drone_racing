"""Simple straight-line waypoint follower with interpolation for DroneRacing-v0."""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
from lsy_drone_racing.control.controller import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SimpleInterpolatedWaypointController(Controller):
    """Follow gates using straight-line waypoints with small step interpolation."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        # --- Waypoints from gates ---
        if "gates" not in config.env.track or len(config.env.track.gates) == 0:
            raise RuntimeError("No gates found in config.env.track.gates.")

        gates_pos = np.array([g["pos"] for g in config.env.track.gates])
        start_pos = obs["pos"].copy()

        # Interpolated path construction
        self.waypoints = [start_pos]
        step_length = 0.4  # each step 40 cm
        for gate_pos in gates_pos:
            start = self.waypoints[-1]
            vec = gate_pos - start
            distance = np.linalg.norm(vec)
            if distance == 0:
                continue
            direction = vec / distance
            num_steps = max(int(distance / step_length), 1)
            for i in range(1, num_steps + 4):
                point = start + direction * (i * step_length)
                self.waypoints.append(point)

        self.waypoints = np.array(self.waypoints)
        self._current_idx = 0
        self._finished = False
        self._tolerance = 0.14  # 0.14 m tolerance

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        if self._finished:
            return np.zeros(13, dtype=np.float32)

        target_pos = self.waypoints[self._current_idx]
        pos_error = target_pos - obs["pos"]

        # Switch to the next interpolation point upon reaching the current one
        if np.linalg.norm(pos_error) < self._tolerance:
            self._current_idx += 1
            if self._current_idx >= len(self.waypoints):
                self._finished = True
                return np.zeros(13, dtype=np.float32)
            target_pos = self.waypoints[self._current_idx]

        # Output minimal action, only controlling position
        action = np.zeros(13, dtype=np.float32)
        action[0:3] = target_pos
        return action

    def step_callback(
        self, action: NDArray[np.floating], obs: dict[str, NDArray[np.floating]],
        reward: float, terminated: bool, truncated: bool, info: dict
    ) -> bool:
        return self._finished

    def episode_callback(self):
        self._current_idx = 0
        self._finished = False
