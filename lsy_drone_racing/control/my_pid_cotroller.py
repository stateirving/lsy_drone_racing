"""PID-based Attitude Controller for DroneRacing-v0.

Tracks the next gate position using a 3-axis PID controller and outputs
collective thrust and attitude commands (roll, pitch, yaw).
"""

from __future__ import annotations
import math
import numpy as np
from typing import TYPE_CHECKING
from scipy.spatial.transform import Rotation as R
from drone_models.core import load_params
from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PIDGateController(Controller):
    """PID controller for flying through gates using thrust & attitude commands."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]
        self.g = 9.81

        # --- PID gains (tuned for stability and responsiveness) ---
        self.kp = np.array([0.4, 0.4, 1.25])
        self.ki = np.array([0.05, 0.05, 0.05])
        self.kd = np.array([0.2, 0.2, 0.4])
        self.i_error = np.zeros(3)
        self.ki_limit = np.array([1.5, 1.5, 0.4])

        # --- Load all gate positions ---
        self.gates_pos = np.array([g["pos"] for g in config.env.track.gates])
        self._current_gate_idx = 0
        self._tolerance = 0.1
        self._finished = False

    # ================================================================
    # Core control law
    # ================================================================
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute attitude + thrust to reach the next gate."""
        if self._finished:
            return np.zeros(4, dtype=np.float32)

        # --- Determine current target gate ---
        target_pos = self.gates_pos[self._current_gate_idx]
        pos_error = target_pos - obs["pos"]
        vel_error = -obs["vel"]  # desired velocity = 0 at the gate center

        # --- Check if we reached the current gate ---
        if np.linalg.norm(pos_error) < self._tolerance:
            self._current_gate_idx += 1
            if self._current_gate_idx >= len(self.gates_pos):
                self._finished = True
                return np.zeros(4, dtype=np.float32)
            target_pos = self.gates_pos[self._current_gate_idx]
            pos_error = target_pos - obs["pos"]

        # --- PID computation ---
        dt = 1.0 / self._freq
        self.i_error += pos_error * dt
        self.i_error = np.clip(self.i_error, -self.ki_limit, self.ki_limit)

        pid_term = (
            self.kp * pos_error + self.ki * self.i_error + self.kd * vel_error
        )

        # --- Add gravity compensation in Z ---
        pid_term[2] += self.drone_mass * self.g

        # --- Compute thrust and orientation from desired force vector ---
        z_axis_des = pid_term / np.linalg.norm(pid_term)
        des_yaw = 0.0  # keep level flight
        x_c = np.array([math.cos(des_yaw), math.sin(des_yaw), 0.0])
        y_axis_des = np.cross(z_axis_des, x_c)
        y_axis_des /= np.linalg.norm(y_axis_des)
        x_axis_des = np.cross(y_axis_des, z_axis_des)

        R_des = np.vstack([x_axis_des, y_axis_des, z_axis_des]).T
        euler_des = R.from_matrix(R_des).as_euler("xyz", degrees=False)

        # --- Compute thrust along body z-axis ---
        z_axis = R.from_quat(obs["quat"]).as_matrix()[:, 2]
        thrust_des = pid_term.dot(z_axis)

        # --- Final action [roll, pitch, yaw, thrust] ---
        action = np.array(
            [euler_des[0], euler_des[1], euler_des[2], thrust_des], dtype=np.float32
        )

        return action

    # ================================================================
    # Callbacks
    # ================================================================
    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        return self._finished

    def episode_callback(self):
        """Reset the internal state each episode."""
        self.i_error[:] = 0
        self._current_gate_idx = 0
        self._finished = False
