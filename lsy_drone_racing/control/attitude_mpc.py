from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control.controller import Controller
from rrt_algorithms.rrt.rrt_star import RRTStar
from rrt_algorithms.search_space.search_space import SearchSpace
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

if TYPE_CHECKING:
    from numpy.typing import NDArray

# --------------------- RRT + Waypoint Path Planning ---------------------
class RRTWaypointController(Controller):
    """RRT-based waypoint generator for drone racing."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self.gates_count = 4
        self.pos_limit = np.array([[-2.5, 2.5], [-1.5, 1.5], [-1e-3, 2.0]])
        self.previous_gates = None
        self.previous_obstacles = None
        self.current_gate_idx = 0
        self._gate_stage = 0

        # RRT params
        self.length_tree_edge = 0.05
        self.smallest_edge = 0.01
        self.max_samples = 5000
        self.prc = 0.3

        self.path = None
        self.path_points_idx = 0
        self._step_tolerance = 0.20
        self._pre_dist = 0.40
        self._post_dist = 0.6
        self.gate_width = 0.4
        self.gate_height = 1.2
        self.edge_radius = 0.05
        self.obstacle_radius = 0.15
        self.obstacle_height = 1.52

    def _quat_to_yaw(self, quat: NDArray[np.floating]) -> float:
        x, y, z, w = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def _quat_to_forward(self, quat: NDArray[np.floating]) -> NDArray[np.floating]:
        yaw = self._quat_to_yaw(quat)
        return np.array([np.cos(yaw), np.sin(yaw), 0.0])

    def _generate_obstacles_except_current(self, obs: dict, current_gate_idx: int):
        rrt_search_space = SearchSpace(self.pos_limit)
        rrt_obstacles = []

        # gates
        for idx, gpos in enumerate(obs["gates_pos"]):
            if idx == current_gate_idx:
                continue
            cx, cy, cz = np.array(gpos)
            yaw = self._quat_to_yaw(obs["gates_quat"][idx])
            half_w = self.gate_width / 2.0
            dz = self.gate_height / 2.0
            corners = [(-half_w, -half_w), (-half_w, half_w), (half_w, -half_w), (half_w, half_w)]
            for lx, ly in corners:
                x = cx + lx * np.cos(yaw) - ly * np.sin(yaw)
                y = cy + lx * np.sin(yaw) + ly * np.cos(yaw)
                z_bottom = cz - dz
                z_top = cz + dz
                ob = np.append([x - self.edge_radius, y - self.edge_radius, z_bottom],
                               [x + self.edge_radius, y + self.edge_radius, z_top])
                rrt_obstacles.append(tuple(ob))
                rrt_search_space.obs.insert(0, tuple(ob), tuple(ob))  # uuid替换为0简化

        # environment obstacles
        for top_pos in obs["obstacles_pos"]:
            x, y, z_top = np.array(top_pos)
            z_bottom = z_top - self.obstacle_height
            ob = np.append([x - self.obstacle_radius, y - self.obstacle_radius, z_bottom],
                           [x + self.obstacle_radius, y + self.obstacle_radius, z_top])
            rrt_obstacles.append(tuple(ob))
            rrt_search_space.obs.insert(0, tuple(ob), tuple(ob))

        return rrt_search_space

    def generate_goal_position(self, gate_pos: NDArray[np.floating], gate_quat: NDArray[np.floating]) -> NDArray[np.floating]:
        gate_forward = self._quat_to_forward(gate_quat)
        if self._gate_stage == 0:
            return gate_pos - gate_forward * self._pre_dist
        elif self._gate_stage == 1:
            return gate_pos
        else:
            return gate_pos + gate_forward * self._post_dist

    def check_replan_condition(self, obs: dict) -> bool:
        if (not np.array_equal(obs["gates_pos"], self.previous_gates)
            or not np.array_equal(obs["obstacles_pos"], self.previous_obstacles)):
            self.previous_gates = obs["gates_pos"]
            self.previous_obstacles = obs["obstacles_pos"]
            return True
        return False

    def plan_path(self, obs: dict, current_pos: NDArray[np.floating]) -> NDArray[np.floating]:
        gate_pos = np.array(obs["gates_pos"][self.current_gate_idx])
        gate_quat = np.array(obs["gates_quat"][self.current_gate_idx])
        goal_pos = self.generate_goal_position(gate_pos, gate_quat)
        rrt_space = self._generate_obstacles_except_current(obs, self.current_gate_idx)
        rrt = RRTStar(rrt_space, self.length_tree_edge, tuple(current_pos.tolist()), tuple(goal_pos.tolist()),
                      self.max_samples, self.smallest_edge, self.prc)
        path = rrt.rrt_search()
        if path is None or len(path) == 0:
            path = np.linspace(current_pos, goal_pos, 5)
        self.path_points_idx = 0
        self.path = np.array(path)
        return self.path

    def get_next_waypoint(self, current_pos: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.path is None or self.path_points_idx >= len(self.path):
            return current_pos
        wp = self.path[self.path_points_idx]
        if np.linalg.norm(current_pos - wp) < self._step_tolerance:
            self.path_points_idx += 1
        return wp


# --------------------- MPC Controller for Path Tracking ---------------------
class RRTMPCController(Controller):
    """Integrates RRT path planning with ACADOS MPC trajectory tracking."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._dt = 1 / config.env.freq
        self._N = 25
        self._tick = 0
        self._finished = False

        # RRT waypoint generator
        self.rrt_controller = RRTWaypointController(obs, info, config)

        # Load drone params and create ACADOS solver
        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._T_HORIZON = self._N * self._dt
        self._acados_ocp_solver, self._ocp = create_ocp_solver(self._T_HORIZON, self._N, self.drone_params)

        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        current_pos = np.array(obs["pos"], dtype=float)

        # Replan RRT path if necessary
        if self.rrt_controller.check_replan_condition(obs) or self.rrt_controller.path is None:
            self.rrt_controller.plan_path(obs, current_pos)

        # Get next RRT waypoint as MPC reference
        next_wp = self.rrt_controller.get_next_waypoint(current_pos)

        # Build MPC reference
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))

        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = next_wp  # repeat next waypoint as reference
        yref[:, 5] = 0  # yaw
        yref[:, 6:9] = np.zeros(3)  # desired velocity
        yref[:, 15] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]

        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])
        yref_e = np.zeros(self._ny_e)
        yref_e[0:3] = next_wp
        yref_e[5] = 0
        yref_e[6:9] = np.zeros(3)
        self._acados_ocp_solver.set(self._N, "y_ref", yref_e)

        self._acados_ocp_solver.solve()
        u0 = self._acados_ocp_solver.get(0, "u")
        return u0

    def step_callback(self, action: NDArray[np.floating], obs: dict[str, NDArray[np.floating]],
                      reward: float, terminated: bool, truncated: bool, info: dict) -> bool:
        return self._finished

    def episode_callback(self):
        self._tick = 0
        self._finished = False
        self.rrt_controller.path = None
        self.rrt_controller.path_points_idx = 0
        self.rrt_controller.current_gate_idx = 0
