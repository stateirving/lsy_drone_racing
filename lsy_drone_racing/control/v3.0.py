from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import sys
sys.path.append("/home/miao/repos/lsy_drone_racing")
from rrt_algorithms.rrt.rrt_star import RRTStar
from rrt_algorithms.search_space.search_space import SearchSpace
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from lsy_drone_racing.control.controller import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


def create_acados_model(parameters: dict) -> AcadosModel:
    X_dot, X, U, _ = symbolic_dynamics_euler(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        J=parameters["J"],
        J_inv=parameters["J_inv"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
    )
    model = AcadosModel()
    model.name = "combined_rrt_mpc"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U
    return model


def create_ocp_solver(Tf: float, N: int, parameters: dict, verbose: bool = False) -> tuple[AcadosOcpSolver, AcadosOcp]:
    ocp = AcadosOcp()
    ocp.model = create_acados_model(parameters)
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx
    ocp.solver_options.N_horizon = N

    # Cost weights
    Q = np.diag([50, 50, 400, 1, 1, 1, 10, 10, 10, 5, 5, 5])
    R_mat = np.diag([1, 1, 1, 50])
    ocp.cost.W = np.block([[Q, np.zeros((nx, nu))], [np.zeros((nu, nx)), R_mat]])
    ocp.cost.W_e = Q
    Vx = np.zeros((ny, nx))
    Vx[0:nx, 0:nx] = np.eye(nx)
    ocp.cost.Vx = Vx
    Vu = np.zeros((ny, nu))
    Vu[nx:nx+nu, :] = np.eye(nu)
    ocp.cost.Vu = Vu
    ocp.cost.Vx_e = Vx[0:nx, :]
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # Constraints
    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.zeros(nx)

    # Solver options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tol = 1e-6
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50
    ocp.solver_options.tf = Tf

    acados_solver = AcadosOcpSolver(ocp, json_file="c_generated_code/combined_rrt_mpc.json",
                                    verbose=verbose, build=True, generate=True)
    return acados_solver, ocp


class CombinedRRTMPCController(Controller):
    """Single controller combining RRT waypoint planning and MPC tracking."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        # ---------------- RRT params ----------------
        self.gates_count = 4
        self.pos_limit = np.array([[-2.5, 2.5], [-1.5, 1.5], [-1e-3, 2.0]])
        self.current_gate_idx = 0
        self._gate_stage = 0
        self.previous_gates = None
        self.previous_obstacles = None
        self.previous_gate_idx = -1
        self.previous_gate_stage = -1
        self.previous_going_to_extra = False
        self._step_tolerance = 0.2
        self._pre_dist = 0.4
        self._post_dist = 0.6
        self.gate_width = 0.4
        self.gate_height = 1.2
        self.edge_radius = 0.05
        self.obstacle_radius = 0.15
        self.obstacle_height = 1.52
        self.length_tree_edge = 0.1
        self.smallest_edge = 0.01
        self.max_samples = 5000
        self.prc = 0.3
        self.path = None
        self.path_points_idx = 0
        self.take_off = True
        self.going_to_extra = False
        self.extra_point = np.array([-1.5, -1.0, 0.7], dtype=float)

        # ---------------- MPC params ----------------
        self._dt = 1 / config.env.freq
        self._N = 25
        self._T_HORIZON = self._N * self._dt
        self._finished = False
        self._tick = 0

        # Load drone parameters and create solver
        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(self._T_HORIZON, self._N, self.drone_params)
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

    # ----------------- RRT helpers -----------------
    def _quat_to_yaw(self, quat: NDArray[np.floating]) -> float:
        x, y, z, w = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def _quat_to_forward(self, quat: NDArray[np.floating]) -> NDArray[np.floating]:
        yaw = self._quat_to_yaw(quat)
        return np.array([np.cos(yaw), np.sin(yaw), 0.0])

    def _generate_obstacles(self, obs: dict, current_gate_idx: int):
        rrt_space = SearchSpace(self.pos_limit)
        for idx, gpos in enumerate(obs["gates_pos"]):
            if idx == current_gate_idx:
                continue
            cx, cy, cz = np.array(gpos)
            yaw = self._quat_to_yaw(obs["gates_quat"][idx])
            half_w = self.gate_width / 2
            dz = self.gate_height / 2
            corners = [(-half_w, -half_w), (-half_w, half_w), (half_w, -half_w), (half_w, half_w)]
            for lx, ly in corners:
                x = cx + lx * np.cos(yaw) - ly * np.sin(yaw)
                y = cy + lx * np.sin(yaw) + ly * np.cos(yaw)
                z_bottom = cz - dz
                z_top = cz + dz
                ob = np.append([x - self.edge_radius, y - self.edge_radius, z_bottom],
                               [x + self.edge_radius, y + self.edge_radius, z_top])
                rrt_space.obs.insert(0, tuple(ob), tuple(ob))
        for top_pos in obs["obstacles_pos"]:
            x, y, z_top = np.array(top_pos)
            z_bottom = z_top - self.obstacle_height
            ob = np.append([x - self.obstacle_radius, y - self.obstacle_radius, z_bottom],
                           [x + self.obstacle_radius, y + self.obstacle_radius, z_top])
            rrt_space.obs.insert(0, tuple(ob), tuple(ob))
        return rrt_space

    def generate_goal_position(self, gate_pos: NDArray[np.floating], gate_quat: NDArray[np.floating]) -> NDArray[np.floating]:
        fwd = self._quat_to_forward(gate_quat)
        if self._gate_stage == 0:
            return gate_pos - fwd * self._pre_dist
        elif self._gate_stage == 1:
            return gate_pos
        else:
            return gate_pos + fwd * self._post_dist

    def check_replan_condition(self, obs: dict) -> bool:
        if (not np.array_equal(obs["gates_pos"], self.previous_gates)
            or not np.array_equal(obs["obstacles_pos"], self.previous_obstacles)
            or self.current_gate_idx != self.previous_gate_idx 
            or self._gate_stage != self.previous_gate_stage 
            or self.path is None
            or self.going_to_extra != self.previous_going_to_extra):
            self.previous_gates = obs["gates_pos"]
            self.previous_obstacles = obs["obstacles_pos"]
            self.previous_gate_idx = self.current_gate_idx
            self.previous_gate_stage = self._gate_stage
            self.previous_going_to_extra = self.going_to_extra
            return True
        return False

    def plan_rrt_path(self, obs: dict, current_pos: NDArray[np.floating]):
        gate_pos = np.array(obs["gates_pos"][self.current_gate_idx])
        gate_quat = np.array(obs["gates_quat"][self.current_gate_idx])
        goal_pos = self.generate_goal_position(gate_pos, gate_quat)
        rrt_space = self._generate_obstacles(obs, self.current_gate_idx)
        rrt = RRTStar(rrt_space, self.length_tree_edge, tuple(current_pos.tolist()), tuple(goal_pos.tolist()),
                      self.max_samples, self.smallest_edge, self.prc)
        path = rrt.rrt_search()
        if path is None or len(path) == 0:
            path = np.linspace(current_pos, goal_pos, 5)
        self.path = np.array(path)
        self.path_points_idx = 0

    def get_next_waypoint(self, current_pos: NDArray[np.floating]) -> NDArray[np.floating]:
        if self.take_off:
            target = np.array([current_pos[0], current_pos[1], 0.7])
            if np.linalg.norm(current_pos - target) < self._step_tolerance:
                self.take_off = False
            return target.astype(np.float32)
        
        if self.going_to_extra:
            if np.linalg.norm(current_pos - self.extra_point) < self._step_tolerance:
                self.going_to_extra = False
            return self.extra_point.astype(np.float32)
        
        if self.path is None or len(self.path) == 0:
            return current_pos.astype(np.float32)
        
        wp = self.path[self.path_points_idx]
        if np.linalg.norm(current_pos - wp) < self._step_tolerance:
            if (self.path_points_idx < len(self.path) - 1):
                self.path_points_idx += 1
                wp = self.path[self.path_points_idx]
            else:
                # Reached the end of the path
                if self._gate_stage < 2:
                    self._gate_stage += 1
                    self.path_points_idx = 0
                else:
                    # Finished all stages for current gate
                    self.current_gate_idx += 1
                    if self.current_gate_idx >= self.gates_count:
                        self._finished = True
                        return current_pos.astype(np.float32)
                    self._gate_stage = 0
                    self.path_points_idx = 0
                    if self.current_gate_idx == 2:
                        self._step_tolerance = 0.15  
                        self._post_dist = 0.1
                    if self.current_gate_idx == 3:
                        self._gate_stage = 1
                        self.going_to_extra = True
                        return self.extra_point.astype(np.float32)
        return wp

    # ----------------- MPC control -----------------
    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        current_pos = np.array(obs["pos"], dtype=float)
        if self.check_replan_condition(obs) and (not self.take_off):
            self.plan_rrt_path(obs, current_pos)

        next_wp = self.get_next_waypoint(current_pos)

        # Build MPC state
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))

        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        yref = np.zeros((self._N, self._ny))
        for j in range(self._N):
            yref[j, 0:3] = next_wp
            yref[j, 5] = 0
            yref[j, 6:9] = 0
            yref[j, 15] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
            self._acados_ocp_solver.set(j, "yref", yref[j])

        yref_e = np.zeros(self._ny_e)
        yref_e[0:3] = next_wp
        yref_e[5] = 0
        yref_e[6:9] = 0
        self._acados_ocp_solver.set(self._N, "y_ref", yref_e)

        self._acados_ocp_solver.solve()
        return self._acados_ocp_solver.get(0, "u")

    def step_callback(self, action: NDArray[np.floating], obs: dict[str, NDArray[np.floating]],
                      reward: float, terminated: bool, truncated: bool, info: dict) -> bool:
        return self._finished

    def episode_callback(self):
        self._finished = False
        self._tick = 0
        self.path = None
        self.path_points_idx = 0
        self.current_gate_idx = 0
        self._gate_stage = 0
        self.take_off = True
        self.going_to_extra = False
