
from __future__ import annotations
from typing import *

import os
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, DM, cos, sin, vertcat, dot, norm_2, floor, if_else, exp, Function, power
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.tools.race_objects import Gate, Obstacle
from lsy_drone_racing.control.fresssack_controller import FresssackController
from lsy_drone_racing.tools.ext_tools import TrajectoryTool
from lsy_drone_racing.utils.utils import draw_line


if TYPE_CHECKING:
    from numpy.typing import NDArray

class FresssackMPCC:
    traj_path : str

    num_gates : int
    gates : List[Gate]
    gates_init : List[Gate]
    gates_pos_init : List[NDArray]
    num_obstacles: int
    obstacles : List[Obstacle]
    obstacles_init : List[Obstacle]
    obstacles_pos_init : List[NDArray]

    solver : AcadosOcpSolver
    model : AcadosModel
    model_syms : FresssackDroneModel

    n_x : int
    n_u : int

    
    qp_solver : str
    hessian_approx : str
    integrator_type : str
    nlp_solver_type : str
    tol : np.floating
    qp_solver_warm_start : int
    qp_solver_iter_max : int
    nlp_solver_max_iter : int

    T_f : np.floating
    N : int

    model_traj_length : np.floating
    model_arc_length : np.floating

    q_l:np.floating
    q_l_peak:List[np.floating]
    q_c:np.floating
    q_c_peak:List[np.floating]
    q_c_sigma1:List[np.floating]
    q_c_sigma2:List[np.floating]
    gate_interp_peak : List[np.floating]
    gate_interp_sigma1:List[np.floating]
    gate_interp_sigma2:List[np.floating]
    Q_w:DM
    R_df:DM
    miu:np.floating
    obst_w:np.floating
    d_extend:np.floating
    lb_vel:np.floating
    ub_vel:np.floating
    
    traj_tool : TrajectoryTool
    arc_trajectory : CubicSpline
    arc_trajectory_offset : CubicSpline
    gate_theta_list : NDArray[np.integer]
    gate_theta_list_offset : NDArray[np.integer]

    x_guess : List[np.floating]
    u_guess : List[np.floating]
    traj_update_gate : int

    # For safety in real world implementation
    pos_bound : List[NDArray[np.floating]] = None
    velocity_bound : NDArray[np.floating] = None

    # Compiled MPC path
    model_name : str = 'mpcc_traj_translation'
    compile_path : str = 'lsy_drone_racing/acados_solvers/mpcc_prescripted.json'

    def __init__(self,
                param_dict : Dict[str, Union[List[np.floating], np.floating]],
                ):
        
        self.traj_path = param_dict['traj_path']

    
        self.gates_init = param_dict['gates']
        self.gates = self.gates_init
        self.gates_pos_init = [gate.pos.copy() for gate in self.gates_init]
        self.num_gates = len(self.gates)
        self.obstacles_init = param_dict['obstacles']
        self.obstacles = self.obstacles_init 
        self.obstacles_pos_init = [obstacle.pos.copy() for obstacle in self.obstacles_init]
        self.num_obstacles = len(self.obstacles)
        
    
        # Solver Options
        self.T_f = param_dict['T_f']
        self.N = param_dict['N']

        self.qp_solver = param_dict['qp_solver']
        self.hessian_approx = param_dict['hessian_approx']
        self.integrator_type = param_dict['integrator_type']
        self.nlp_solver_type = param_dict['nlp_solver_type']
        self.tol = param_dict['tol']
        self.qp_solver_warm_start = param_dict['qp_solver_warm_start']
        self.qp_solver_iter_max = param_dict['qp_solver_iter_max']
        self.nlp_solver_max_iter = param_dict['nlp_solver_max_iter']

        # Trajectory Settings
        self.model_traj_length = param_dict['model_traj_length']
        self.model_arc_length = param_dict['model_arc_length']

        # MPCC Cost Weights
        self.q_l = param_dict['q_l']
        self.q_l_peak = param_dict['q_l_peak']
        self.q_c = param_dict['q_c']
        self.q_c_peak = param_dict['q_c_peak']
        self.q_c_sigma1 = param_dict['q_c_sigma1']
        self.q_c_sigma2 = param_dict['q_c_sigma2']
        self.gate_interp_peak = param_dict['gate_interp_peak']
        self.gate_interp_sigma1 = param_dict['gate_interp_sigma1']
        self.gate_interp_sigma2 = param_dict['gate_interp_sigma2']
        self.Q_w = param_dict['Q_w']
        self.R_df = param_dict['R_df']
        self.miu = param_dict['miu']

        # Obstacle relavent
        self.obst_w = param_dict['obst_w']
        self.d_extend = param_dict['d_extend'] # extend distance to supress q_c
        # Velocity bounds
        # TODO: any way to discard lower bound?
        self.lb_vel = param_dict['lb_vel']
        self.ub_vel = param_dict['ub_vel']

        # Model and Symbolics
        self.model_name = param_dict.get('model_name', self.model_name)
        self.model_syms = FresssackDroneModel(model_traj_N = int(self.model_traj_length/self.model_arc_length),
                                    num_gates = self.num_gates,
                                    num_obstacles = self.num_obstacles,
                                    init_model = True,
                                    name = self.model_name
                                    )
        self.model = self.model_syms.model

        # Safety parameters
        self.pos_bound = param_dict.get('pos_bound', None) # List of 3
        self.velocity_bound = param_dict.get('vel_bound', None) # List of 2D

        # MPC path
        self.compile_path = param_dict.get('compile_path', self.compile_path)

        # Read Trajectory
        self.read_traj(path = self.traj_path)

        # Create Solver
        self.create_ocp_solver_external(path = self.compile_path)

        # Reset Solver
        self.reset_solver()
        
    # region Reset
    def reset_solver(self):
        # Reset race object positions
        self.gates = self.gates_init
        self.obstacles = self.obstacles_init

        # Reset trajectories and theta list of gates
        self.arc_trajectory_offset = self.arc_trajectory
        self.gate_theta_list_offset = self.gate_theta_list

        # Reset warm starts
        self.x_guess = None
        self.u_guess = None
        
        # Reset drone trajectory states
        self.traj_update_gate = None

        # Reset cylinder lists
        self.cylinder_list = None

        # Reset parameters in solver
        p_traj = self.get_updated_traj_param(self.arc_trajectory)
        p_obst = self.get_cylinder_param()
        p_gate_offset = self.get_curr_gate_offset(0, self.gates[0].norm_vec)
        p_full = np.concatenate([p_traj, p_obst, p_gate_offset])
        for i in range(self.N):
            self.solver.set(i, "p", p_full)

    def out_of_pos_bound(self, pos : NDArray[np.floating]) -> bool:
        """Check if the position is out of bounds.
        
        Args:
            pos: 3D position vector.
        
        Returns:
            bool: True if out of bounds, False otherwise.
        """
        if self.pos_bound is None:
            return False
        for i in range(3):
            if pos[i] < self.pos_bound[i][0] or pos[i] > self.pos_bound[i][1]:
                return True
        return False

    def out_of_vel_bound(self, vel : NDArray[np.floating]) -> bool:
        """Check if the velocity is out of bounds.

        Args:
            vel: 3D velocity vector.

        Returns:
            bool: True if out of bounds, False otherwise.
        """
        if self.velocity_bound is None:
            return False
        else:
            velocity = np.linalg.norm(vel)
            return not(self.velocity_bound[0] < velocity < self.velocity_bound[1])

    # region Control Step
    def control_step(self, x : NDArray,
                    last_theta : np.floating,
                    need_gate_update : bool,
                    need_obs_update : bool,
                    gates : List[Gate],
                    obstacles : List[Obstacle],
                    next_gate : int = 0):
        # Update race objects
        self.gates = gates
        self.obstacles = obstacles

        # Take a guess
        if self.x_guess is None or self.u_guess is None:
            self.x_guess = [x for _ in range(self.N + 1)]
            self.u_guess = [np.zeros(self.n_u) for _ in range(self.N)]
        else:
            # self.x_guess = self.x_guess[1:] + [self.x_guess[-1]]
            # self.u_guess = self.u_guess[1:] + [self.u_guess[-1]]
            pass
        
        # Set guess
        for i in range(self.N):
            self.solver.set(i, "x", self.x_guess[i])
            self.solver.set(i, "u", self.u_guess[i])
        self.solver.set(self.N, "x", self.x_guess[self.N])



        # Setup trajectory update gate index
        if self.traj_update_gate is None:
            self.traj_update_gate = 0
            self.gate_waypoint_tangent = self.gates[0].norm_vec
        
        # wait a while then reset to norminal trajectory
        if next_gate > self.traj_update_gate and last_theta - self.gate_theta_list_offset[self.traj_update_gate] > self.gate_interp_sigma2[self.traj_update_gate]:
            self.traj_update_gate = next_gate
            self.arc_trajectory_offset = self.arc_trajectory
            p_traj = self.get_updated_traj_param(self.arc_trajectory_offset)
            p_obst = self.get_cylinder_param()
            p_gate_offset = self.get_curr_gate_offset(self.traj_update_gate, self.gate_waypoint_tangent)
            p_full = np.concatenate([p_traj, p_obst, p_gate_offset])
            for i in range(self.N):
                self.solver.set(i, "p", p_full)

        self.curr_gate_offset = self.get_curr_gate_offset(self.traj_update_gate, self.gate_waypoint_tangent if hasattr(self, "gate_waypoint_tangent") else None)
        
        if need_gate_update or need_obs_update:
            # recompute gate_theta
            self.gate_theta_list_offset, _ = self.traj_tool.find_gate_waypoint(self.arc_trajectory, [gate.pos for gate in self.gates])
            if self.traj_update_gate != 2:
                self.gate_waypoint_tangent = self.arc_trajectory.derivative()(self.gate_theta_list_offset[self.traj_update_gate])
            else:
                self.gate_waypoint_tangent = self.gates[2].norm_vec
            # translate trajectory
            p_traj = self.get_updated_traj_param(self.arc_trajectory)
            p_obst = self.get_cylinder_param()
            p_gate_offset = self.get_curr_gate_offset(self.traj_update_gate, self.gate_waypoint_tangent)
            p_full = np.concatenate([p_traj, p_obst, p_gate_offset])
            for i in range(self.N):
                self.solver.set(i, "p", p_full)
            self.arc_trajectory_offset = self.translate_cubicspline(self.arc_trajectory, p_gate_offset) # only for visualization
            self.cylinder_list = self._gen_pillar_cylinder() # only for visualization
            self.curr_gate_offset = p_gate_offset # only for visualization

        # Set initial states
        self.solver.set(0, "lbx", x)
        self.solver.set(0, "ubx", x)

        # Solve for solution
        status = self.solver.solve()
        qp_iter = self.solver.get_stats("qp_iter")[1]
        success = (status == 0) or (qp_iter >= 10) 

        if not success:
            x_result = self.x_guess
            u_result = self.u_guess
            p_result = [self.solver.get(i, "p") for i in range(self.N)]
            return x_result, u_result, p_result, success
        else:
            # Get results
            x_result = [self.solver.get(i, "x") for i in range(self.N + 1)]
            u_result = [self.solver.get(i, "u") for i in range(self.N)]
            p_result = [self.solver.get(i, "p") for i in range(self.N)]

            self.x_guess = x_result
            self.u_guess = u_result

            return x_result, u_result, p_result, success


    def translate_cubicspline(self, trajectory: CubicSpline, offset: np.ndarray) -> CubicSpline:
        """translate trajectory
        """
        theta_list = trajectory.x
        waypoints = trajectory(theta_list)
        waypoints_offset = waypoints + offset
        trajectory_offset = CubicSpline(theta_list, waypoints_offset)
        return trajectory_offset

    def read_traj(self, path : str):
        t, pos, vel = FresssackController.read_trajectory(path)

        trajectory = CubicSpline(t, pos)

        # trajectory reparameterization
        self.traj_tool = TrajectoryTool()
        trajectory = self.traj_tool.extend_trajectory(trajectory, extend_length = 1.0)
        self.arc_trajectory = self.traj_tool.arclength_reparameterize(trajectory, arc_length = self.model_arc_length)
        self.arc_trajectory_offset = self.arc_trajectory
        self.gate_theta_list, _ = self.traj_tool.find_gate_waypoint(self.arc_trajectory, [gate.pos for gate in self.gates])
        self.gate_theta_list_offset = self.gate_theta_list

    # region Ocp Solver
    def create_ocp_solver_external(
            self,
            verbose: bool = False,
            path : str = 'lsy_drone_racing/acados_solvers/mpcc_prescripted.json', 
        ) -> AcadosOcpSolver:
            """Creates an acados Optimal Control Problem and Solver."""
            ocp = AcadosOcp()
            # set model
            ocp.model = self.model
            # Get Dimensions
            self.n_x = ocp.model.x.rows()
            self.n_u = ocp.model.u.rows()

            # Set dimensions
            ocp.solver_options.N_horizon = self.N


            # Cost Type
            ocp.cost.cost_type = "EXTERNAL"
            ocp.model.cost_expr_ext_cost = self.mpcc_cost()

            ## Set Csolver_wrapper.traints.ubx = np.array([0.55, 0.55, 1.57, 1.57, 1.57])
            ocp.constraints.lbx = np.array([0.1, 0.1, -1.57, -1.57, -1.57])
            ocp.constraints.ubx = np.array([0.55, 0.55, 1.57, 1.57, 1.57])
            ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

            # Set Input Constraints
            ocp.constraints.lbu = np.array([-10.0, -10.0, -10.0, -10.0, self.lb_vel]) # set a speed lower bound to provide it from stopping at obstacles
            ocp.constraints.ubu = np.array([10.0, 10.0, 10.0, 10.0, self.ub_vel])
            ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

            # We have to set x0 even though we will overwrite it later on.
            ocp.constraints.x0 = np.zeros((self.n_x))
            # Set initial reference trajectory
            p_traj = self.get_updated_traj_param(self.arc_trajectory)
            # Set initial obstacle parameters
            p_obst = self.get_cylinder_param()
            # Set initial gate offset = 0
            p_gate_offset = self.get_curr_gate_offset(0)
            # stuff parameters 
            p_full = np.concatenate([p_traj, p_obst, p_gate_offset])
            ocp.parameter_values = p_full


            # Solver Options
            ocp.solver_options.qp_solver = self.qp_solver  
            ocp.solver_options.hessian_approx = self.hessian_approx
            ocp.solver_options.integrator_type = self.integrator_type
            ocp.solver_options.nlp_solver_type = self.nlp_solver_type
            ocp.solver_options.tol = self.tol

            ocp.solver_options.qp_solver_cond_N = self.N
            ocp.solver_options.qp_solver_warm_start = self.qp_solver_warm_start

            ocp.solver_options.qp_solver_iter_max = self.qp_solver_iter_max
            ocp.solver_options.nlp_solver_max_iter = self.nlp_solver_max_iter

            # set prediction horizon
            ocp.solver_options.tf = self.T_f
            self.solver = AcadosOcpSolver(ocp, json_file=path, verbose=verbose)
    
    # region Obst Param
    def _gen_pillar_cylinder(self):
        """init pillar cylinder from self.obstacles
        Returns:
            List of horizontal coordinates
        """
        cylinder_list = []
        for obst in self.obstacles:
            x, y = obst.pos[:2]
            r = obst.safe_radius
            cylinder_list.append([x, y, r])
        return cylinder_list

    def get_cylinder_param(self) -> NDArray[np.floating]:
        """put all cylinders into a flat array to write to model.p
        Returns:
            NDArray of cylinders parameters like [x, y, r, x, y, r, ...]
        """
        cylinder_list = self._gen_pillar_cylinder()

        cylinder_params = []
        for x, y, r in cylinder_list:
            cylinder_params.append(x)
            cylinder_params.append(y)
            cylinder_params.append(r)
        return np.array(cylinder_params, dtype=np.float32)

    # region Traj Param
    def get_updated_traj_param(self, trajectory: CubicSpline):
        """get updated trajectory parameters upon replaning"""
        # construct pd/tp lists from current trajectory
        theta_list = np.arange(0, self.model_traj_length, self.model_arc_length)
        pd_list = trajectory(theta_list)
        tp_list = trajectory.derivative(1)(theta_list)
        qc_dyn_list = np.zeros(theta_list.shape[0])
        ql_dyn_list = np.zeros(theta_list.shape[0])
        gate_interp_list = np.zeros(theta_list.shape[0])
        for idx, gate in enumerate(self.gates):
            # distances = np.linalg.norm(pd_list - gate.pos, axis=-1) # spacial distance
            distances = theta_list - self.gate_theta_list_offset[idx] # progress distance
            qc_dyn_gate_front = np.exp(-distances**2 / (0.5*self.q_c_sigma1[idx])**2) # gaussian
            qc_dyn_gate_behind = np.exp(-distances**2 / (0.5*self.q_c_sigma2[idx])**2) # gaussian
            qc_dyn_list += self.q_c_peak[idx] * qc_dyn_gate_front * ((theta_list < self.gate_theta_list_offset[idx]) & (theta_list > self.gate_theta_list_offset[idx] - self.q_c_sigma1[idx])) \
                         + self.q_c_peak[idx] * qc_dyn_gate_behind * ((theta_list >= self.gate_theta_list_offset[idx]) & (theta_list < self.gate_theta_list_offset[idx] + self.q_c_sigma2[idx]))
            ql_dyn_list += self.q_l_peak[idx] * qc_dyn_gate_front * ((theta_list < self.gate_theta_list_offset[idx]) & (theta_list > self.gate_theta_list_offset[idx] - self.q_c_sigma1[idx])) \
                         + self.q_l_peak[idx] * qc_dyn_gate_behind * ((theta_list >= self.gate_theta_list_offset[idx]) & (theta_list < self.gate_theta_list_offset[idx] + self.q_c_sigma2[idx]))
            gate_interp_gate_front  = self.gate_interp_peak[idx] * np.exp(-distances**2 / (0.5*self.gate_interp_sigma1[idx])**2) # gaussian
            gate_interp_gate_behind = self.gate_interp_peak[idx] * np.exp(-distances**2 / (0.5*self.gate_interp_sigma2[idx])**2) # gaussian
            gate_interp_list += gate_interp_gate_front * ((theta_list < self.gate_theta_list_offset[idx]) & (theta_list > self.gate_theta_list_offset[idx] - self.gate_interp_sigma1[idx])) \
                              + gate_interp_gate_behind * ((theta_list >= self.gate_theta_list_offset[idx]) & (theta_list < self.gate_theta_list_offset[idx] + self.gate_interp_sigma2[idx]))
            
        self.gate_interp_list = CubicSpline(theta_list, gate_interp_list)
        p_vals = np.concatenate([pd_list.flatten(), tp_list.flatten(), qc_dyn_list.flatten(), ql_dyn_list.flatten(), gate_interp_list.flatten()])
        return p_vals
    
    def get_curr_gate_offset(self, curr_gate, curr_gate_norm=None):
        """return current gate position change
        run detect pos change outside in control loop
        Returns:
            NDArray(3): position change of current target gate
        """
        curr_gate_offset = self.gates[curr_gate].pos - self.gates_pos_init[curr_gate]
        if curr_gate_norm is not None: # NOTE: EXP: translate trajectory only on normal plane of gate
            curr_gate_offset = curr_gate_offset - np.dot(curr_gate_offset, curr_gate_norm) * curr_gate_norm
        return curr_gate_offset
    
    def casadi_linear_interp(self, theta : MX, theta_list : MX, p_flat : MX, dim=3):
        """Manually interpolate a 3D path using CasADi symbolic expressions.
        
        Args:
            theta: CasADi symbol, scalar progress variable (0 ~ model_traj_length)
            theta_list: list or array, thetas of path points [0, 0.1, 0.2, ...]
            p_flat: CasADi symbol, 1D flattened path points [x0,y0,z0, x1,y1,z1, ...]
            dim: int, dimension of a single point (default=3)
        Returns:
            p_interp: CasADi 3x1 vector, interpolated path point at given theta
        """
        M = len(theta_list)
        
        # Find index interval
        # Normalize theta to index scale
        idx_float = (theta - theta_list[0]) / (theta_list[-1] - theta_list[0]) * (M - 1)

        idx_lower = floor(idx_float)
        idx_upper = idx_lower + 1
        alpha = idx_float - idx_lower

        # Handle boundary cases (clamping)
        idx_lower = if_else(idx_lower < 0, 0, idx_lower)
        idx_upper = if_else(idx_upper >= M, M-1, idx_upper)

        # Gather points
        p_lower = vertcat(*[p_flat[dim * idx_lower + d] for d in range(dim)])
        p_upper = vertcat(*[p_flat[dim * idx_upper + d] for d in range(dim)])

        # Interpolated point
        p_interp = (1 - alpha) * p_lower + alpha * p_upper

        return p_interp
    
    def calc_obst_distance(self, pos : MX, cyl_xy : MX):
        """calculate distances of pos to every obstacles with casadi
        Args:
            pos: CasADi 3x1 
            cyl_xy: cylinder center
        Returns:
            distance to closest point and vector from closest point to pos
        """
        vec = pos[:2] - cyl_xy
        dist = norm_2(vec)
        return dist
    
    # region MPCC Cost
    def mpcc_cost_components(self):
        pos = vertcat(self.model_syms.px, self.model_syms.py, self.model_syms.pz)
        ang = vertcat(self.model_syms.roll, self.model_syms.pitch, self.model_syms.yaw)
        control_input = vertcat(self.model_syms.f_collective_cmd, self.model_syms.dr_cmd, self.model_syms.dp_cmd, self.model_syms.dy_cmd)

        # interpolate spline dynamically
        theta_list = np.arange(0, self.model_traj_length, self.model_arc_length)
        pd_theta = self.casadi_linear_interp(self.model_syms.theta, theta_list, self.model_syms.pd_list)
        tp_theta = self.casadi_linear_interp(self.model_syms.theta, theta_list, self.model_syms.tp_list)
        qc_dyn_theta = self.casadi_linear_interp(self.model_syms.theta, theta_list, self.model_syms.qc_dyn, dim=1)
        ql_dyn_theta = self.casadi_linear_interp(self.model_syms.theta, theta_list, self.model_syms.ql_dyn, dim=1)
        gate_interp_theta = self.casadi_linear_interp(self.model_syms.theta, theta_list, self.model_syms.gate_interp, dim=1)
        tp_theta_norm = tp_theta / norm_2(tp_theta)
        # apply offset on pd_theta
        pd_theta_offset = pd_theta + gate_interp_theta * self.model_syms.gate_offset_param
        e_theta = pos - pd_theta
        e_l = dot(tp_theta_norm, e_theta) * tp_theta_norm
        # e_c = e_theta - e_l
        # NOTE: handle e_l and e_c separately, only e_c on offset trajectory
        e_theta_offset = pos - pd_theta_offset
        e_l_offset = dot(tp_theta_norm, e_theta_offset) * tp_theta_norm
        e_c = e_theta_offset - e_l_offset

        # cost for obstacles
        q_c_supress = 0.0
        obst_cost = 0.0
        for i in range(self.num_obstacles):
            idx = i * 3
            cyl_xy = self.model_syms.obst_list[idx     : idx + 2] # extract params from model.p
            cyl_r =  self.model_syms.obst_list[idx + 2 : idx + 3]
            dis = self.calc_obst_distance(pd_theta_offset, cyl_xy) # EXP: use trajectory collision to supress q_c
            # trick: to supress q_c & miu when running into obstacle extended surfaces
            q_c_supress += exp( -power(dis/(0.5*(self.d_extend+cyl_r)), 2) )
            # soft punish when getting into safe range: when , cost = gaussian(distance) if outside surface else 1.0
            dis = self.calc_obst_distance(pos, cyl_xy)
            obst_cost += exp( -power(dis/(0.5*cyl_r), 2) )

        q_c_factor = 1 - 0.6 * q_c_supress  # supress q_c based on trajectory collision
        miu_factor = 1 - 0.9 * q_c_supress  # supress miu 

        # Break down the costs
        C_l = self.q_l + ql_dyn_theta
        e_l_cost = dot(e_l, e_l)
        cost_l = C_l * q_c_factor * e_l_cost

        C_c = self.q_c + qc_dyn_theta
        e_c_cost = dot(e_c, e_c)
        cost_c = C_c * e_c_cost

        ang_cost = ang.T @ self.Q_w @ ang
        ctrl_cost = control_input.T @ self.R_df @ control_input

        cost_obs = self.obst_w * obst_cost

        miu_cost = (-self.miu) * miu_factor * self.model_syms.v_theta_cmd

        mpcc_cost = cost_l + cost_c + ang_cost + ctrl_cost + cost_obs + miu_cost

        return {
            'total': mpcc_cost,
            'cost_l': cost_l,
            'C_l': C_l,
            'e_l_cost': e_l_cost,
            'cost_c': cost_c,
            'C_c': C_c,
            'e_c_cost': e_c_cost,
            'ang_cost': ang_cost,
            'ctrl_cost': ctrl_cost,
            'cost_obs': cost_obs,
            'miu_cost': miu_cost,
        }

    def mpcc_cost(self):
        """calculate mpcc cost function"""
        return self.mpcc_cost_components()['total']

class FresssackDroneModel:
    model : AcadosModel
    model_traj_N : int

    px : MX
    py : MX
    vx : MX
    vy : MX
    vz : MX
    roll : MX
    pitch : MX
    yaw : MX
    f_collective : MX

    f_collective_cmd : MX
    r_cmd : MX
    p_cmd : MX
    y_cmd : MX

    df_cmd : MX
    dr_cmd : MX
    dp_cmd : MX
    dy_cmd : MX

    theta : MX
    v_theta_cmd : MX

    x : MX
    u : MX
    p : MX

    def __init__(self,
                model_traj_N : int,
                num_gates : int = 4,
                num_obstacles : int = 4,
                init_model = False,
                name : str = "mpcc_traj_translation"
                ):
        self.model_traj_N = model_traj_N
        self.num_gates = num_gates
        self.num_obstacles = num_obstacles
        if init_model:
            self.init_quadrotor_model(name = name)

    def init_quadrotor_model(self, name : str= "mpcc_traj_translation") -> AcadosModel:
            """Symbolic Quadrotor Model."""
            # Define name of solver to be used in script
            model_name = name

            # Define Gravitational Acceleration
            GRAVITY = 9.806

            # Sys ID Params
            params_pitch_rate = [-6.003842038081178, 6.213752925707588]
            params_roll_rate = [-3.960889336015948, 4.078293254657104]
            params_yaw_rate = [-0.005347588299390372, 0.0]
            params_acc = [20.907574256269616, 3.653687545690674]

            """Model setting"""
            # define basic variables in state and input vector
            self.px = MX.sym("px")  # 0
            self.py = MX.sym("py")  # 1
            self.pz = MX.sym("pz")  # 2
            self.vx = MX.sym("vx")  # 3
            self.vy = MX.sym("vy")  # 4
            self.vz = MX.sym("vz")  # 5
            self.roll = MX.sym("r")  # 6
            self.pitch = MX.sym("p")  # 7
            self.yaw = MX.sym("y")  # 8
            self.f_collective = MX.sym("f_collective")

            self.f_collective_cmd = MX.sym("f_collective_cmd")
            self.r_cmd = MX.sym("r_cmd")
            self.p_cmd = MX.sym("p_cmd")
            self.y_cmd = MX.sym("y_cmd")

            self.df_cmd = MX.sym("df_cmd")
            self.dr_cmd = MX.sym("dr_cmd")
            self.dp_cmd = MX.sym("dp_cmd")
            self.dy_cmd = MX.sym("dy_cmd")

            # expanded observation state
            self.theta = MX.sym("theta")
            # self.v_theta = MX.sym("v_theta")
            self.v_theta_cmd = MX.sym("v_theta_cmd")

            # define state and input vector
            states = vertcat(
                self.px,
                self.py,
                self.pz,
                self.vx,
                self.vy,
                self.vz,
                self.roll,
                self.pitch,
                self.yaw,
                self.f_collective,
                self.f_collective_cmd,
                self.r_cmd,
                self.p_cmd,
                self.y_cmd,
                self.theta
            )
            inputs = vertcat(
                self.df_cmd, 
                self.dr_cmd, 
                self.dp_cmd, 
                self.dy_cmd, 
                self.v_theta_cmd
            )

            # Define nonlinear system dynamics
            f = vertcat(
                self.vx,
                self.vy,
                self.vz,
                (params_acc[0] * self.f_collective + params_acc[1])
                * (cos(self.roll) * sin(self.pitch) * cos(self.yaw) + sin(self.roll) * sin(self.yaw)),
                (params_acc[0] * self.f_collective + params_acc[1])
                * (cos(self.roll) * sin(self.pitch) * sin(self.yaw) - sin(self.roll) * cos(self.yaw)),
                (params_acc[0] * self.f_collective + params_acc[1]) * cos(self.roll) * cos(self.pitch) - GRAVITY,
                params_roll_rate[0] * self.roll + params_roll_rate[1] * self.r_cmd,
                params_pitch_rate[0] * self.pitch + params_pitch_rate[1] * self.p_cmd,
                params_yaw_rate[0] * self.yaw + params_yaw_rate[1] * self.y_cmd,
                10.0 * (self.f_collective_cmd - self.f_collective),
                self.df_cmd,
                self.dr_cmd,
                self.dp_cmd,
                self.dy_cmd,
                self.v_theta_cmd,
            )

            # define dynamic trajectory input & obstacle parameters input
            self.pd_list = MX.sym("pd_list", 3 * self.model_traj_N)
            self.tp_list = MX.sym("tp_list", 3 * self.model_traj_N)
            self.qc_dyn = MX.sym("qc_dyn", 1 * self.model_traj_N)
            self.ql_dyn = MX.sym("ql_dyn", 1 * self.model_traj_N)
            self.gate_interp = MX.sym("gate_interp", 1 * self.model_traj_N)
            self.obst_list = MX.sym("obst_list", self.num_obstacles * 3) # 4 * 3 = 12
            self.gate_offset_param = MX.sym("gate_offset_param", 3)
            params = vertcat(
                self.pd_list,
                self.tp_list,
                self.qc_dyn,
                self.ql_dyn,
                self.gate_interp,
                self.obst_list,
                self.gate_offset_param
            )

            # For ease of use, contatenate x, u, and p
            self.x = states
            self.u = inputs
            self.p = params

            # Initialize the nonlinear model for NMPC formulation
            model = AcadosModel()
            model.name = model_name
            model.f_expl_expr = f
            model.f_impl_expr = None
            model.x = states
            model.u = inputs
            model.p = params

            self.model = model