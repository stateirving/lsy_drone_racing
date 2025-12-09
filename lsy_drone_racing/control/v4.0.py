from __future__ import annotations
import numpy as np
import casadi as ca
from typing import TYPE_CHECKING
from dataclasses import dataclass

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append("/home/miao/repos/lsy_drone_racing")

from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from lsy_drone_racing.control.controller import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ============================================================
#  工具：四元数 → yaw
# ============================================================

def quat_to_yaw(quat: NDArray[np.floating]) -> float:
    x, y, z, w = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


# ============================================================
#  工具：门点 → centerline (Ein, G, Eout)
# ============================================================

def build_centerline_points(
    gates_pos: NDArray[np.floating],
    gates_quat: NDArray[np.floating],
    L_pre: float = 0.4,
    L_post: float = 0.4,
):
    pts = []
    for i in range(len(gates_pos)):
        G = np.array(gates_pos[i])
        yaw = quat_to_yaw(gates_quat[i])
        fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])

        Ein = G - fwd * L_pre
        Eout = G + fwd * L_post

        pts.append(Ein)
        pts.append(G)
        pts.append(Eout)

    return np.asarray(pts)


# ============================================================
#  spline / Frenet track
# ============================================================

@dataclass
class FrenetTrack:
    s_tab: NDArray[np.floating]
    p_tab: NDArray[np.floating]
    t_tab: NDArray[np.floating]
    n_tab: NDArray[np.floating]


def build_spline(centerline_pts):
    u = np.linspace(0, 1, len(centerline_pts))
    cs_x = CubicSpline(u, centerline_pts[:, 0])
    cs_y = CubicSpline(u, centerline_pts[:, 1])
    cs_z = CubicSpline(u, centerline_pts[:, 2])
    return cs_x, cs_y, cs_z


def sample_frenet(cs_x, cs_y, cs_z, M=400) -> FrenetTrack:
    u = np.linspace(0, 1, M)
    px = cs_x(u)
    py = cs_y(u)
    pz = cs_z(u)
    p = np.vstack([px, py, pz]).T

    # -------- 弧长 --------
    s = np.zeros(M)
    for i in range(1, M):
        s[i] = s[i-1] + np.linalg.norm(p[i]-p[i-1])

    # -------- 切向量 --------
    dx = cs_x(u, 1)
    dy = cs_y(u, 1)
    dz = cs_z(u, 1)
    t = np.vstack([dx, dy, dz]).T
    t /= np.linalg.norm(t, axis=1, keepdims=True)

    # -------- 法向量（2D）--------
    n = np.zeros_like(t)
    n[:, 0] = -t[:, 1]
    n[:, 1] =  t[:, 0]
    n[:, 2] =  0

    return FrenetTrack(s_tab=s, p_tab=p, t_tab=t, n_tab=n)


def frenet_lookup(track: FrenetTrack, theta: float):
    s_tab = track.s_tab
    s_max = s_tab[-1]

    # wrap
    s = np.mod(theta, s_max)

    idx = np.searchsorted(s_tab, s)
    if idx <= 0:
        return (track.p_tab[0], track.t_tab[0], track.n_tab[0])
    if idx >= len(s_tab):
        return (track.p_tab[-1], track.t_tab[-1], track.n_tab[-1])

    s0 = s_tab[idx-1]
    s1 = s_tab[idx]
    w = (s - s0) / (s1 - s0 + 1e-9)

    p = (1-w)*track.p_tab[idx-1] + w*track.p_tab[idx]
    t = (1-w)*track.t_tab[idx-1] + w*track.t_tab[idx]
    n = (1-w)*track.n_tab[idx-1] + w*track.n_tab[idx]

    return p, t/np.linalg.norm(t), n/np.linalg.norm(n)


# ============================================================
#  ACADOS model with MPCC cost
# ============================================================

def create_acados_model_mpcc(params):

    # ---------- 调用你的动力学 ----------
    X_dot, X, U, Y = symbolic_dynamics_euler(
        model_rotor_vel=False,
        mass=params["mass"],
        gravity_vec=params["gravity_vec"],
        J=params["J"],
        J_inv=params["J_inv"],
        acc_coef=params["acc_coef"],
        cmd_f_coef=params["cmd_f_coef"],
        rpy_coef=params["rpy_coef"],
        rpy_rates_coef=params["rpy_rates_coef"],
        cmd_rpy_coef=params["cmd_rpy_coef"],
    )

    # ---------- 增加 P：track 参数 ----------
    # p_ref(0:3), t_ref(3:6), n_ref(6:9)
    P = ca.SX.sym("P", 9)

    model = AcadosModel()
    model.name = "mpcc_drone"

    model.x = X        # 12
    model.u = U        # 4
    model.xdot = X_dot
    model.p = P
    model.z = ca.SX.sym("z", 0)
    model.f_expl_expr = X_dot
    model.f_impl_expr = None

    # ---------- 定义 MPCC cost ----------
    pos = X[0:3]
    vel = X[6:9]

    p_ref = P[0:3]
    t_ref = P[3:6]
    n_ref = P[6:9]

    e_vec = pos - p_ref
    e_c = ca.dot(n_ref, e_vec)
    e_l = ca.dot(t_ref, e_vec)
    v_theta = ca.dot(t_ref, vel)

    Qc = 50
    Ql = 5
    mu = 0.4

    Ru = ca.diag(ca.vertcat(1,1,1,40))
    u_cost = ca.mtimes(U.T, ca.mtimes(Ru, U))

    stage_cost = Qc*e_c**2 + Ql*e_l**2 + u_cost - mu*v_theta
    terminal_cost = Qc*e_c**2 + Ql*e_l**2

    model.cost_expr_ext_cost = stage_cost
    model.cost_expr_ext_cost_e = terminal_cost

    return model


def create_ocp_solver_mpcc(Tf, N, params):

    ocp = AcadosOcp()
    ocp.model = create_acados_model_mpcc(params)

    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]

    # external cost
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # constraints（与你原来的 MPC 一致）
    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
    ocp.constraints.ubx = np.array([ 0.5,  0.5,  0.5])
    ocp.constraints.idxbx = np.array([3,4,5])

    ocp.constraints.lbu = np.array([-0.5,-0.5,-0.5, params["thrust_min"]*4])
    ocp.constraints.ubu = np.array([ 0.5, 0.5, 0.5, params["thrust_max"]*4])
    ocp.constraints.idxbu = np.array([0,1,2,3])

    ocp.constraints.x0 = np.zeros(nx)

    # solver options
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50
    ocp.solver_options.qp_solver_warm_start = 1

    solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/mpcc_drone.json",
        generate=True,
        build=True,
    )
    return solver, ocp


# ============================================================
#  MPCC Controller
# ============================================================

class MPCCController(Controller):

    def __init__(self, obs, info, config):
        super().__init__(obs, info, config)

        self._dt = 1.0 / config.env.freq
        self._N = 25
        self._T_HORIZON = self._N * self._dt

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self.solver, self.ocp = create_ocp_solver_mpcc(
            self._T_HORIZON, self._N, self.drone_params
        )

        # ====== build track from gates ======
        gates_pos = np.array(obs["gates_pos"])
        gates_quat = np.array(obs["gates_quat"])
        center_pts = build_centerline_points(gates_pos, gates_quat)
        cs_x, cs_y, cs_z = build_spline(center_pts)
        self.track = sample_frenet(cs_x, cs_y, cs_z)

        self.theta = 0.0
        self.finished = False

    # ---------------------------
    def compute_control(self, obs, info=None):

        pos = np.array(obs["pos"])
        quat = np.array(obs["quat"])
        vel = np.array(obs["vel"])
        ang_vel = np.array(obs["ang_vel"])

        rpy = R.from_quat(quat).as_euler("xyz")
        drpy = ang_vel2rpy_rates(quat, ang_vel)

        x0 = np.concatenate([pos, rpy, vel, drpy])

        # 初始状态
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

        # progress 预测
        v = np.linalg.norm(vel)
        v_nom = max(0.3, v)

        # 设置 stage 参数 P
        for j in range(self._N):
            theta_j = self.theta + v_nom * self._dt * j
            p_ref, t_ref, n_ref = frenet_lookup(self.track, theta_j)
            P_vec = np.hstack([p_ref, t_ref, n_ref]).astype(np.float64)
            self.solver.set(j, "p", P_vec)

        theta_T = self.theta + v_nom * self._dt * self._N
        p_ref_T, t_ref_T, n_ref_T = frenet_lookup(self.track, theta_T)
        self.solver.set(self._N, "p", np.hstack([p_ref_T, t_ref_T, n_ref_T]))

        status = self.solver.solve()
        if status != 0:
            print("[MPCC] solver fail:", status)
            u = np.zeros(4)
            u[3] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
        else:
            u = self.solver.get(0, "u")

        # update θ
        p_now, t_now, _ = frenet_lookup(self.track, self.theta)
        v_theta = np.dot(vel, t_now)
        self.theta += max(0, v_theta) * self._dt

        # 判断是否走完轨道
        if self.theta > self.track.s_tab[-1]:
            self.finished = True

        return u.astype(np.float32)

    # ---------------------------
    def step_callback(self, *args):
        return self.finished

    def episode_callback(self):
        self.theta = 0.0
        self.finished = False
