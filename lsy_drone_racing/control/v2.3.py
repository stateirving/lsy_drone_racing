from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
import sys

sys.path.append("/home/miao/repos/lsy_drone_racing")

from rrt_algorithms.rrt.rrt_star import RRTStar
from rrt_algorithms.rrt.rrt_star_bid import RRTStarBidirectional
from rrt_algorithms.search_space.search_space import SearchSpace
from lsy_drone_racing.control.controller import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class RRT2DStateController(Controller):
    """
    2D RRT 控制器：
    - 只在 x-y 平面规划
    - z 高度由 gate 决定
    - 输出仍然是 3D position command
    """

    def __init__(self, obs, info, config):
        super().__init__(obs, info, config)

        # ---------------- parameters ----------------
        self.gates_count = 4
        self.current_gate_idx = 0
        self._gate_stage = 0  # 0: pre, 1: center, 2: post

        # 2D search space
        self.xy_limit = np.array([
            [-2.5, 2.5],
            [-1.5, 1.5]
        ])

        self.step_tol = 0.15
        self.pre_dist = 0.4
        self.post_dist = 0.4

        self.gate_width = 1.0
        self.obstacle_radius = 0.1

        self.length_tree_edge = 0.4
        self.smallest_edge = 0.01
        self.max_samples = 500
        self.prc = 0.1

        # path
        self.path_xy = None
        self.path_idx = 0
        self.target_z = 0.4
        
        # --- previous observation for replan check ---
        self.previous_gates = None
        self.previous_obstacles = None
        self.previous_gate_idx = -1
        self.previous_gate_stage = -1
        # self.previous_going_to_extra = False    

        self._finished = False

    # ------------------------------------------------
    # utils
    # ------------------------------------------------
    def _quat_to_yaw(self, quat):
        x, y, z, w = quat
        return np.arctan2(
            2 * (w * z + x * y),
            1 - 2 * (y * y + z * z)
        )

    def _gate_forward_xy(self, quat):
        yaw = self._quat_to_yaw(quat)
        return np.array([np.cos(yaw), np.sin(yaw)])

    # ------------------------------------------------
    # obstacle generation (2D)
    # ------------------------------------------------
    def _generate_2d_space(self, obs):
        space = SearchSpace(self.xy_limit)

#用小矩形元代替整块AABB门板
        # --- gates as blocking rectangles ---
        # for gpos, gquat in zip(obs["gates_pos"], obs["gates_quat"]):
        #     cx, cy, _ = gpos
        #     yaw = self._quat_to_yaw(gquat)

        #     half_w = self.gate_width / 2
        #     thickness = 0.05

        #     corners = [
        #         (-half_w, -thickness),
        #         (-half_w, thickness),
        #         (half_w, -thickness),
        #         (half_w, thickness),
        #     ]

        #     pts = []
        #     for lx, ly in corners:
        #         x = cx + lx * np.cos(yaw) - ly * np.sin(yaw)
        #         y = cy + lx * np.sin(yaw) + ly * np.cos(yaw)
        #         pts.append([x, y])
        #     pts = np.array(pts)

        #     xmin, ymin = pts.min(axis=0)
        #     xmax, ymax = pts.max(axis=0)

        #     ob = (xmin, ymin, xmax, ymax)
        #     space.obs.insert(0, ob, ob)

        # --- obstacles as circles → AABB ---
        for pos in obs["obstacles_pos"]:
            x, y, _ = pos
            r = self.obstacle_radius
            ob = (x - r, y - r, x + r, y + r)
            space.obs.insert(0, ob, ob)

        return space

    # ------------------------------------------------
    # goal generation
    # ------------------------------------------------
    def _goal_xy(self, gate_pos, gate_quat):
        gate_xy = gate_pos[:2]
        fwd = self._gate_forward_xy(gate_quat)

        if self._gate_stage == 0:
            return gate_xy - fwd * self.pre_dist
        elif self._gate_stage == 1:
            return gate_xy
        else:
            return gate_xy + fwd * self.post_dist


    # ----------------- replanning 条件 -----------------
    def check_replan_condition(self, obs: dict) -> bool:
        if (not np.array_equal(obs["gates_pos"], self.previous_gates)
            or not np.array_equal(obs["obstacles_pos"], self.previous_obstacles)
            or self.current_gate_idx != self.previous_gate_idx
            or self._gate_stage != self.previous_gate_stage
            or self.path_xy is None
            # or self.going_to_extra != self.previous_going_to_extra
            ):

            self.previous_gates = obs["gates_pos"]
            self.previous_obstacles = obs["obstacles_pos"]
            self.previous_gate_idx = self.current_gate_idx
            self.previous_gate_stage = self._gate_stage
            # self.previous_going_to_extra = self.going_to_extra
            return True

        return False
        
    # ------------------------------------------------
    # planning
    # ------------------------------------------------
    def plan(self, obs, current_xy):
        gate_pos = np.array(obs["gates_pos"][self.current_gate_idx])
        gate_quat = np.array(obs["gates_quat"][self.current_gate_idx])

        self.target_z = gate_pos[2]
        goal_xy = self._goal_xy(gate_pos, gate_quat)

        space = self._generate_2d_space(obs)

        rrt = RRTStarBidirectional(
            space,
            self.length_tree_edge,
            tuple(current_xy),
            tuple(goal_xy),
            self.max_samples,
            self.smallest_edge,
            self.prc
        )

        path = rrt.rrt_search()

        if path is None or len(path) == 0:
            direction = goal_xy - current_xy
            direction_norm = np.linalg.norm(direction)

            # 如果距离太小，直接认为已经到达
            if direction_norm < 0.2:
                path = np.array([current_xy, goal_xy])
            else:
                # # 缩短步长（小步前进，比如 0.2 米）
                # step = direction / direction_norm * 0.2
                # safe_point = current_xy + step
                # # 生成一个短路径，next RRT 重新规划
                # path = np.array([current_xy, safe_point])
                rrt = RRTStarBidirectional(space, self.length_tree_edge,
                      tuple(current_xy.tolist()), tuple(goal_xy.tolist()),
                      self.max_samples * 10, self.smallest_edge, self.prc)
                path = rrt.rrt_search()
                if path is None or len(path) == 0:
                    print("RRT failed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    path = np.array([current_xy])


        self.path_xy = np.array(path)
        self.path_idx = 0

    # ------------------------------------------------
    # waypoint logic
    # ------------------------------------------------
    def next_waypoint(self, current_xy):
        wp = self.path_xy[self.path_idx]

        if np.linalg.norm(current_xy - wp) < self.step_tol:
            if self.path_idx < len(self.path_xy) - 1:
                self.path_idx += 1
                wp = self.path_xy[self.path_idx]
            else:
                if self._gate_stage < 2:
                    self._gate_stage += 1
                    self.path_idx = 0
                else:
                    self.current_gate_idx += 1
                    self._gate_stage = 0
                    self.path_idx = 0
                    
                    if self.current_gate_idx >= self.gates_count:
                        self._finished = True
                        return current_xy.astype(np.float32)

                    if self.current_gate_idx >= self.gates_count:
                        self._finished = True
                        
                    if self.current_gate_idx ==2:
                        self.post_dist = 0.3

        return wp

    # ------------------------------------------------
    # control
    # ------------------------------------------------
    def compute_control(self, obs, info=None):
        pos = np.array(obs["pos"], dtype=float)
        xy = pos[:2]
        
        # 是否需要重新规划
        if self.check_replan_condition(obs):
            if self._gate_stage == 0:
                self.plan(obs, xy)
            else:
                # 直接飞向 gate 中心或后方点
                self.path_xy = np.array([xy,
                                      self._goal_xy(
                                          np.array(obs["gates_pos"][self.current_gate_idx]),
                                          np.array(obs["gates_quat"][self.current_gate_idx]))])
                self.path_idx = 0

        wp_xy = self.next_waypoint(xy)
        # if wp_xy is None:
        #     self.plan(obs, xy)
        #     wp_xy = self.path_xy[0]

        action = np.zeros(13, dtype=np.float32)
        action[0] = wp_xy[0]
        action[1] = wp_xy[1]
        action[2] = self.target_z

        return action

    # ------------------------------------------------
    def step_callback(self, *args):
        return self._finished

    def episode_callback(self):
        self.current_gate_idx = 0
        self._gate_stage = 0
        self.path_xy = None
        self.path_idx = 0
        self._finished = False
