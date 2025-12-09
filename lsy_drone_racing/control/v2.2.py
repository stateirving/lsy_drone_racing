from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append("/home/miao/repos/lsy_drone_racing")
from rrt_algorithms.rrt.rrt_star import RRTStar
from rrt_algorithms.rrt.rrt_star_bid import RRTStarBidirectional
from rrt_algorithms.rrt.rrt import RRT
from rrt_algorithms.search_space.search_space import SearchSpace
from lsy_drone_racing.control.controller import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class RRTStateController(Controller):
    """降级版控制器：只做路径规划，输出状态控制 action = pos_des"""

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
        self._step_tolerance = 0.15
        self._pre_dist = 0.4
        self._post_dist = 0.4
        self.gate_width = 0.4
        self.gate_height = 1.2
        self.edge_radius = 0.15
        self.obstacle_radius = 0.1
        self.obstacle_height = 1.52
        self.length_tree_edge = 0.4
        self.smallest_edge = 0.01
        self.max_samples = 5000
        self.prc = 0.1

        # 路径状态
        self.path = None
        self.path_points_idx = 0

        # 起飞/额外点逻辑
        # self.take_off = True
        self.take_off = False
        self.going_to_extra = False
        self.extra_point = np.array([-1.5, -1.0, 0.7], dtype=float)

        self._finished = False

    # ----------------- 工具函数 -----------------
    def _quat_to_yaw(self, quat: NDArray[np.floating]) -> float:
        x, y, z, w = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def _quat_to_forward(self, quat: NDArray[np.floating]) -> NDArray[np.floating]:
        yaw = self._quat_to_yaw(quat)
        return np.array([np.cos(yaw), np.sin(yaw), 0.0])

    # ----------------- 障碍物生成 -----------------
    def _generate_obstacles(self, obs: dict, current_gate_idx: int):
        rrt_space = SearchSpace(self.pos_limit)
        
        # 门障碍物
        for idx, gpos in enumerate(obs["gates_pos"]):
            # if idx == current_gate_idx:
            #     continue
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
        
        # 生成整体门面长方体障碍：沿 yaw 旋转的完整 "板子"
        # for idx, gpos in enumerate(obs["gates_pos"]):
        #     # if idx == current_gate_idx:
        #     #     continue

        #     cx, cy, cz = np.array(gpos)
        #     yaw = self._quat_to_yaw(obs["gates_quat"][idx])
        #     width = self.gate_width
        #     thickness = 0.1   # 你门厚度的定义
        #     zmin = 0.0
        #     zmax = 2.0

        #     # 门的局部坐标四个角点（水平面）
        #     # 这里 width 是横向，thickness 是竖向（厚度）
        #     corners_local = [
        #         (-width/2, -thickness/2),
        #         (-width/2,  thickness/2),
        #         ( width/2, -thickness/2),
        #         ( width/2,  thickness/2),
        #     ]

        #     # 将局部角点旋转到全局坐标
        #     corners_world = []
        #     for lx, ly in corners_local:
        #         x = cx + lx * np.cos(yaw) - ly * np.sin(yaw)
        #         y = cy + lx * np.sin(yaw) + ly * np.cos(yaw)
        #         corners_world.append((x, y))

        #     # 生成 8 个顶点（上层 + 下层）
        #     points_3d = []
        #     for x, y in corners_world:
        #         points_3d.append((x, y, zmin))
        #         points_3d.append((x, y, zmax))
        #     points_3d = np.array(points_3d)

        #     # 求 bounding box（因为 R-tree 只能存 AABB）
        #     xmin, ymin, zmin = points_3d.min(axis=0)
        #     xmax, ymax, zmax = points_3d.max(axis=0)

        #     ob = (xmin, ymin, zmin, xmax, ymax, zmax)
        #     rrt_space.obs.insert(0, ob, ob)
        
        # 障碍物
        for top_pos in obs["obstacles_pos"]:
            x, y, z_top = np.array(top_pos)
            z_bottom = z_top - self.obstacle_height
            ob = np.append([x - self.obstacle_radius, y - self.obstacle_radius, z_bottom],
                           [x + self.obstacle_radius, y + self.obstacle_radius, z_top])
            rrt_space.obs.insert(0, tuple(ob), tuple(ob))

        return rrt_space

    # ----------------- 生成目标点 -----------------
    def generate_goal_position(self, gate_pos, gate_quat):
        fwd = self._quat_to_forward(gate_quat)
        if self._gate_stage == 0:
            return gate_pos - fwd * self._pre_dist
        elif self._gate_stage == 1:
            return gate_pos
        else:
            return gate_pos + fwd * self._post_dist

    # ----------------- replanning 条件 -----------------
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

    # ----------------- RRT 规划 -----------------
    def plan_rrt_path(self, obs: dict, current_pos):
        gate_pos = np.array(obs["gates_pos"][self.current_gate_idx])
        gate_quat = np.array(obs["gates_quat"][self.current_gate_idx])

        goal_pos = self.generate_goal_position(gate_pos, gate_quat)
        rrt_space = self._generate_obstacles(obs, self.current_gate_idx)

        rrt = RRTStarBidirectional(rrt_space, self.length_tree_edge,
                      tuple(current_pos.tolist()), tuple(goal_pos.tolist()),
                      self.max_samples, self.smallest_edge, self.prc)

        path = rrt.rrt_search()

        if path is None or len(path) == 0:
            # 退化为直线
            direction = goal_pos - current_pos
            direction_norm = np.linalg.norm(direction)

            # 如果距离太小，直接认为已经到达
            if direction_norm < 0.2:
                path = np.array([current_pos, goal_pos])
            else:
                # # 缩短步长（小步前进，比如 0.2 米）
                # step = direction / direction_norm * 0.2
                # safe_point = current_pos + step
                # # 生成一个短路径，next RRT 重新规划
                # path = np.array([current_pos, safe_point])
                rrt = RRTStar(rrt_space, self.length_tree_edge,
                      tuple(current_pos.tolist()), tuple(goal_pos.tolist()),
                      self.max_samples * 6, self.smallest_edge, self.prc)
                path = rrt.rrt_search()

        self.path = np.array(path)
        self.path_points_idx = 0

    # ----------------- waypoint 逻辑 -----------------
    def get_next_waypoint(self, current_pos):

        # 起飞高度控制
        if self.take_off:
            target = np.array([current_pos[0], current_pos[1], 0.7])
            if np.linalg.norm(current_pos - target) < self._step_tolerance:
                self.take_off = False
            return target.astype(np.float32)

        # 额外点
        if self.going_to_extra:
            if np.linalg.norm(current_pos - self.extra_point) < self._step_tolerance:
                self.going_to_extra = False
                self._post_dist = 0.4
            return self.extra_point.astype(np.float32)

        # 无 path
        if self.path is None or len(self.path) == 0:
            return current_pos.astype(np.float32)

        wp = self.path[self.path_points_idx]

        if np.linalg.norm(current_pos - wp) < self._step_tolerance:
            if self.path_points_idx < len(self.path) - 1:
                self.path_points_idx += 1
                wp = self.path[self.path_points_idx]
            else:
                # 进入下一阶段
                if self._gate_stage < 2:
                    self._gate_stage += 1
                    self.path_points_idx = 0
                else:
                    # 完成当前 gate
                    self.current_gate_idx += 1
                    if self.current_gate_idx >= self.gates_count:
                        self._finished = True
                        return current_pos.astype(np.float32)

                    self._gate_stage = 0
                    self.path_points_idx = 0

                    # gate 2 之后参数变化
                    if self.current_gate_idx == 2:
                        # self._step_tolerance = 0.15
                        self._post_dist = 0.2
                        

                    if self.current_gate_idx == 3:
                        self.going_to_extra = True
                        return self.extra_point.astype(np.float32)

        return wp

    # ----------------- 控制输出 -----------------
    def compute_control(self, obs, info=None) -> NDArray[np.floating]:
        current_pos = np.array(obs["pos"], dtype=float)

        # 是否需要重新规划
        if self.check_replan_condition(obs) and (not self.going_to_extra):
            if self._gate_stage == 0:
                self.plan_rrt_path(obs, current_pos)
            else:
                # 直接飞向 gate 中心或后方点
                self.path = np.array([current_pos,
                                      self.generate_goal_position(
                                          np.array(obs["gates_pos"][self.current_gate_idx]),
                                          np.array(obs["gates_quat"][self.current_gate_idx]))])
                self.path_points_idx = 0

        # 获取下一 waypoint
        next_wp = self.get_next_waypoint(current_pos)

        # ---- 这里是最核心的降级：只有状态控制 ----
        action = np.zeros(13, dtype=np.float32)
        action[0:3] = next_wp.astype(np.float32)
        return action

    # ----------------- callback -----------------
    def step_callback(self, action, obs, reward, terminated, truncated, info):
        return self._finished

    def episode_callback(self):
        self._finished = False
        self.path = None
        self.path_points_idx = 0
        self.current_gate_idx = 0
        self._gate_stage = 0
        self.take_off = True
        self.going_to_extra = False
