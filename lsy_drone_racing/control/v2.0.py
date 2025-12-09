"""Sequential Dynamic Waypoint Controller fully using obs for DroneRacing-v0 with gate orientation based extension.
"""

from __future__ import annotations
import os
import sys
import uuid
import numpy as np
from typing import TYPE_CHECKING

from lsy_drone_racing.control.controller import Controller

# --- External dependency: PythonRobotics (RRT path planner) ---

sys.path.append("/home/miao/repos/lsy_drone_racing")
from rrt_algorithms.rrt.rrt import RRT
from rrt_algorithms.search_space.search_space import SearchSpace

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ObsBasedWaypointController(Controller):
    """Follow gates sequentially using obs, with three-point gate traversal and obstacle-aware RRT."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        # 控制参数
        self._tolerance = 0.14
        self._pre_dist = 0.2
        self._post_dist = 0.4
        self._stage_threshold = 0.10

        # RRT 参数
        self.length_tree_age = 0.2
        self.smallest_edge = 0.05
        self.max_samples = 1024
        self.prc = 0.3

        # 搜索空间
        self.pos_limit = np.array([[-2.5, 2.5], [-1.5, 1.5], [-1e-3, 2.0]])
        self.rrt_search_space = SearchSpace(self.pos_limit)

        # 状态机
        self._finished = False
        self.current_gate_idx = None
        self._gate_stage = 0

        self.rrt_obstacles = []

        # 额外点逻辑
        self._going_extra = False
        self._extra_point = np.array([-1.5, -1.0, 0.7], dtype=float)

    def _quat_to_yaw(self, quat: NDArray[np.floating]) -> float:
        x, y, z, w = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def _quat_to_forward(self, quat: NDArray[np.floating]) -> NDArray[np.floating]:
        yaw = self._quat_to_yaw(quat)
        return np.array([np.cos(yaw), np.sin(yaw), 0.0])

    def _generate_obstacles_except_current(self, obs: dict, current_gate_idx: int):
        self.rrt_obstacles.clear()
        self.rrt_search_space = SearchSpace(self.pos_limit)

        gate_width = 0.4
        gate_height = 1.2
        edge_radius = 0.05

        for idx, gpos in enumerate(obs["gates_pos"]):
            if idx == current_gate_idx:
                continue
            cx, cy, cz = np.array(gpos)
            yaw = self._quat_to_yaw(obs["gates_quat"][idx])
            half_w = gate_width / 2.0
            dz = gate_height / 2.0
            corners = [(-half_w, -half_w), (-half_w, half_w), (half_w, -half_w), (half_w, half_w)]
            for lx, ly in corners:
                x = cx + lx * np.cos(yaw) - ly * np.sin(yaw)
                y = cy + lx * np.sin(yaw) + ly * np.cos(yaw)
                z_bottom = cz - dz
                z_top = cz + dz
                min_corner = np.array([x - edge_radius, y - edge_radius, z_bottom])
                max_corner = np.array([x + edge_radius, y + edge_radius, z_top])
                ob = np.append(min_corner, max_corner)
                self.rrt_obstacles.append(tuple(ob))
                self.rrt_search_space.obs.insert(uuid.uuid4().int, tuple(ob), tuple(ob))

        env_radius = 0.1
        env_height = 1.52
        for top_pos in obs["obstacles_pos"]:
            x, y, z_top = np.array(top_pos)
            z_bottom = z_top - env_height
            min_corner = np.array([x - env_radius, y - env_radius, z_bottom])
            max_corner = np.array([x + env_radius, y + env_radius, z_top])
            ob = np.append(min_corner, max_corner)
            self.rrt_obstacles.append(tuple(ob))
            self.rrt_search_space.obs.insert(uuid.uuid4().int, tuple(ob), tuple(ob))

    def compute_control(self, obs: dict, info: dict = None) -> NDArray[np.float32]:
        if self.current_gate_idx is None:
            try:
                self.current_gate_idx = int(obs.get("target_gate", 0))
            except Exception:
                self.current_gate_idx = 0

        gates_count = len(obs["gates_pos"])
        if self.current_gate_idx >= gates_count:
            self._finished = True
            return np.zeros(13, dtype=np.float32)

        current_pos = np.array(obs["pos"], dtype=float)

        if self._going_extra:
            desired = self._extra_point
            if np.linalg.norm(current_pos - self._extra_point) < self._tolerance:
                self._going_extra = False
                self._gate_stage = 0
                self.current_gate_idx += 1
                if self.current_gate_idx >= gates_count:
                    self._finished = True
                    return np.zeros(13, dtype=np.float32)
                self._generate_obstacles_except_current(obs, self.current_gate_idx)
                gate_pos = np.array(obs["gates_pos"][self.current_gate_idx], dtype=float)
                gate_quat = np.array(obs["gates_quat"][self.current_gate_idx], dtype=float)
                yaw = self._quat_to_yaw(gate_quat)
                forward = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)
                desired = gate_pos - self._pre_dist * forward
        else:
            self._generate_obstacles_except_current(obs, self.current_gate_idx)
            gate_pos = np.array(obs["gates_pos"][self.current_gate_idx], dtype=float)
            gate_quat = np.array(obs["gates_quat"][self.current_gate_idx], dtype=float)
            yaw = self._quat_to_yaw(gate_quat)
            forward = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)

            pre_pt = gate_pos - self._pre_dist * forward
            mid_pt = gate_pos.copy()
            # 如果是 gate idx=2，post_pt 不用原来的，直接固定值
            if self.current_gate_idx == 2:
                post_pt = gate_pos - 0.2 * forward  # post_pt 仅作为中间标记，实际直接去 extra_point
            else:
                post_pt = gate_pos + self._post_dist * forward

            if self._gate_stage == 0:
                desired = pre_pt
                if np.linalg.norm(current_pos - pre_pt) < self._tolerance:
                    self._gate_stage = 1
            elif self._gate_stage == 1:
                desired = mid_pt
                if np.linalg.norm(current_pos - mid_pt) < self._stage_threshold:
                    if self.current_gate_idx == 2:
                        self._going_extra = True
                        desired = self._extra_point
                    else:
                        self._gate_stage = 2
            else:  # stage == 2
                desired = post_pt
                if np.linalg.norm(current_pos - post_pt) < self._tolerance:
                    self._gate_stage = 0
                    self.current_gate_idx += 1
                    if self.current_gate_idx >= gates_count:
                        self._finished = True
                        return np.zeros(13, dtype=np.float32)
                    self._generate_obstacles_except_current(obs, self.current_gate_idx)
                    gate_pos = np.array(obs["gates_pos"][self.current_gate_idx], dtype=float)
                    gate_quat = np.array(obs["gates_quat"][self.current_gate_idx], dtype=float)
                    yaw = self._quat_to_yaw(gate_quat)
                    forward = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)
                    desired = gate_pos - self._pre_dist * forward

        # RRT path planning
        start = tuple(current_pos.tolist())
        goal = tuple(desired.tolist())
        rrt = RRT(self.rrt_search_space, self.length_tree_age, start, goal,
                  self.max_samples, self.smallest_edge, self.prc)
        path = rrt.rrt_search()

        if path is not None and len(path) > 1:
            next_pt = np.array(path[1], dtype=float)
        else:
            next_pt = desired

        action = np.zeros(13, dtype=np.float32)
        action[0:3] = next_pt.astype(np.float32)
        return action

    def step_callback(self, action: NDArray[np.float32], obs: dict, reward: float,
                      terminated: bool, truncated: bool, info: dict) -> bool:
        return self._finished

    def episode_callback(self):
        self._finished = False
        self.current_gate_idx = None
        self._gate_stage = 0
        self._going_extra = False
        self.rrt_obstacles.clear()
        self.rrt_search_space = SearchSpace(self.pos_limit)
