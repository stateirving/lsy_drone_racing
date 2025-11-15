"""Sequential Dynamic Waypoint Controller fully using obs for DroneRacing-v0 with gate orientation based extension.
"""

from __future__ import annotations
import os
import sys
import uuid
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from typing import TYPE_CHECKING

from lsy_drone_racing.control.controller import Controller

# --- External dependency: PythonRobotics (RRT path planner) ---
from rrt_algorithms.rrt.rrt_star import RRTStar
from rrt_algorithms.search_space.search_space import SearchSpace

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ObsBasedWaypointController(Controller):
    """Follow gates sequentially using obs, with three-point gate traversal and obstacle-aware RRT."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        
        # environment parameters
        self._freq = config.env.freq
        self.gates_count = 4
        self.pos_limit = np.array([[-2.5, 2.5], [-1.5, 1.5], [-1e-3, 2.0]])   # search space

        # store gate and obstacle info
        self.previous_gates = None
        self.previous_obstacles = None
        self.previous_gate_idx = None
        self.previous_gate_stage = None

        # RRT parameters
        self.length_tree_edge = 0.05
        self.smallest_edge = 0.01
        self.max_samples = 5000
        self.prc = 0.3

        # state machine
        self._finished = False
        self.current_gate_idx = 0
        self._gate_stage = 0
        
        # gate extension parameters
        self._step_tolerance = 0.20
        self._pre_dist = 0.40
        self._post_dist = 0.6
        
        # path planning
        self.path = None
        self.path_points_idx = 0

        # visualize
        self._vis_fig = None
        self._vis_ax = None
        
        # gate geometry parameters
        self.gate_width = 0.4
        self.gate_height = 1.2
        self.edge_radius = 0.05
        
        # obstacle geometry parameters
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
        
        # generate gate obstacles
        for idx, gpos in enumerate(obs["gates_pos"]):
            if idx == current_gate_idx:
                continue

            cx, cy, cz = np.array(gpos)
            yaw = self._quat_to_yaw(obs["gates_quat"][idx])

            half_w = self.gate_width / 2.0
            dz = self.gate_height / 2.0
            corners = [(-half_w, -half_w), (-half_w, half_w),
                       (half_w, -half_w), (half_w, half_w)]

            for lx, ly in corners:
                x = cx + lx * np.cos(yaw) - ly * np.sin(yaw)
                y = cy + lx * np.sin(yaw) + ly * np.cos(yaw)
                z_bottom = cz - dz
                z_top = cz + dz
                min_corner = np.array([x - self.edge_radius, y - self.edge_radius, z_bottom])
                max_corner = np.array([x + self.edge_radius, y + self.edge_radius, z_top])
                ob = tuple(np.append(min_corner, max_corner))
                rrt_search_space.obs.insert(uuid.uuid4().int, ob, ob)

        # generate environment obstacles
        for top_pos in obs["obstacles_pos"]:
            x, y, z_top = np.array(top_pos)
            z_bottom = z_top - self.obstacle_height
            min_corner = np.array([x - self.obstacle_radius, y - self.obstacle_radius, z_bottom])
            max_corner = np.array([x + self.obstacle_radius, y + self.obstacle_radius, z_top])
            ob = tuple(np.append(min_corner, max_corner))
            rrt_search_space.obs.insert(uuid.uuid4().int, ob, ob)

        return rrt_search_space
    
    def check_replan_condition(self, obs: dict) -> bool:
        change_detected = (
            not np.array_equal(obs["gates_pos"], self.previous_gates) or
            not np.array_equal(obs["obstacles_pos"], self.previous_obstacles) or
            self.current_gate_idx != self.previous_gate_idx or
            self._gate_stage != self.previous_gate_stage
        )

        if change_detected:
            self.previous_gates = obs["gates_pos"]
            self.previous_obstacles = obs["obstacles_pos"]
            self.previous_gate_idx = self.current_gate_idx
            self.previous_gate_stage = self._gate_stage

        return change_detected
    
    def generate_goal_position(self, gate_pos, gate_quat):
        gate_forward = self._quat_to_forward(gate_quat)

        if self._gate_stage == 0:
            return gate_pos - gate_forward * self._pre_dist
        elif self._gate_stage == 1:
            return gate_pos
        else:
            return gate_pos + gate_forward * self._post_dist
    
    # RRT path planning
    def RRT_path_planning(self, obs, start, goal):
        rrt_search_space = self._generate_obstacles_except_current(obs, self.current_gate_idx)
        
        rrt = RRTStar(
            rrt_search_space,
            self.length_tree_edge,
            tuple(start.tolist()), tuple(goal.tolist()),
            self.max_samples,
            self.smallest_edge,
            self.prc
        )

        path = rrt.rrt_search()
        if path is None or len(path) == 0:
            path = np.linspace(start, goal, 5)

        self.visualize_rrt_path(start, goal, path, rrt_search_space)
        return path
        
    # ========== FIXED SINGLE-WINDOW VISUALIZATION ==========
    def visualize_rrt_path(self, start, goal, path, rrt_search_space):

        # --- create persistent window ---
        if self._vis_fig is None:
            plt.ion()
            self._vis_fig = plt.figure(figsize=(8, 6))
            self._vis_ax = self._vis_fig.add_subplot(111, projection="3d")

        ax = self._vis_ax
        ax.clear()

        # bounds
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.pos_limit
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)

        # obstacles (correct reading through .obs.objects)
        for key, ob_pair in rrt_search_space.obs.objects.items():
            ob = ob_pair[0]
            x1, y1, z1, x2, y2, z2 = ob
            ax.bar3d(x1, y1, z1, x2-x1, y2-y1, z2-z1, alpha=0.2)

        # path
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], linewidth=2)

        # start/goal
        ax.scatter(*start, s=40)
        ax.scatter(*goal, s=40)

        ax.set_title("RRT Path Planning Visualization")

        plt.pause(0.001)

    # main control
    def compute_control(self, obs, info=None):

        if self.current_gate_idx >= self.gates_count:
            self._finished = True
            return np.zeros(13, dtype=np.float32)

        current_pos = np.array(obs["pos"], dtype=float)
        gate_pos = np.array(obs["gates_pos"][self.current_gate_idx], dtype=float)
        gate_quat = np.array(obs["gates_quat"][self.current_gate_idx], dtype=float)
        
        # trigger replan
        if self.check_replan_condition(obs):
            goal_pos = self.generate_goal_position(gate_pos, gate_quat)
            self.path = np.array(self.RRT_path_planning(obs, current_pos, goal_pos))
            self.path_points_idx = 0

        # waypoint progression
        if np.linalg.norm(current_pos - self.path[self.path_points_idx]) < self._step_tolerance:
            if self.path_points_idx < len(self.path) - 1:
                self.path_points_idx += 1
            else:
                if self._gate_stage < 2:
                    self._gate_stage += 1
                    self.path_points_idx = 0
                else:
                    self.current_gate_idx += 1
                    if self.current_gate_idx >= self.gates_count:
                        self._finished = True
                        return np.zeros(13, dtype=np.float32)

                    self._gate_stage = 0
                    self.path_points_idx = 0

                    if self.current_gate_idx >= 2:
                        self.length_tree_edge = 0.03
            
        action = np.zeros(13, dtype=np.float32)
        action[0:3] = self.path[self.path_points_idx].astype(np.float32)
        return action

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        return self._finished

    def episode_callback(self):
        self._finished = False
        self.current_gate_idx = 0
        self._gate_stage = 0
