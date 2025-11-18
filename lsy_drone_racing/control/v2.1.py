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


from rrt_algorithms.rrt.rrt import RRT
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

        # 额外点逻辑
        self._going_extra = False
        self._extra_point = np.array([-1.5, -1.0, 0.7], dtype=float)
        
        # gete geometry parameters
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
        rrt_obstacles = []
        rrt_search_space = SearchSpace(self.pos_limit)
        
        # generate gate obstacles
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
                min_corner = np.array([x - self.edge_radius, y - self.edge_radius, z_bottom])
                max_corner = np.array([x + self.edge_radius, y + self.edge_radius, z_top])
                ob = np.append(min_corner, max_corner)
                rrt_obstacles.append(tuple(ob))
                rrt_search_space.obs.insert(uuid.uuid4().int, tuple(ob), tuple(ob))

        # generate environment obstacles
        for top_pos in obs["obstacles_pos"]:
            x, y, z_top = np.array(top_pos)
            z_bottom = z_top - self.obstacle_height
            min_corner = np.array([x - self.obstacle_radius, y - self.obstacle_radius, z_bottom])
            max_corner = np.array([x + self.obstacle_radius, y + self.obstacle_radius, z_top])
            ob = np.append(min_corner, max_corner)
            rrt_obstacles.append(tuple(ob))
            rrt_search_space.obs.insert(uuid.uuid4().int, tuple(ob), tuple(ob))
            
        return rrt_search_space
    
    # check if gate or obstacle positions have changed by comparing with previous observations
    def check_replan_condition(self, obs: dict) -> bool:
        change_detected = False
        
        current_gates = obs["gates_pos"]
        current_obstacles = obs["obstacles_pos"]
        
        if (not np.array_equal(current_gates, self.previous_gates)
            or not np.array_equal(current_obstacles, self.previous_obstacles)
            or self.current_gate_idx != self.previous_gate_idx 
            or self._gate_stage != self.previous_gate_stage ):
            change_detected = True
            self.previous_gates = current_gates
            self.previous_obstacles = current_obstacles
            self.previous_gate_idx = self.current_gate_idx
            self.previous_gate_stage = self._gate_stage
            
        return change_detected
    
    def generate_goal_position(self, gate_pos: NDArray[np.floating], gate_quat: NDArray[np.floating]) -> NDArray[np.floating]:
        gate_forward = self._quat_to_forward(gate_quat)
        match self._gate_stage:
            case 0:
                goal_pos = gate_pos - gate_forward * self._pre_dist
            case 1:
                goal_pos = gate_pos
            case 2:
                goal_pos = gate_pos + gate_forward * self._post_dist
                
        return goal_pos 
    
    # RRT path planning function
    def RRT_path_planning(self, obs: dict, start: NDArray[np.floating], goal: NDArray[np.floating]) -> NDArray[np.floating]:
        rrt_search_space = self._generate_obstacles_except_current(obs, self.current_gate_idx)
        
        rrt = RRTStar(rrt_search_space, self.length_tree_edge, tuple(start.tolist()), tuple(goal.tolist()),
                  self.max_samples, self.smallest_edge, self.prc)
        path = rrt.rrt_search()
        if path is None or len(path) == 0:
            # fallback: straight line path if RRT fails
            path = np.linspace(start, goal, 5)
        return path
        

    def compute_control(self, obs: dict, info: dict = None) -> NDArray[np.float32]:

        if self.current_gate_idx >= self.gates_count:
            self._finished = True
            return np.zeros(13, dtype=np.float32)

        current_pos = np.array(obs["pos"], dtype=float)
        gate_pos = np.array(obs["gates_pos"][self.current_gate_idx], dtype=float)
        gate_quat = np.array(obs["gates_quat"][self.current_gate_idx], dtype=float)
        
        # Check for gate or obstacle changes
        if self.check_replan_condition(obs):
            goal_pos = self.generate_goal_position(gate_pos, gate_quat)
            self.path = np.array(self.RRT_path_planning(obs, current_pos, goal_pos))
            self.path_points_idx = 0

        if np.linalg.norm(current_pos - self.path[self.path_points_idx]) < self._step_tolerance:
            if self.path_points_idx < len(self.path) - 1:
                self.path_points_idx += 1
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
                        return np.zeros(13, dtype=np.float32)
                    self._gate_stage = 0
                    self.path_points_idx = 0
                    if self.current_gate_idx >= 2:
                        self.length_tree_edge = 0.03  # adjust RRT parameter after gate 2
            
        
        action = np.zeros(13, dtype=np.float32)
        action[0:3] = self.path[self.path_points_idx].astype(np.float32)
        return action

    def step_callback(self, action: NDArray[np.float32], obs: dict, reward: float,
                      terminated: bool, truncated: bool, info: dict) -> bool:
        return self._finished

    def episode_callback(self):
        self._finished = False
        self.current_gate_idx = 0
        self._gate_stage = 0
        self._going_extra = False
        self.rrt_search_space = SearchSpace(self.pos_limit)
