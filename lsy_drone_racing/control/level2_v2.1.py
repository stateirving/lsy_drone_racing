"""Sequential Dynamic Waypoint Controller fully using obs for DroneRacing-v0 with gate orientation based extension."""

from __future__ import annotations
import os
import sys
import uuid
import numpy as np
from typing import TYPE_CHECKING

from lsy_drone_racing.control.controller import Controller

# --- External dependency: PythonRobotics (RRT path planner) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
external_path = os.path.join(current_dir, "../../external/rrt-algorithms")
sys.path.append(os.path.abspath(external_path))

from rrt_algorithms.rrt.rrt import RRT
from rrt_algorithms.search_space.search_space import SearchSpace

if TYPE_CHECKING:
    from numpy.typing import NDArray

class ObsBasedWaypointController(Controller):
    """Follow gates sequentially using obs, with gate orientation based extension and pre/mid/post logic."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        
        # control parameters
        self._tolerance = 0.14       

        # state variables
        self._finished = False
        self.current_target_gate = 0  # index of the current target gate
        self._gate_stage = 0  # 0=pre,1=mid,2=post
        self.gate_quat = 0
        
        # RRT planner parameters
        self.pos_limit = np.array([[-2.5, 2.5], [-1.5, 1.5], [-1e-3, 2.0]])
        self.rrt_search_space = SearchSpace(self.pos_limit)
        self.rrt_obstacles = []  # list of obstacles for RRT
        self.length_tree_age = 0.3
        self.smallest_edge = 0.01
        self.max_samples = 1024
        self.prc = 0.1

    def _quat_to_yaw(self, quat: NDArray[np.floating]) -> float:
        """Convert quaternion to yaw angle."""
        x, y, z, w = quat
        siny_cosp = 2*(w*z + x*y)
        cosy_cosp = 1 - 2*(y*y + z*z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def _generate_obstacles(self, obs: dict, current_gate_idx: int, edge_radius=0.05):
        """Generate rectangular edge obstacles for all gates except the one being traversed."""
        self.rrt_obstacles.clear()
        self.rrt_search_space = SearchSpace(self.pos_limit)
        for idx, gate_pos in enumerate(obs["gates_pos"]):
            if idx == current_gate_idx:
                continue  # skip current gate
            cx, cy, cz = gate_pos
            width = 0.4  # gate width (x/y)
            height = 0.9  # half height
            local_edges = [(-width/2, -width/2), (-width/2, width/2), 
                           (width/2, -width/2), (width/2, width/2)]
            yaw = self._quat_to_yaw(obs["gates_quat"][idx])
            dz = height / 2
            for lx, ly in local_edges:
                x = cx + lx * np.cos(yaw) - ly * np.sin(yaw)
                y = cy + lx * np.sin(yaw) + ly * np.cos(yaw)
                z = cz
                min_corner = np.array([x - edge_radius, y - edge_radius, z - dz])
                max_corner = np.array([x + edge_radius, y + edge_radius, z + dz])
                edge_obstacle = np.append(min_corner, max_corner)
                self.rrt_obstacles.append(tuple(edge_obstacle))
                self.rrt_search_space.obs.insert(uuid.uuid4().int, tuple(edge_obstacle), tuple(edge_obstacle))

        # Environment obstacles: obs["obstacles_pos"] gives position of top marker
        # obstacle height (top measured from ground)
        env_radius = 0.4
        env_height = 1.52
        for top_pos in obs["obstacles_pos"]:
            x, y, z_top = np.array(top_pos)
            z_bottom = z_top - env_height
            min_corner = np.array([x - env_radius, y - env_radius, z_bottom])
            max_corner = np.array([x + env_radius, y + env_radius, z_top])
            ob = np.append(min_corner, max_corner)
            # print(ob)
            self.rrt_obstacles.append(tuple(ob))
            self.rrt_search_space.obs.insert(uuid.uuid4().int, tuple(ob), tuple(ob))

    def compute_control(self, obs: dict, info: dict = None) -> NDArray[np.float32]:

        gate_pos = obs["gates_pos"][self.current_target_gate]
        gate_quat = obs["gates_quat"][self.current_target_gate]
        current_pos = obs["pos"].copy()
        current_pos_tuple = tuple(current_pos)
        gate_pos_tuple = tuple(gate_pos)
        
        # Generate obstacles
        self._generate_obstacles(obs, self.current_target_gate)
        
        # Pre/mid/post points
        yaw = self._quat_to_yaw(gate_quat)
        dir_forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        pre_gate = gate_pos - 0.2 * dir_forward
        mid_gate = gate_pos
        post_gate = gate_pos + 0.4 * dir_forward
        
        if self._gate_stage == 0:
            target_pos = pre_gate
            if np.linalg.norm(current_pos - pre_gate) < 0.15:
                self._gate_stage = 1
        elif self._gate_stage == 1:
            target_pos = mid_gate
            if np.linalg.norm(current_pos - mid_gate) < 0.15:
                self._gate_stage = 2
        elif self._gate_stage == 2:
            target_pos = post_gate
            if np.linalg.norm(current_pos - post_gate) < 0.2:
                self._gate_stage = 0
                # 切换到下一个门
                self.current_target_gate += 1
                
        # Use RRT to plan (current_pos -> desired), but pass tuple inputs
        start = tuple(current_pos.tolist())
        goal = tuple(target_pos.tolist())

        # Build RRT and search
        rrt = RRT(self.rrt_search_space, self.length_tree_age, start, goal,
                  self.max_samples, self.smallest_edge, self.prc)
        path = rrt.rrt_search()

        # choose next small step from path if available (avoid large jumps)
        if path is not None and len(path) > 1:
            next_pt = np.array(path[1], dtype=float)
        else:
            next_pt = target_pos  # fallback

        # Build action (position control)
        action = np.zeros(13, dtype=np.float32)
        action[0:3] = next_pt.astype(np.float32)
        return action

    def step_callback(self, action: NDArray[np.float32], obs: dict, reward: float,
                     terminated: bool, truncated: bool, info: dict) -> bool:
        return self._finished

    def episode_callback(self):
        self._finished = False
        self._gate_stage = 0
        self.rrt_obstacles.clear()
