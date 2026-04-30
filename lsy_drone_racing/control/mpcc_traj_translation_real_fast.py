"""MPCC with trajectory translation
handle obstacles with integrated costs
but handle gates by computing contour error on translated trajectory
when approching the gates, shift the trajectory towards the gate offset

NOTE This set of parameters acchieves 48/100 success rate, with an average of 4.80s
NOTE This file is simplified version, core modules including cost definition are in "lsy_drone_racing/tools/mpcc_controller/fresssack_mpcc"
"""

from __future__ import annotations  # Python 3.10 type hints


from typing import *

import os
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, DM, cos, sin, vertcat, dot, norm_2, floor, if_else, exp, Function, power
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.tools.race_objects import Gate, Obstacle
from lsy_drone_racing.control.fresssack_controller import FresssackController
from lsy_drone_racing.tools.mpcc_controller.fresssack_mpcc import FresssackMPCC, FresssackDroneModel
from lsy_drone_racing.utils.utils import draw_line

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.publisher import Publisher
    from rclpy.publisher import MsgType
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from nav_msgs.msg import Path
    from tf2_ros import TransformBroadcaster
    from visualization_msgs.msg import Marker, MarkerArray
    from lsy_drone_racing.control.fresssack_controller import MultiArrayTx, MarkerArrayTx, CapsuleMarkerTx, MeshMarkerTx, PathTx, TFTx

    from transformations import quaternion_from_euler, euler_from_quaternion
    ROS_AVAILABLE = True
except:
    ROS_AVAILABLE = False
ROS_AVAILABLE = False

# region MPCC Params
solver_param_dict : Dict[str, Union[List[np.floating], np.floating]] = {
    'pos_bound': [np.array([-2.0, 2.0]), np.array([-2.0, 2.0]), np.array([0.0, 2.0])],
    'vel_bound': [-1.0, 5.0],
    'compile_path' : r"lsy_drone_racing/tools/mpcc_controller/our_mpcc.json",
    'traj_path' : r"lsy_drone_racing/planned_trajectories/traj_22_k.csv",

    'starting_pos' : [1.0, 1.5, 0.07],

    'gates' : [
        Gate(pos = np.array([0.45, -0.5, 0.56]), quat = np.array([0.,0.,0.92268986,0.38554308]),),
        Gate(pos = np.array([1.0, -1.05, 1.11]), quat = np.array([0.,0.,-0.3801884,0.92490906]),),
        Gate(pos = np.array([0.0, 1.0, 0.56]), quat = np.array([0.,0.,0.,1.]),),
        Gate(pos = np.array([-0.5, 0.0, 1.11]), quat = np.array([0.,0.,1.,0.]),),

    ],
    'obstacles' : [
        Obstacle(pos = np.array([1.0, 0.0, 1.4]), safe_radius = 0.15),
        Obstacle(pos = np.array([0.5, -1.0, 1.4]), safe_radius = 0.20),
        Obstacle(pos = np.array([0.0, 1.5, 1.4]), safe_radius = 0.01),
        Obstacle(pos = np.array([-0.5, 0.5, 1.4]), safe_radius = 0.01),
    ],


    'qp_solver' : 'FULL_CONDENSING_HPIPM', # # FULL_CONDENSING_QPOASES
    'hessian_approx' : 'GAUSS_NEWTON',
    'integrator_type' : 'ERK',
    'nlp_solver_type' : 'SQP_RTI',
    'tol' : 1e-5,
    'qp_solver_warm_start' : 1,
    'qp_solver_iter_max' : 20,
    'nlp_solver_max_iter' :50,

    'T_f' : 0.60,
    'N' : 40,

    'model_arc_length' : 0.05,
    'model_traj_length' : 12,
    'model_traj_end' : 9.0,

    'q_l' : 200,
    'q_l_peak' : [800, 800, 800, 1400],
    'q_c':100,
    'q_c_peak':[1000, 100, 1400, 0],
    'q_c_sigma1':[0.9, 0.6, 1.0, 0.2],
    'q_c_sigma2':[0.3, 0.4, 0.5, 0.1],
    'gate_interp_peak':[1.2, 1.2, 1.2, 1.4],
    'gate_interp_sigma1': [0.5, 0.5, 0.9, 0.5],
    'gate_interp_sigma2':[0.3, 0.8, 0.8, 2.0],
    'Q_w':1 * DM(np.eye(3)),
    'R_df':DM(np.diag([1,0.4,0.4,0.4])),
    'miu':1.0,
    'obst_w':10,
    'd_extend':0.15,
    'lb_vel': 0.99,
    'ub_vel':4.2,
}

our_mpcc : FresssackMPCC = FresssackMPCC(param_dict = solver_param_dict)


class MPCC(FresssackController):
    """Implementation of MPCC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict, env=None):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config, ros_tx_freq = 20)
        self.freq = config.env.freq
        self._tick = 0

        self.env = env
        self.init_gates(obs=obs,
                        gate_outer_size=[0.8, 0.8, 0.8, 0.8],   # capsule built in between inner & outer
                        thickness=[0.1, 0.1, 0.1, 0.1],     # thickness of capsule, gaussian cost
                        # thickness=[0.2, 0.2, 0.2, 0.2],     # thickness of capsule, gaussian cost
                        )
        # region Cylinder radius
        self.init_obstacles(obs=obs, 
                            obs_safe_radius=[0.25, 0.25, 0.3, 0.10])#[0.3, 0.3, 0.3, 0.3])
        

        # build model & create solver
        self.our_mpcc = our_mpcc
        self.our_mpcc.reset_solver()
        self.acados_ocp_solver = self.our_mpcc.solver
        
      
        # region Global params
        self.dt = self.our_mpcc.T_f / self.our_mpcc.N
        # self.gates_norm_list = [gate.norm_vec for gate in self.gates] # 4 * 3 = 12

        # initialize ros tx
        self.init_ros_tx()
        
        # initialize debug function for MPC
        cost_dict = self.our_mpcc.mpcc_cost_components()
        self.cost_func_debug = Function(
            'mpcc_cost_debug',
            [self.our_mpcc.model_syms.x, self.our_mpcc.model_syms.u, self.our_mpcc.model_syms.p],
            [cost_dict[k] for k in cost_dict],
            ['x', 'u', 'p'],
            list(cost_dict.keys()
            )  # ['total', 'cost_l', ..., 'miu_cost']    
        )

        # initialize
        # NOTE: EXP: update initial last theta
        self.last_theta, _ = self.our_mpcc.traj_tool.find_nearest_waypoint(self.our_mpcc.arc_trajectory, obs['pos'], 0.5)
        self.last_f_collective = 0.3
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.3
        self.config = config
        self.finished = False
        self.fail_counter = 1
        self.interp_weight = 1 / self.config.env.freq / self.dt

    
            

    # region Compute Control
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [t_des, r_des, p_des, y_des] as a numpy array.
        """
        need_gate_update, _ = self.update_gate_if_needed(obs)
        need_obs_update, _ = self.update_obstacle_if_needed(obs)
        
        q = obs["quat"]
        r = R.from_quat(q)
        # Convert to Euler angles in XYZ order
        rpy = r.as_euler("xyz", degrees=False)  # Set degrees=False for radians

        xcurrent = np.concatenate(
            (
                obs["pos"],
                obs["vel"],
                rpy,
                np.array([self.last_f_collective, self.last_f_cmd]),
                self.last_rpy_cmd,
                np.array([self.last_theta])
            )
        )

        x_result, u_result, p_result, success = self.our_mpcc.control_step(x = xcurrent,
                                                                        last_theta = self.last_theta,
                                                                        need_gate_update = need_gate_update,
                                                                        need_obs_update = need_obs_update,
                                                                        next_gate = obs["target_gate"],
                                                                        gates = self.gates,
                                                                        obstacles = self.obstacles)
        if not success:
            self.fail_counter += 1
            if self.fail_counter * self.dt > 0.5:
                self.finished = True
                print("Solver failure force quit")
        else:
            self.fail_counter = 1

        x_result = np.array(x_result)
        u_result = np.array(u_result)
        x_interp = np.array([
            np.interp(self.fail_counter * self.interp_weight, np.array([idx for idx in range(len(x_result))]),x_result[:, i])
            for i in range(x_result.shape[1])
        ])

        self.last_f_collective = x_interp[9]
        self.last_theta = x_interp[14]
        self.last_f_cmd = x_interp[10]
        self.last_rpy_cmd = x_interp[11:14]

        # For safety and stability of the solver
        if self.last_theta > solver_param_dict['model_traj_end']:
            self.finished = True
            print("Quit-finished")
        if self.our_mpcc.out_of_pos_bound(self.pos):
            self.finished = True
            print("Quit-flying out of safe area")
        # if self.our_mpcc.out_of_vel_bound(self.vel):
        #     self.finished = True
        #     print("Quit-out of safe velocity range")
            
        cmd = np.concatenate(
            (
                np.array([self.last_f_cmd]),
                self.last_rpy_cmd
            )
        )

        # Calculate costs every steps for tuning
        debug_costs : Dict[str, List[np.floating]] = {key : [] for key in self.mpc_costs_tx_keys}
        for i in range(self.our_mpcc.N):
            result = self.cost_func_debug(x_result[i], u_result[i], p_result[i])
            for idx, value in enumerate(result):
                debug_costs[self.mpc_costs_tx_keys[idx]].append(float(value))


        #region Visualization
        pos_traj = np.array([x_result[i][:3] for i in range(self.our_mpcc.N+1)])
        if self.need_ros_tx():
            # TODO: uncomment it if visualization is needed in rviz
            # self.mpcc_traj_tx.publish(
            #     raw_data = {
            #         'traj' : pos_traj,
            #         'frame_id' : 'map'
            #     }
            # )
            # TODO: uncomment it if you need to check the costs
            # self.mpc_costs_tx.publish(raw_data = debug_costs)
            pass
        if self.need_ros_tx(slow=True):
            pass

        if not hasattr(self, "trajectory_record"):
            self.trajectory_interp = self.our_mpcc.arc_trajectory(self.our_mpcc.arc_trajectory.x)
            self.trajectory_record = []
        self.trajectory_record.append(self.pos)
        if need_gate_update:
            update_mask = (self.our_mpcc.arc_trajectory.x > self.our_mpcc.gate_theta_list_offset[self.our_mpcc.traj_update_gate] - self.our_mpcc.gate_interp_sigma1[self.our_mpcc.traj_update_gate]) \
                        & (self.our_mpcc.arc_trajectory.x < self.our_mpcc.gate_theta_list_offset[self.our_mpcc.traj_update_gate] + self.our_mpcc.gate_interp_sigma2[self.our_mpcc.traj_update_gate])
            self.trajectory_interp[update_mask] = self.our_mpcc.arc_trajectory(self.our_mpcc.arc_trajectory.x[update_mask]) \
                    + np.tile(self.our_mpcc.curr_gate_offset, (sum(update_mask), 1)) * self.our_mpcc.gate_interp_list(self.our_mpcc.arc_trajectory.x[update_mask])[:, None]

        try:

            draw_line(self.env, self.our_mpcc.arc_trajectory(self.our_mpcc.arc_trajectory.x), self.hex2rgba("#ffffff0f")) # original trajectory
            # draw_line(self.env, self.arc_trajectory_offset(self.arc_trajectory_offset.x), rgba=self.hex2rgba("#2b2b2b7d")) # translated trajectory
            draw_line(self.env, self.trajectory_interp, rgba=self.hex2rgba("#ff9500da")) # interp trajectory
            draw_line(self.env, np.array(self.trajectory_record), rgba=self.hex2rgba("#2BFF00F9")) # recorded trajectory
            # draw_line(self.env, self.our_mpcc.arc_trajectory(self.our_mpcc.gate_theta_list_offset), rgba=self.hex2rgba("#F209FE8A")) # gate theta
            draw_line(self.env, np.stack([self.our_mpcc.arc_trajectory(self.our_mpcc.gate_theta_list_offset[self.our_mpcc.traj_update_gate]), self.our_mpcc.arc_trajectory(self.our_mpcc.gate_theta_list_offset[self.our_mpcc.traj_update_gate])+self.our_mpcc.curr_gate_offset]), rgba=self.hex2rgba("#002aff55"))
            # draw_line(self.env, np.stack([self.arc_trajectory_offset(self.last_theta), obs["pos"]]), rgba=self.hex2rgba("#002aff55"))
            draw_line(self.env, pos_traj[0:-1:5],rgba=self.hex2rgba("#ffff00a0"))
            
            
            # obstacles: plot a line from pos to the cylinder when dist < self.d_safe
            if self.our_mpcc.cylinder_list is not None:
                for x,y,r in self.our_mpcc.cylinder_list:
                    closest = np.array([x,y,self.pos[2]])
                    dist = np.linalg.norm(self.pos - closest) - r
                    if dist < self.our_mpcc.d_extend:
                        draw_line(self.env, np.stack([self.pos, closest]), rgba=np.array([0.8, 0.0, 1.0, 0.3]))
        except:
            pass

        return cmd

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter."""
        self.step_update(obs = obs)
        self.update_target_gate(obs = obs)
        self.ros_transmit(obs = obs)

        return self.finished

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0
    
    def init_ros_tx(self):
        self.mpc_costs_tx_keys = ['total',
                        'cost_l',
                        'C_l',
                        'e_l_cost',
                        'cost_c',
                        'C_c',
                        'e_c_cost',
                        'ang_cost',
                        'ctrl_cost',
                        'cost_obs',
                        'miu_cost',
                        'cost_g_c']
        if ROS_AVAILABLE and self.ros_tx:
            # self.mpcc_traj_tx = PathTx(
            #     node_name = 'mpcc_path_tx',
            #     topic_name = 'mpcc_traj',
            #     queue_size = 10
            # )
            # self.path_tx = PathTx(
            #     node_name = 'global_path_tx',
            #     topic_name = 'global_path',
            #     queue_size = 10
            # )
            # self.gate_tf_txs = [
            #     TFTx(node_name = 'gate_' + str(i) + '_TF_tx', topic_name = 'gate_' + str(i))
            #     for i in range(len(self.gates))
            # ]
            
            # self.obstacle_tf_txs = [
            #     TFTx(node_name = 'obstacle_' + str(i) + '_TF_tx', topic_name = 'obstacle_' + str(i))
            #     for i in range(len(self.obstacles))
            # ]
            # current_dir = os.path.dirname(os.path.abspath(__file__))

            # gate_mesh_file = os.path.join(current_dir, '..', 'ros', 'rviz','meshes', 'gate.dae')
            # gate_unobserved_mesh_file = os.path.join(current_dir, '..', 'ros', 'rviz','meshes', 'gate_not_sure.dae')
            # gate_passed_mesh_file = os.path.join(current_dir, '..', 'ros', 'rviz','meshes', 'gate_passed.dae')
            # self.gate_marker_txs = [
            #         MeshMarkerTx(
            #         node_name = 'gate_' + str(i) + '_marker_tx',
            #         topic_name = 'gate_' + str(i) + '_marker',
            #         mesh_path = [os.path.abspath(gate_unobserved_mesh_file),
            #                       os.path.abspath(gate_mesh_file),
            #                       os.path.abspath(gate_passed_mesh_file)],
            #         frame_id = 'gate_' + str(i),
            #         queue_size = 1
            #     )
            #     for i in range(len(self.gates))
            # ]

            # self.gate_marker_array_tx = MarkerArrayTx(
            #     node_name = 'gate_marker_array_tx',
            #     topic_name = 'gate_marker_array',
            #     queue_size = 5
            # )
            
            # obstacle_mesh_file = os.path.join(current_dir, '..', 'ros', 'rviz','meshes', 'obstacle.dae')
            # obstacle_unobserved_mesh_file = os.path.join(current_dir, '..', 'ros', 'rviz','meshes', 'obstacle_not_sure.dae')
            # self.obstacle_marker_txs = [
            #         MeshMarkerTx(
            #         node_name = 'obstacle_' + str(i) + '_marker_tx',
            #         topic_name = 'obstacle_' + str(i) + '_marker',
            #         mesh_path = [os.path.abspath(obstacle_unobserved_mesh_file),
            #                      os.path.abspath(obstacle_mesh_file)],
            #         frame_id = 'obstacle_' + str(i),
            #         queue_size = 1
            #     )
            #     for i in range(len(self.obstacles))
            # ]
            # self.obstacle_marker_array_tx = MarkerArrayTx(
            #     node_name = 'obstacle_marker_array_tx',
            #     topic_name = 'obstacle_marker_array',
            #     queue_size = 5
            # )

            # self.collision_capsule_tx =  CapsuleMarkerTx(
            #         node_name = 'collision_capsule' + '_tx',
            #         topic_name = 'collision_capsule',
            #         frame_id = 'map',
            #         queue_size = 20
            #     )            

            # self.capsule_marker_array_tx = MarkerArrayTx(
            #     node_name = 'capsule_marker_array_tx',
            #     topic_name = 'capsule_marker_array',
            #     queue_size = 5,
            # )
            
            self.mpc_costs_tx = MultiArrayTx(
                node_name = 'cost_debug_tx', 
                queue_size = 10,
                keys = self.mpc_costs_tx_keys,
                topic_prefix = 'mpcc_cost'
            )

    def ros_transmit(self, obs):
        # if self.need_ros_tx(): 
        #     self.path_tx.publish(
        #         raw_data = {
        #             'traj' : self.arc_trajectory(self.arc_trajectory.x),
        #             'frame_id' : 'map'
        #         }
        #     )
        #     for idx, tx in enumerate(self.gate_tf_txs):
        #         tx.publish(
        #             raw_data = 
        #             {
        #                 'pos' : self.gates[idx].pos,
        #                 'quat' : self.gates[idx]._quat,
        #                 'frame_id' : 'map'
        #             }
        #         )
        #     for idx, tx in enumerate(self.obstacle_tf_txs):
        #         tx.publish(
        #             raw_data = 
        #             {
        #                 'pos' : self.obstacles[idx].pos,
        #                 'quat' : [0,0,0,1],
        #                 'frame_id' : 'map'
        #             }
        #         )
        # if self.need_ros_tx(slow = True): 
        #     gate_marker_list : List[Marker] = []
        #     for idx, tx in enumerate(self.gate_marker_txs):
        #         marker : Marker = None
        #         if self.next_gate > idx:
        #             marker = tx.process_data(
        #                 {
        #                     'idx' : 2
        #                 }
        #             )
        #         elif self.gates_visited[idx]:
        #             marker = tx.process_data(
        #                 {
        #                     'idx' : 1
        #                 }
        #             )
        #         else:
        #             marker = tx.process_data(
        #                 {
        #                     'idx' : 0
        #                 }
        #             )
        #         gate_marker_list.append(marker)
        #     self.gate_marker_array_tx.publish(
        #         {
        #             'markers' : gate_marker_list
        #         }
        #     )

        #     obstacle_marker_list = []
        #     for idx, tx in enumerate(self.obstacle_marker_txs):
        #         marker : Marker = None
        #         if self.obstacles_visited[idx]:
        #             marker = tx.process_data(
        #                 {
        #                     'idx' : 1
        #                 }
        #                 )
        #         else:
        #             marker = tx.process_data(
        #                 {
        #                     'idx' : 0
        #                 }
        #                 )
        #         obstacle_marker_list.append(marker)
        #     self.obstacle_marker_array_tx.publish(
        #         {
        #             'markers' : obstacle_marker_list
        #         }
        #     )
        #     capsule_list:List[Marker] = []
        #     for idx, item in enumerate(self.capsule_list):
        #         a,b,r = item
        #         marker_arr : MarkerArray = self.collision_capsule_tx.process_data(
        #             {
        #                 'a' : a,
        #                 'b' : b,
        #                 'r' : r,
        #                 'rgba': [1.0,1.0,0.0,0.3],
        #                 'base_idx' : 3 * idx
        #             }
        #         )
        #         capsule_list = capsule_list + [marker for marker in marker_arr.markers]
        #     self.capsule_marker_array_tx.publish(
        #          {
        #             'markers' : capsule_list
        #         }
        #     )
        pass
