from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import Controller
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray


# -------------------------
# Trajectory generator
# -------------------------
class TrajectoryGenerator:
    def __init__(self, waypoints: NDArray[np.floating], total_time: float = 15.0):
        self._t_total = float(total_time)
        self._waypoints = waypoints.astype(np.float32).copy()
        self._times = self._compute_times(self._waypoints)
        self._spline = self._build_spline()

    def _compute_times(self, waypoints: NDArray[np.floating]) -> NDArray[np.floating]:
        if len(waypoints) > 1:
            distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        else:
            distances = np.array([], dtype=np.float32)
        cumdist = np.concatenate(([0.0], np.cumsum(distances))) if distances.size > 0 else np.array([0.0], dtype=np.float32)
        total = max(float(cumdist[-1]), 1e-6)
        return (cumdist / total * self._t_total).astype(np.float32)

    def _build_spline(self) -> CubicSpline:
        if len(self._waypoints) < 2:
            wp = np.vstack([self._waypoints, self._waypoints]) if len(self._waypoints) == 1 else np.zeros((2, 3), dtype=np.float32)
            t = np.array([0.0, self._t_total], dtype=np.float32)
            return CubicSpline(t, wp, bc_type="clamped")
        return CubicSpline(self._times, self._waypoints, bc_type="clamped")

    def evaluate(self, t: float) -> NDArray[np.floating]:
        t = float(np.clip(t, 0.0, self._t_total))
        return self._spline(t).astype(np.float32)


# -------------------------
# State controller
# -------------------------
class StateController(Controller):
    def __init__(self, obs: dict, info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        self._nominal_waypoints = np.array(
            [
                [-1.5, 0.75, 0.05],
                [-1.0, 0.55, 0.4],
                [0.3, 0.35, 0.7],
                [1.3, -0.15, 0.9],
                [0.85, 0.85, 1.2],
                [-0.5, -0.05, 0.7],
                [-1.2, -0.2, 0.8],
                [-1.2, -0.2, 1.2],
                [0.0, -0.7, 1.2],
                [0.5, -0.75, 1.2],
            ],
            dtype=np.float32,
        )

        self._trajectory = TrajectoryGenerator(self._nominal_waypoints, total_time=15.0)

        self._tick = 0
        self._finished = False
        self._mode: str = "global"

        self._next_gate_idx: int = 0
        self._current_gate_idx: Optional[int] = None
        self._passed_gates: set[int] = set()

        self._enter_thresh = 0.7
        self._lookahead = 0.2

        self._local_traj: Optional[TrajectoryGenerator] = None
        self._local_t: float = 0.0
        self._local_total_time: float = 1.5

        # Cooldown logic
        self._local_cooldown = False
        self._current_wp_idx = 0
        
        self._safety_margin = 0.6
        self._corridor_thresh_extra = 0.0
        self._past_gate_window = 0.6
        self._gate_plane_tol = 0.2
        self._local_total_time = 2.5
        self._gate_width = 0.7
        self._gate_height = 0.7
        self._gate_margin = 0.1


    # -------------------------
    # Helpers
    # -------------------------
    def _now(self) -> float:
        return min(self._tick / float(self._freq), self._trajectory._t_total)

    def _get_gate_pos(self, obs: dict, idx: int | None) -> Optional[np.ndarray]:
        if idx is None:
            return None
        if "gates_pos" in obs and obs["gates_pos"] is not None and idx < len(obs["gates_pos"]):
            gp = obs["gates_pos"][idx]
            return None if gp is None else np.array(gp, dtype=np.float32)
        return None

    def _mark_gate_passed(self, gate_idx: Optional[int]):
        if gate_idx is not None:
            self._passed_gates.add(int(gate_idx))
            print(f"[DEBUG] Marked gate {gate_idx} as PASSED. Passed gates={sorted(list(self._passed_gates))}")

    def _is_gate_passed(self, gate_idx: Optional[int]) -> bool:
        return gate_idx is not None and int(gate_idx) in self._passed_gates

    def _has_passed_gate(self, drone_pos: np.ndarray, gate_pos: np.ndarray) -> bool:
        dist = float(np.linalg.norm(drone_pos - gate_pos))
        passed = dist < 0.2
        print(f"[DEBUG] HasPassedGate? dist={dist:.3f}, passed={passed}")
        return passed

    

    def _get_gate_normal(self, obs: dict, idx: int) -> np.ndarray:
        """
        Extract the gate's forward/normal vector from its quaternion.
        Assumes the gate's local +x axis points through the opening.
        """
        if "gates_quat" in obs and obs["gates_quat"] is not None:
            quat = obs["gates_quat"][idx]
            rot = R.from_quat(quat)
            normal = rot.apply([1, 0, 0])  # adjust axis if your env uses a different convention
            return normal / (np.linalg.norm(normal) + 1e-6)
        # Fallback: assume gates face along +x
        return np.array([1, 0, 0], dtype=np.float32)


    def _build_local(self, drone_pos: np.ndarray, gate_pos: np.ndarray, obs: dict | None = None):
        """
        Local trajectory with:
        - Extended obstacle check (before & slightly past gate plane)
        - Double detour points for smoother avoidance
        - Special case: obstacle on/near gate plane -> side-pass through gate
        - Special case: obstacle just past gate -> stop if needed or post-gate side-pass
        - Shaping waypoints through gate and blending back to global
        - NEW: gate aperture clearance check for side-pass offsets
        """
        drone_pos = drone_pos.astype(np.float32)
        gate_center = np.array(gate_pos, dtype=np.float32)

        # Approach vector
        v = gate_center - drone_pos
        v_norm = float(np.linalg.norm(v))
        v_dir = v / (v_norm + 1e-6)

        # Parameters
        lookahead_past_gate = 0.8
        detour_axial = 0.3

        # Gate aperture dimensions (tune to your sim)
        gate_width = getattr(self, "_gate_width", 1.0)
        gate_height = getattr(self, "_gate_height", 1.0)

        # Build lateral axis
        lateral = np.cross(v_dir, np.array([0, 0, 1], dtype=np.float32))
        if np.linalg.norm(lateral) < 1e-6:
            lateral = np.cross(v_dir, np.array([0, 1, 0], dtype=np.float32))
        lateral /= (np.linalg.norm(lateral) + 1e-6)

        # Gate frame axes for clearance check
        gate_normal = self._get_gate_normal(obs, self._current_gate_idx or 0)
        up = np.array([0, 0, 1], dtype=np.float32)
        gate_lateral = np.cross(up, gate_normal)
        if np.linalg.norm(gate_lateral) < 1e-6:
            gate_lateral = np.cross([0, 1, 0], gate_normal)
        gate_lateral /= (np.linalg.norm(gate_lateral) + 1e-6)
        gate_vertical = np.cross(gate_normal, gate_lateral)

        def inside_gate(pt):
            rel = pt - gate_center
            lat = np.dot(rel, gate_lateral)
            vert = np.dot(rel, gate_vertical)
            return (abs(lat) < self._gate_width/2 - self._gate_margin and
                abs(vert) < self._gate_height/2 - self._gate_margin)



        waypoints = [drone_pos]
        detours_added = 0

        side_pass_gate_offset = None
        post_gate_side_offset = None
        need_pre_gate_brake = False

        vel = np.array(obs.get("vel", [0.0, 0.0, 0.0]), dtype=np.float32) if obs is not None else np.zeros(3, dtype=np.float32)
        speed = float(np.linalg.norm(vel))
        max_decel = float(getattr(self, "_max_decel", 2.5))

        # --- Obstacle processing ---
        if obs is not None and "obstacles" in obs and obs["obstacles"] is not None:
            for i, obstacle in enumerate(obs["obstacles"]):
                obs_pos = np.array(obstacle["pos"], dtype=np.float32)
                obs_rad = float(obstacle.get("radius", 0.2))
                corridor_thresh = obs_rad + self._safety_margin + self._corridor_thresh_extra

                w = obs_pos - drone_pos
                proj = float(np.dot(w, v_dir))
                proj_point = drone_pos + proj * v_dir
                dist_line = float(np.linalg.norm(obs_pos - proj_point))


                # Special case A: obstacle at gate plane
                if abs(proj - v_norm) < self._gate_plane_tol and dist_line < corridor_thresh:
                    obs_lateral_sign = np.sign(np.dot(obs_pos - gate_center, lateral))
                    chosen_side = -obs_lateral_sign if obs_lateral_sign != 0 else 1.0
                    offset_mag = corridor_thresh
                    candidate_offset = chosen_side * lateral * offset_mag

                    # Gate clearance check
                    if inside_gate(gate_center + candidate_offset):
                        side_pass_gate_offset = candidate_offset
                        print(f"[DEBUG] Gate-plane side-pass: Obs#{i} offset={offset_mag:.2f}")
                    else:
                        need_pre_gate_brake = True
                        print(f"[DEBUG] Gate-plane obstacle but offset not feasible -> braking")
                    continue

                # Special case B: obstacle just past gate
                if (proj > v_norm) and (proj < v_norm + self._past_gate_window):
                    dist_past_gate = proj - v_norm
                    stopping_distance = (speed * speed) / (2.0 * max(max_decel, 1e-3))
                    aligned = dist_line < corridor_thresh
                    if aligned:
                        if stopping_distance >= dist_past_gate:
                            obs_lateral_sign = np.sign(np.dot(obs_pos - gate_center, lateral))
                            chosen_side = -obs_lateral_sign if obs_lateral_sign != 0 else 1.0
                            candidate_offset = chosen_side * lateral * corridor_thresh
                            if inside_gate(gate_center + candidate_offset):
                                post_gate_side_offset = candidate_offset
                                print(f"[DEBUG] Post-gate side-pass: Obs#{i} offset ok")
                            else:
                                need_pre_gate_brake = True
                                print(f"[DEBUG] Post-gate offset not feasible -> braking")
                        else:
                            need_pre_gate_brake = True
                            print(f"[DEBUG] Pre-gate braking: Obs#{i}")
                    continue

                # General detour
                if 0.0 < proj < (v_norm + lookahead_past_gate) and dist_line < corridor_thresh:
                    detour_center = drone_pos + (proj - corridor_thresh) * v_dir
                    detour_before = detour_center - v_dir * detour_axial
                    detour_point  = detour_center + lateral * corridor_thresh
                    detour_after  = detour_center + v_dir * detour_axial
                    waypoints.extend([detour_before, detour_point, detour_after])
                    detours_added += 1
                    print(f"[DEBUG] OA detour Obs#{i}")

        # --- Shape path through gate ---
        pre_far_base   = gate_center - v_dir * 0.6
        pre_near_base  = gate_center - v_dir * 0.3
        post_near_base = gate_center + v_dir * 0.3
        post_far_base  = gate_center + v_dir * 0.6

        if side_pass_gate_offset is not None:
            gate_mid  = gate_center + side_pass_gate_offset
            pre_far   = pre_far_base + side_pass_gate_offset
            pre_near  = pre_near_base + side_pass_gate_offset
            post_near = post_near_base + side_pass_gate_offset
            post_far  = post_far_base + side_pass_gate_offset
        else:
            gate_mid  = gate_center
            pre_far   = pre_far_base
            pre_near  = pre_near_base
            post_near = post_near_base
            post_far  = post_far_base

        if post_gate_side_offset is not None:
            post_near = post_near + post_gate_side_offset
            post_far  = post_far + post_gate_side_offset

        if need_pre_gate_brake:
            brake_point = pre_near - v_dir * 0.25
            waypoints.append(brake_point.astype(np.float32))
            print(f"[DEBUG] Inserted pre-gate brake point")

        # Blend toward global
        t_future = min(self._now() + 0.5, self._trajectory._t_total)
        global_dir = self._trajectory.evaluate(t_future) - gate_center
        global_dir /= (np.linalg.norm(global_dir) + 1e-6)

        blend_anchor = post_far
        blend_point = blend_anchor + global_dir * max(self._lookahead, 0.25)

        waypoints.extend([pre_far, pre_near, gate_mid, post_near, post_far, blend_point])
        waypoints = np.vstack(waypoints)

        local_time_base = max(self._local_total_time, 2.0)
        local_time_bonus = 0.3 if (need_pre_gate_brake or side_pass_gate_offset is not None or post_gate_side_offset is not None) else 0.0
        local_time = float(local_time_base + local_time_bonus)

        self._local_traj = TrajectoryGenerator(waypoints, total_time=local_time)
        self._local_t = 0.0

        print(
        f"[DEBUG] LocalTraj built gate={self._current_gate_idx} "
        f"center={gate_center.tolist()} wps={len(waypoints)} detours={detours_added} "
        f"side_pass={'yes' if side_pass_gate_offset is not None else 'no'} "
        f"post_side={'yes' if post_gate_side_offset is not None else 'no'} "
        f"brake={'yes' if need_pre_gate_brake else 'no'}"
                )





    def _advance_next_gate_idx(self):
        g = 0
        while self._is_gate_passed(g):
            g += 1
        self._next_gate_idx = g

    # -------------------------
    # Control loop
    # -------------------------
    def compute_control(self, obs: dict, info: dict | None = None) -> NDArray[np.floating]:
        print(f"[DEBUG] Tick={self._tick}, Mode={self._mode}, NextGateIdx={self._next_gate_idx}, "
              f"CurrentGateIdx={self._current_gate_idx}, PassedGates={sorted(list(self._passed_gates))}, "
              f"Cooldown={self._local_cooldown}")

        t = self._now()
        if t >= self._trajectory._t_total and self._mode == "global":
            self._finished = True

        drone_pos = np.array(obs.get("pos", np.zeros(3)), dtype=np.float32)

        # Update next gate index
        self._advance_next_gate_idx()
        gate_idx = self._next_gate_idx
        gate_pos = self._get_gate_pos(obs, gate_idx)

        # Default: global trajectory
        des_pos = self._trajectory.evaluate(t)

        # --- Local mode ---
        if self._mode == "local" and self._local_traj is not None:
            self._local_t = min(self._local_t + 1.0 / float(self._freq), self._local_traj._t_total)
            des_pos = self._local_traj.evaluate(self._local_t)

            # Exit local mode only when local spline is finished
            if self._local_t >= self._local_traj._t_total:
                print(f"[DEBUG] EXIT local mode at gate {self._current_gate_idx}")
                self._mark_gate_passed(self._current_gate_idx)
                self._mode = "global"
                self._current_gate_idx = None
                self._local_traj = None
                self._local_t = 0.0
                self._local_cooldown = True

        # --- Global mode ---
        elif self._mode == "global":
            if (not self._local_cooldown) and gate_pos is not None and not self._is_gate_passed(gate_idx):
                dist = float(np.linalg.norm(drone_pos - gate_pos))
                print(f"[DEBUG] EntryCheck: gate={gate_idx}, dist={dist:.3f}, gate_pos={gate_pos}")
                if dist < self._enter_thresh:
                    print(f"[DEBUG] ENTER local mode at gate {gate_idx}")
                    self._mode = "local"
                    self._current_gate_idx = gate_idx
                    self._build_local(drone_pos, gate_pos, obs)
                    des_pos = self._local_traj.evaluate(0.0)

        # --- Clear cooldown once we’ve advanced to the next gate ---
        if self._local_cooldown and self._current_gate_idx is None:
            if self._next_gate_idx > max(self._passed_gates, default=-1):
                self._local_cooldown = False
                print(f"[DEBUG] Cleared local cooldown at gate {self._next_gate_idx}")

        # Final action vector
        action = np.concatenate((des_pos.astype(np.float32),
                                 np.zeros(10, dtype=np.float32)), dtype=np.float32)
        return action

    # -------------------------
    # Callbacks
    # -------------------------
    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        # advance simulation tick
        self._tick += 1

        # mark finished if environment says so
        if terminated or truncated:
            self._finished = True

        return self._finished

    def episode_callback(self):
        print("[DEBUG] Episode end: resetting controller.")
        # reset counters and state
        self._tick = 0
        self._finished = False

        # global/local modes
        self._mode = "global"
        self._current_gate_idx = None
        self._local_traj = None
        self._local_t = 0.0

        # gates and cooldown
        self._passed_gates.clear()
        self._next_gate_idx = 0
        self._local_cooldown = False

        # rebuild trajectory from nominal waypoints
        self._trajectory = TrajectoryGenerator(
            self._nominal_waypoints, total_time=self._trajectory._t_total
        )
