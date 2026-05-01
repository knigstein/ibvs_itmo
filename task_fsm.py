"""FSM: IBVS-подход → выравнивание → захват (weld) → перенос (интерполяция q) → отпускание."""
from __future__ import annotations

from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from ibvs import IBVS
from vision.aruco_ranging import ArucoMotionRanging
from vision.cube_segmentation import CubeSegmenter
from vision.depth_map import build_depth_provider

if TYPE_CHECKING:
    from sim_env import MuJoCoArmSim


class Phase(Enum):
    IDLE = auto()
    CALIBRATION_MARKER = auto()
    CALIBRATION_DESCEND = auto()
    CALIBRATION_ASCEND = auto()
    IBVS_APPROACH = auto()
    FINAL_ALIGN = auto()
    SEARCH = auto()
    GRASP_CLOSE = auto()
    DESCENT_TO_GRASP = auto()
    LIFT_AFTER_GRASP = auto()
    TRANSPORT = auto()
    RELEASE = auto()
    DONE = auto()


class PickPlaceFSM:
    def __init__(
        self,
        ibvs: IBVS,
        segmenter: CubeSegmenter,
        cfg: Dict[str, Any],
        camera_cfg: Optional[Dict[str, Any]] = None,
        *,
        on_phase: Optional[Callable[[Phase], None]] = None,
    ):
        self.ibvs = ibvs
        self.segmenter = segmenter
        self.cfg = cfg
        self.on_phase = on_phase
        self.phase = Phase.IDLE
        self._align_count = 0
        self._move_t = 0.0
        self._move_duration = float(cfg.get("move_j_duration_s", 4.0))
        self._q0: Optional[np.ndarray] = None
        self._q1 = np.asarray(cfg.get("place_joint_q"), dtype=float).reshape(6)
        self._max_speed = float(cfg.get("max_ibvs_speed", 0.35))
        self._final_scale = float(cfg.get("final_align_error_scale", 0.85))
        self._align_hold = int(cfg.get("align_hold_steps", 8))
        self._grasp_dist = float(cfg.get("grasp_distance", 0.14))
        self.default_Z = float(cfg.get("default_Z", 0.5))
        self._depth_offset = float(cfg.get("ibvs_depth_offset_m", 0.0))
        self._z_fallback = self.default_Z + self._depth_offset
        self._depth_map, self._depth_mode = build_depth_provider(cfg)
        zc = cfg.get("z_calibration") or {}
        cam_for_ranging = camera_cfg or {"focal_length": np.asarray(getattr(self.ibvs, "_focal_length", [600.0, 600.0])).tolist()}
        self._aruco_ranging = ArucoMotionRanging(zc, cam_for_ranging)
        self._calib_probe_speed = abs(float(zc.get("probe_speed_mps", 0.03)))
        self._descent_sign = float(zc.get("descent_direction_sign", 1.0))
        if abs(self._descent_sign) < 1e-9:
            self._descent_sign = 1.0
        self._calib_move_remaining_m = 0.0

        self._grasp_descent_m = max(0.0, float(zc.get("grasp_descent_m", 0.12)))
        self._grasp_lift_m = max(0.0, float(zc.get("lift_after_grasp_m", self._grasp_descent_m)))
        self._grasp_descent_speed = abs(float(zc.get("grasp_descent_speed_mps", 0.035)))
        self._grasp_move_remaining_m = 0.0

        s_cfg = cfg.get("search") or {}
        self._lost_threshold = int(s_cfg.get("lost_frames_to_search", 15))
        self._search_dwell = float(s_cfg.get("dwell_s", 0.5))
        wps = s_cfg.get("waypoints") or []
        self._search_wps: List[np.ndarray] = [np.asarray(w, dtype=float).reshape(6) for w in wps]
        self._search_wp_idx = 0
        self._search_hold_t = 0.0
        self._lost_streak = 0

    def reset(self) -> None:
        self.phase = Phase.IDLE
        self._align_count = 0
        self._move_t = 0.0
        self._q0 = None
        self._search_wp_idx = 0
        self._search_hold_t = 0.0
        self._lost_streak = 0
        self._calib_move_remaining_m = 0.0
        self._grasp_move_remaining_m = 0.0
        self.segmenter.reset_smoothing()

    def start(self) -> None:
        self.phase = Phase.CALIBRATION_MARKER if self._aruco_ranging.enabled else Phase.IBVS_APPROACH
        self._lost_streak = 0
        self._emit(self.phase)

    def _emit(self, p: Phase) -> None:
        if self.on_phase:
            self.on_phase(p)

    def _enter_search(self) -> None:
        self.phase = Phase.SEARCH
        self._align_count = 0
        self._search_wp_idx = 0
        self._search_hold_t = 0.0
        self._lost_streak = 0
        self.segmenter.reset_smoothing()
        self._emit(Phase.SEARCH)

    def _note_visibility(self, seg_ok: bool) -> None:
        if seg_ok:
            self._lost_streak = 0
        else:
            self._lost_streak += 1

    def _lost_search_triggered(self) -> bool:
        return self._lost_streak >= self._lost_threshold

    def _ibvs_Z(self, corners: np.ndarray, depth_m: Optional[np.ndarray]):
        """Скаляр или вектор (4,) глубин для IBVS; при ошибке — fallback из конфига."""
        z_meas: Optional[float] = None
        if self._depth_map is not None and depth_m is not None:
            z_meas = self._depth_map.Z_for_ibvs(corners, depth_m)
        if z_meas is not None:
            z = float(z_meas) * self._aruco_ranging.depth_scale + self._depth_offset
            return np.full(4, z, dtype=float)
        return self._z_fallback

    def _start_transport(self, sim: "MuJoCoArmSim") -> None:
        self.phase = Phase.TRANSPORT
        self._move_t = 0.0
        self._q0 = sim.get_q()
        self._emit(Phase.TRANSPORT)

    def step(
        self,
        sim: "MuJoCoArmSim",
        img_bgr: np.ndarray,
        *,
        depth_m: Optional[np.ndarray] = None,
        dt: float = 0.01,
    ) -> np.ndarray:
        seg = self.segmenter.detect(img_bgr)
        seg_ok = bool(seg.ok and seg.corners is not None)
        aruco = self._aruco_ranging.detect(img_bgr)
        v = np.zeros(6, dtype=float)

        if self.phase in (Phase.IDLE, Phase.DONE):
            return v

        if self.phase == Phase.CALIBRATION_MARKER:
            if aruco.ok and aruco.distance_m is not None:
                self._aruco_ranging.begin_probe(aruco)
                self._calib_move_remaining_m = self._aruco_ranging.probe_descent_m
                self._z_fallback = self._aruco_ranging.known_start_distance_m + self._depth_offset
                if self._calib_move_remaining_m <= 1e-6 or self._calib_probe_speed <= 1e-6:
                    self._aruco_ranging.finalize_probe()
                    if self._aruco_ranging.calibrated_start_distance_m is not None:
                        self._z_fallback = self._aruco_ranging.calibrated_start_distance_m + self._depth_offset
                    self.phase = Phase.IBVS_APPROACH
                    self._emit(Phase.IBVS_APPROACH)
                else:
                    self.phase = Phase.CALIBRATION_DESCEND
                    self._emit(Phase.CALIBRATION_DESCEND)
            return v

        if self.phase == Phase.CALIBRATION_DESCEND:
            v[2] = self._descent_sign * self._calib_probe_speed
            self._calib_move_remaining_m -= abs(v[2]) * max(dt, 1e-4)
            if self._calib_move_remaining_m <= 0.0:
                if aruco.ok and aruco.distance_m is not None:
                    self._aruco_ranging.capture_probe_bottom(aruco)
                self._calib_move_remaining_m = self._aruco_ranging.probe_descent_m
                self.phase = Phase.CALIBRATION_ASCEND
                self._emit(Phase.CALIBRATION_ASCEND)
            return v

        if self.phase == Phase.CALIBRATION_ASCEND:
            v[2] = -self._descent_sign * self._calib_probe_speed
            self._calib_move_remaining_m -= abs(v[2]) * max(dt, 1e-4)
            if self._calib_move_remaining_m <= 0.0:
                self._aruco_ranging.finalize_probe()
                if self._aruco_ranging.calibrated_start_distance_m is not None:
                    self._z_fallback = self._aruco_ranging.calibrated_start_distance_m + self._depth_offset
                self.phase = Phase.IBVS_APPROACH
                self._emit(Phase.IBVS_APPROACH)
            return v

        if self.phase == Phase.SEARCH:
            if seg_ok:
                self.phase = Phase.IBVS_APPROACH
                self._lost_streak = 0
                self._emit(Phase.IBVS_APPROACH)
            return v

        if self.phase == Phase.IBVS_APPROACH:
            if not seg_ok:
                self._note_visibility(False)
                if self._lost_search_triggered():
                    self._enter_search()
                return v
            self._note_visibility(True)
            Z = self._ibvs_Z(seg.corners, depth_m)
            v_c, e, _ = self.ibvs.step(seg.corners, Z)
            n = np.linalg.norm(v_c)
            if n > self._max_speed and n > 1e-9:
                v_c = v_c * (self._max_speed / n)
            if self.ibvs.is_converged(e):
                self.phase = Phase.FINAL_ALIGN
                self._align_count = 0
                self._emit(Phase.FINAL_ALIGN)
            return v_c

        if self.phase == Phase.FINAL_ALIGN:
            if not seg_ok:
                self._note_visibility(False)
                if self._lost_search_triggered():
                    self._enter_search()
                return v
            self._note_visibility(True)
            Z = self._ibvs_Z(seg.corners, depth_m)
            v_c, e, _ = self.ibvs.step(seg.corners, Z)
            thr = self.ibvs.exit_threshold * self._final_scale
            if float(np.linalg.norm(e)) < thr:
                self._align_count += 1
            else:
                self._align_count = 0
            n = np.linalg.norm(v_c)
            if n > self._max_speed * 0.5 and n > 1e-9:
                v_c = v_c * (self._max_speed * 0.5 / n)
            if self._align_count >= self._align_hold:
                self.phase = Phase.GRASP_CLOSE
                self._lost_streak = 0
                self._emit(Phase.GRASP_CLOSE)
            return v_c * 0.5

        if self.phase == Phase.GRASP_CLOSE:
            v_c = np.zeros(6, dtype=float)
            if seg_ok:
                Z = self._ibvs_Z(seg.corners, depth_m)
                v_c, _, _ = self.ibvs.step(seg.corners, Z)
                n = np.linalg.norm(v_c)
                if n > self._max_speed * 0.2 and n > 1e-9:
                    v_c = v_c * (self._max_speed * 0.2 / n)
            self._grasp_move_remaining_m = self._grasp_descent_m
            self.phase = Phase.DESCENT_TO_GRASP
            self._emit(Phase.DESCENT_TO_GRASP)
            return v_c * 0.2

        if self.phase == Phase.DESCENT_TO_GRASP:
            if seg_ok:
                Z = self._ibvs_Z(seg.corners, depth_m)
                v_xy, _, _ = self.ibvs.step(seg.corners, Z)
                v[:2] = 0.15 * v_xy[:2]
            v[2] = self._descent_sign * self._grasp_descent_speed
            self._grasp_move_remaining_m -= abs(v[2]) * max(dt, 1e-4)
            if sim.eef_cube_distance < self._grasp_dist or self._grasp_move_remaining_m <= 0.0:
                sim.set_grasp_weld(True)
                sim.mj_forward()
                self._grasp_move_remaining_m = self._grasp_lift_m
                self.phase = Phase.LIFT_AFTER_GRASP
                self._emit(Phase.LIFT_AFTER_GRASP)
            return v

        if self.phase == Phase.LIFT_AFTER_GRASP:
            v[2] = -self._descent_sign * self._grasp_descent_speed
            self._grasp_move_remaining_m -= abs(v[2]) * max(dt, 1e-4)
            if self._grasp_move_remaining_m <= 0.0:
                self._start_transport(sim)
            return v

        return v

    def joint_target_for_transport(self, sim: "MuJoCoArmSim", dt: float) -> Optional[np.ndarray]:
        if self.phase != Phase.TRANSPORT:
            return None
        if self._q0 is None:
            self._q0 = sim.get_q()
        self._move_t += dt
        alpha = min(1.0, self._move_t / max(self._move_duration, 1e-6))
        q = (1.0 - alpha) * self._q0 + alpha * self._q1
        if alpha >= 1.0:
            self.phase = Phase.RELEASE
            self._emit(Phase.RELEASE)
        return q

    def joint_target_for_search(self, sim: "MuJoCoArmSim", dt: float) -> Optional[np.ndarray]:
        if self.phase != Phase.SEARCH:
            return None
        q_now = sim.get_q()
        if not self._search_wps:
            q_tgt = q_now.copy()
            q_tgt[0] += 0.25 * np.sin(self._search_hold_t * 0.8)
            self._search_hold_t += dt
            return q_tgt

        idx = self._search_wp_idx % len(self._search_wps)
        q_tgt = self._search_wps[idx]
        self._search_hold_t += dt
        if self._search_hold_t >= self._search_dwell:
            self._search_hold_t = 0.0
            self._search_wp_idx += 1
        return np.asarray(q_tgt, dtype=float).reshape(6)

    def finish_release(self, sim: "MuJoCoArmSim") -> None:
        if self.phase != Phase.RELEASE:
            return
        sim.set_grasp_weld(False)
        sim.set_gripper_open()
        sim.mj_forward()
        self.phase = Phase.DONE
        self._emit(Phase.DONE)
