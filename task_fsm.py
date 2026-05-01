"""Конечный автомат: IBVS-подход → выравнивание → захват (weld) → перенос (moveJ) → отпускание."""
from __future__ import annotations

from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

import numpy as np

from ibvs import IBVS
from vision.cube_segmentation import CubeSegmenter

if TYPE_CHECKING:
    from sim_env import MuJoCoArmSim


class Phase(Enum):
    IDLE = auto()
    IBVS_APPROACH = auto()
    FINAL_ALIGN = auto()
    GRASP_CLOSE = auto()
    TRANSPORT = auto()
    RELEASE = auto()
    DONE = auto()


class PickPlaceFSM:
    def __init__(
        self,
        ibvs: IBVS,
        segmenter: CubeSegmenter,
        cfg: Dict[str, Any],
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

    def reset(self) -> None:
        self.phase = Phase.IDLE
        self._align_count = 0
        self._move_t = 0.0
        self._q0 = None
        self.segmenter.reset_smoothing()

    def start(self) -> None:
        self.phase = Phase.IBVS_APPROACH
        self._emit(Phase.IBVS_APPROACH)

    def _emit(self, p: Phase) -> None:
        if self.on_phase:
            self.on_phase(p)

    def step(self, sim: "MuJoCoArmSim", img_bgr: np.ndarray) -> np.ndarray:
        """Один такт логики: вернуть винт камеры (6,) для IBVS или нули; для TRANSPORT вызывать physics_step_joint снаружи."""
        seg = self.segmenter.detect(img_bgr)
        v = np.zeros(6, dtype=float)

        if self.phase in (Phase.IDLE, Phase.DONE):
            return v

        if self.phase == Phase.IBVS_APPROACH:
            if not seg.ok or seg.corners is None:
                return v
            v_c, e, _ = self.ibvs.step(seg.corners, self.default_Z)
            n = np.linalg.norm(v_c)
            if n > self._max_speed and n > 1e-9:
                v_c = v_c * (self._max_speed / n)
            if self.ibvs.is_converged(e):
                self.phase = Phase.FINAL_ALIGN
                self._align_count = 0
                self._emit(Phase.FINAL_ALIGN)
            return v_c

        if self.phase == Phase.FINAL_ALIGN:
            if seg.ok and seg.corners is not None:
                v_c, e, _ = self.ibvs.step(seg.corners, self.default_Z)
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
                    self._emit(Phase.GRASP_CLOSE)
                return v_c * 0.5
            return v

        if self.phase == Phase.GRASP_CLOSE:
            if sim.eef_cube_distance < self._grasp_dist:
                sim.set_grasp_weld(True)
                sim.mj_forward()
                self.phase = Phase.TRANSPORT
                self._move_t = 0.0
                self._q0 = sim.get_q()
                self._emit(Phase.TRANSPORT)
            return v

        if self.phase == Phase.TRANSPORT:
            return v

        if self.phase == Phase.RELEASE:
            return v

        return v

    def joint_target_for_transport(self, sim: "MuJoCoArmSim", dt: float) -> Optional[np.ndarray]:
        """Если фаза TRANSPORT — интерполяция к place_joint_q; иначе None."""
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

    def finish_release(self, sim: "MuJoCoArmSim") -> None:
        if self.phase != Phase.RELEASE:
            return
        sim.set_grasp_weld(False)
        sim.mj_forward()
        self.phase = Phase.DONE
        self._emit(Phase.DONE)
