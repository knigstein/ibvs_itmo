"""Логика переходов FSM без полного MuJoCo (заглушка sim)."""
import json
from pathlib import Path

import numpy as np
import pytest

from ibvs import IBVS
from task_fsm import Phase, PickPlaceFSM
from vision import CubeSegmenter

_ROOT = Path(__file__).resolve().parents[1]


class _StubSim:
    def __init__(self):
        self.grasp_on = False
        self._dist = 0.5

    @property
    def eef_cube_distance(self) -> float:
        return self._dist

    def set_grasp_weld(self, active: bool) -> None:
        self.grasp_on = bool(active)

    def mj_forward(self) -> None:
        pass

    def get_q(self) -> np.ndarray:
        return np.zeros(6)


@pytest.fixture
def fsm_and_img():
    with open(_ROOT / "config" / "camera.json", "r", encoding="utf8") as f:
        cam = json.load(f)
    with open(_ROOT / "config" / "robot.json", "r", encoding="utf8") as f:
        robot = json.load(f)
    ibvs = IBVS(cam)
    seg = CubeSegmenter(robot.get("vision", {}))
    fsm = PickPlaceFSM(ibvs, seg, robot)
    img = np.full((480, 640, 3), (120, 90, 60), dtype=np.uint8)
    return fsm, img, cam


def test_start_goes_to_approach(fsm_and_img):
    fsm, img, _ = fsm_and_img
    fsm.start()
    assert fsm.phase == Phase.IBVS_APPROACH


def test_grasp_when_close(fsm_and_img):
    fsm, img, cam = fsm_and_img
    fsm.phase = Phase.GRASP_CLOSE
    stub = _StubSim()
    stub._dist = 0.05
    fsm._grasp_dist = 0.14
    fsm.step(stub, img)
    assert fsm.phase == Phase.TRANSPORT
    assert stub.grasp_on


def test_transport_then_release_joint_alpha(fsm_and_img):
    fsm, _, _ = fsm_and_img
    stub = _StubSim()
    fsm.phase = Phase.TRANSPORT
    fsm._move_t = 0.0
    fsm._move_duration = 0.1
    fsm._q0 = np.zeros(6)
    fsm._q1 = np.ones(6)
    dt = 0.05
    q1 = fsm.joint_target_for_transport(stub, dt)
    assert q1 is not None
    assert fsm.phase == Phase.TRANSPORT
    q2 = fsm.joint_target_for_transport(stub, dt)
    assert fsm.phase == Phase.RELEASE


def test_finish_release_done(fsm_and_img):
    fsm, _, _ = fsm_and_img
    stub = _StubSim()
    stub.grasp_on = True
    fsm.phase = Phase.RELEASE
    fsm.finish_release(stub)
    assert fsm.phase == Phase.DONE
    assert not stub.grasp_on
