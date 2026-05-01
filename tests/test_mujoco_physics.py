"""Несколько шагов симуляции и инварианты weld (нужен mujoco)."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from sim_env import MuJoCoArmSim

_ROOT = Path(__file__).resolve().parents[1]
_SCENE = str(_ROOT / "universal_robots_ur5e" / "IBVS_Scene.xml")


def test_grasp_weld_starts_disabled():
    sim = MuJoCoArmSim(model_path=_SCENE)
    if sim.grasp_eq_id < 0:
        pytest.skip("нет grasp_weld в модели")
    assert sim.data.eq_active[sim.grasp_eq_id] == 0


def test_physics_steps_zero_ibvs_finite_state():
    sim = MuJoCoArmSim(model_path=_SCENE)
    q0 = sim.get_q().copy()
    for _ in range(20):
        sim.physics_step_ibvs(np.zeros(6))
    assert np.all(np.isfinite(sim.data.qpos))
    assert np.all(np.isfinite(sim.data.qvel))
    assert np.isfinite(sim.eef_cube_distance)


def test_joint_pd_moves_toward_target():
    sim = MuJoCoArmSim(model_path=_SCENE)
    q0 = sim.get_q().copy()
    q_goal = q0 + np.array([0.05, 0, 0, 0, 0, 0])
    for _ in range(80):
        sim.physics_step_joint(q_goal)
    err_after = np.linalg.norm(sim.get_q() - q_goal)
    err_before = np.linalg.norm(q0 - q_goal)
    assert err_after < err_before


def test_grasp_weld_toggle():
    sim = MuJoCoArmSim(model_path=_SCENE)
    if sim.grasp_eq_id < 0:
        pytest.skip("нет grasp_weld")
    sim.set_grasp_weld(True)
    sim.mj_forward()
    assert sim.data.eq_active[sim.grasp_eq_id] == 1
    sim.set_grasp_weld(False)
    sim.mj_forward()
    assert sim.data.eq_active[sim.grasp_eq_id] == 0
