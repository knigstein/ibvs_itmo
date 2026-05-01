import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_scene_loads_and_ids():
    import mujoco

    from sim_env import MuJoCoArmSim

    with open(ROOT / "config" / "camera.json", "r", encoding="utf8") as f:
        cam_cfg = json.load(f)
    sim = MuJoCoArmSim(
        model_path=str(ROOT / "universal_robots_ur5e" / "IBVS_Scene.xml"),
        camera_cfg=cam_cfg,
    )
    assert sim.cam_site_id >= 0
    assert sim.work_geom_id >= 0
    assert sim.model.neq >= 1
    assert sim.grasp_eq_id >= 0


if __name__ == "__main__":
    test_scene_loads_and_ids()
    print("test_mujoco_scene OK")
