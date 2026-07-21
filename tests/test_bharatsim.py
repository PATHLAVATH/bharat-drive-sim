"""Integration tests for bharat-drive-sim."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from bharatsim.maps.library import MAPS
from bharatsim.scenarios.builtin import SPECS
from bharatsim.scenarios.dsl import build_from_dict, load_scenario
from bharatsim.eval.env import BharatEnv, run_scenario, Agent, Control
from bharatsim.eval.reference import RiskFieldAgent, ConstantAgent
from bharatsim.sensors.bev import BEV_GRID, N_CLASSES
from bharatsim.core.spatial import SpatialHash


def test_maps_build():
    for n, f in MAPS.items():
        m = f(0)
        assert m.route.total > 20 and m.speed_limit > 0
    print(f"{len(MAPS)} maps OK")


def test_all_scenarios_run():
    for n in SPECS:
        w = build_from_dict(SPECS[n], 0)
        for _ in range(20):
            if w.step({"steer": 0, "throttle": 0.3, "brake": 0}):
                break
        assert w.result is None or isinstance(w.result, str)
    print(f"{len(SPECS)} built-in scenarios run OK")


def test_gym_and_agent_api():
    env = BharatEnv("cow_on_road", 0)
    obs = env.reset()
    assert obs.bev.shape == (BEV_GRID, BEV_GRID)
    assert obs.future_occ.shape == (4, BEV_GRID, BEV_GRID)
    o, r, d, info = env.step(Control(0, 0.3, 0))
    assert "result" in info
    row = run_scenario(RiskFieldAgent(), "pothole_field", 0)
    assert row["result"] and 0 <= row["ds"] <= 100
    print("gym + agent API OK")


def test_yaml_dsl():
    here = os.path.dirname(__file__)
    w = load_scenario(os.path.join(here, "..", "scenarios", "custom_example.yaml"), 0)
    assert len(w.entities) > 0
    print(f"YAML DSL builds {len(w.entities)} entities OK")


def test_spatial_hash():
    sh = SpatialHash(2.0)
    for i in range(50):
        sh.insert(i * 0.5, 0, i)
    got = set(sh.query(5.0, 0.0, 1.5))
    assert got and all(abs(p * 0.5 - 5) <= 1.5 for p in got)
    print("spatial hash OK")




# ---- v0.2 module tests ------------------------------------------------------
def test_cameras_and_route_sensor():
    from bharatsim.sensors.camera import render_cameras, IMG_H, IMG_W
    from bharatsim.sensors.route_sensor import route_features
    w = build_from_dict(SPECS["junction_freeforall"], 0)
    for _ in range(15):
        w.step({"steer": 0, "throttle": 0.3, "brake": 0})
    cam = render_cameras(w)
    assert cam.rgb.shape == (3, IMG_H, IMG_W, 3)
    assert cam.depth.shape == (3, IMG_H, IMG_W)
    assert cam.intrinsics.shape == (3, 3, 3)
    rf = route_features(w)
    assert "heading_errors" in rf and len(rf["heading_errors"]) == 3
    print("cameras + route sensor OK")


def test_traffic_signals():
    from bharatsim.core.signals import TrafficLight, JunctionControl
    lt = TrafficLight(stop_s=90, path_name="route", phase0=0.0)
    seen = set()
    for _ in range(140):
        seen.add(lt.state); lt.step(0.1)
    assert {"G", "Y", "R"} <= seen
    jc = JunctionControl([TrafficLight(90, "route")])
    assert jc.stopline_gap("route", 80) in (None,) or jc.stopline_gap("route", 80) >= 0
    print("traffic signals OK")


def test_expert_runs():
    from bharatsim.eval.expert import ExpertAutopilot
    r = run_scenario(ExpertAutopilot(), "highway_mixed", 0)
    assert r["result"] and 0 <= r["ds"] <= 100
    print(f"expert autopilot OK ({r['result']}, DS {r['ds']:.0f})")


def test_recorder_and_logger(tmp="/tmp/_bds_test"):
    import os
    from bharatsim.eval.expert import ExpertAutopilot
    from bharatsim.viz.recorder import Recorder
    from bharatsim.eval.logger import DataLogger
    rec = Recorder(); log = DataLogger(out_dir=tmp, shard_size=10**9)
    run_scenario(ExpertAutopilot(), "cow_on_road", 0, recorder=rec, logger=log)
    assert len(rec.frames) > 0
    p = log.flush()
    assert p and os.path.exists(p)
    print(f"recorder ({len(rec.frames)} frames) + logger OK")


def test_generator_and_leaderboard():
    from bharatsim.scenarios.generator import generate
    from bharatsim.eval.leaderboard import run_leaderboard
    from bharatsim.eval.reference import ConstantAgent
    b = generate(n=200, seed=0)
    assert len(b) == 200
    rep = run_leaderboard(ConstantAgent, n=6, seed=0, out_dir="/tmp/_bds_lb",
                          verbose=False)
    assert rep["overall"]["n"] >= 1 and "per_map" in rep
    print(f"generator (200 routes) + leaderboard OK")


if __name__ == "__main__":
    test_maps_build()
    test_all_scenarios_run()
    test_gym_and_agent_api()
    test_yaml_dsl()
    test_spatial_hash()
    test_cameras_and_route_sensor()
    test_traffic_signals()
    test_expert_runs()
    test_recorder_and_logger()
    test_generator_and_leaderboard()
    print("ALL BHARATSIM TESTS PASSED")
