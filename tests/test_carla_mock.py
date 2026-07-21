"""Verify the CARLA integration control flow against an in-process mock server."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tests.mock_carla as mock
mock.install()          # register fake `carla` before importing the bridge


def test_bridge_runs_full_episode():
    from bharatsim.carla.bridge import run_scenario_carla
    from bharatsim.eval.reference import ConstantAgent
    spec = {"map": "narrow_street", "ego_speed": 6,
            "actors": [{"type": "two_wheeler", "path": "same", "s": 20, "count": 3},
                       {"type": "auto", "path": "same", "s": 40, "count": 2}],
            "vrus": [{"cross_at": 45, "side": -1, "count": 2, "speed": 1.3}],
            "hazards": [{"type": "pothole", "count": 5},
                        {"type": "speedbump", "at": 60}],
            "environment": {"label": "monsoon"}}
    row = run_scenario_carla(ConstantAgent(throttle=0.5), spec, seed=0,
                             max_steps=60, cameras=True)
    assert row["result"] in ("success", "timeout", "collision_vehicle",
                             "collision_ped", "collision_static")
    assert 0 <= row["ds"] <= 100
    print(f"CARLA bridge episode OK -> {row['result']} DS {row['ds']:.0f} "
          f"t {row['sim_time']:.1f}s")


def test_sensor_rig_assembly():
    import numpy as np
    from bharatsim.carla.client import CarlaClient
    from bharatsim.carla.sensors import CarlaCameraRig
    import carla
    cc = CarlaClient(town="Town01")
    ego = cc.world.spawn_actor(cc.world.get_blueprint_library().filter("vehicle.*")[0],
                               carla.Transform())
    rig = CarlaCameraRig(cc.carla, cc.world, ego, use_lidar=True)
    obs = rig.read()
    assert obs.rgb.shape[0] == 3 and obs.intrinsics.shape == (3, 3, 3)
    rig.destroy()
    print(f"sensor rig assembled {len(rig.sensors) if rig.sensors else 0} + read OK")


def test_leaderboard_wrapper_sensors_spec():
    from bharatsim.carla.leaderboard_agent import BharatsimLeaderboardAgent, get_entry_point
    a = BharatsimLeaderboardAgent.__new__(BharatsimLeaderboardAgent)
    spec = a.sensors()
    ids = {s["id"] for s in spec}
    assert {"left", "front", "right", "speed"} <= ids
    assert get_entry_point() == "BharatsimLeaderboardAgent"
    print(f"leaderboard sensor spec OK ({len(spec)} sensors)")


def test_blueprint_and_weather_tables():
    from bharatsim.carla.blueprints import BLUEPRINT_MAP, resolve
    from bharatsim.carla.client import WEATHER, TOWN_FOR_MAP
    import carla
    lib = list(carla.Client("", 0).get_world().get_blueprint_library())
    assert resolve(lib, BLUEPRINT_MAP["two_wheeler"]) is not None
    assert "monsoon" in WEATHER and TOWN_FOR_MAP["highway"] == "Town04"
    print("blueprint resolve + weather/town tables OK")


if __name__ == "__main__":
    test_blueprint_and_weather_tables()
    test_sensor_rig_assembly()
    test_leaderboard_wrapper_sensors_spec()
    test_bridge_runs_full_episode()
    print("ALL CARLA-MOCK TESTS PASSED")
