"""CARLA Leaderboard / Bench2Drive adapter.

Wraps any bharatsim Agent as a leaderboard `AutonomousAgent`, so you can submit
the exact same policy you benchmarked in the pure-Python sim to the official
CARLA Leaderboard 2.0 or Bench2Drive evaluation.

Usage (leaderboard evaluator expects a module exposing `get_entry_point`):

    export TEAM_AGENT=bharatsim/carla/leaderboard_agent.py
    export TEAM_CONFIG=configs/carla_agent.yaml   # {agent: "module:Class"}
    leaderboard_evaluator.py --agent=$TEAM_AGENT --agent-config=$TEAM_CONFIG ...

The base autonomous_agent import is guarded so this file also imports in a plain
Python env for inspection/testing.
"""
from __future__ import annotations
import importlib
import math
import numpy as np

try:
    from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
    _LB = True
except ImportError:
    _LB = False
    class AutonomousAgent:      # shim for import without the leaderboard installed
        def setup(self, path): ...
    class Track:                # noqa
        SENSORS = "SENSORS"


def get_entry_point():
    return "BharatsimLeaderboardAgent"


SENSOR_YAWS = [("left", -60.0), ("front", 0.0), ("right", 60.0)]


class BharatsimLeaderboardAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS
        self._policy = self._load_policy(path_to_conf_file)
        self._policy.reset()

    def _load_policy(self, conf):
        import yaml, os
        spec = "bharatsim.eval.reference:RiskFieldAgent"
        if conf and os.path.exists(conf):
            spec = yaml.safe_load(open(conf)).get("agent", spec)
        mod, cls = spec.split(":")
        return getattr(importlib.import_module(mod), cls)()

    def sensors(self):
        """CARLA Leaderboard sensor spec: 3 RGB + speed + IMU/GNSS."""
        s = []
        for name, yaw in SENSOR_YAWS:
            s.append({"type": "sensor.camera.rgb", "id": name,
                      "x": 1.2, "y": 0.0, "z": 1.5, "roll": 0, "pitch": 0, "yaw": yaw,
                      "width": 200, "height": 120, "fov": 75})
        s.append({"type": "sensor.speedometer", "id": "speed", "reading_frequency": 20})
        s.append({"type": "sensor.other.imu", "id": "imu",
                  "x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0,
                  "sensor_tick": 0.05})
        return s

    def run_step(self, input_data, timestamp):
        from bharatsim.eval.env import Observation, Control
        from bharatsim.sensors.bev import BEV_GRID
        rgb = []
        for name, _ in SENSOR_YAWS:
            frame = input_data[name][1][:, :, :3][:, :, ::-1]   # BGRA->RGB
            rgb.append(frame)
        speed = float(input_data["speed"][1]["speed"])
        from bharatsim.sensors.camera import CameraObs
        rgb = np.stack(rgb)
        cam = CameraObs(rgb, np.full(rgb.shape[:3], np.inf, np.float32),
                        np.zeros(rgb.shape[:3], np.uint8),
                        np.zeros((3, 3, 3), np.float32), np.zeros((3, 4, 4), np.float32))
        obs = Observation(np.zeros((BEV_GRID, BEV_GRID), np.uint8),
                          np.zeros((4, BEV_GRID, BEV_GRID), np.uint8),
                          speed, 0, 1.0, timestamp, camera=cam,
                          route={"heading_errors": [0, 0, 0], "lateral_offset": 0,
                                 "curvature": 0, "dist_to_goal": 0})
        c = self._policy.act(obs)
        import carla
        return carla.VehicleControl(throttle=float(c.throttle), steer=float(c.steer),
                                    brake=float(c.brake))
