"""Simulation core: fixed-step world with a kinematic-bicycle ego, an entity
list, a spatial hash, and CARLA-style termination + infraction tracking.
"""
import numpy as np
from bharatsim.core.geometry import obb_corners, obb_overlap
from bharatsim.core.spatial import SpatialHash

DT = 0.1


class Ego:
    L, W, WB = 4.4, 1.9, 2.6

    def __init__(self, x, y, yaw, v=0.0):
        self.x, self.y, self.yaw, self.v = x, y, yaw, v

    def step(self, steer, throttle, brake, dt):
        delta = np.clip(steer, -1, 1) * np.radians(38)
        a = np.clip(throttle, 0, 1) * 3.0 - np.clip(brake, 0, 1) * 8.0
        self.v = max(0.0, self.v + a * dt)
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / self.WB * np.tan(delta) * dt

    def corners(self):
        return obb_corners(self.x, self.y, self.yaw, self.L, self.W)


class World:
    def __init__(self, mapdata, route, ego_speed0=5.0, max_time=None,
                 environment=None):
        self.map = mapdata
        self.route = route
        self.speed_limit = mapdata.speed_limit
        p = route.point_at(0.0)
        self.ego = Ego(float(p[0]), float(p[1]), route.heading_at(0.0), ego_speed0)
        self.entities = []            # traffic, VRUs, hazards (all Entity)
        self.environment = environment
        self.t = 0.0
        self.max_time = max_time or float(min(route.total / 4.5 + 12, 70))
        self.done = False
        self.result = None
        self.events = []
        self.hash = SpatialHash(4.0)
        self._dev = False
        self.junction = None      # optional core.signals.JunctionControl
        self.ran_red = False

    # -- helpers --
    def add(self, e):
        self.entities.append(e); return e

    def ego_xy(self):
        return np.array([self.ego.x, self.ego.y])

    @property
    def progress(self):
        s, _ = self.route.project(self.ego_xy())
        return float(np.clip(s / self.route.total, 0, 1))

    def command(self):
        s, _ = self.route.project(self.ego_xy())
        dh = self.route.heading_at(s + 18) - self.route.heading_at(s + 2)
        dh = (dh + np.pi) % (2 * np.pi) - np.pi
        return 1 if dh > 0.3 else 2 if dh < -0.3 else 0

    def rebuild_hash(self):
        self.hash.clear()
        for e in self.entities:
            if e.alive:
                self.hash.insert(e.x, e.y, e)

    # -- step --
    def step(self, control, dt=DT):
        if self.done:
            return True
        self.rebuild_hash()
        self.ego.step(control.get("steer", 0), control.get("throttle", 0),
                      control.get("brake", 0), dt)
        if self.junction is not None:
            self.junction.step(dt)
        for e in self.entities:
            if e.alive and e.dynamic:
                e.step(self, dt)
        self.t += dt
        self._check()
        return self.done

    def _check(self):
        er = self.ego.corners()
        for e in self.entities:
            if not e.alive:
                continue
            if np.hypot(e.x - self.ego.x, e.y - self.ego.y) > 9:
                continue
            if e.hard and obb_overlap(er, e.corners()):
                self.done = True
                self.result = ("collision_ped" if e.kind == "vru"
                               else "collision_static" if not e.dynamic
                               else "collision_vehicle")
                return
        s, lat = self.route.project(self.ego_xy())
        if self.junction is not None and not self.ran_red:
            for lt in self.junction.lights:
                if (lt.path_name in ("route", "same") and lt.state == "R"
                        and s > lt.stop_s and s - lt.stop_s < 2.0):
                    self.ran_red = True
                    self.events.append("red_light")
        if abs(lat) > 5.0:
            self.done = True; self.result = "offroad"; return
        if abs(lat) > 3.0 and not self._dev:
            self._dev = True; self.events.append("lane_deviation")
        if s > self.route.total - 3.0:
            self.done = True; self.result = "success"; return
        if self.t >= self.max_time:
            self.done = True; self.result = "timeout"
