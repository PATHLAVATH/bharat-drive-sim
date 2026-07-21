"""Minimal in-process mock of the CARLA PythonAPI.

Implements just enough of the surface bharatsim.carla.* uses so the bridge,
sensors, and leaderboard wrapper can be exercised WITHOUT a real simulator or
GPU. Install this as `carla` in sys.modules for tests. It is NOT a physics sim —
it validates the integration control flow (spawn/tick/observe/apply/teardown).
"""
import sys, types
import numpy as np


class Location:
    def __init__(self, x=0.0, y=0.0, z=0.0): self.x, self.y, self.z = x, y, z
    def distance(self, o): return ((self.x-o.x)**2+(self.y-o.y)**2+(self.z-o.z)**2)**0.5

class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0): self.pitch, self.yaw, self.roll = pitch, yaw, roll

class Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or Location(); self.rotation = rotation or Rotation()

class Vector3D:
    def __init__(self, x=0, y=0, z=0): self.x, self.y, self.z = x, y, z

class VehicleControl:
    def __init__(self, throttle=0.0, steer=0.0, brake=0.0):
        self.throttle, self.steer, self.brake = throttle, steer, brake

class WeatherParameters:
    def __init__(self, **kw): self.__dict__.update(kw)


class _BP:
    def __init__(self, bid): self.id = bid; self._attrs = {}
    def set_attribute(self, k, v): self._attrs[k] = v
    def has_attribute(self, k): return True

class _BPLib:
    _IDS = ["vehicle.tata.nano", "vehicle.yamaha.yzf", "vehicle.nissan.micra",
            "vehicle.bh.crossbike", "vehicle.volkswagen.t2", "vehicle.carlamotors.carlacola",
            "vehicle.micro.microlino", "walker.pedestrian.0001", "walker.pedestrian.0005",
            "controller.ai.walker", "sensor.other.collision", "sensor.camera.rgb",
            "sensor.camera.depth", "sensor.camera.semantic_segmentation",
            "sensor.lidar.ray_cast", "static.prop.dirtdebris01",
            "static.prop.streetbarrier", "static.prop.container"]
    def __iter__(self): return iter(_BP(i) for i in self._IDS)
    def find(self, bid): return _BP(bid)
    def filter(self, pat):
        import fnmatch; return [_BP(i) for i in self._IDS if fnmatch.fnmatch(i, pat)]


class _Waypoint:
    def __init__(self, x): self._x = x; self.transform = Transform(Location(x, 0, 0), Rotation())
    def next(self, step): return [_Waypoint(self._x + step)]

class _Map:
    def get_spawn_points(self): return [Transform(Location(i*5, 0, 0)) for i in range(20)]
    def get_waypoint(self, loc): return _Waypoint(getattr(loc, "x", 0.0))

class _Snapshot:
    def __init__(self, t): self.timestamp = types.SimpleNamespace(elapsed_seconds=t)


class _Sensor:
    def __init__(self, bp): self.bp = bp; self._cb = None
    def listen(self, cb): self._cb = cb
    def stop(self): pass
    def destroy(self): pass

class _Vehicle:
    def __init__(self, bp): self.bp = bp; self._loc = Location(0, 0, 0)
    def set_autopilot(self, *a): pass
    def apply_control(self, c): self._loc.x += c.throttle * 2.0
    def get_velocity(self): return Vector3D(8.0, 0, 0)
    def get_location(self): return self._loc
    def get_transform(self): return Transform(self._loc, Rotation())
    def destroy(self): pass

class _Walker(_Vehicle):
    pass

class _Controller(_Sensor):
    def start(self): pass
    def go_to_location(self, l): pass
    def set_max_speed(self, s): pass


class _Settings:
    def __init__(self): self.synchronous_mode = False; self.fixed_delta_seconds = 0.0

class _TM:
    def get_port(self): return 8000
    def set_synchronous_mode(self, b): pass
    def vehicle_percentage_speed_difference(self, *a): pass
    def distance_to_leading_vehicle(self, *a): pass
    def ignore_lights_percentage(self, *a): pass
    def ignore_signs_percentage(self, *a): pass
    def random_left_lanechange_percentage(self, *a): pass
    def random_right_lanechange_percentage(self, *a): pass
    def vehicle_lane_offset(self, *a): pass

class _World:
    def __init__(self): self._settings = _Settings(); self._t = 0.0
    def get_blueprint_library(self): return _BPLib()
    def get_map(self): return _Map()
    def get_settings(self): return self._settings
    def apply_settings(self, s): self._settings = s
    def set_weather(self, w): pass
    def get_snapshot(self): return _Snapshot(self._t)
    def get_random_location_from_navigation(self): return Location(5, 5, 0)
    def spawn_actor(self, bp, tf, attach_to=None):
        return _make_actor(bp)
    def try_spawn_actor(self, bp, tf, attach_to=None):
        return _make_actor(bp)
    def tick(self): self._t += 0.05

def _make_actor(bp):
    bid = bp.id
    if bid.startswith("sensor") or bid.startswith("controller"):
        return _Controller(bp) if "controller" in bid else _Sensor(bp)
    if bid.startswith("walker"):
        return _Walker(bp)
    return _Vehicle(bp)

class Client:
    def __init__(self, host, port): pass
    def set_timeout(self, t): pass
    def get_world(self): return _World()
    def load_world(self, town): return _World()
    def get_trafficmanager(self, *a): return _TM()


def install():
    m = types.ModuleType("carla")
    for name in ("Location", "Rotation", "Transform", "Vector3D", "VehicleControl",
                 "WeatherParameters", "Client"):
        setattr(m, name, globals()[name])
    sys.modules["carla"] = m
    return m
