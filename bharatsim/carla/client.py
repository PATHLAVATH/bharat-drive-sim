"""CARLA client + synchronous world setup + Indian weather presets.

`import carla` is done lazily so the rest of bharatsim works without CARLA
installed. Install CARLA 0.9.15 (or matching PythonAPI) on a GPU machine and
run `CarlaUE4.sh -RenderOffScreen` before using anything here.
"""
from __future__ import annotations
import time


def _carla():
    try:
        import carla  # noqa
        return carla
    except ImportError as e:
        raise RuntimeError(
            "CARLA PythonAPI not found. Install CARLA 0.9.15 and add its "
            "PythonAPI/carla/dist/*.egg (or `pip install carla`) to PYTHONPATH. "
            "See docs/CARLA.md."
        ) from e


# Indian conditions map to CARLA WeatherParameters
WEATHER = {
    "clear":   dict(cloudiness=10, precipitation=0, sun_altitude_angle=70,
                    fog_density=0, wetness=0),
    "monsoon": dict(cloudiness=90, precipitation=80, precipitation_deposits=60,
                    sun_altitude_angle=35, fog_density=20, wetness=70),
    "dust":    dict(cloudiness=60, precipitation=0, sun_altitude_angle=25,
                    fog_density=40, wetness=0),
    "night":   dict(cloudiness=30, precipitation=0, sun_altitude_angle=-15,
                    fog_density=10, wetness=0),
    "dawn":    dict(cloudiness=40, precipitation=0, sun_altitude_angle=8,
                    fog_density=15, wetness=10),
}

# CARLA towns that best resemble Indian road classes (closest available stock maps).
TOWN_FOR_MAP = {
    "narrow_street": "Town01", "arterial_4lane": "Town10HD",
    "uncontrolled_junction": "Town03", "roundabout": "Town03",
    "village_curvy": "Town07", "highway": "Town04",
}


class CarlaClient:
    def __init__(self, host="localhost", port=2000, timeout=20.0,
                 fixed_dt=0.05, town=None):
        self.carla = _carla()
        self.client = self.carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.fixed_dt = fixed_dt
        self.world = None
        if town:
            self.load_town(town)
        else:
            self.world = self.client.get_world()
        self._orig_settings = None

    def load_town(self, town):
        self.world = self.client.load_world(town)
        return self.world

    def set_weather(self, label="clear"):
        w = self.carla.WeatherParameters(**WEATHER.get(label, WEATHER["clear"]))
        self.world.set_weather(w)

    def enable_sync(self):
        self._orig_settings = self.world.get_settings()
        s = self.world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = self.fixed_dt
        self.world.apply_settings(s)
        tm = self.client.get_trafficmanager()
        tm.set_synchronous_mode(True)
        return tm

    def tick(self):
        self.world.tick()

    def restore(self):
        if self._orig_settings is not None:
            self.world.apply_settings(self._orig_settings)
