"""Scenario DSL — author scenarios in YAML, no code required.

A scenario file:

    name: cow_and_swarm
    map: narrow_street
    ego_speed: 5.0
    environment: {visibility: 1.0, label: clear}
    actors:
      - {type: cow, path: same, s: 30, lat: 0.0}
      - {type: two_wheeler, path: same, s: 20, lat: -1.5, count: 4, spread_s: 6}
    vrus:
      - {cross_at: 45, side: -1, speed: 1.3, trigger: 18, count: 3, spread: 3}
    hazards:
      - {type: pothole, along: [0.2, 0.9], count: 8}
      - {type: speedbump, at: 60}
      - {type: encroach, at: 50, side: 1}

`count`/`spread` fields expand into multiple randomized instances (seeded).
"""
import numpy as np
import yaml

from bharatsim.core.world import World
from bharatsim.core.entities import StaticHazard
from bharatsim.core.geometry import obb_corners
from bharatsim.agents.traffic import TrafficAgent, VRU
from bharatsim.maps.library import MAPS


def _env(cfg):
    if not cfg:
        return None
    return type("Env", (), {"visibility": cfg.get("visibility", 1.0),
                            "label": cfg.get("label", "clear")})()


def build_from_dict(spec, seed=0):
    rng = np.random.default_rng(seed + 777)
    mapdata = MAPS[spec["map"]](seed)
    world = World(mapdata, mapdata.route, ego_speed0=spec.get("ego_speed", 5.0),
                  environment=_env(spec.get("environment")))

    for a in spec.get("actors", []):
        path = mapdata.paths[a.get("path", "same")]
        count = a.get("count", 1)
        for i in range(count):
            s = a["s"] + i * a.get("spread_s", 6) + float(rng.uniform(-2, 2))
            lat = a.get("lat", 0.0) + float(rng.uniform(-0.6, 0.6)) * (count > 1)
            beh = dict(a.get("behavior", {}))
            world.add(TrafficAgent(path, a["type"], s0=s, lat=lat,
                                   seed=int(rng.integers(1 << 30)), behavior=beh))

    for v in spec.get("vrus", []):
        s0 = v["cross_at"]; side = v.get("side", -1)
        p = mapdata.route.point_at(s0); h = mapdata.route.heading_at(s0)
        n = np.array([-np.sin(h), np.cos(h)])
        for i in range(v.get("count", 1)):
            off = side * (4 + rng.uniform(0, v.get("spread", 2)))
            start = p + off * n + h * 0
            world.add(VRU(start[0] + rng.uniform(-3, 3), start[1],
                          direction=-side * n, speed=v.get("speed", 1.2),
                          trigger=v.get("trigger", 20), radius=v.get("radius", 0.35),
                          seed=int(rng.integers(1 << 30))))

    for hz in spec.get("hazards", []):
        t = hz["type"]
        if t == "pothole":
            lo, hi = hz.get("along", [0.15, 0.9]); route = mapdata.route
            for _ in range(hz.get("count", 6)):
                s = float(rng.uniform(lo, hi)) * route.total
                pp = route.point_at(s); hh = route.heading_at(s)
                nn = np.array([-np.sin(hh), np.cos(hh)])
                c = pp + rng.uniform(-2.5, 2.5) * nn
                world.add(StaticHazard(c[0], c[1], hh, L=rng.uniform(0.6, 1.4),
                                       W=rng.uniform(0.6, 1.4), kind="pothole",
                                       hard=False))
        elif t == "speedbump":
            s = hz["at"]; pp = mapdata.route.point_at(s); hh = mapdata.route.heading_at(s)
            world.add(StaticHazard(pp[0], pp[1], hh, L=1.0, W=hz.get("width", 6.0),
                                   kind="speedbump", hard=False, slow_to=3.0))
        elif t == "encroach":
            s = hz["at"]; side = hz.get("side", 1)
            pp = mapdata.route.point_at(s); hh = mapdata.route.heading_at(s)
            nn = np.array([-np.sin(hh), np.cos(hh)]); c = pp + side * 1.4 * nn
            world.add(StaticHazard(c[0], c[1], hh, L=3.5, W=2.0, kind="encroach",
                                   hard=True))
    return world


def load_scenario(path, seed=0):
    with open(path) as f:
        return build_from_dict(yaml.safe_load(f), seed)
