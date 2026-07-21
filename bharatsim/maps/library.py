"""Map library. Each builder returns a MapData with a route + named traffic
paths + drivable geometry. Extend by adding a builder and registering it.
"""
import numpy as np
from bharatsim.core.geometry import Path, offset


class MapData:
    def __init__(self, route, paths, roads, speed_limit, name="map"):
        self.route = route
        self.paths = paths
        self.roads = roads              # [(centerline ndarray, width)]
        self.speed_limit = speed_limit
        self.name = name


def _two_way(length, width, speed, name):
    center = np.array([[-15., 0.], [length, 0.]])
    route = Path([[-8., -width/4], [length - 8, -width/4]])
    paths = {"same": Path([[-8., -width/4], [length, -width/4]]),
             "oncoming": Path([[length, width/4], [-15., width/4]])}
    return MapData(route, paths, [(center, width)], speed, name)


def narrow_street(seed=0):
    return _two_way(180, 7.0, 8.0, "narrow_street")


def arterial_4lane(seed=0):
    center = np.array([[-15., 0.], [300., 0.]])
    route = Path([[-8., -3.0], [300., -3.0]])
    paths = {"l0": Path([[-8., -5.0], [300., -5.0]]),
             "l1": Path([[-8., -1.5], [300., -1.5]]),
             "onc0": Path([[300., 1.5], [-15., 1.5]]),
             "onc1": Path([[300., 5.0], [-15., 5.0]]),
             "same": Path([[-8., -3.0], [300., -3.0]])}
    return MapData(route, paths, [(center, 14.0)], 14.0, "arterial_4lane")


def uncontrolled_junction(seed=0):
    A = np.array([[-30., 0.], [200., 0.]]); B = np.array([[90., -80.], [90., 120.]])
    route = Path([[-8., -1.6], [180., -1.6]])
    paths = {"same": Path([[-8., -1.6], [200., -1.6]]),
             "oncoming": Path([[200., 1.6], [-30., 1.6]]),
             "cross_a": Path([[90., 120.], [90., -80.]]),
             "cross_b": Path([[86., -80.], [86., 120.]])}
    return MapData(route, paths, [(A, 9.0), (B, 9.0)], 8.5, "uncontrolled_junction")


def roundabout(seed=0):
    th = np.linspace(np.pi, -np.pi/2, 30)
    r = 16.0
    ring = np.stack([r*np.cos(th)+0, r*np.sin(th)+0], axis=1)
    entry = np.array([[-40., -3.], [-r-2, -3.]])
    route = Path(np.vstack([entry, ring, [[r+2, 2.], [40., 2.]]]))
    ring_path = Path(np.vstack([np.stack([r*np.cos(t) for t in np.linspace(np.pi,-np.pi,40)],),
                                np.stack([r*np.sin(t) for t in np.linspace(np.pi,-np.pi,40)])]).T)
    paths = {"ring": ring_path, "same": route}
    roads = [(np.array([[-40.,-3.],[-r,-3.]]), 7.0),
             (ring, 8.0), (np.array([[r,2.],[40.,2.]]), 7.0)]
    return MapData(route, paths, roads, 7.0, "roundabout")


def village_curvy(seed=0):
    x = np.linspace(0, 220, 70)
    center = np.stack([x, 8*np.sin(x/24)], axis=1)
    route = Path(offset(center, -1.3))
    paths = {"same": Path(offset(center, -1.3)),
             "oncoming": Path(offset(center, 1.3)[::-1])}
    return MapData(route, paths, [(center, 6.5)], 6.5, "village_curvy")


def highway(seed=0):
    center = np.array([[-20., 0.], [420., 0.]])
    route = Path([[-10., -2.0], [420., -2.0]])
    paths = {"same": Path([[-10., -2.0], [420., -2.0]]),
             "fast": Path([[-10., 2.0], [420., 2.0]]),
             "shoulder": Path([[-10., -5.0], [420., -5.0]])}
    return MapData(route, paths, [(center, 12.0)], 18.0, "highway")


MAPS = {"narrow_street": narrow_street, "arterial_4lane": arterial_4lane,
        "uncontrolled_junction": uncontrolled_junction, "roundabout": roundabout,
        "village_curvy": village_curvy, "highway": highway}
