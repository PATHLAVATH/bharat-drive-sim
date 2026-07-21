"""BEV semantic + occupancy sensors. Lightweight (numpy+cv2), ego-centric.

Provides the observation surface a driving model consumes: a semantic BEV
raster and a stack of future dynamic-occupancy frames, plus optional camera
placeholders (extend with a 3D renderer or CARLA bridge for photoreal)."""
import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

BEV_RANGE = 40.0
BEV_RES = 0.5
BEV_GRID = int(2 * BEV_RANGE / BEV_RES)     # 160

CLASSES = ["free", "road", "vehicle", "vru", "hazard_soft", "hazard_hard"]
N_CLASSES = len(CLASSES)


def _to_ego(pts, ego):
    rel = np.asarray(pts, float) - np.array([ego.x, ego.y])
    c, s = np.cos(-ego.yaw), np.sin(-ego.yaw)
    return np.stack([c*rel[..., 0] - s*rel[..., 1],
                     s*rel[..., 0] + c*rel[..., 1]], axis=-1)


def _cells(pts_ego):
    g = (pts_ego + BEV_RANGE) / BEV_RES
    return np.round(np.stack([g[..., 1], g[..., 0]], axis=-1)).astype(np.int32)


def _quad_strip(center, width, step=4.0):
    from bharatsim.core.geometry import resample, offset
    p = resample(center, step); l = offset(p, width/2); r = offset(p, -width/2)
    return [np.array([l[i], l[i+1], r[i+1], r[i]]) for i in range(len(p)-1)]


def render_bev(world):
    """Ego-centric semantic BEV, uint8 (BEV_GRID, BEV_GRID)."""
    grid = np.zeros((BEV_GRID, BEV_GRID), np.uint8)
    ego = world.ego
    if _CV2:
        for center, width in world.map.roads:
            for q in _quad_strip(center, width):
                cv2.fillPoly(grid, [_cells(_to_ego(q, ego))], 1)
    for e in world.entities:
        if not e.alive:
            continue
        if e.kind == "vehicle":
            cls = 2
        elif e.kind == "vru":
            cls = 3
        elif getattr(e, "hard", True) is False:
            cls = 4
        else:
            cls = 5
        if _CV2:
            cv2.fillPoly(grid, [_cells(_to_ego(e.corners(), ego))], cls)
    return grid


def render_future(world, steps=4, dt=0.5):
    """Future dynamic occupancy (steps, G, G) uint8, current ego frame."""
    ego = world.ego
    out = np.zeros((steps, BEV_GRID, BEV_GRID), np.uint8)
    if not _CV2:
        return out
    for k in range(1, steps + 1):
        layer = out[k-1]
        for e in world.entities:
            if not e.alive or not e.dynamic:
                continue
            if hasattr(e, "path"):
                s_f = e.s + e.v * k * dt
                p = e.path.point_at(s_f); h = e.path.heading_at(s_f)
                n = np.array([-np.sin(h), np.cos(h)]); c = p + e.lat * n
                from bharatsim.core.geometry import obb_corners
                cv2.fillPoly(layer, [_cells(_to_ego(obb_corners(c[0], c[1], h, e.L, e.W), ego))], 1)
            elif hasattr(e, "vel"):
                fut = np.array([e.x, e.y]) + (e.vel * k * dt if e.active else 0)
                cv2.circle(layer, tuple(_cells(_to_ego(fut[None], ego))[0]), 1, 1, -1)
    return out
