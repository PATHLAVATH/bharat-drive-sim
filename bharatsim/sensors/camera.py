"""Multi-camera semantic + depth sensor.

Renders the world to per-camera RGB semantic images and metric depth maps via a
pinhole model with painter's-algorithm compositing and near-plane clipping. This
is the sensor surface a perception model consumes (in addition to BEV). It is
2.5D flat-shaded (not photoreal) — swap this module for a 3D/CARLA backend later
while keeping the CameraObs contract.

Conventions: camera x-right, y-down, z-forward; ego x-forward, y-left, z-up.
"""
from dataclasses import dataclass
import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

IMG_H, IMG_W = 120, 200
FOV_DEG = 75.0
CAM_YAWS = np.radians([60.0, 0.0, -60.0])       # left, front, right
CAM_POS = np.array([1.2, 0.0, 1.5])             # ego frame (x fwd, y left, z up)

# semantic palette (RGB) + class id used in the SEMANTIC channel
PALETTE = {
    "sky":   (135, 206, 235, 0),
    "ground":(70, 70, 70, 1),      # road/ground
    "veh":   (200, 40, 40, 2),
    "vru":   (40, 40, 220, 3),
    "haz_s": (150, 110, 30, 4),
    "haz_h": (120, 20, 120, 5),
    "signal_r": (230, 30, 30, 6),
    "signal_g": (30, 200, 30, 6),
}


@dataclass
class CameraObs:
    rgb: np.ndarray        # (N_cam, H, W, 3) uint8 semantic-colored
    depth: np.ndarray      # (N_cam, H, W) float32 metres (inf = sky)
    semantic: np.ndarray   # (N_cam, H, W) uint8 class ids
    intrinsics: np.ndarray # (N_cam, 3, 3)
    cam2ego: np.ndarray    # (N_cam, 4, 4)


def intrinsics():
    f = IMG_W / (2 * np.tan(np.radians(FOV_DEG) / 2))
    K = np.array([[f, 0, IMG_W / 2], [0, f, IMG_H / 2], [0, 0, 1]], np.float32)
    return np.stack([K] * len(CAM_YAWS))


def cam2ego_mats():
    mats = []
    for yaw in CAM_YAWS:
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[s, 0, c], [-c, 0, s], [0, -1, 0]], float)  # cam->ego basis
        m = np.eye(4); m[:3, :3] = R; m[:3, 3] = CAM_POS
        mats.append(m)
    return np.stack(mats).astype(np.float32)


def _to_ego(pts, ego):
    rel = np.asarray(pts, float) - np.array([ego.x, ego.y])
    c, s = np.cos(-ego.yaw), np.sin(-ego.yaw)
    return np.stack([c * rel[..., 0] - s * rel[..., 1],
                     s * rel[..., 0] + c * rel[..., 1]], axis=-1)


def _quad_strip(center, width, step=4.0):
    from bharatsim.core.geometry import resample, offset
    p = resample(center, step); l = offset(p, width / 2); r = offset(p, -width / 2)
    return [np.array([l[i], l[i + 1], r[i + 1], r[i]]) for i in range(len(p) - 1)]


def _clip_near(poly_cam, z=0.4):
    out = []
    n = len(poly_cam)
    for i in range(n):
        a, b = poly_cam[i], poly_cam[(i + 1) % n]
        ain, bin_ = a[2] >= z, b[2] >= z
        if ain:
            out.append(a)
        if ain != bin_:
            t = (z - a[2]) / (b[2] - a[2]); out.append(a + t * (b - a))
    return np.array(out) if len(out) >= 3 else None


def render_cameras(world):
    """-> CameraObs. Requires OpenCV; returns zeros if unavailable."""
    K = intrinsics(); c2e = cam2ego_mats()
    N = len(CAM_YAWS)
    rgb = np.zeros((N, IMG_H, IMG_W, 3), np.uint8)
    depth = np.full((N, IMG_H, IMG_W), np.inf, np.float32)
    sem = np.zeros((N, IMG_H, IMG_W), np.uint8)
    if not _CV2:
        return CameraObs(rgb, depth, sem, K, c2e)
    ego = world.ego

    # gather 3D primitives in ego frame: (mean_dist, corners3d, rgb, class)
    prims = []
    for center, width in world.map.roads:
        for q in _quad_strip(center, width):
            qe = _to_ego(q, ego)
            q3 = np.concatenate([qe, np.zeros((len(qe), 1))], 1)
            prims.append((np.hypot(*qe.mean(0)), q3, PALETTE["ground"][:3], 1))
    for e in world.entities:
        if not e.alive:
            continue
        ce = _to_ego(e.corners(), ego)
        d = np.hypot(*ce.mean(0))
        if d > 60:
            continue
        h = 1.6 if e.kind == "vehicle" else 1.7 if e.kind == "vru" else 0.3
        col, cls = ((PALETTE["veh"][:3], 2) if e.kind == "vehicle" else
                    (PALETTE["vru"][:3], 3) if e.kind == "vru" else
                    (PALETTE["haz_h"][:3], 5) if getattr(e, "hard", True)
                    else (PALETTE["haz_s"][:3], 4))
        bot = np.concatenate([ce, np.zeros((4, 1))], 1)
        top = bot + np.array([0, 0, h])
        prims.append((d, np.concatenate([bot, top]), col, cls))
    prims.sort(key=lambda p: -p[0])   # far first (painter's)

    for ci in range(N):
        R = c2e[ci, :3, :3]; t = c2e[ci, :3, 3]
        img = rgb[ci]; img[:IMG_H // 2] = PALETTE["sky"][:3]
        img[IMG_H // 2:] = (90, 90, 90)
        for dist, corners, col, cls in prims:
            cam = (corners - t) @ R
            uvs, zs = [], []
            for pt in cam:
                if pt[2] >= 0.4:
                    uv = K[ci] @ (pt / pt[2]); uvs.append(uv[:2]); zs.append(pt[2])
            if len(uvs) < 3:
                continue
            hull = cv2.convexHull(np.round(np.array(uvs)).astype(np.int32))
            cv2.fillConvexPoly(img, hull, col)
            m = np.zeros((IMG_H, IMG_W), np.uint8)
            cv2.fillConvexPoly(m, hull, 1)
            mask = m.astype(bool)
            zval = float(np.mean(zs))
            closer = mask & (zval < depth[ci])
            depth[ci][closer] = zval
            sem[ci][closer] = cls
    return CameraObs(rgb, depth, sem, K, c2e)
