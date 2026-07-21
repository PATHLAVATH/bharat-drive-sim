"""Photoreal CARLA sensor rig -> bharatsim Observation.

Attaches RGB + depth + semantic-segmentation cameras (3-cam ring by default,
extendable to surround) plus an optional LiDAR, buffers their async callbacks,
and assembles the SAME Observation fields the pure-Python sim produces — so any
bharatsim Agent runs unchanged against photoreal CARLA.

CARLA semantic-seg tags are remapped to bharatsim classes (0 free,1 road,
2 vehicle,3 vru,4 soft-hazard,5 hard-hazard) so a model trained in sim transfers.
"""
from __future__ import annotations
import math
import numpy as np

# 3-cam ring (deg yaw) matching bharatsim.sensors.camera
CAM_LAYOUT = [("left", 60.0), ("front", 0.0), ("right", -60.0)]
IMG_H, IMG_W, FOV = 120, 200, 75.0
CAM_XYZ = (1.2, 0.0, 1.5)

# CARLA CityScapes-style seg tag -> bharatsim class id
SEG_REMAP = {
    0: 0,    # unlabeled -> free
    1: 1,    # roads
    24: 1,   # road line -> road
    7: 1,    # (0.9.15 road id variants); harmless if absent
    14: 2, 15: 2, 16: 2,      # car/truck/bus -> vehicle
    12: 3, 13: 3,             # pedestrian/rider -> vru
    19: 5, 20: 5,             # static/dynamic objects -> hard hazard
}


def _bgra_to_rgb(image):
    a = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)
    return a[:, :, :3][:, :, ::-1].copy()


def _depth_meters(image):
    a = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)
    r, g, b = a[:, :, 2].astype(np.float32), a[:, :, 1].astype(np.float32), a[:, :, 0].astype(np.float32)
    norm = (r + g * 256 + b * 256 * 256) / (256 ** 3 - 1)
    return norm * 1000.0


def _remap_seg(image):
    a = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)
    tags = a[:, :, 2]                     # CARLA stores class in R
    out = np.zeros_like(tags)
    for k, v in SEG_REMAP.items():
        out[tags == k] = v
    return out


class CarlaCameraRig:
    def __init__(self, carla, world, ego, img_h=IMG_H, img_w=IMG_W, fov=FOV,
                 use_lidar=False):
        self.carla = carla
        self.world = world
        self.ego = ego
        self.h, self.w, self.fov = img_h, img_w, fov
        self.sensors = []
        self.buffers = {}      # (name, modality) -> latest ndarray
        self.use_lidar = use_lidar
        self._spawn()

    def _cam_bp(self, kind):
        bl = self.world.get_blueprint_library()
        bp = bl.find(kind)
        bp.set_attribute("image_size_x", str(self.w))
        bp.set_attribute("image_size_y", str(self.h))
        bp.set_attribute("fov", str(self.fov))
        return bp

    def _spawn(self):
        carla = self.carla
        for name, yaw in CAM_LAYOUT:
            tf = carla.Transform(carla.Location(x=CAM_XYZ[0], y=0.0, z=CAM_XYZ[2]),
                                 carla.Rotation(yaw=yaw))
            for modality, kind, cb in (
                ("rgb", "sensor.camera.rgb", _bgra_to_rgb),
                ("depth", "sensor.camera.depth", _depth_meters),
                ("seg", "sensor.camera.semantic_segmentation", _remap_seg),
            ):
                s = self.world.spawn_actor(self._cam_bp(kind), tf, attach_to=self.ego)
                key = (name, modality)
                s.listen(lambda data, k=key, fn=cb: self.buffers.__setitem__(k, fn(data)))
                self.sensors.append(s)
        if self.use_lidar:
            bl = self.world.get_blueprint_library()
            lb = bl.find("sensor.lidar.ray_cast")
            lb.set_attribute("range", "60"); lb.set_attribute("rotation_frequency", "20")
            lb.set_attribute("channels", "32"); lb.set_attribute("points_per_second", "300000")
            tf = carla.Transform(carla.Location(x=0.0, z=1.8))
            s = self.world.spawn_actor(lb, tf, attach_to=self.ego)
            s.listen(lambda d: self.buffers.__setitem__(("top", "lidar"),
                     np.frombuffer(d.raw_data, np.float32).reshape(-1, 4).copy()))
            self.sensors.append(s)

    def intrinsics(self):
        f = self.w / (2 * math.tan(math.radians(self.fov) / 2))
        K = np.array([[f, 0, self.w / 2], [0, f, self.h / 2], [0, 0, 1]], np.float32)
        return np.stack([K] * len(CAM_LAYOUT))

    def cam2ego(self):
        mats = []
        for _, yaw in CAM_LAYOUT:
            r = math.radians(yaw); c, s = math.cos(r), math.sin(r)
            R = np.array([[s, 0, c], [-c, 0, s], [0, -1, 0]], float)
            m = np.eye(4); m[:3, :3] = R; m[:3, 3] = CAM_XYZ
            mats.append(m)
        return np.stack(mats).astype(np.float32)

    def read(self):
        """Assemble stacked RGB/depth/semantic in CAM_LAYOUT order (front-centered)."""
        rgb, depth, sem = [], [], []
        for name, _ in CAM_LAYOUT:
            rgb.append(self.buffers.get((name, "rgb"), np.zeros((self.h, self.w, 3), np.uint8)))
            depth.append(self.buffers.get((name, "depth"), np.full((self.h, self.w), np.inf, np.float32)))
            sem.append(self.buffers.get((name, "seg"), np.zeros((self.h, self.w), np.uint8)))
        from bharatsim.sensors.camera import CameraObs
        return CameraObs(np.stack(rgb), np.stack(depth), np.stack(sem),
                         self.intrinsics(), self.cam2ego())

    def destroy(self):
        for s in self.sensors:
            try:
                s.stop(); s.destroy()
            except Exception:
                pass
        self.sensors = []
