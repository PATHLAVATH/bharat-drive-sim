"""Top-down episode recorder — watch what the agent did (CARLA spectator-style).

Captures a bird's-eye frame each tick (roads, entities, ego, route, signals) and
writes an animated GIF or PNG sequence. Pure OpenCV; no display needed.
"""
import os
import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

PX = 6.0            # pixels per metre
SIZE = 700


class Recorder:
    def __init__(self, follow=True, size=SIZE, px=PX):
        self.frames = []
        self.follow = follow
        self.size = size
        self.px = px

    def _w2p(self, pt, cx, cy):
        return (int(self.size / 2 + (pt[0] - cx) * self.px),
                int(self.size / 2 - (pt[1] - cy) * self.px))

    def capture(self, world):
        if not _CV2:
            return
        cx, cy = (world.ego.x, world.ego.y) if self.follow else (60, 0)
        img = np.full((self.size, self.size, 3), 30, np.uint8)
        # roads
        from bharatsim.core.geometry import resample, offset
        for center, width in world.map.roads:
            p = resample(center, 3.0)
            l = offset(p, width / 2); r = offset(p, -width / 2)
            poly = np.array([self._w2p(q, cx, cy) for q in np.vstack([l, r[::-1]])])
            cv2.fillPoly(img, [poly], (70, 70, 70))
        # route line
        rp = world.route.pts[::3]
        for i in range(len(rp) - 1):
            cv2.line(img, self._w2p(rp[i], cx, cy), self._w2p(rp[i + 1], cx, cy),
                     (0, 120, 0), 1)
        # signals
        if world.junction is not None:
            for lt in world.junction.lights:
                p = world.route.point_at(lt.stop_s)
                col = {"R": (0, 0, 230), "Y": (0, 200, 230), "G": (0, 200, 0)}[lt.state]
                cv2.circle(img, self._w2p(p, cx, cy), 4, col, -1)
        # entities
        for e in world.entities:
            if not e.alive:
                continue
            col = ((40, 40, 200) if e.kind == "vru" else
                   (200, 160, 40) if e.kind == "vehicle" else
                   (120, 40, 120) if getattr(e, "hard", True) else (40, 110, 150))
            pts = np.array([self._w2p(c, cx, cy) for c in e.corners()])
            cv2.fillConvexPoly(img, pts, col)
        # ego
        ego_pts = np.array([self._w2p(c, cx, cy) for c in world.ego.corners()])
        cv2.fillConvexPoly(img, ego_pts, (60, 230, 60))
        cv2.putText(img, f"t={world.t:4.1f}s v={world.ego.v:4.1f} rc={world.progress*100:4.0f}%",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
        self.frames.append(img)

    def save_gif(self, path, fps=10):
        if not self.frames:
            return None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            import imageio
            imageio.mimsave(path, [f[:, :, ::-1] for f in self.frames], fps=fps)
            return path
        except ImportError:
            # fallback: dump key PNGs
            base = path.rsplit(".", 1)[0]
            for i in range(0, len(self.frames), max(len(self.frames) // 12, 1)):
                cv2.imwrite(f"{base}_{i:04d}.png", self.frames[i])
            return base + "_*.png"
