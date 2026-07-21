"""Vector + polyline geometry. Arc-length parametrized paths (lane graph edges)."""
import numpy as np


def resample(poly, step=1.0):
    poly = np.asarray(poly, float)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    n = max(int(cum[-1] / step), 2)
    s = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(s, cum, poly[:, 0]),
                     np.interp(s, cum, poly[:, 1])], axis=1)


def offset(poly, d):
    poly = np.asarray(poly, float)
    g = np.gradient(poly, axis=0)
    h = np.arctan2(g[:, 1], g[:, 0])
    return poly + d * np.stack([-np.sin(h), np.cos(h)], axis=1)


class Path:
    """Arc-length polyline with projection, used for routes and lane centrelines."""

    def __init__(self, pts, step=1.0):
        self.pts = resample(pts, step)
        seg = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        self.cum = np.concatenate([[0], np.cumsum(seg)])
        self.total = float(self.cum[-1])

    def point_at(self, s):
        s = np.clip(s, 0, self.total)
        return np.array([np.interp(s, self.cum, self.pts[:, 0]),
                         np.interp(s, self.cum, self.pts[:, 1])])

    def heading_at(self, s):
        s = float(np.clip(s, 0, self.total))
        a = self.point_at(max(s - 0.5, 0)); b = self.point_at(min(s + 0.5, self.total))
        d = b - a
        return float(np.arctan2(d[1], d[0]))

    def project(self, p):
        p = np.asarray(p, float)
        i = int(np.argmin(((self.pts - p) ** 2).sum(1)))
        s = float(self.cum[i]); h = self.heading_at(s); rel = p - self.pts[i]
        lon = np.cos(h) * rel[0] + np.sin(h) * rel[1]
        lat = -np.sin(h) * rel[0] + np.cos(h) * rel[1]
        return s + float(lon), float(lat)


def obb_corners(cx, cy, yaw, L, W):
    c, s = np.cos(yaw), np.sin(yaw)
    loc = np.array([[L/2, W/2], [L/2, -W/2], [-L/2, -W/2], [-L/2, W/2]])
    return loc @ np.array([[c, -s], [s, c]]).T + np.array([cx, cy])


def obb_overlap(r1, r2):
    for rect in (r1, r2):
        for i in range(4):
            e = rect[(i + 1) % 4] - rect[i]
            ax = np.array([-e[1], e[0]])
            p1, p2 = r1 @ ax, r2 @ ax
            if p1.max() < p2.min() or p2.max() < p1.min():
                return False
    return True
