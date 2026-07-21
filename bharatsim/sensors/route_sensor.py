"""Route/navigation sensor: local goal direction + lateral offset + curvature.

Gives a driving policy the minimal map context a real vehicle has (a route from
a navigation system) without leaking full privileged state.
"""
import numpy as np


def route_features(world, lookahead=(5.0, 10.0, 20.0)):
    r = world.route; ego = world.ego
    s, lat = r.project(np.array([ego.x, ego.y]))
    feats = {"lateral_offset": float(lat)}
    dirs = []
    for la in lookahead:
        h = r.heading_at(min(s + la, r.total))
        dirs.append(float((h - ego.yaw + np.pi) % (2 * np.pi) - np.pi))
    feats["heading_errors"] = dirs
    feats["curvature"] = float(dirs[-1] - dirs[0])
    feats["dist_to_goal"] = float(r.total - s)
    return feats
