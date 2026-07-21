"""Privileged expert autopilot — the upper-bound baseline (PDM-Lite-style).

Uses ground-truth world state (not sensors): follows the route, brakes for the
nearest hazard in its corridor (vehicles, VRUs, encroachments, red lights),
respects speed bumps, and does simple gap-based lateral avoidance. Serves as the
reference ceiling and as the demonstration policy for imitation-learning data.
"""
import numpy as np
from bharatsim.eval.env import Agent, Control


class ExpertAutopilot(Agent):
    def __init__(self, wheelbase=2.6, max_steer_deg=38.0):
        self.L = wheelbase; self.max_steer = max_steer_deg

    def act(self, obs):
        # obs carries privileged handles injected by the runner
        w = obs._world
        route, ego = w.route, w.ego
        s_e, lat_e = route.project(np.array([ego.x, ego.y]))

        # --- hazard scan in a corridor ahead ---
        gap, lead_v, want_lat = None, 0.0, 0.0
        for e in w.entities:
            if not e.alive:
                continue
            if getattr(e, "dynamic", True) is False and not getattr(e, "hard", True):
                continue                        # soft hazard: not a braking obstacle
            s, lat = route.project((e.x, e.y))
            for dt_ in (0.0, 0.8, 1.6):     # anticipate motion
                if hasattr(e, "path"):
                    sf = e.s + e.v * dt_; p = e.path.point_at(sf)
                    s2, lat2 = route.project(p)
                elif getattr(e, "active", False):
                    p = np.array([e.x, e.y]) + e.vel * dt_
                    s2, lat2 = route.project(p)
                else:
                    s2, lat2 = s, lat
                if s2 > s_e + 0.5 and s2 - s_e < 45 and abs(lat2 - lat_e) < 2.2:
                    g = s2 - s_e - (getattr(e, "L", 1) + ego.L) / 2
                    if gap is None or g < gap:
                        gap, lead_v = max(g, 0.0), getattr(e, "v", 0.0)
                        want_lat = np.clip(lat_e + np.sign(lat2 - lat_e + 1e-3) * -2.0,
                                           -2.0, 2.0)
                    break
        # red light as phantom hazard
        if w.junction is not None:
            sg = w.junction.stopline_gap("route", s_e) or w.junction.stopline_gap("same", s_e)
            if sg is not None and (gap is None or sg < gap):
                gap, lead_v = sg, 0.0

        # --- longitudinal (IDM-ish) ---
        v = ego.v; v0 = w.speed_limit
        if gap is None:
            target = v0
        else:
            T, s0 = 1.4, 4.0
            safe = s0 + v * T
            target = 0.0 if gap < s0 else min(v0, max(gap - s0, 0) / max(T, 0.7) + lead_v)
        # speed bump advisory
        for e in w.entities:
            if getattr(e, "kind", "") == "speedbump":
                s, _ = route.project((e.x, e.y))
                if 0 < s - s_e < 8:
                    target = min(target, 3.0)

        err = target - v
        thr = float(np.clip(err * 0.5, 0, 1)); brk = float(np.clip(-err * 0.4, 0, 1)) if err < -0.2 else 0.0

        # --- lateral: pursue a point on route, biased by avoidance gap ---
        la = min(max(3.0 + 0.6 * v, 4.0), 12.0)
        pt = route.point_at(s_e + la)
        rel = pt - np.array([ego.x, ego.y])
        c, sn = np.cos(-ego.yaw), np.sin(-ego.yaw)
        ex = c * rel[0] - sn * rel[1]; ey = sn * rel[0] + c * rel[1]
        if gap is not None and gap < 15:
            ey += (want_lat - lat_e) * 0.5      # nudge toward clear side
        import math
        alpha = math.atan2(ey, max(ex, 1e-3))
        delta = math.atan2(2 * self.L * math.sin(alpha), max(math.hypot(ex, ey), 1e-3))
        steer = float(np.clip(math.degrees(delta) / self.max_steer, -1, 1))
        return Control(steer, thr, brk)
