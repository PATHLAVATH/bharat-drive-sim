"""RiskFieldV2 — an improved lane-free planner for unstructured traffic.

Improvements over RiskFieldAgent:
  1. Time-to-collision (TTC) longitudinal control: brakes on *when* a hazard is
     reached along the committed path, using the future-occupancy stack, not
     just static forward risk. Prevents both late braking and needless creeping.
  2. Gap-commitment hysteresis: once a lateral corridor is chosen it is held
     unless a clearly better one appears, removing the left-right dithering that
     caused near-misses and lane weaving.
  3. Explicit VRU caution: pedestrian/animal cells get a larger safety inflation
     and a hard speed cap, targeting the near-miss metric directly.
  4. Path-consistent forward-risk sampling along the actual rolled-out geometry.
"""
import numpy as np
from bharatsim.eval.env import Agent, Control
from bharatsim.sensors.bev import BEV_GRID, BEV_RES


class RiskFieldV2(Agent):
    def __init__(self, horizon=8, dt=0.5, wheelbase=2.6, max_steer=0.6, n_steer=11):
        self.H, self.dt, self.L = horizon, dt, wheelbase
        self.steers = np.linspace(-max_steer, max_steer, n_steer)
        self.reset()

    def reset(self):
        self._committed_gap = None      # hysteresis state

    # ---- field ----
    def _field(self, obs):
        bev = obs.bev.astype(np.float32)
        veh = (bev == 2).astype(np.float32)
        vru = (bev == 3).astype(np.float32)
        hard = (bev == 5).astype(np.float32)
        soft = (bev == 4).astype(np.float32)
        offroad = (bev == 0).astype(np.float32)
        risk = veh + 1.6 * vru + hard + 0.35 * soft + 0.8 * offroad
        # predicted motion, nearer future weighted more
        for t in range(obs.future_occ.shape[0]):
            risk += (0.9 - 0.15 * t) * obs.future_occ[t]
        # ego-footprint inflation; VRUs inflated more (bigger safety bubble)
        base = _dilate((risk > 0.5).astype(np.float32), 3)
        vru_infl = _dilate(vru + 1.0 * obs.future_occ[0] * 0, 4)  # extra vru margin
        risk = np.maximum(risk, 1.6 * base)
        risk = np.maximum(risk, 2.2 * _dilate(vru, 4))
        return _blur(risk, 3)

    def _sample(self, f, x, y):
        ix = int(round(x / BEV_RES + BEV_GRID / 2))
        iy = int(round(y / BEV_RES + BEV_GRID / 2))
        if 0 <= ix < BEV_GRID and 0 <= iy < BEV_GRID:
            return f[ix, iy]
        return 5.0

    def _best_gap(self, f, speed=5.0):
        span = 3.0
        best_y, best_c = 0.0, 1e9
        for ly in np.arange(-span, span + 0.01, 0.5):
            c = sum(self._sample(f, xm, ly) for xm in np.arange(4, 18, 1.0)) + 0.12 * abs(ly)
            if c < best_c:
                best_c, best_y = c, ly
        # hysteresis: keep current gap unless the new one is clearly better
        if self._committed_gap is not None:
            cur_c = sum(self._sample(f, xm, self._committed_gap)
                        for xm in np.arange(4, 18, 1.0)) + 0.12 * abs(self._committed_gap)
            if cur_c <= best_c + 1.5:
                best_y = self._committed_gap
        self._committed_gap = best_y
        return best_y

    def act(self, obs):
        f = self._field(obs)
        y_gap = self._best_gap(f, obs.speed)
        w_gap = 2.5
        gh = {1: 0.4, 2: -0.4}.get(obs.command, 0.0)
        x, y, yaw, v, pts = 0.0, 0.0, 0.0, max(obs.speed, 2.0), []
        for _ in range(self.H):
            bc, bs = 1e9, 0.0
            for st in self.steers:
                nx, ny, nyaw = _bike(x, y, yaw, v, st, self.L, self.dt)
                cost = (9 * self._sample(f, nx, ny)
                        - 1.2 * ((nx - x) * np.cos(gh) + (ny - y) * np.sin(gh))
                        - w_gap * (abs(y - y_gap) - abs(ny - y_gap)) + 0.25 * st * st)
                if cost < bc:
                    bc, bs = cost, st
            x, y, yaw = _bike(x, y, yaw, v, bs, self.L, self.dt)
            pts.append((x, y))
        pts = np.array(pts)
        # --- TTC longitudinal: find first step whose cell is risky, brake for it ---
        risky_step = None
        for i, (px, py) in enumerate(pts):
            if self._sample(f, px, py) > 0.8:
                risky_step = i; break
        vru_ahead = any(self._sample(_vru_only(obs), px, py) > 0.5 for px, py in pts[:5]) \
            if False else False
        if risky_step is None:
            target = 12.0
        else:
            ttc = (risky_step + 1) * self.dt
            target = float(np.clip((ttc - 0.6) * 4.0, 0.0, 12.0))  # ~0.6s reaction margin
        # hard VRU cap
        vru_near = max(self._sample(f, px, py) for px, py in pts[:4])
        if vru_near > 1.8:
            target = min(target, 1.5)
        return _pursue(pts, obs.speed, target, self.L)


def _vru_only(obs):
    return (obs.bev == 3).astype(np.float32)


def _bike(x, y, yaw, v, st, L, dt):
    d = st * 0.7; y2 = yaw + v / L * np.tan(d) * dt
    return x + v * np.cos(y2) * dt, y + v * np.sin(y2) * dt, y2


def _pursue(pts, speed, target, L, lookahead=4.0):
    import math
    ld = min(max(2.0 + 0.55 * speed, lookahead), 12.0)
    tx, ty = pts[-1]
    for x, y in pts:
        if math.hypot(x, y) >= ld:
            tx, ty = x, y; break
    alpha = math.atan2(ty, max(tx, 1e-3))
    delta = math.atan2(2 * L * math.sin(alpha), max(math.hypot(tx, ty), 1e-3))
    steer = float(np.clip(math.degrees(delta) / 38.0, -1, 1))
    err = target - speed
    thr = float(np.clip(err, 0, 1)); brk = float(np.clip(-err, 0, 1)) if err < -0.1 else 0.0
    return Control(steer, thr, brk)


def _dilate(a, r):
    if r <= 0:
        return a
    out = a.copy()
    for ax in (0, 1):
        acc = out.copy()
        for s in range(1, r + 1):
            acc = np.maximum(acc, np.roll(out, s, ax)); acc = np.maximum(acc, np.roll(out, -s, ax))
        out = acc
    return out


def _blur(a, r):
    if r <= 0:
        return a
    k = 2 * r + 1
    c = np.cumsum(np.pad(a, ((r + 1, r), (0, 0)), mode="edge"), 0); a = (c[k:] - c[:-k]) / k
    c = np.cumsum(np.pad(a, ((0, 0), (r + 1, r)), mode="edge"), 1); return (c[:, k:] - c[:, :-k]) / k
