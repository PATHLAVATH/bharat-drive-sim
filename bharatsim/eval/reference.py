"""Reference agents. RiskFieldAgent = lane-free risk potential-field planner
over the BEV occupancy, the recommended baseline for unstructured traffic."""
import numpy as np
from bharatsim.eval.env import Agent, Control
from bharatsim.sensors.bev import BEV_GRID, BEV_RES, BEV_RANGE


class ConstantAgent(Agent):
    def __init__(self, throttle=0.3):
        self.throttle = throttle
    def act(self, obs):
        return Control(0.0, self.throttle, 0.0)


class RiskFieldAgent(Agent):
    """Continuous risk field from BEV occupancy + gap-seeking + speed governor."""

    def __init__(self, horizon=8, dt=0.5, wheelbase=2.6, max_steer=0.6, n_steer=9):
        self.H, self.dt, self.L = horizon, dt, wheelbase
        self.steers = np.linspace(-max_steer, max_steer, n_steer)

    def _field(self, obs):
        bev = obs.bev
        risk = np.zeros(bev.shape, np.float32)
        risk += (bev == 2)                      # vehicle
        risk += 1.2 * (bev == 3)                # vru (weigh higher)
        risk += 1.0 * (bev == 5)                # hard hazard
        risk += 0.4 * (bev == 4)                # soft hazard
        risk += 0.8 * obs.future_occ.max(0)     # predicted motion
        risk += 0.8 * (bev == 0)                # off-road (non-road) penalty
        # inflate by ego footprint + blur
        risk = _dilate((risk > 0.5).astype(np.float32), 3) * 1.6 + risk
        return _blur(risk, 3)

    def _sample(self, f, x, y):
        ix = int(round(x / BEV_RES + BEV_GRID / 2))
        iy = int(round(y / BEV_RES + BEV_GRID / 2))
        if 0 <= ix < BEV_GRID and 0 <= iy < BEV_GRID:
            return f[ix, iy]
        return 5.0

    def act(self, obs):
        f = self._field(obs)
        # gap: clearest lateral corridor ahead
        best_y, best_c = 0.0, 1e9
        for ly in np.arange(-3.0, 3.01, 0.5):
            c = sum(self._sample(f, xm, ly) for xm in np.arange(4, 18, 1.0)) + 0.15*abs(ly)
            if c < best_c:
                best_c, best_y = c, ly
        gh = {1: 0.4, 2: -0.4}.get(obs.command, 0.0)
        x, y, yaw, v, pts = 0.0, 0.0, 0.0, max(obs.speed, 2.0), []
        for _ in range(self.H):
            bc, bs = 1e9, 0.0
            for st in self.steers:
                nx, ny, nyaw = _bike(x, y, yaw, v, st, self.L, self.dt)
                cost = (9*self._sample(f, nx, ny) - 1.2*((nx-x)*np.cos(gh)+(ny-y)*np.sin(gh))
                        - 2.5*(abs(y-best_y)-abs(ny-best_y)) + 0.25*st*st)
                if cost < bc:
                    bc, bs = cost, st
            x, y, yaw = _bike(x, y, yaw, v, bs, self.L, self.dt); pts.append((x, y))
        fwd = max(self._sample(f, px, py) for px, py in pts[:5])
        tgt = 0.0 if fwd > 1.4 else 2.0 if fwd > 0.8 else 4.0 if fwd > 0.45 else 10.0
        return _pursue(np.array(pts), obs.speed, tgt, self.L)


def _bike(x, y, yaw, v, st, L, dt):
    d = st * 0.7; y2 = yaw + v / L * np.tan(d) * dt
    return x + v*np.cos(y2)*dt, y + v*np.sin(y2)*dt, y2


def _pursue(pts, speed, target, L, lookahead=4.0):
    import math
    ld = min(max(2.0 + 0.55 * speed, lookahead), 12.0)
    tx, ty = pts[-1]
    for x, y in pts:
        if math.hypot(x, y) >= ld:
            tx, ty = x, y; break
    alpha = math.atan2(ty, max(tx, 1e-3))
    delta = math.atan2(2 * L * math.sin(alpha), max(math.hypot(tx, ty), 1e-3))
    steer = np.clip(math.degrees(delta) / 38.0, -1, 1)
    err = target - speed
    thr = float(np.clip(err, 0, 1)); brk = float(np.clip(-err, 0, 1)) if err < -0.1 else 0.0
    return Control(steer, thr, brk)


def _dilate(a, r):
    if r <= 0:
        return a
    out = a.copy()
    for ax in (0, 1):
        acc = out.copy()
        for s in range(1, r+1):
            acc = np.maximum(acc, np.roll(out, s, ax)); acc = np.maximum(acc, np.roll(out, -s, ax))
        out = acc
    return out


def _blur(a, r):
    if r <= 0:
        return a
    k = 2*r+1
    c = np.cumsum(np.pad(a, ((r+1, r), (0, 0)), mode="edge"), 0); a = (c[k:]-c[:-k])/k
    c = np.cumsum(np.pad(a, ((0, 0), (r+1, r)), mode="edge"), 1); return (c[:, k:]-c[:, :-k])/k
