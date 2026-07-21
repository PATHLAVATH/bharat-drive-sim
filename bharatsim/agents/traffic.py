"""Traffic + VRU behavior models for Indian unstructured roads.

TrafficAgent: lane-free gap-seeking lateral drift + IDM longitudinal, per
archetype. VRU: proximity-triggered stochastic-heading pedestrian/animal.
Both are Entities.
"""
import numpy as np
from bharatsim.core.entities import Entity

# (L, W, v_des, aggression, lat_freedom)
ARCHETYPES = {
    "car":         (4.2, 1.8, 9.0,  0.4, 1.2),
    "auto":        (2.6, 1.4, 7.0,  0.7, 2.2),
    "two_wheeler": (1.8, 0.7, 9.5,  0.9, 3.2),
    "cycle":       (1.7, 0.6, 4.5,  0.6, 3.0),
    "bus":         (10.5, 2.5, 7.5, 0.3, 0.8),
    "truck":       (8.0, 2.5, 7.0,  0.3, 0.8),
    "cart":        (3.0, 1.5, 2.2,  0.2, 0.6),
    "cow":         (2.2, 1.0, 0.4,  0.05, 0.3),
    "auto_stand":  (2.6, 1.4, 0.0,  0.0, 0.2),   # parked auto
}


def idm(v, v0, gap, lead_v, aggr=0.5, b=3.0):
    if v0 <= 0.2:
        return -6.0
    a_max = 1.5 + 2.5 * aggr
    T = 1.6 - 0.9 * aggr
    s0 = 1.5 - 0.7 * aggr
    free = a_max * (1 - (v / v0) ** 4)
    if gap is None:
        return float(np.clip(free, -8, a_max))
    s_star = s0 + max(v * T + v * (v - lead_v) / (2 * np.sqrt(a_max * b)), 0)
    return float(np.clip(a_max * (1 - (v / v0) ** 4 - (s_star / max(gap, 0.4)) ** 2),
                         -8, a_max))


class TrafficAgent(Entity):
    kind = "vehicle"

    def __init__(self, path, kind="car", s0=0.0, lat=0.0, seed=0, behavior=None,
                 path_name="route", obey_prob=0.85):
        L, W, vdes, aggr, latf = ARCHETYPES[kind]
        super().__init__(L=L, W=W)
        self.archetype = kind
        self.path, self.s, self.lat = path, float(s0), float(lat)
        self.v_des, self.aggr, self.latf = vdes, aggr, latf
        self.v = vdes * 0.6
        self.behavior = dict(behavior or {})
        self.rng = np.random.default_rng(seed)
        self._lat_tgt, self._retarget = lat, 0.0
        self.path_name = path_name
        self.obey = self.rng.random() < obey_prob
        self.dynamic = self.v_des > 0.1 or kind == "cow"
        self.hard = True
        self._sync_pose()

    def _sync_pose(self):
        p = self.path.point_at(self.s); h = self.path.heading_at(self.s)
        n = np.array([-np.sin(h), np.cos(h)])
        c = p + self.lat * n
        self.x, self.y, self.yaw = float(c[0]), float(c[1]), h

    def _leader_gap(self, world):
        best, bv = None, 0.0
        for other in world.hash.query(self.x, self.y, 40):
            if other is self:
                continue
            s, lat = self.path.project((other.x, other.y))
            if s > self.s + 0.5 and s - self.s < 40 and abs(lat - self.lat) < 2.2:
                gap = s - self.s - (other.L + self.L) / 2
                if best is None or gap < best:
                    best, bv = gap, getattr(other, "v", 0.0)
        # ego as obstacle
        s, lat = self.path.project((world.ego.x, world.ego.y))
        if s > self.s + 0.5 and s - self.s < 40 and abs(lat - self.lat) < 2.2:
            gap = s - self.s - (world.ego.L + self.L) / 2
            if best is None or gap < best:
                best, bv = gap, world.ego.v
        return best, bv

    def step(self, world, dt):
        if not self.alive:
            return
        b = self.behavior
        if "trigger_dist" in b and not b.get("_fired"):
            if np.hypot(world.ego.x - self.x, world.ego.y - self.y) < b["trigger_dist"]:
                b["_fired"] = True
                if "trigger_v" in b:
                    self.v_des = b["trigger_v"]
                if "trigger_lat" in b:
                    self._lat_tgt = b["trigger_lat"]
        if self.s >= self.path.total - 2.0:
            self.alive = False; return
        # lateral gap-seeking drift
        self._retarget -= dt
        if self._retarget <= 0 and "trigger_lat" not in b:
            self._retarget = self.rng.uniform(0.8, 2.5)
            drift = self.rng.uniform(-1, 1) * self.latf * (0.3 + self.aggr)
            self._lat_tgt = float(np.clip(self.lat + drift, -self.latf, self.latf))
        rate = (0.6 + 1.8 * self.aggr)
        self.lat += float(np.clip(self._lat_tgt - self.lat, -rate * dt, rate * dt))
        # longitudinal IDM (signal acts as a phantom leader if obeyed)
        gap, lv = self._leader_gap(world)
        if self.obey and world.junction is not None:
            sg = world.junction.stopline_gap(self.path_name, self.s)
            if sg is not None and (gap is None or sg < gap):
                gap, lv = sg, 0.0
        self.v = max(0.0, self.v + idm(self.v, self.v_des, gap, lv, self.aggr) * dt)
        self.s += self.v * dt
        self._sync_pose()


class VRU(Entity):
    kind = "vru"

    def __init__(self, x, y, direction, speed=1.2, trigger=20.0, radius=0.35,
                 seed=0, wander=0.5):
        super().__init__(x, y, 0.0, L=radius * 2, W=radius * 2)
        d = np.array(direction, float); self.dir = d / (np.linalg.norm(d) + 1e-9)
        self.speed, self.trigger, self.radius = speed, trigger, radius
        self.rng = np.random.default_rng(seed); self.wander = wander
        self.active = False
        self.vel = self.dir * speed

    def step(self, world, dt):
        if not self.active:
            if np.hypot(world.ego.x - self.x, world.ego.y - self.y) < self.trigger:
                self.active = True
            return
        if self.rng.random() < self.wander * dt * 3:
            a = self.rng.uniform(-0.8, 0.8); c, s = np.cos(a), np.sin(a)
            self.dir = np.array([c * self.dir[0] - s * self.dir[1],
                                 s * self.dir[0] + c * self.dir[1]])
        paused = self.rng.random() < 0.15 * dt * 3
        self.vel = self.dir * (0.0 if paused else self.speed)
        self.x += self.vel[0] * dt; self.y += self.vel[1] * dt
