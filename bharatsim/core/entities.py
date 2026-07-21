"""Entity system: everything in the world is an Entity with a footprint and a
step() hook. Ego, traffic agents, VRUs, and static hazards share the interface,
so the World loop and collision system treat them uniformly.
"""
import numpy as np
from bharatsim.core.geometry import obb_corners


class Entity:
    kind = "entity"
    dynamic = True
    hard = True            # hard = terminal collision on overlap

    def __init__(self, x=0.0, y=0.0, yaw=0.0, L=1.0, W=1.0):
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)
        self.L, self.W = float(L), float(W)
        self.v = 0.0
        self.alive = True

    def pose(self):
        return self.x, self.y, self.yaw

    def corners(self):
        return obb_corners(self.x, self.y, self.yaw, self.L, self.W)

    def step(self, world, dt):
        pass


class StaticHazard(Entity):
    dynamic = False

    def __init__(self, x, y, yaw=0.0, L=1.0, W=1.0, kind="hazard", hard=True,
                 slow_to=None):
        super().__init__(x, y, yaw, L, W)
        self.kind = kind
        self.hard = hard
        self.slow_to = slow_to     # advisory speed (speed bump), None = n/a
