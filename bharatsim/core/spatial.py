"""Uniform-grid spatial hash: amortized O(1) insert, O(k) radius query.

Rebuilt each tick over the (small, dense) dynamic-entity set for neighbor
lookups in behavior + collision — faster than a rebuilt KD-tree at this scale.
"""
import math
from collections import defaultdict


class SpatialHash:
    __slots__ = ("cell", "grid")

    def __init__(self, cell_size=4.0):
        self.cell = cell_size
        self.grid = defaultdict(list)

    def _key(self, x, y):
        return (math.floor(x / self.cell), math.floor(y / self.cell))

    def insert(self, x, y, payload):
        self.grid[self._key(x, y)].append((x, y, payload))

    def query(self, x, y, r):
        cr = int(math.ceil(r / self.cell)); cx, cy = self._key(x, y); r2 = r * r
        for i in range(cx - cr, cx + cr + 1):
            for j in range(cy - cr, cy + cr + 1):
                for ox, oy, pl in self.grid.get((i, j), ()):
                    if (ox - x) ** 2 + (oy - y) ** 2 <= r2:
                        yield pl

    def clear(self):
        self.grid.clear()
