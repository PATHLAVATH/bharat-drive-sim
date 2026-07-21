"""Procedural route/scenario generator — Bench2Drive-scale benchmark.

Samples map + traffic density + hazard mix + VRU events to synthesize many
distinct, seeded scenario specs (DSL dicts). Combined with the 19 hand-authored
scenarios, this gives a large, reproducible benchmark suite.
"""
import numpy as np

from bharatsim.maps.library import MAPS
from bharatsim.agents.traffic import ARCHETYPES

_VEH = ["car", "auto", "two_wheeler", "cycle", "bus", "truck", "cart"]
_PATHS = {
    "narrow_street": ["same", "oncoming"],
    "arterial_4lane": ["l0", "l1", "onc0", "onc1"],
    "uncontrolled_junction": ["same", "oncoming", "cross_a", "cross_b"],
    "roundabout": ["ring", "same"],
    "village_curvy": ["same", "oncoming"],
    "highway": ["same", "fast", "shoulder"],
}
_SPEED0 = {"narrow_street": 6, "arterial_4lane": 11, "uncontrolled_junction": 5,
           "roundabout": 5, "village_curvy": 5, "highway": 14}


def generate(n=200, seed=0):
    """Return list of (name, spec, seed) — a reproducible benchmark."""
    rng = np.random.default_rng(seed)
    maps = list(MAPS)
    out = []
    for i in range(n):
        mp = maps[i % len(maps)]
        density = rng.choice(["light", "medium", "heavy"], p=[0.3, 0.4, 0.3])
        n_actor = {"light": 2, "medium": 5, "heavy": 9}[density]
        spec = {"map": mp, "ego_speed": _SPEED0[mp], "actors": [], "vrus": [],
                "hazards": []}
        paths = _PATHS[mp]
        for _ in range(n_actor):
            kind = str(rng.choice(_VEH, p=_veh_probs(mp)))
            spec["actors"].append({
                "type": kind, "path": str(rng.choice(paths)),
                "s": float(rng.uniform(15, 120)),
                "lat": float(rng.uniform(-1.5, 1.5)),
                "count": int(rng.integers(1, 3)), "spread_s": 8})
        # VRUs (denser on slow urban maps)
        if mp in ("narrow_street", "uncontrolled_junction", "village_curvy"):
            for _ in range(int(rng.integers(0, 4))):
                spec["vrus"].append({
                    "cross_at": float(rng.uniform(25, 100)),
                    "side": int(rng.choice([-1, 1])), "count": int(rng.integers(1, 4)),
                    "speed": float(rng.uniform(0.9, 2.4)),
                    "trigger": float(rng.uniform(12, 24)), "spread": 3})
        # hazards
        if rng.random() < 0.6:
            spec["hazards"].append({"type": "pothole",
                                    "count": int(rng.integers(3, 12))})
        if rng.random() < 0.25:
            spec["hazards"].append({"type": "speedbump",
                                    "at": float(rng.uniform(30, 90))})
        if rng.random() < 0.2 and mp == "narrow_street":
            spec["hazards"].append({"type": "encroach",
                                    "at": float(rng.uniform(35, 80)),
                                    "side": int(rng.choice([-1, 1]))})
        if rng.random() < 0.2:
            spec["environment"] = {"visibility": float(rng.uniform(0.4, 0.7)),
                                   "label": "lowvis"}
        out.append((f"gen_{mp}_{i:03d}", spec, int(rng.integers(1 << 30))))
    return out


def _veh_probs(mp):
    # Indian mix skews to two-wheelers/autos on urban roads; trucks on highway
    if mp == "highway":
        p = {"car": .3, "auto": .05, "two_wheeler": .2, "cycle": .02,
             "bus": .13, "truck": .28, "cart": .02}
    else:
        p = {"car": .18, "auto": .22, "two_wheeler": .34, "cycle": .1,
             "bus": .05, "truck": .04, "cart": .07}
    return [p[k] for k in _VEH]
