"""Map bharatsim archetypes to CARLA vehicle/walker blueprints.

CARLA lacks native auto-rickshaws/carts/cows; we approximate with the closest
stock blueprints and note where custom Indian assets should be imported (see
docs/CARLA.md, 'Indian asset pack'). Dimensions/behaviors still come from
bharatsim archetypes so the traffic *behaves* Indian even on stock meshes.
"""

# archetype -> list of candidate CARLA blueprint ids (first available is used)
BLUEPRINT_MAP = {
    "car":         ["vehicle.tata.*", "vehicle.nissan.micra", "vehicle.seat.leon",
                    "vehicle.audi.a2"],
    "auto":        ["vehicle.bajaj.*", "vehicle.piaggio.*",       # if asset pack
                    "vehicle.micro.microlino", "vehicle.nissan.micra"],  # fallback
    "two_wheeler": ["vehicle.hero.*", "vehicle.bajaj.*",
                    "vehicle.yamaha.yzf", "vehicle.kawasaki.ninja",
                    "vehicle.harley-davidson.low_rider"],
    "cycle":       ["vehicle.bh.crossbike", "vehicle.gazelle.omafiets",
                    "vehicle.diamondback.century"],
    "bus":         ["vehicle.ashok.*", "vehicle.mitsubishi.fusorosa",
                    "vehicle.volkswagen.t2"],
    "truck":       ["vehicle.tata.*", "vehicle.carlamotors.carlacola",
                    "vehicle.carlamotors.firetruck"],
    "cart":        ["vehicle.micro.microlino"],   # approximation; import cart asset
    "cow":         ["static.prop.*cow*", "walker.pedestrian.0001"],  # asset/placeholder
    "auto_stand":  ["vehicle.micro.microlino"],
}

WALKER_BLUEPRINTS = ["walker.pedestrian.0001", "walker.pedestrian.0005",
                     "walker.pedestrian.0007", "walker.pedestrian.0012"]

HAZARD_PROPS = {
    "pothole":   "static.prop.dirtdebris01",      # or import pothole decal asset
    "speedbump": "static.prop.streetbarrier",
    "encroach":  "static.prop.container",
    "cone":      "static.prop.constructioncone",
}


def resolve(bp_lib, candidates):
    """Return the first blueprint in the library matching any candidate pattern."""
    import fnmatch
    all_ids = [bp.id for bp in bp_lib]
    for pat in candidates:
        for bid in all_ids:
            if fnmatch.fnmatch(bid, pat):
                return bid
    return None


def pick_vehicle(bp_lib, archetype, rng=None):
    bid = resolve(bp_lib, BLUEPRINT_MAP.get(archetype, ["vehicle.*"]))
    return bp_lib.find(bid) if bid else None
