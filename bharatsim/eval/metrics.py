"""CARLA/Bench2Drive-style metrics + per-episode records."""
import numpy as np

PENALTY = {"collision_vru": 0.45, "collision_ped": 0.45, "collision_vehicle": 0.55,
           "collision_static": 0.60, "offroad": 0.65, "lane_deviation": 0.9}


def score_episode(world, jerks, near):
    rc = world.progress * 100.0
    infr = list(world.events)
    res = world.result or "timeout"
    if res.startswith("collision") or res == "offroad":
        infr.append(res)
    pen = 1.0
    for i in infr:
        pen *= PENALTY.get(i, 1.0)
    return {"result": res, "rc": rc, "ds": rc * pen, "infractions": infr,
            "vru_near_miss": int(near),
            "mean_jerk": float(np.mean(jerks) if jerks else 0.0),
            "sim_time": float(world.t)}


def aggregate(rows):
    m = lambda k: float(np.mean([r[k] for r in rows]))
    rate = lambda f: 100.0 * float(np.mean([f(r) for r in rows]))
    return {
        "driving_score": m("ds"), "route_completion": m("rc"),
        "success_rate": rate(lambda r: r["result"] == "success"),
        "collision_rate": rate(lambda r: str(r["result"]).startswith("collision")),
        "offroad_rate": rate(lambda r: r["result"] == "offroad"),
        "timeout_rate": rate(lambda r: r["result"] == "timeout"),
        "vru_near_miss_avg": m("vru_near_miss"), "mean_jerk": m("mean_jerk"),
        "episodes": len(rows),
    }
