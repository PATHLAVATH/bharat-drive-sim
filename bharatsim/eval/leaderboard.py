"""Leaderboard runner: evaluate an agent over a large generated benchmark and
write a markdown + JSON report with overall and per-map aggregates."""
import json
import os
import time
from collections import defaultdict

import numpy as np

from bharatsim.eval.env import run_scenario
from bharatsim.eval.metrics import aggregate
from bharatsim.scenarios.generator import generate


def run_leaderboard(agent_factory, n=200, seed=0, out_dir="leaderboard",
                    time_budget=None, cameras=False, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    bench = generate(n=n, seed=seed)
    rows_path = os.path.join(out_dir, "rows.json")
    rows = json.load(open(rows_path)).get("rows", []) if os.path.exists(rows_path) else []
    done = {r["name"] for r in rows}
    agent = agent_factory()
    t0 = time.time()
    complete = True
    for name, spec, s in bench:
        if name in done:
            continue
        if time_budget and time.time() - t0 > time_budget:
            complete = False
            break
        r = run_scenario(agent, spec=spec, seed=s, cameras=cameras)
        r["name"] = name; r["map"] = spec["map"]
        rows.append(r)
        json.dump({"rows": rows}, open(rows_path, "w"))
        if verbose:
            print(f"{name:26s} DS {r['ds']:6.1f} {r['result']}", flush=True)
    overall = aggregate(rows); overall["complete"] = complete; overall["n"] = len(rows)
    per_map = {}
    by = defaultdict(list)
    for r in rows:
        by[r["map"]].append(r)
    for mp, rs in by.items():
        per_map[mp] = aggregate(rs)
    report = {"overall": overall, "per_map": per_map}
    json.dump(report, open(os.path.join(out_dir, "report.json"), "w"), indent=1)
    _write_md(report, os.path.join(out_dir, "REPORT.md"))
    if verbose:
        print(json.dumps(overall, indent=2))
    return report


def _write_md(report, path):
    o = report["overall"]
    L = ["# bharat-drive-sim Leaderboard Report", "",
         f"Episodes: **{o['n']}**  ·  complete: {o['complete']}", "",
         "## Overall", "",
         "| Metric | Value |", "|---|---|",
         f"| Driving Score | {o['driving_score']:.1f} |",
         f"| Route Completion % | {o['route_completion']:.1f} |",
         f"| Success Rate % | {o['success_rate']:.1f} |",
         f"| Collision Rate % | {o['collision_rate']:.1f} |",
         f"| VRU Near-Miss (avg) | {o['vru_near_miss_avg']:.2f} |",
         f"| Offroad % | {o['offroad_rate']:.1f} |",
         f"| Timeout % | {o['timeout_rate']:.1f} |",
         f"| Mean Jerk (m/s²) | {o['mean_jerk']:.2f} |", "",
         "## Per-map", "",
         "| Map | DS | Coll % | VRU near | Succ % | n |", "|---|---|---|---|---|---|"]
    for mp, m in sorted(report["per_map"].items()):
        L.append(f"| {mp} | {m['driving_score']:.1f} | {m['collision_rate']:.1f} | "
                 f"{m['vru_near_miss_avg']:.2f} | {m['success_rate']:.1f} | {m['episodes']} |")
    open(path, "w").write("\n".join(L) + "\n")
