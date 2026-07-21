"""Gym-style environment + Agent contract + closed-loop runner.

Two ways to use bharatsim:
  1. Gym loop:   obs = env.reset(); obs, rew, done, info = env.step(action)
  2. Agent API:  run_scenario(MyAgent(), "cow_on_road")   (CARLA-style)
"""
from dataclasses import dataclass
import numpy as np

from bharatsim.sensors.bev import render_bev, render_future, BEV_GRID, N_CLASSES
from bharatsim.sensors.camera import render_cameras
from bharatsim.sensors.route_sensor import route_features
from bharatsim.scenarios.dsl import build_from_dict
from bharatsim.scenarios.builtin import SPECS
from bharatsim.eval.metrics import score_episode, aggregate


@dataclass
class Observation:
    bev: np.ndarray          # (BEV_GRID, BEV_GRID) uint8 semantic
    future_occ: np.ndarray   # (4, BEV_GRID, BEV_GRID) uint8
    speed: float
    command: int             # 0 follow, 1 left, 2 right
    visibility: float
    t: float
    # camera + navigation sensors (populated when cameras=True)
    camera: object = None    # bharatsim.sensors.camera.CameraObs or None
    route: dict = None       # route_sensor.route_features output


@dataclass
class Control:
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    def as_dict(self):
        return {"steer": float(np.clip(self.steer, -1, 1)),
                "throttle": float(np.clip(self.throttle, 0, 1)),
                "brake": float(np.clip(self.brake, 0, 1))}


class Agent:
    def reset(self):
        pass
    def act(self, obs: Observation) -> Control:
        raise NotImplementedError


def _observe(world, cameras=False):
    vis = getattr(world.environment, "visibility", 1.0) if world.environment else 1.0
    obs = Observation(render_bev(world), render_future(world),
                      float(world.ego.v), int(world.command()), vis, float(world.t),
                      camera=render_cameras(world) if cameras else None,
                      route=route_features(world))
    obs._world = world     # privileged handle (expert/oracle only; ignore in models)
    return obs


class BharatEnv:
    """Minimal Gym-style wrapper (no external gym dependency)."""

    metadata = {"render_modes": ["bev"]}

    def __init__(self, scenario="cow_on_road", seed=0, spec=None, cameras=False):
        self.spec = spec or SPECS[scenario]
        self.seed = seed
        self.cameras = cameras
        self.world = None

    def reset(self, seed=None):
        self.world = build_from_dict(self.spec, seed if seed is not None else self.seed)
        self._prev_v = self.world.ego.v
        return _observe(self.world, self.cameras)

    def step(self, action):
        c = action.as_dict() if isinstance(action, Control) else action
        done = self.world.step(c)
        obs = _observe(self.world, self.cameras)
        reward = self.world.ego.v * 0.01 * self.world.progress
        if done and self.world.result != "success":
            reward -= 1.0
        info = {"result": self.world.result, "progress": self.world.progress,
                "t": self.world.t}
        return obs, reward, done, info


def run_scenario(agent, scenario=None, seed=0, spec=None, dt=0.1, max_steps=1400,
                 cameras=False, recorder=None, logger=None):
    env = BharatEnv(scenario=scenario, seed=seed, spec=spec, cameras=cameras)
    obs = env.reset(seed)
    agent.reset()
    prev_v, jerks, near, steps = obs.speed, [], 0, 0
    done = False
    while not done and steps < max_steps:
        ctrl = agent.act(obs)
        if logger is not None:
            logger.record(obs, ctrl)
        if recorder is not None:
            recorder.capture(env.world)
        obs, _, done, _ = env.step(ctrl)
        jerks.append(abs(obs.speed - prev_v) / dt); prev_v = obs.speed
        for e in env.world.entities:
            if e.kind == "vru" and getattr(e, "active", False):
                if np.hypot(e.x - env.world.ego.x, e.y - env.world.ego.y) < 2.0:
                    near += 1
        steps += 1
    row = score_episode(env.world, jerks, near)
    row.update(scenario=scenario or "custom", seed=seed)
    return row


def run_suite(agent_factory, scenarios=None, seeds=5, out_json=None,
              time_budget=None, verbose=True):
    import json, os, time
    scenarios = scenarios or list(SPECS)
    rows, t0 = [], time.time()
    if out_json and os.path.exists(out_json):
        rows = json.load(open(out_json)).get("rows", [])
    done = {(r["scenario"], r["seed"]) for r in rows}
    agent = agent_factory()
    complete = True
    for name in scenarios:
        for seed in range(seeds):
            if (name, seed) in done:
                continue
            if time_budget and time.time() - t0 > time_budget:
                complete = False; break
            rows.append(run_scenario(agent, scenario=name, seed=seed))
            if out_json:
                json.dump({"rows": rows}, open(out_json, "w"))
            if verbose:
                r = rows[-1]
                print(f"{r['scenario']:24s} s{r['seed']} DS {r['ds']:6.1f} "
                      f"RC {r['rc']:6.1f} near {r['vru_near_miss']:2d} {r['result']}",
                      flush=True)
        if not complete:
            break
    summary = aggregate(rows); summary["complete"] = complete
    if verbose:
        print(json.dumps(summary, indent=2))
    if out_json:
        json.dump({"summary": summary, "rows": rows}, open(out_json, "w"), indent=1)
    return summary, rows
