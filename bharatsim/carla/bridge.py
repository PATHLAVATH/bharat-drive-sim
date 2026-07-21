"""Scenario bridge: instantiate a bharatsim DSL scenario inside CARLA and run
any bharatsim Agent closed-loop with photoreal sensors + CARLA-Leaderboard metrics.

The SAME YAML/dict scenarios and the SAME Agent API used by the pure-Python sim
run here — only the world/sensors are swapped for CARLA. Unstructured Indian
behavior (gap-seeking, aggression) is applied via per-vehicle target-speed and
lane-offset control through the TrafficManager, since CARLA's default TM is
lane-disciplined.
"""
from __future__ import annotations
import math
import numpy as np

from bharatsim.carla.client import CarlaClient, TOWN_FOR_MAP
from bharatsim.carla.blueprints import pick_vehicle, WALKER_BLUEPRINTS, HAZARD_PROPS, resolve
from bharatsim.carla.sensors import CarlaCameraRig
from bharatsim.agents.traffic import ARCHETYPES
from bharatsim.eval.env import Observation, Control
from bharatsim.sensors.bev import BEV_GRID
from bharatsim.eval.metrics import PENALTY


class CarlaScenario:
    def __init__(self, spec, seed=0, host="localhost", port=2000, cameras=True,
                 use_lidar=False):
        self.spec = spec
        self.rng = np.random.default_rng(seed + 991)
        town = TOWN_FOR_MAP.get(spec.get("map", "narrow_street"), "Town01")
        self.client = CarlaClient(host=host, port=port, town=town)
        self.client.set_weather(spec.get("environment", {}).get("label", "clear")
                                if isinstance(spec.get("environment"), dict) else "clear")
        self.tm = self.client.enable_sync()
        self.carla = self.client.carla
        self.world = self.client.world
        self.cameras = cameras
        self.use_lidar = use_lidar
        self.ego = None
        self.rig = None
        self.actors = []
        self.walkers = []
        self.props = []
        self.collision = False
        self._collision_kind = None
        self.route = None
        self._start_loc = None

    # ---- build ----
    def _spawn_ego(self):
        bl = self.world.get_blueprint_library()
        bp = bl.filter("vehicle.*")[0]
        sp = self.world.get_map().get_spawn_points()
        tf = sp[int(self.rng.integers(len(sp)))]
        self.ego = self.world.spawn_actor(bp, tf)
        self._start_loc = tf.location
        # collision sensor
        cbp = bl.find("sensor.other.collision")
        self._csensor = self.world.spawn_actor(cbp, self.carla.Transform(), attach_to=self.ego)
        self._csensor.listen(self._on_collision)
        # route = a forward waypoint chain
        self.route = self._build_route(tf)
        if self.cameras:
            self.rig = CarlaCameraRig(self.carla, self.world, self.ego,
                                      use_lidar=self.use_lidar)

    def _build_route(self, start_tf, n=40, step=5.0):
        m = self.world.get_map()
        wp = m.get_waypoint(start_tf.location)
        pts = [wp]
        for _ in range(n):
            nxt = pts[-1].next(step)
            if not nxt:
                break
            pts.append(nxt[0])
        return pts

    def _on_collision(self, event):
        self.collision = True
        oa = getattr(event, "other_actor", None)
        tid = getattr(oa, "type_id", "") if oa else ""
        self._collision_kind = ("collision_ped" if tid.startswith("walker")
                                else "collision_static" if tid.startswith("static")
                                else "collision_vehicle")

    def _spawn_traffic(self):
        bl = self.world.get_blueprint_library()
        sp = self.world.get_map().get_spawn_points()
        for a in self.spec.get("actors", []):
            kind = a["type"]
            for _ in range(a.get("count", 1)):
                bp = pick_vehicle(bl, kind, self.rng)
                if bp is None:
                    continue
                tf = sp[int(self.rng.integers(len(sp)))]
                v = self.world.try_spawn_actor(bp, tf)
                if v is None:
                    continue
                v.set_autopilot(True, self.tm.get_port())
                self._apply_indian_behavior(v, kind)
                self.actors.append(v)

    def _apply_indian_behavior(self, vehicle, archetype):
        """Turn CARLA's lane-disciplined TM into Indian gap-seeking via TM params."""
        _, _, vdes, aggr, latf = ARCHETYPES[archetype]
        tm = self.tm
        # aggressive agents: less speed difference, closer following, more lane changes
        tm.vehicle_percentage_speed_difference(vehicle, -int(20 * aggr))
        try:
            tm.distance_to_leading_vehicle(vehicle, max(0.5, 2.5 - 2.0 * aggr))
            tm.ignore_lights_percentage(vehicle, int(60 * aggr))
            tm.ignore_signs_percentage(vehicle, int(70 * aggr))
            tm.random_left_lanechange_percentage(vehicle, int(60 * aggr))
            tm.random_right_lanechange_percentage(vehicle, int(60 * aggr))
            tm.vehicle_lane_offset(vehicle, float(self.rng.uniform(-1, 1) * min(latf, 1.5)))
        except Exception:
            pass  # older TM API subset

    def _spawn_walkers(self):
        bl = self.world.get_blueprint_library()
        for v in self.spec.get("vrus", []):
            for _ in range(v.get("count", 1)):
                bp = bl.find(str(self.rng.choice(WALKER_BLUEPRINTS)))
                loc = self.world.get_random_location_from_navigation()
                if loc is None:
                    continue
                w = self.world.try_spawn_actor(bp, self.carla.Transform(loc))
                if w is None:
                    continue
                ctrl_bp = bl.find("controller.ai.walker")
                ctrl = self.world.spawn_actor(ctrl_bp, self.carla.Transform(), attach_to=w)
                ctrl.start()
                ctrl.go_to_location(self.world.get_random_location_from_navigation())
                ctrl.set_max_speed(float(v.get("speed", 1.3)))
                self.walkers.append((w, ctrl))

    def _spawn_hazards(self):
        bl = self.world.get_blueprint_library()
        for hz in self.spec.get("hazards", []):
            prop_id = HAZARD_PROPS.get(hz["type"])
            bid = resolve(bl, [prop_id]) if prop_id else None
            if not bid:
                continue
            n = hz.get("count", 1)
            for _ in range(n):
                wp = self.route[int(self.rng.integers(len(self.route)))]
                loc = wp.transform.location
                p = self.world.try_spawn_actor(bl.find(bid),
                                               self.carla.Transform(loc))
                if p:
                    self.props.append(p)

    def build(self):
        self._spawn_ego(); self._spawn_traffic(); self._spawn_walkers(); self._spawn_hazards()
        for _ in range(10):
            self.client.tick()   # settle
        return self

    # ---- observation / control ----
    def _command(self):
        return 0   # extendable: derive from route curvature (see pure-sim World.command)

    def observe(self):
        v = self.ego.get_velocity()
        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        cam = self.rig.read() if self.rig else None
        bev = np.zeros((BEV_GRID, BEV_GRID), np.uint8)   # optional: project seg->BEV
        fut = np.zeros((4, BEV_GRID, BEV_GRID), np.uint8)
        vis = 1.0
        obs = Observation(bev, fut, speed, self._command(), vis,
                          self.world.get_snapshot().timestamp.elapsed_seconds,
                          camera=cam, route=self._route_features())
        obs._world = None
        return obs

    def _route_features(self):
        t = self.ego.get_transform()
        yaw = math.radians(t.rotation.yaw)
        errs = []
        for la in (5.0, 10.0, 20.0):
            idx = min(int(la / 5.0), len(self.route) - 1)
            wp = self.route[idx].transform
            dh = math.radians(wp.rotation.yaw) - yaw
            errs.append(float((dh + math.pi) % (2 * math.pi) - math.pi))
        return {"heading_errors": errs, "lateral_offset": 0.0,
                "curvature": errs[-1] - errs[0],
                "dist_to_goal": 5.0 * len(self.route)}

    def apply(self, control: Control):
        self.ego.apply_control(self.carla.VehicleControl(
            throttle=float(control.throttle), steer=float(control.steer),
            brake=float(control.brake)))

    def progress(self):
        loc = self.ego.get_location(); goal = self.route[-1].transform.location
        d0 = self._start_loc.distance(goal) + 1e-6
        return float(np.clip(1 - loc.distance(goal) / d0, 0, 1))

    def destroy(self):
        if self.rig:
            self.rig.destroy()
        for a in self.actors:
            try: a.destroy()
            except Exception: pass
        for w, c in self.walkers:
            try: c.stop(); w.destroy()
            except Exception: pass
        for p in self.props:
            try: p.destroy()
            except Exception: pass
        try: self._csensor.destroy(); self.ego.destroy()
        except Exception: pass
        self.client.restore()


def run_scenario_carla(agent, spec, seed=0, max_steps=2000, host="localhost",
                       port=2000, cameras=True, recorder=None):
    sc = CarlaScenario(spec, seed=seed, host=host, port=port, cameras=cameras).build()
    agent.reset()
    prev_v, jerks, steps = 0.0, [], 0
    try:
        while steps < max_steps:
            obs = sc.observe()
            ctrl = agent.act(obs)
            sc.apply(ctrl)
            sc.client.tick()
            jerks.append(abs(obs.speed - prev_v) / sc.client.fixed_dt); prev_v = obs.speed
            if sc.collision:
                result = sc._collision_kind; break
            if sc.progress() > 0.98:
                result = "success"; break
            steps += 1
        else:
            result = "timeout"
        rc = sc.progress() * 100.0
        pen = PENALTY.get(result, 1.0) if result != "success" else 1.0
        return {"result": result, "rc": rc, "ds": rc * pen,
                "mean_jerk": float(np.mean(jerks) if jerks else 0.0),
                "sim_time": steps * sc.client.fixed_dt}
    finally:
        sc.destroy()
