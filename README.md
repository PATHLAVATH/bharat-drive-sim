# bharat-drive-sim

A **CARLA/Bench2Drive-style driving simulator for Indian roads** — for developing
and benchmarking ADAS / autonomous-driving stacks in the conditions Western
simulators ignore: **unstructured, lane-free, negotiation-based traffic** with
mixed vehicles, unpredictable pedestrians and animals, degraded roads, and
traffic signals with variable compliance.

Pure Python (numpy + OpenCV). **No game engine, no GPU** — runs on any laptop,
~0.5–3 s per scenario. Clean sensor/physics interfaces so a photoreal or CARLA
backend can be swapped in later.

```bash
pip install -r requirements.txt
python tests/test_bharatsim.py                                  # verify (11 tests)
python bin/bharatsim-bench --agent riskfield --seeds 5          # scenario suite
python bin/bharatsim-leaderboard --agent expert --n 200         # full benchmark + report
```

## Feature set (v0.2)

**Sensors**
- **Multi-camera** semantic + metric-depth renderer (3 cams, RGB + depth +
  semantic class maps, calibrated K & extrinsics) — `bharatsim/sensors/camera.py`
- **BEV** semantic raster + 4-step future dynamic occupancy — `sensors/bev.py`
- **Route/navigation** sensor (goal heading, lateral offset, curvature)

**World & traffic**
- Kinematic-bicycle ego, uniform-grid spatial hash, unified Entity system
- **9 agent archetypes** (car, auto, two-wheeler, cycle, bus, truck, cart, cow,
  parked auto) with **lane-free gap-seeking** behavior + IDM, aggression-tuned
- **Wandering VRUs** (stochastic heading + pauses) for adults/children/vendors
- **Degraded roads**: potholes, speed bumps, encroachment, low-visibility
- **Traffic signals + junction right-of-way** with per-agent compliance;
  red-light-running is scored

**Maps (6)**: narrow two-way street, 4-lane arterial, uncontrolled junction,
roundabout, curvy village road, highway.

**Scenarios**
- **19 hand-authored** built-in scenarios across all maps
- **YAML scenario DSL** — author new scenarios with no code
- **Procedural generator** — Bench2Drive-scale (200+ seeded routes)

**Evaluation**
- Gym-style `BharatEnv` (reset/step) **and** CARLA-style `Agent.act(obs)->Control`
- **Expert autopilot** privileged baseline (upper bound / IL demonstrator)
- Reference **risk-field** agent (lane-free planner) + constant baseline
- **Leaderboard runner** → markdown + JSON report with per-map aggregates
- CARLA/Bench2Drive **metrics**: Driving Score, route completion, collision rate,
  **VRU near-miss**, success/offroad/timeout, red-light, comfort (jerk)

**Tooling**
- **Episode recorder** → top-down GIF/PNG (watch what the agent did)
- **Imitation-learning data logger** → compressed `.npz` shards (obs + expert action)
- Two CLIs: `bin/bharatsim-bench`, `bin/bharatsim-leaderboard`

## Bring your own model

```python
from bharatsim.eval.env import Agent, Control

class MyAgent(Agent):
    def act(self, obs) -> Control:
        # obs.bev (160x160), obs.future_occ (4x160x160), obs.camera (RGB+depth+semantic),
        # obs.route (heading/offset/curvature), obs.speed, obs.command, obs.visibility
        return Control(steer, throttle, brake)
```

```bash
python bin/bharatsim-bench --agent my_module:MyAgent --seeds 5 --cameras
```

Template: `examples/my_agent.py`. Full walkthrough: `docs/GUIDE.md`.

## Sample leaderboard (expert baseline, n=40)

Overall DS 44.9 · collision 47.5% · VRU near-miss 0.35 · success 27.5%. The
expert is over-cautious in dense unstructured traffic — an honest signal that
**negotiation-based Indian driving is a genuinely open problem**, not a solved
one. Full table: `docs/SAMPLE_LEADERBOARD.md`.

## Photoreal (CARLA) backend

The same scenarios and the same `Agent` run against **CARLA** (Unreal, photoreal
cameras + LiDAR + semantic seg) and can be **submitted to the CARLA Leaderboard
2.0 / Bench2Drive** — see `docs/CARLA.md`. Needs a GPU + CARLA; the integration
(client, sync world, Indian-weather presets, archetype→blueprint mapping, sensor
rig, traffic/VRU/hazard spawning, tick loop, leaderboard `AutonomousAgent`
wrapper) is implemented and verified against an in-process mock server
(`tests/test_carla_mock.py`). Run it on a CUDA box for photoreal frames + scores.

```python
from bharatsim.carla.bridge import run_scenario_carla
from bharatsim.eval.reference import RiskFieldAgent
from bharatsim.scenarios.builtin import SPECS
run_scenario_carla(RiskFieldAgent(), SPECS["junction_freeforall"], cameras=True)
```

## Honest limitations

- The built-in fast sim is **2.5D flat-shaded, not photoreal** — use the CARLA
  backend (`docs/CARLA.md`) for photoreal frames and leaderboard scores.
- Agent behaviors are hand-tuned heuristics, not learned from Indian driving logs.
- No multi-lane change negotiation or roundabout yield logic yet.
- The expert baseline is not strong in dense chaos (see above).

These are the prioritized next steps in `docs/DESIGN.md`.

## Docs

- `docs/GUIDE.md` — setup / test / evaluate / metrics / BYO-model
- `docs/DESIGN.md` — architecture, extension points, limitations
- `docs/CARLA.md` — **photoreal CARLA backend + Leaderboard/Bench2Drive submission**
- `docs/SAMPLE_LEADERBOARD.md` — example benchmark report

## License

Apache-2.0
