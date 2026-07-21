# bharat-drive-sim — Setup, Test, Evaluate, Metrics

## 1. Install
```bash
git clone https://github.com/PATHLAVATH/bharat-drive-sim.git
cd bharat-drive-sim
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # numpy, opencv-headless, pyyaml
```
CPU-only, no GPU. `pip install -e .` to import `bharatsim` anywhere.

## 2. Verify
```bash
python tests/test_bharatsim.py       # -> ALL BHARATSIM TESTS PASSED
```

## 3. Evaluate
```bash
python bin/bharatsim-bench --agent riskfield --seeds 5                    # full suite
python bin/bharatsim-bench --agent constant  --scenarios cow_on_road      # subset
python bin/bharatsim-bench --agent riskfield --seeds 5 --budget 60 --out r.json  # resumable
```
Flags: `--agent` (`riskfield`|`constant`|`module:Class`), `--seeds`,
`--scenarios a,b,c|all`, `--scenario-file file.yaml`, `--budget SECONDS`, `--out`.

## 4. Bring your own model
Subclass `Agent`, return `Control(steer, throttle, brake)` from `act(obs)`:

- `obs.bev` (160×160 uint8): 0 free, 1 road, 2 vehicle, 3 vru, 4 soft hazard,
  5 hard hazard. Ego at grid centre, facing +x (up).
- `obs.future_occ` (4×160×160): predicted dynamic occupancy at +0.5..2.0 s.
- `obs.speed` (m/s), `obs.command` (0 follow/1 left/2 right), `obs.visibility`.

If your model outputs waypoints, use pure-pursuit + PID (see
`bharatsim/eval/reference.py::_pursue`). Run with
`--agent your_module:YourAgent`. Template in `examples/my_agent.py`.

## 5. Author scenarios (YAML DSL)
No code needed — see `scenarios/custom_example.yaml`:
```yaml
map: uncontrolled_junction
ego_speed: 5.0
actors:  [{type: auto, path: cross_a, s: 25, count: 3, spread_s: 15}]
vrus:    [{cross_at: 78, side: -1, count: 2, speed: 1.2, trigger: 18}]
hazards: [{type: pothole, count: 6}, {type: speedbump, at: 60}]
```
Run: `python bin/bharatsim-bench --scenario-file my.yaml --agent riskfield`.
Actor types: car, auto, two_wheeler, cycle, bus, truck, cart, cow, auto_stand.
Maps: narrow_street, arterial_4lane, uncontrolled_junction, roundabout,
village_curvy, highway.

## 6. Metrics
| Metric | Meaning | Direction |
|---|---|---|
| collision_rate % | episodes ending in a collision | lower (#1) |
| vru_near_miss_avg | avg ticks within 2 m of an active VRU | lower, ~0 |
| driving_score | route completion × infraction penalties (CARLA DS) | higher |
| route_completion % | fraction of route reached | higher |
| success_rate % | clean goal arrivals | higher |
| offroad_rate / timeout_rate % | failure breakdown | lower |
| mean_jerk m/s² | ride comfort (potholes/bumps) | lower |

Penalties: collision_vru 0.45, collision_vehicle 0.55, collision_static 0.60,
offroad 0.65, lane_deviation 0.90.

For Indian conditions read **collision_rate + vru_near_miss before DS** — a
reckless agent scores higher DS by completing more route while crashing more.

## 7. Cameras, recording, and data logging (v0.2)

Enable camera sensors (adds RGB + depth + semantic per camera to `obs.camera`):
```bash
python bin/bharatsim-bench --agent riskfield --cameras --scenarios cow_on_road
```

Record a top-down GIF of an episode and log expert data for imitation learning:
```python
from bharatsim.eval.env import run_scenario
from bharatsim.eval.expert import ExpertAutopilot
from bharatsim.viz.recorder import Recorder
from bharatsim.eval.logger import DataLogger

rec, log = Recorder(), DataLogger(out_dir="dataset")
run_scenario(ExpertAutopilot(), "junction_freeforall", 0, recorder=rec, logger=log)
rec.save_gif("episode.gif"); log.flush()   # -> dataset/shard_0000.npz
```

## 8. Full benchmark + leaderboard
```bash
python bin/bharatsim-leaderboard --agent expert     --n 200
python bin/bharatsim-leaderboard --agent my_mod:Cls --n 200 --budget 600   # resumable
```
Writes `leaderboard/REPORT.md` (+ `report.json`, `rows.json`) with overall and
per-map Driving Score, collision rate, VRU near-miss, success, and jerk.

## 9. Traffic signals
Add a `JunctionControl` with `TrafficLight`s to a World (see
`bharatsim/core/signals.py`); agents obey with a per-agent probability, and the
ego is scored for red-light running. Wire lights in a scenario builder or extend
the DSL.
