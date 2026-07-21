# Photoreal Evaluation with CARLA

This is the **photoreal backend** for bharat-drive-sim. The same scenarios and
the same `Agent` API you use in the pure-Python sim run against **CARLA** (Unreal
Engine, photoreal cameras + LiDAR + semantic seg), and the same policy can be
submitted to the **CARLA Leaderboard 2.0 / Bench2Drive**.

> **Requires a GPU + CARLA.** The pure-Python sim needs neither. This backend
> was developed with the integration verified against an in-process mock server
> (`tests/test_carla_mock.py`); run the real thing on a CUDA GPU to get photoreal
> frames and leaderboard scores.

## 1. Hardware / software
- NVIDIA GPU, 8 GB+ VRAM (16 GB+ recommended for dense scenes).
- Ubuntu 20.04/22.04, CUDA-capable driver.
- **CARLA 0.9.15** (`https://github.com/carla-simulator/carla/releases`).
- Python 3.8–3.10 matching the CARLA PythonAPI egg.

## 2. Install
```bash
# 1) CARLA
wget https://.../CARLA_0.9.15.tar.gz && tar -xf CARLA_0.9.15.tar.gz -C carla
# 2) PythonAPI on PYTHONPATH (or pip)
pip install carla==0.9.15
# 3) bharat-drive-sim
pip install -e .        # numpy, opencv, pyyaml
# 4) (leaderboard) clone scenario_runner + leaderboard 2.0 per their READMEs
```

## 3. Start the simulator
```bash
./CarlaUE4.sh -RenderOffScreen -quality-level=Epic     # headless server
```

## 4. Run a scenario (photoreal, closed-loop)
```python
from bharatsim.carla.bridge import run_scenario_carla
from bharatsim.eval.reference import RiskFieldAgent
from bharatsim.scenarios.builtin import SPECS

row = run_scenario_carla(RiskFieldAgent(), SPECS["junction_freeforall"],
                         seed=0, cameras=True)   # host/port default localhost:2000
print(row)   # {result, rc, ds, mean_jerk, sim_time}
```
Every bharatsim scenario (built-in dict, generated route, or YAML DSL) works —
the bridge maps the map to the closest CARLA town, sets weather from the
scenario's `environment` label (clear/monsoon/dust/night/dawn), spawns Indian
mixed traffic + pedestrians + hazards, and runs your agent with photoreal
sensors.

## 5. Indian traffic behavior in CARLA
CARLA's TrafficManager is lane-disciplined by default. `bridge._apply_indian_behavior`
re-parameterizes each vehicle from its bharatsim archetype (aggression →
speed-difference, following distance, lane-offset, signal/sign ignore %, random
lane changes) to reproduce gap-filling and weaving. Archetypes map to the closest
stock blueprints in `blueprints.py`.

> **Indian asset pack (recommended):** stock CARLA lacks auto-rickshaws, carts,
> and cattle. Import Indian meshes as CARLA blueprints and extend
> `BLUEPRINT_MAP` / `HAZARD_PROPS`. Datasets like IDD (India Driving Dataset)
> can be used to fine-tune perception and to author assets.

## 6. Sensors
`bharatsim/carla/sensors.py` attaches a 3-camera ring (RGB + depth + semantic
seg) + optional LiDAR and assembles the **same `Observation`** the pure sim
produces. CARLA semantic tags are remapped to bharatsim classes, so a model
trained in the pure sim transfers directly. Configure in `configs/sensors.yaml`.

## 7. Submit to the CARLA Leaderboard / Bench2Drive
`bharatsim/carla/leaderboard_agent.py` is a Leaderboard-2.0
`AutonomousAgent`. Point the evaluator at it and select which policy to run via
`configs/carla_agent.yaml`:
```yaml
agent: "bharatsim.eval.reference:RiskFieldAgent"   # or your trained model
```
```bash
export TEAM_AGENT=$(pwd)/bharatsim/carla/leaderboard_agent.py
export TEAM_CONFIG=$(pwd)/configs/carla_agent.yaml
python leaderboard/leaderboard_evaluator.py \
    --agent=$TEAM_AGENT --agent-config=$TEAM_CONFIG \
    --routes=leaderboard/data/routes_devtest.xml --track=SENSORS
```
For **Bench2Drive**, use its 220-route set with the same agent wrapper.

## 8. Train, then transfer
1. Generate imitation data in the fast pure sim (`DataLogger` + `ExpertAutopilot`).
2. Train your model on it (behavior cloning / DAgger).
3. Wrap it as an `Agent`, benchmark in the pure sim (`bharatsim-leaderboard`).
4. Switch on cameras and run the SAME agent here in CARLA; fine-tune perception
   on real IDD data to close the sim-to-real gap.
5. Submit via the leaderboard wrapper.

## What is verified vs. what needs your GPU
- **Verified here (mock server):** full control flow — client/sync/weather,
  blueprint resolution, sensor rig spawn + frame assembly, traffic/VRU/hazard
  spawning, tick loop, observe→act→apply, metrics, teardown, leaderboard sensor
  spec (`tests/test_carla_mock.py`).
- **Needs a real CARLA + GPU:** photoreal frames, physics accuracy, and actual
  leaderboard scores. Nothing in the API changes — only the backend.
