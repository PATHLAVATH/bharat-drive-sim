# Design

## Layout
```
bharatsim/
  core/      geometry (Path/OBB), spatial hash, entity system, World (physics)
  agents/    TrafficAgent (lane-free gap-seeking + IDM), VRU, 9 archetypes
  maps/      6 map builders (MapData: route + named traffic paths + geometry)
  sensors/   BEV semantic raster + future-occupancy (extend for photoreal)
  scenarios/ YAML DSL + 19 built-in specs
  eval/      Gym env + Agent API + suite runner + CARLA-style metrics
bin/         bharatsim-bench CLI
scenarios/   example YAML   examples/  BYO agent   tests/  docs/
```

## Principles
- **Entity uniformity.** Ego, traffic, VRUs, hazards all implement the same
  `Entity` interface (pose, corners, step), so the World loop, spatial hash, and
  collision system treat them uniformly — adding a new object is one class.
- **Lane-free first.** Behavior is a lateral gap-seeking drift over IDM, not a
  lane-keeper. This is the core Indian-driving primitive.
- **Sensor abstraction.** Models consume `Observation` (BEV + future occupancy +
  ego state). Swapping in a 3D/photoreal renderer or a CARLA bridge means
  reimplementing `sensors/` only — the agent/eval code is unchanged.
- **Efficiency.** Uniform-grid spatial hash for O(1) neighbor queries; vectorized
  numpy rasters; ~0.5 s per scenario on CPU.

## Extending
- New map: add a builder returning `MapData`, register in `maps/library.py::MAPS`.
- New agent type: add a row to `agents/traffic.py::ARCHETYPES`.
- New scenario: a YAML file, or a dict in `scenarios/builtin.py::SPECS`.
- New sensor/metric: `sensors/bev.py` / `eval/metrics.py`.
- Photoreal / CARLA: implement an alternative `sensors` backend and (optionally)
  a physics bridge; keep the `Observation`/`Control` contract.

## Honest limitations
2.5D semantic rendering (not photoreal); heuristic agent behaviors (not learned
from logs); no signals/roundabout-yield logic yet; visibility is a scalar. These
are the prioritized next steps.
