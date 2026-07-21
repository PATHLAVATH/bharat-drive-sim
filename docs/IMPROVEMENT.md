# Separate Improvement Experiment — RiskFieldV2

A focused attempt to build a *better* planner than the current risk-field agent,
benchmarked honestly on a representative 8-scenario × 2-seed slice of the Bharat
suite (cow, cart-block, two-wheeler swarm, jaywalk swarm, potholes, highway,
junction, arterial).

## What RiskFieldV2 changes
1. **Time-to-collision longitudinal control** — brakes based on *when* a hazard
   is reached along the committed path (using the future-occupancy stack),
   instead of static forward risk.
2. **Gap-commitment hysteresis** — holds a chosen lateral corridor unless a
   clearly better one appears, removing left-right dithering.
3. **Explicit VRU caution** — pedestrian/animal cells get a larger safety
   inflation and a hard speed cap, targeting the VRU-near-miss metric directly.

## Result (honest)

| Agent | Driving Score | Collision rate | VRU near-miss |
|---|---|---|---|
| RiskFieldAgent (v1) | **65.9** | **25.0 %** | 1.19 |
| RiskFieldV2 (safety) | 56.5 | 31.2 % | **0.31** |

On the denser urban-only slice the VRU effect is starker: near-miss **3.17 → 0.00**
at equal collision rate.

## Interpretation — a genuine (partly negative) finding
RiskFieldV2 **substantially improves VRU safety** (the metric that matters most
on Indian roads) but does **not** improve — and slightly regresses — collision
rate and overall Driving Score, because the extra caution lowers route
completion and the stronger gap-seeking can wander on fast/wide roads.

Repeated hand-tuning (speed-gating, road-keeping weights) reshuffled metrics
without a net win. **That is the important conclusion:** classical planner
tuning trades one metric for another; a true Pareto improvement on unstructured
Indian traffic needs a **learned planner trained on data**, not more heuristics.
This validates the roadmap in `opendrive-e2e/docs/PROJECT_REPORT.md` §5:
generate demonstrations (expert + this sim's data logger), train a policy
(behavior cloning → DAgger → RL in the fast sim), then transfer to CARLA.

## Status
`RiskFieldV2` ships as a clearly-labeled **experimental, safety-focused**
variant (`bharatsim/eval/reference_v2.py`), available in the CLI as
`--agent riskfield_v2`. Use it when VRU safety is the priority and lower speed /
completion is acceptable; use `riskfield` otherwise. Small sample (16 episodes)
with high seed variance — treat magnitudes as indicative.
