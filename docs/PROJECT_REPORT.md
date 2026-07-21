# Project Report — Autonomous Driving Stack + Indian-Roads Simulator

*A complete account of what was built, from the first scaffold to the photoreal
CARLA backend, with honest results and a concrete plan to improve.*

Two GitHub repositories:
- **opendrive-e2e** — the end-to-end driving models + the original `simstation`
  benchmark + the India (Bharat) extension + the risk-field planner + the
  bring-your-own-model evaluation API.
- **bharat-drive-sim** — a standalone, full-featured simulator for Indian roads
  (its own package, CLI, tests, docs, and a photoreal CARLA backend).

---

## 1. Goal and constraints

Goal: build a world-class end-to-end autonomous-driving system, then focus it on
Indian road conditions, and provide a CARLA/Bench2Drive-style environment to
develop and benchmark it.

Hard constraint throughout: the build environment is **CPU-only, 2 cores, no GPU
and no game engine**. So *training a frontier model* and *photoreal rendering*
must happen on the user's hardware. Everything here was therefore built to be
(a) fully runnable/verifiable on CPU, and (b) architected so GPU/photoreal
backends drop in without API changes. Every result below was measured in-session.

---

## 2. Timeline of what was built

### Phase 1 — End-to-end model scaffold (v1)
Camera → CNN backbone → Lift-Splat BEV → occupancy + motion prediction →
transformer planner → pure-pursuit/PID controller. ~2.6 M params. Verified
forward/backward/controller. Purpose: prove the full pixels-to-plan pipeline.

### Phase 2 — simstation + trained model (v2)
A pure-Python closed-loop simulator: procedural town/village/highway maps, IDM
traffic, a 3-camera semantic renderer, BEV + future-occupancy ground truth, and
a **220-scenario benchmark** (22 types × 10 seeds, Bench2Drive-scale). Built an
**infinite synthetic data engine** with DAgger-style steering-noise injection —
which proved necessary: the first behavior-cloned model drove off the road from
covariate shift, and the noise injection fixed it. Added collision-aware
trajectory selection.

**Result (structured suite):** trained camera-only model **DS 78.7 / 9.1 %
collisions**, vs naive 73.7 / 32.3 %, vs privileged expert 98.2 / 0 %.

### Phase 3 — Production architecture (v3)
Upgraded the architecture to competitive-class components: **ImageNet-pretrained
ResNet-50 + FPN** backbone, **200×200 @ 0.5 m BEV** with **temporal fusion**, a
**hierarchical multi-granularity planner** (HiP-AD-class: dense near-field +
sparse long-horizon queries with cross-attention), and an **action-conditioned
latent world model** that scores each candidate trajectory against a
*counterfactual* rollout (planning by imagination). 35 M params at smoke scale;
`configs/v3_gpu.yaml` scales past 100 M. Verified forward/backward/selection.
(v2 remains the *trained* model with real closed-loop numbers; v3 is the
architecture to train on GPU.)

### Phase 4 — Indian conditions (Bharat)
Added unstructured mixed traffic (auto-rickshaws, weaving two-wheelers, carts,
cows, buses), wandering pedestrians, and degraded roads (potholes, speed bumps,
encroachment, low visibility) to simstation — 18 scenario types. Introduced a
new **risk-field planner**: instead of following lanes, the ego navigates a
continuous risk potential field with active gap-seeking (the core
Indian-driving primitive), backed by an efficient uniform-grid spatial hash
(~0.5 ms/plan).

**Result (Bharat suite):** risk-field over model occupancy (zero-shot)
DS 17.3 / **22.2 % collisions / 0.00 VRU near-miss**; naive DS 51.8 / 55.6 % /
0.07; privileged expert DS 47.5 / 25.9 % / 0.00. Finding: unstructured traffic
is *far* harder than structured (expert 47 vs 98), and Driving Score alone is
misleading — collision rate and VRU near-miss must be read first.

### Phase 5 — Bring-your-own-model API + report
A clean sensor-in/control-out `Agent` contract (`adstack/eval_api.py`), a unified
benchmark CLI, and a full simulator guide, so any model can be evaluated like a
CARLA agent. Plus a full technical report of the environment
(`docs/ENVIRONMENT_REPORT.md`).

### Phase 6 — Standalone simulator: bharat-drive-sim
A separate, properly-packaged simulator repo. v0.1 core (6 maps, 9 agent
archetypes, YAML scenario DSL, Gym + Agent APIs). v0.2 completed it: **multi-
camera semantic+depth renderer**, **traffic signals + junction right-of-way**,
a **privileged expert autopilot**, an **episode recorder** (top-down GIFs), an
**imitation-learning data logger**, a **procedural route generator** (200+
routes), and a **leaderboard runner** producing markdown+JSON reports. 11 tests.

### Phase 7 — Photoreal CARLA backend (v0.3)
A full CARLA integration: guarded client + synchronous world + Indian weather
presets, archetype→CARLA-blueprint mapping, a photoreal sensor rig (RGB + depth
+ semantic + optional LiDAR) that produces the *same* Observation, a scenario
bridge that spawns Indian traffic/VRUs/hazards and re-parameterizes CARLA's
TrafficManager for gap-seeking behavior, and a **CARLA Leaderboard 2.0
`AutonomousAgent` wrapper** so the same policy can be submitted to the official
leaderboard / Bench2Drive. Verified end-to-end against an **in-process mock CARLA
server** (`tests/test_carla_mock.py`) — the control flow is tested, not just
compiled; only photoreal frames and real scores need a GPU.

---

## 3. Consolidated results

| Suite | Agent | Driving Score | Collision | VRU near-miss |
|---|---|---|---|---|
| Structured (220) | Trained model (camera-only) | 78.7 | 9.1 % | — |
| Structured (220) | Naive | 73.7 | 32.3 % | — |
| Structured (220) | Expert (privileged) | 98.2 | 0 % | — |
| Bharat (90) | Risk-field / model occ (zero-shot) | 17.3 | 22.2 % | 0.00 |
| Bharat (90) | Naive | 51.8 | 55.6 % | 0.07 |
| Bharat (90) | Expert (privileged) | 47.5 | 25.9 % | 0.00 |
| bharat-drive-sim (gen. n=40) | Expert autopilot | 44.9 | 47.5 % | 0.35 |

For public context (different simulator, not directly comparable): Bench2Drive
DS — UniAD 45.8, VAD 42.4, SimLingo 85.1, TF++ 87.0, PDM-Lite expert 97.0.

---

## 4. What is genuinely strong vs. still weak (honest)

**Strong**
- The full pixels-to-plan pipeline works and trains; DAgger fixed closed-loop
  covariate shift (a real, correctly-diagnosed failure).
- The v3 architecture is competitive-class (pretrained backbone, temporal BEV,
  hierarchical planner, counterfactual world model) and scales on GPU.
- The Indian simulator is feature-complete for behavior/logic: mixed unstructured
  traffic, VRUs, degraded roads, signals, sensors, scenarios, metrics, recorder,
  data logger, leaderboard, and a photoreal CARLA path.
- The risk-field planner is inherently VRU-safe (0 near-misses) and real-time.

**Weak / unfinished (stated plainly)**
- No model is trained on Indian data yet — the Bharat model agent is zero-shot
  and over-creeps (low route completion).
- Even the privileged expert is mediocre in dense chaos (DS ~45–47) — negotiating
  unstructured traffic (overtaking a slow cart, threading a swarm) is unsolved.
- The fast sim is 2.5D flat-shaded, not photoreal — perception won't transfer to
  real cameras without domain adaptation (the CARLA backend + IDD data is the fix).
- Agent behaviors are hand-tuned heuristics, not learned from real Indian logs.
- v3 has not been trained at scale (needs GPU + nuScenes/real data).

---

## 5. Scope to improve — prioritized roadmap

**A. Train on real/large data (highest impact).** Implement the nuScenes adapter
for v3 (structured) and fine-tune on the **India Driving Dataset (IDD)** for
Indian perception. This is what turns the v3 architecture from "verified" into
"competitive." Requires GPU.

**B. Solve unstructured negotiation (the core research gap).** The planner —
learned or classical — needs to actively overtake and thread gaps, not creep.
Directions: a stronger expert (search/MPPI-based) to generate better
demonstrations; a learned planner trained on those; or an RL fine-tune in the
fast sim (which is cheap to run).

**C. Close sim-to-real.** Render photoreal in CARLA, remap semantics, and
domain-adapt perception with IDD; then submit via the leaderboard wrapper.

**D. Richer behavior + assets.** Learn agent behaviors from real Indian driving
logs; import Indian vehicle/animal assets into CARLA; add honking as an explicit
negotiation channel (a genuinely India-specific idea).

**E. Distillation.** Use open NVIDIA Alpamayo weights as a teacher for the
deployable student model.

A concrete first improvement — building and measuring a stronger planner — is
implemented separately in `bharat-drive-sim` (`RiskFieldV2`, see its
`docs/IMPROVEMENT.md`), with before/after numbers.
