# Project Guide: Autonomous A-to-B Navigation Robot

**Module:** Cranfield MSc Robotics, N-RBT-AIR AI for Robotics (group assignment)
**Deadline:** 12:00, 18 December 2026 (full-time)
**Status:** Week 1, planning and setup. Last updated 2026-11-12.

This is the living plan. Read it cold and you should know what we are building,
why, and what happens next. Anything not yet confirmed is marked **[OPEN]** and
is listed in §11. Every significant decision is recorded in the decision log (§12).

---

## 1. Requirements

### 1.1 Task
Navigate from **Point A** to **Point B** in a test arena with **static obstacles**
that represent buildings or other road users.

The system must demonstrate:
- optimised A-to-B navigation
- detection and avoidance of static obstacles
- safe operation in safety-critical situations
- robust, repeatable performance shown by systematic testing

### 1.2 AI requirements
- **ML classification** of objects or environment states from sensor data.
  *Confirmed with the lecturer:* classification may be implicit, fused into one
  model, **as long as behaviour differs by obstacle type** (e.g. steering wider
  around the human picture than around the car or building picture).
- **Reinforcement learning**, implemented separately, for decision-making
  (navigation / obstacle avoidance).
- **Safety, ethics, responsible AI** (10% of the mark): bias, reliability,
  robustness, and behaviour in novel situations.

### 1.3 Deliverables
| Deliverable | Notes |
|---|---|
| Narrated MP4 demo video (group) | architecture, detection, RL, navigation, challenges and fixes, safety/ethics, evidence of task completion |
| Technical report (group), max 3,000 words | Intro & Lit Review; System Design & Methodology; AI Development & Implementation; Results & Critical Discussion; Safety, Ethics & Responsible AI; Conclusion & Future Work; References; Appendices |
| Individual reflection, 600 words ±10% | contribution, decisions & trade-offs, challenges, learning, limitations. Analytical, not a diary |
| ZIP via Canvas | data prep scripts, prepared datasets, AI source code, docs, **failure scenarios and test cases** |
| Originality statement | in **both** the report and the source code |
| GenAI declaration | what was used and to what extent; declared in the report |

### 1.4 Marking weights
Literature & problem definition 20% · System design & justification 20% ·
AI development & implementation 25% · Evaluation & critical analysis 25% ·
Safety, ethics & responsible AI 10%.

Implication: evaluation (25%) is worth as much as implementation. A working robot
with weak metrics scores worse than a slightly worse robot with rigorous,
honest evaluation. Budget the time accordingly (see §10).

### 1.5 Suggested metrics (from the brief)
Classification accuracy · precision/recall/F1 · navigation success rate ·
collision rate · task completion time · RL convergence · repeatability.

---

## 2. Hardware

| Item | Status |
|---|---|
| Raspberry Pi single-board computer | assumed **Pi 5** **[OPEN]** |
| Raspberry Pi Pico | available. Role: independent e-stop + watchdog (§7) |
| Camera | Pi Camera Module 3 |
| Ultrasonic sensor | available. Model, mounting and cone angle **[OPEN]** |
| Infrared sensor | available. Type (forward proximity vs downward) and mounting **[OPEN]** |
| Mobile robot platform | drive type and motor driver **[OPEN]** |
| IMU | not in the kit. Adding one (gyro yaw rate) is **[OPEN]** |

Compute:
- **Laptop (dev):** Linux, Python 3.11 venv, MuJoCo 4.0.1, AMD Vega iGPU, no CUDA.
  Headless rendering through Mesa EGL. Used for coding and small checks.
- **Desktop (proposed main training machine):** Ryzen 7 9800X3D (8C/16T) +
  Radeon RX 9070 XT. GPU training needs PyTorch on ROCm. The desktop's OS is
  **[OPEN]** (Q18); ROCm is far more reliable on Linux.
- **Google Colab:** secondary / team-shared option (D-010).

Workload note: in this project the bottleneck is probably **env stepping and
rendering**, not network training. The policy is a small MLP and the encoder is
small and frozen during RL, so CPU core count matters more than the GPU for the
RL stage. The GPU mainly helps encoder pretraining and batched rendering.
Confirm with the FPS benchmark from the headless-render check.

---

## 3. Arena

- Inner size **1.5 m × 3.0 m**. Units in metres, **z up, origin at arena centre**.
  - *Proposed convention:* **x along the 3 m length, y across the 1.5 m width**,
    so the inner arena spans x ∈ [−1.5, 1.5], y ∈ [−0.75, 0.75]. **[OPEN]**: confirm.
- **Floor:** 3 white plaques, each 1.5 m × 1.0 m, laid along the 3 m length (seams
  every 1 m). Each plaque gets its own photo texture (placeholders first).
- **Walls:** each of the 4 sides is either
  - (A) a flat textured wall, or
  - (B) a row of cardboard boxes, 0.5 m long × 0.1 m thick × 0.3 m tall, standing
    on the 0.5 × 0.1 face. Box dimensions are **[OPEN]** until measured.
  The wall type can be chosen per side.
- **Obstacles:** printed pictures on paper, upright on small stands, modelled as
  thin 2D panes. Three classes with **one fixed image each**: **human, car,
  building**. Positions are random.
- **Goal (Point B):** proposed to be marked with an **ArUco marker** so the robot
  can estimate bearing and distance to it. Positions of A and B are **[OPEN]**.
- **Environment risk:** the real arena is next to a large window. Strong glare on
  the glossy white floor and changing daylight are the main perception risks.

---

## 4. System architecture

### 4.1 Design A (primary): pixel features + PPO

```
Pi Camera ──► crop band ──► downsample ~160×48 RGB ──► CNN encoder ──► z (~128)
                                                            │
                                                            ├──► [aux] obstacle head: 8 bearing bins × (presence, class, distance)
                                                            └──► [aux] floor head: free-floor distance in ~8 image columns
Ultrasonic (dist + validity flag) ───────┐
Goal bearing/distance (ArUco) ───────────┤
Previous action ─────────────────────────┼──► Policy MLP (PPO) ──► drive command ──► Pico ──► motors
(gyro yaw rate, if IMU added) ───────────┘                                            ▲
z ───────────────────────────────────────┘                     ultrasonic/IR e-stop ──┘ (overrides Pi)
```

- **Camera preprocessing:** crop part of the frame (currently "top half"),
  downsample to about 160×48, keep colour.
  **[OPEN]**: which band to keep depends on camera height and tilt. It must
  contain the panes *and* enough near floor for the floor head.
- **Encoder:** small CNN, frame → feature vector (~128).
- **Policy:** MLP trained with PPO. Inputs: encoder features, forward ultrasonic
  distance + validity flag, goal bearing/distance, previous action (+ gyro yaw
  rate if an IMU is added). Output: drive command. **Discrete vs continuous is
  [OPEN]** (§11).
- **Auxiliary heads** stay attached for evaluation (classification metrics) and
  for the confidence-based slowdown (§7), even though the policy does not read them.
- **How class-dependent behaviour arises:** the class information sits in z
  (forced there by the obstacle head's class loss). The policy learns to use it
  because the reward has a class-dependent clearance penalty (§6.3). We must
  *show* this by measuring minimum clearance per class (§8).

### 4.2 Design B (fallback): detector outputs + PPO
- Policy input = obstacle-head outputs (bearing, distance, class per bin) +
  ultrasonic + goal.
- Policy trained in a **fast 2D numpy simulator** with detector-like noise and
  dropouts (no rendering).
- Same encoder and heads; only the policy changes.
- **Decision point: end of week 3 (2 Dec).** If pixel-feature RL is not
  converging, switch to B. Convergence criteria **[OPEN]**: agree them before
  week 3, e.g. sim success rate ≥ X% after N steps.

### 4.3 Compute split on the robot **[OPEN]**
- Pi: camera capture, encoder + policy (ONNX Runtime) at 10 Hz, ArUco detection,
  logging.
- Pico: e-stop (ultrasonic/IR), command watchdog, possibly motor PWM.
- Still to decide: which board reads the ultrasonic and the motors, and the
  Pi↔Pico link (UART / USB serial). The ultrasonic is needed by **both** the
  policy (Pi) and the e-stop (Pico).

---

## 5. Simulation

### 5.1 Engine and scene
- **MuJoCo**, all-mesh arena (no Gaussian splats): floor planes, box cuboids,
  pane planes, and background photo planes behind the walls.
- Textures come from our own photos:
  - floor, shot straight down without glare
  - box sides
  - pane images (human / car / building)
  - background at robot height
- Render **RGB + segmentation** (geom IDs). Labels for the aux heads are generated
  automatically from segmentation and known poses.
- Must run **headless** (EGL; OSMesa as the fallback) on the dev PC and possibly
  on Colab.

### 5.2 Environment API: Gymnasium vs RLGym v2
**Recommendation: Gymnasium** (proposed, see D-009), with the code *organised*
into RLGym-style components (state mutators, obs builder, action parser,
reward, done conditions) as plain modules.
- **For Gymnasium:** standard API. Works with Stable-Baselines3 / CleanRL,
  vectorised envs, `gymnasium.utils.env_checker`, and TensorBoard logging with no
  glue code. The Design B 2D sim drops in behind the same interface.
  Teammates and markers recognise it.
- **For RLGym v2:** a clean separation of concerns and a good fit for multi-agent
  or game-style setups. But its ecosystem (rlgym-ppo etc.) is built around
  Rocket League-style use. We would write adapter code for a single-agent robot
  and get fewer reference examples when debugging under deadline.
- Splitting the code into components gives us RLGym's main benefit: ablations
  such as swapping the reward or replacing the ultrasonic with noise become a
  one-line config change.

### 5.3 Domain randomisation (per episode)
- wall type per side and box layout
- pane positions and rotations (including near edge-on)
- pane curl/tilt and brightness
- light direction and intensity
- floor tint and shininess (glare)
- background distractor images
- camera mounting: height, tilt, yaw
- small robot dynamics noise and action delay
- measured real end-to-end latency, added once known (§9)

### 5.4 Sensor models
- **Ultrasonic:** several rays across a ~15–30° cone, take the minimum.
  Per-episode offset and gain; per-step noise; dropouts (more likely at steep
  pane angles, because paper reflects sound away); 0–2 step latency. Outputs a
  distance + validity flag.
- **Goal bearing/distance:** in the real robot it comes from ArUco detection, so
  the sim must model marker-not-visible cases (out of field of view, occluded).
  **[OPEN]**: what the policy receives when the marker is not seen (last-seen value
  + age, odometry / dead reckoning, or a validity flag).
- **IR:** model only if it feeds the policy. Otherwise it is e-stop only (Pico).

---

## 6. Training plan

### 6.1 Stage 1: supervised encoder pretraining
- Dataset rendered from the randomised sim, labels from segmentation and poses.
- Obstacle head: 8 bearing bins × (presence, class, distance).
- Floor head: free-floor distance in ~8 image columns.
- Validate on synthetic held-out data **and** the real test set (§8.1) to measure
  the sim-to-real gap early.

### 6.2 Stage 2: RL with the encoder frozen
- PPO on the policy MLP, encoder frozen.
- Log convergence (return, success rate, collision rate, episode length) to TensorBoard.

### 6.3 Reward (initial idea, to iterate)
- **+** progress towards the goal (change in distance)
- **−** time penalty per step
- **−** collision penalty (and episode end)
- **+** success bonus on reaching B
- **−** class-dependent clearance penalty: **human > car > building**
Weights and clearance radii are **[OPEN]**. Record each reward change in the
decision log (the report must justify the reward design).

### 6.4 Stage 3 (optional): fine-tune
- Unfreeze the encoder at ~10× lower learning rate.
- Keep the result **only if real-test-set head scores do not drop**.

---

## 7. Safety, ethics and responsible AI

### 7.1 Safety architecture (defence in depth)
1. **Independent e-stop on the Pico** using ultrasonic/IR. It overrides the Pi's
   commands, so it still works if the vision pipeline or the policy crashes.
2. **Watchdog:** the Pico stops the motors if the Pi stops sending commands
   (timeout **[OPEN]**).
3. **Confidence-based slowdown:** reduce speed when detections are uncertain
   (e.g. low obstacle-head confidence or high entropy). Implemented as a
   runtime layer between policy and motors.
4. Class-aware clearance learned through the reward (human gets the widest margin).

### 7.2 Topics for the report
- **Bias:** lighting (window glare, time of day), class imbalance in the
  generated data, and having one image per class. The model may learn
  "this picture" rather than "human".
- **Reliability and robustness:** sim-to-real gap, ultrasonic dropouts on
  angled paper, glossy-floor glare.
- **Novel situations:** unseen object types, a moving person, a missing ArUco
  marker. What does the system do, and how does it fail safe?
- **Responsible AI:** honest reporting of failure cases, the limits of a
  picture-on-paper proxy for real road users, and the GenAI declaration.

---

## 8. Evaluation plan

### 8.1 Perception (aux heads)
- **Real test set:** 50–100 frames from the robot camera. Ground truth from ArUco
  markers on the pane stands (hidden or cropped from the scored images).
- Metrics: precision/recall/F1 for presence, class accuracy + confusion matrix,
  bearing and distance error, floor-distance error.
- Report **synthetic vs real** for every metric (the sim-to-real gap).

### 8.2 RL
- Convergence curves (TensorBoard), several seeds if compute allows.
- Sim success / collision rate on held-out randomisation seeds.

### 8.3 Real trials
- Success rate, collision rate, completion time.
- **Minimum clearance per obstacle class**, the evidence for class-dependent behaviour.
- Repeatability across many runs (number of trials and layouts **[OPEN]**).
- Log conditions per trial (time of day / lighting) for the bias discussion.

### 8.4 Ablations
- Ultrasonic input replaced by noise.
- (Optional) frozen vs fine-tuned encoder; Design A vs B if both exist.

### 8.5 Failure scenarios and test cases
Maintained in `docs/FAILURE_SCENARIOS.md` (required deliverable).

---

## 9. Deployment
- Export encoder + policy to **ONNX** and run them with **ONNX Runtime** on the Pi at **10 Hz**.
- **Lock the camera's exposure and white balance** (Picamera2) so the input
  distribution is stable.
- **Measure real end-to-end latency early** (capture → inference → motor
  command) and add it to the sim's action delay.

---

## 10. Timeline and milestones

| Week | Dates | Milestones |
|---|---|---|
| 1 | 12–18 Nov | Repo + docs; MuJoCo arena with headless rendering; arena measurements and texture photos; hardware bring-up (motors, camera, ultrasonic, Pico e-stop) |
| 2 | 19–25 Nov | Panes + randomisation + dataset generation; encoder pretraining; real test set captured and labelled |
| 3 | 26 Nov–2 Dec | RL training. **Decision A vs B on 2 Dec** |
| 4 | 3–9 Dec | Deployment on the robot; sim-to-real fixes; first real trials |
| 5 | 10–18 Dec | Evaluation trials; video; report; reflections; ZIP. **Submit by 12:00, 18 Dec** |

Risk notes:
- Week 4 is the only slot for sim-to-real fixes. Hardware bring-up and latency
  measurement in week 1 protect it.
- CPU-only training: test PPO throughput early (steps/s with rendering) and decide
  on Colab before week 3.
- Write the report alongside the work (lit review in weeks 1–2), not all in week 5.

### 10.1 Suggested workstreams (team size **[OPEN]**)
| Stream | Scope |
|---|---|
| S1 Simulation | MuJoCo arena, textures, randomisation, sensor models, Gymnasium env |
| S2 Perception | dataset generation, encoder + heads, real test set, perception metrics |
| S3 RL | reward design, PPO training, Design B 2D sim, convergence analysis |
| S4 Hardware & safety | motors, camera, Pico e-stop + watchdog, ONNX deployment, latency |
| S5 Evaluation & report | trial protocol, logging, failure scenarios, lit review, video |

- 3 people: S1+S2 / S3+S5 / S4
- 4 people: S1 / S2 / S3 / S4, with S5 shared
- 5 people: one stream each

Everyone owns at least one clearly attributable technical piece, which the
individual reflection needs.

---

## 11. Open questions

| # | Question | Blocks |
|---|---|---|
| Q1 | Raspberry Pi model (assumed Pi 5)? | ONNX speed, camera stack |
| Q2 | Drive type (differential / other) and motor control on the Pi or the Pico? | action space, sim dynamics, safety design |
| Q3 | Add an IMU (gyro yaw rate)? | policy input, odometry |
| Q4 | Camera height and tilt on the robot? | sim camera, crop band |
| Q5 | Exact box dimensions? | wall model |
| Q6 | Action space: discrete or continuous? | PPO setup, deployment |
| Q7 | Team size and roles? | workstreams |
| Q8 | Where are Point A and Point B? Fixed or varied? | env reset, evaluation |
| Q9 | Any arena rules for the demo (time limits, allowed markers, number of obstacles)? | ArUco use, evaluation |
| Q10 | Coordinate convention: x along the 3 m length? | sim, labels |
| Q11 | Which camera band to crop (depends on Q4)? | encoder input |
| Q12 | Goal input when the ArUco marker is not visible? | obs design, sim |
| Q13 | Ultrasonic and IR models, mounting, and which board reads them? | sensor model, e-stop |
| Q14 | Pi↔Pico link and watchdog timeout? | safety |
| Q15 | Convergence criteria for the week-3 A/B decision? | D-001 |
| Q16 | Reward weights and per-class clearance radii? | RL |
| Q17 | Number of real trials and layouts for evaluation? | evaluation |
| Q18 | Desktop OS (Linux / Windows / dual boot)? Does PyTorch ROCm work on the 9070 XT there? | D-010 |

---

## 12. Decision log

Format: **ID · date · decision**. Why, alternatives, status.
Status: *accepted*, *proposed* (pending team agreement) or *superseded by D-xxx*.

**D-001 · 2026-11-12 · Design A (pixel features → PPO) primary, Design B (detector outputs → PPO in 2D sim) fallback**
- Why: A is a richer, more original use of learned features and is end-to-end
  learnable. B de-risks RL convergence because it trains quickly without
  rendering. Sharing the encoder and heads means the fallback costs little.
- Alternatives: B only (lower risk, less original); end-to-end pixel RL without
  pretraining (very sample-hungry on CPU).
- Status: accepted. Switch decision at the end of week 3.

**D-002 · 2026-11-12 · Classification fused into the model (implicit), shown through class-dependent behaviour**
- Why: confirmed with the lecturer as acceptable. Avoids a separate detector
  pipeline. The aux obstacle head still gives explicit class metrics.
- Alternatives: separate explicit classifier feeding a rule-based or RL planner.
- Status: accepted.

**D-003 · 2026-11-12 · MuJoCo with an all-mesh arena (no Gaussian splats)**
- Why: fast headless rendering, built-in segmentation for automatic labels, and
  full control of randomisation.
- Alternatives: Gaussian-splat scene reconstruction (rejected: harder to randomise
  and to get labels from).
- Status: accepted.

**D-004 · 2026-11-12 · Staged training: supervised encoder pretraining with aux heads → RL with frozen encoder → optional fine-tune**
- Why: supervised labels from the sim are cheap and make RL far more
  sample-efficient. The aux heads provide classification metrics.
  Fine-tuning is kept only if real-test scores hold.
- Alternatives: end-to-end RL from pixels; RL from raw sensors only.
- Status: accepted.

**D-005 · 2026-11-12 · Independent Pico e-stop + watchdog overriding the Pi**
- Why: safety must not depend on the ML stack. Hardware-level separation of
  safety from autonomy.
- Alternatives: software-only stop on the Pi.
- Status: accepted.

**D-006 · 2026-11-12 · ONNX Runtime on the Pi at 10 Hz; camera exposure and white balance locked**
- Why: portable, efficient CPU inference. Locked camera settings reduce
  input-distribution shift under changing daylight.
- Alternatives: PyTorch on the Pi; TFLite. **[to evaluate if ONNX causes problems]**
- Status: accepted.

**D-007 · 2026-11-12 · Goal (Point B) marked with an ArUco marker**
- Why: gives bearing and distance to the goal from the camera alone.
- Alternatives: odometry only; fixed known goal + dead reckoning.
- Status: proposed. Depends on demo rules (Q9).

**D-008 · 2026-11-12 · Real test-set ground truth from ArUco markers on pane stands**
- Why: objective bearing/distance labels without manual measurement. Markers are
  hidden or cropped from scored images to avoid leaking the answer.
- Status: accepted.

**D-009 · 2026-11-12 · Environment API: Gymnasium, code organised in RLGym-style components**
- Why: see §5.2. Standard tooling (SB3/CleanRL, vector envs, checker) and a
  lower integration risk under deadline.
- Alternatives: RLGym v2 TransitionEngine.
- Status: proposed, awaiting team agreement.

**D-010 · 2026-11-12 · Train on the 9800X3D/9070 XT desktop; keep the code Colab-portable**
- Why: 16 threads run many parallel MuJoCo envs. The free Colab tier offers few
  CPU cores, session time limits and idle disconnects, which suits CPU-bound RL
  poorly. Portability (requirements file, `MUJOCO_GL=egl`, relative paths,
  config files, checkpoint/resume) still lets teammates and extra seeds or
  ablations run on Colab.
- Alternatives: Colab as the main trainer (weak CPU, session limits); laptop CPU
  only (slow).
- Status: proposed. Depends on Q18 (desktop OS / ROCm).

**D-011 · 2026-11-12 · Training stays in Python; no C++ port. Speed comes from profiling and parallelism**
- Why: the heavy compute already runs in C/C++ under Python (MuJoCo physics and
  rendering, PyTorch kernels, ONNX Runtime). Python is only the glue. A port
  would cost weeks for, at best, a few-times speed-up on the per-step glue code,
  and the deadline is 18 Dec.
- How to speed up instead: profile first; run envs in parallel processes;
  vectorise per-step Python logic with numpy; keep rendering resolution small.
- Alternatives: C++ training loop (LibTorch + MuJoCo C API). Rejected for time
  and risk; revisit only if profiling shows Python glue dominates *and* the
  fixes above are exhausted.
- Status: proposed.

---

## 13. Repository structure

```
README.md                 how to install and run (later)
requirements.txt          PC / training dependencies
requirements-pi.txt       robot runtime dependencies (later)
configs/                  YAML configs (sim randomisation, training, reward)
docs/
  PROJECT_GUIDE.md        this file
  FAILURE_SCENARIOS.md    failure scenarios and test cases
  measurements.md         arena, robot and camera measurements (to create)
assets/
  textures/{floor,boxes,panes,background}/
  placeholders/           generated placeholder textures
sim/                      MuJoCo arena builder, rendering, randomisation, sensors, Gymnasium env
perception/               dataset generation, encoder + heads, pretraining, head evaluation
rl/                       PPO training, policy, reward components
  sim2d/                  Design B numpy simulator
deploy/
  pi/                     camera capture, ONNX inference loop, ArUco, logging
  pico/                   MicroPython e-stop + watchdog
eval/                     real test set tools, trial logging, analysis
scripts/                  one-off checks and utilities
tests/                    unit tests (sensor models, labels, reward)
data/                     generated / captured data (large files git-ignored; curated subset goes in the ZIP)
runs/                     TensorBoard logs and checkpoints (git-ignored)
```
