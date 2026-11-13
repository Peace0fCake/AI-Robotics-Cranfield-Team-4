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
| Camera | Pi Camera Module 3, standard lens assumed (66° horizontal, 41° vertical FOV; sensor 4608×2592, full-FOV binned mode 2304×1296). Used at **4:3**: a central full-height crop, about 53° horizontal, 41° vertical (D-014). Lens 0.145 m above the floor and 0.13 m behind the front bumper (measured). Fixed mount, level, cannot be pitched |
| Ultrasonic sensor | available. Model, mounting and cone angle **[OPEN]** |
| Infrared sensor | available. Type (forward proximity vs downward) and mounting **[OPEN]** |
| Mobile robot platform | Car-like: front-wheel steering, rear-wheel drive. 0.28 m long × 0.17 m wide, wheelbase 0.15 m (measured ±1 cm). Cannot turn on the spot. Maximum steering angle, axle positions and motor driver **[OPEN]** (Q2, Q20) |
| IMU | not in the kit. Adding one (gyro yaw rate) is **[OPEN]** |

Compute:
- **Laptop (dev):** Linux, Python 3.11 venv, MuJoCo 4.0.1, AMD Vega iGPU, no CUDA.
  Headless rendering through Mesa EGL. Used for coding and small checks.
- **Team machines:** a mix of Linux and Windows (D-012). On a Windows 11 dev PC
  the offscreen-render probe (one box on a floor) gave about 2,000 fps at
  640×480 and 2,700 fps at 160×48 with MuJoCo 4.0.1.
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
  - (B) a row of cardboard boxes, measured **0.36 m long × 0.12 m thick ×
    0.26 m tall**, standing on the 0.36 × 0.12 face. A 3 m side takes 8⅓ boxes
    and a 1.5 m side 4⅙, so rows overhang the corners or leave gaps.
  The wall type can be chosen per side.
- **Obstacles:** printed pictures on paper, upright on small stands, modelled as
  thin 2D panes. Three classes with **one fixed image each**: **human, car,
  building**. Printed in **black and white**, and each class is a different
  size (Q19). The picture sits on a larger white sheet with a wide margin, the
  sheet bends forward or backward on its stand, and the back shows the same
  picture, not mirrored, so a pane looks the same from either side.
  Positions are random. Sim model: D-015.
- **Start (Point A):** on the first plaque (the box end wall, x = −1 m), at a
  random spot around the plaque centre, facing down the arena towards +x.
- **Obstacles:** only on the middle and last plaques; the first plaque is clear.
- **Finish (Point B):** crossing the **black line** on the last plaque, about
  5 cm in from its outer edge (x ≈ 1.45 m, estimated). The goal is a line, not
  a point, which changes what the policy needs as its goal input (Q12, D-007).
- **Observed in the first arena photo (2026-11-12, temporary location; all to
  be confirmed by measurement):**
  - one long side is the building wall: cream trunking with sockets at arena
    height, floor-to-ceiling windows directly above it
  - the other long side and one short side are rows of cardboard boxes; the
    boxes look much deeper than the 0.1 m assumed above, some carry a red/white
    sticker, and they sit unevenly with gaps and small rotations
  - the remaining short end has a black line sticker on the last plaque and
    carpet beyond it; whether that end is closed during runs is **[OPEN]** (Q21)
  - the obstacle picture on its stand looks far smaller than A4 (Q19)
  - the arena will move to the lab; the final wall is not photographed yet
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

- **Camera preprocessing:** crop the frame to 4:3, downsample to 256×192,
  keep the **bottom half** (256×96), keep colour (D-014).
  With the lens at 0.145 m and no tilt, the horizon is the middle image row:
  the whole floor and anything shorter than 0.145 m lie in the bottom half.
  Consequences to design around:
  - the nearest visible floor is about 0.39 m ahead of the lens, which is
    0.26 m ahead of the front bumper; closer than that, only
    the part of an object above the bottom edge of the view is seen, so the
    ultrasonic and IR cover the near field
  - anything above 0.145 m is cut, including the upper part of the boxes and of
    any pane taller than that (Q19)
  - the far wall base sits only about 14 rows below the horizon, so a small
    pitch error moves distant objects out of the band. Whether to keep a few
    rows above the horizon as margin is **[OPEN]** (Q11)
  - ArUco detection runs on the uncropped frame, not on the CNN input
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

### 4.2 Design B (fallback): YOLO bounding boxes + PPO
- Detector: an Ultralytics YOLO model trained on rendered frames with
  automatic labels, then real robot frames (D-018). Developed in parallel
  with Design A, not only after a switch.
- Policy input = the detected boxes turned into a fixed-size vector (e.g. the
  N nearest boxes as class, centre, width, height, or bearing bins) +
  ultrasonic + goal.
- Policy trained in a **fast 2D numpy simulator** with detector-like noise and
  dropouts (no rendering).
- Shares the arena, sample generator, reward and evaluation with Design A;
  only the perception model and the policy input differ.
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
- Must run **headless** (EGL; OSMesa as the fallback) on Linux and Colab, and
  offscreen through MuJoCo's default backend on Windows. `import sim` selects
  the backend and must come before `import mujoco` (D-012).
- The arena is built in code with `mujoco.MjSpec` from `configs/arena.yaml`
  (`sim/arena.py`), not from a hand-written XML file (D-013).

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
- wall type per side
- boxes: small position and rotation error on each box, as if placed by hand
- panes: 1–2 of each class (3–6 in total) at random positions in the 2 m
  beyond the start zone; the first metre from the start is normally empty
- panes stand roughly square across the arena (facing the start end and the
  back wall), with a small random yaw and now and then a larger one
- pane lean forward or backward, paper size and margins, brightness
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
- **−** collision penalty (and episode end). This is the baseline. A variant
  where hitting a pane is penalised by class but does not end the episode, so
  the robot can drive over one when the way is blocked, is a stretch
  experiment once the baseline converges (D-016)
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
| Q2 | ~~Drive type?~~ Answered 2026-11-13: car-like, front-wheel steering on a servo, rear-wheel drive. Minimum turning radius about 0.50 m (estimate, 2026-11-13), giving about 16.7° maximum steering in the bicycle model. Still open: which point that radius was measured to, left/right symmetry, the servo-command-to-curvature curve, top speed, and whether motor control runs on the Pi or the Pico | action space, sim dynamics, safety design |
| Q3 | Add an IMU (gyro yaw rate)? | policy input, odometry |
| Q4 | Camera: height 0.145 m (corrected 2026-11-13 from 0.16 m), 0.13 m behind the front bumper, level, fixed mount (answered). Still open: standard or wide lens, and how far off level the mount really is? | sim camera, crop band |
| Q5 | ~~Exact box dimensions?~~ Answered 2026-11-13: 36 × 26 × 12 cm. Still open: how the rows are finished at the corners (overhang, gap, or a box turned sideways)? | wall model |
| Q6 | Action space: discrete or continuous? | PPO setup, deployment |
| Q7 | Team size and roles? | workstreams |
| Q8 | ~~Where are Point A and Point B?~~ Answered 2026-11-13: start around the centre of the first plaque (box end), facing +x; finish = crossing the black line on the last plaque. Still open: the real spread of start positions and headings, and whether the far end is open or walled in the lab | env reset, evaluation |
| Q9 | Any arena rules for the demo (time limits, allowed markers, number of obstacles)? | ArUco use, evaluation |
| Q10 | Coordinate convention: x along the 3 m length? | sim, labels |
| Q11 | Crop band: exactly the bottom half (rows 96–192), or start a few rows above the horizon as margin for pitch error? | encoder input |
| Q12 | Goal input. The finish is a line across the arena, so the robot mainly needs its heading relative to +x and its progress along x, not a bearing to a point. Options: an ArUco marker beyond the line; heading from an IMU or wheel odometry; seeing the line itself. And what does the policy get when that source is lost? | obs design, sim, D-007 |
| Q13 | Ultrasonic and IR models, mounting, and which board reads them? | sensor model, e-stop |
| Q14 | Pi↔Pico link and watchdog timeout? | safety |
| Q15 | Convergence criteria for the week-3 A/B decision? | D-001 |
| Q16 | Reward weights and per-class clearance radii? Proposal to decide: two tiers instead of three (human gets a wider soft margin; car and building share one), with an unreadable pane treated as human. The classifier stays three-class either way | RL |
| Q17 | Number of real trials and layouts for evaluation? | evaluation |
| Q18 | Desktop OS (Linux / Windows / dual boot)? Does PyTorch ROCm work on the 9070 XT there? | D-010 |
| Q19 | Pane sizes. Sheet heights compared with a Pixel 8 Pro: building = phone height (16.3 cm), human ≈ bottom of the camera bar (≈13 cm, estimate), car ≈ bottom of the power button (≈10.5 cm, estimate). The sheet is a white card with a wide margin. Still needed: ruler measurements of each sheet and of the **picture** on it, gap between floor and paper. The picture size drives D-014: with the current margin guesses the human drawing is only ≈ 3 × 8 cm, about 7 px wide at 1 m and 4 px at 2 m at 256×192 | pane model, labels, D-014 |
| Q21 | Which sides are closed during runs? (The black line is the finish line, answered 2026-11-13.) Final lab wall appearance? Temporary trunking wall is estimated from a photo at about 0.26 m high with the socket pattern repeating every 0.38 m | wall model, A/B positions, background |
| Q22 | What clearance applies to a pane whose class cannot be read (seen edge-on, or too far away)? The back is not an issue: it shows the same picture as the front | reward, safety |
| Q23 | Running over panes (D-016): does the brief or the demo allow it, and how is it scored? Should hitting the human end the run? What does the Pico e-stop do when the policy chooses to drive over a pane? What happens when the robot hits a box wall? | reward, e-stop design, evaluation |
| Q20 | ~~Robot footprint?~~ Answered 2026-11-13 (±1 cm): 0.28 × 0.17 m, wheelbase 0.15 m; camera 0.13 m and ultrasonic pivot 0.08 m behind the bumper (sensor on a 2 cm radius, 0.115 m high); line sensor 5 cm ahead of the front axle. An earlier rounded figure of 0.30 × 0.20 m is kept as the padded collision envelope (D-017). Front axle 0.08 m behind the bumper (2026-11-13), so the rear axle is 0.23 m behind it. Still open: whether the ultrasonic sweeps during runs or stays forward | collision checks, dynamics, sensor model |

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
- Status: proposed, under review. The goal turned out to be a finish line,
  not a point (§3), so a marker is no longer the obvious choice (Q12).
  Depends on demo rules (Q9).

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

**D-012 · 2026-11-12 · The code runs on both Linux and Windows; GL backend chosen per OS in `sim/__init__.py`**
- Why: team members use different operating systems. `MUJOCO_GL=egl` is
  Linux-only; on Windows it makes `import mujoco` raise, so the backend cannot
  be hard-coded in each script. Virtual environments are created per machine
  from `requirements.txt` and never copied.
- Alternatives: Linux only (excludes teammates); WSL2 or Docker for Windows
  users (extra setup, GPU/GL passthrough problems, no gain for a pip-only stack).
- Status: accepted.

**D-013 · 2026-11-12 · Arena built programmatically with `mujoco.MjSpec` from a YAML config; panes and robot pre-allocated as mocap bodies**
- Why: wall type per side, box count and pane count are loops over config
  values, which is awkward in hand-written XML. Pre-allocating the maximum
  number of panes and moving them through `data.mocap_pos` lets each episode be
  randomised without recompiling the model. Unconfirmed dimensions live in one
  config file.
- Alternatives: hand-written MJCF file (simple, but every layout change is an
  edit); XML string templating (works, harder to read and validate);
  recompiling a new model per episode (simplest randomisation, slower resets).
- Status: proposed.

**D-014 · 2026-11-13 · Camera pipeline: 2304×1296 sensor mode → central 4:3 crop (1728×1296) → 256×192 → keep the bottom half (256×96)**
- Why 4:3 (team decision, 2026-11-13): it lightens the encoder for Pi 5
  inference and for training. Because the crop keeps the full sensor height,
  the vertical FOV stays 41° and the same pixels cover a narrower 53°
  horizontal view, so each obstacle gets as many pixels as before from a
  smaller image. Measured in the sim (human picture estimated 12 × 4 cm):

  | Render | CNN input | Pixels | Human at 1 m | Human at 2 m | Car at 2 m |
  |---|---|---|---|---|---|
  | 384×216 (16:9) | 384×108 | 41,472 | 13 × 34 | 6 × 17 | 23 × 16 |
  | 320×240 (4:3) | 320×120 | 38,400 | 14 × 38 | 7 × 20 | 26 × 18 |
  | 256×192 (4:3) | 256×96 | 24,576 | 11 × 31 | 6 × 16 | 21 × 15 |

  256×192 matches the 16:9 384×216 at 2 m with 41% fewer pixels.
  Note (2026-11-13): this table assumed a 12 × 4 cm human picture. The sheet
  heights given later (Q19) suggest the drawing is nearer 8 × 3 cm, which
  scales every human figure above by about two-thirds. Revisit once the
  pictures are measured; 320×240 is the first step up if needed.
- Cost: the horizontal view narrows from 66° to 53°. At 1 m ahead the camera
  sees a band about 1.0 m wide instead of 1.3 m, so obstacles beside the
  robot leave the view sooner while it turns past them.
- Why 2304×1296: it is the binned mode that keeps the full field of view;
  smaller sensor modes crop the sensor further, which would not match the sim
  camera. The 4:3 crop must be set explicitly on the robot (Picamera2
  ScalerCrop, central region of the sensor) and checked against a sim render. The camera is level at 0.145 m (first given as 0.16 m; the pixel figures above
  used 0.16 m and change only slightly) and the
  mount cannot be pitched, so the floor and any object shorter than 0.145 m
  project below the image centre; the top half holds mostly box tops and
  window, the main glare source.
- Alternatives: 16:9 at 384×216 (wider view, 1.7× the pixels; the earlier
  choice); 4:3 at 320×240 (slightly larger obstacles, 1.6× the pixels);
  16:9 at 256×144 (human only 3 px wide at 1.85 m); pitch the camera down
  (mount does not allow it); full frame instead of the bottom half (twice the
  pixels, mostly window).
- Status: accepted (4:3). Exact size depends on the real pane sizes (Q19) and
  on encoder speed on the Pi 5, which is unmeasured; exact rows on Q11.
- Checked 2026-11-13 against the turning radius (about 0.5 m, D-017): the
  robot only drives forward and turns gently, so what it can reach in the
  next 0.5–1 m lies mostly inside the 53° view; the 13° lost from 66° matters
  little. Neither view sees beside the robot, so a pane being passed leaves
  the view in both cases; the ultrasonic, or a short memory of past frames,
  covers that. The pixel saving of 4:3 over 16:9 at the same detail per
  degree is only about 15%; the real gain is detail on the human picture.

**D-015 · 2026-11-12 · Pane model: white paper sheet + picture layer on one body, randomised margins and lean; true curvature deferred**
- What: each pane is a white box geom (the sheet) with a thin textured box
  geom (the picture) on each face. The back copy is turned 180° about the
  vertical axis so it reads un-mirrored, as on the real prints. Per episode the sheet size, the
  picture's size and position on the sheet, and the pane's pitch, roll and yaw
  are randomised by editing model and mocap arrays, with no recompile. The
  body origin is on the sheet's bottom edge, so pitch leans it about the floor.
- Why: the real prints have a wide white margin of unknown and varying size,
  so the margin cannot be baked into the texture. Two geoms also give separate
  segmentation IDs for "paper" and for the class picture, which makes the
  labels cleaner. Bending is approximated by leaning the flat sheet: it
  reproduces most of the foreshortening, and MuJoCo primitives cannot curve.
- Alternatives: margin baked into the texture (fixed margin, or one texture
  per variant); curved mesh sheets with several fixed curvatures (closest to
  reality, more work, curvature fixed at compile time); recomposing and
  re-uploading the texture each episode (flexible, untested on 4.0.1).
- Status: proposed. Revisit curvature if the real perception test set shows a
  gap on bent panes. Ranges in `configs/arena.yaml` are placeholders (Q19).
- Blind case: a pane seen edge-on shows no picture from either side, so the
  class cannot be read at that angle.

**D-016 · 2026-11-12 · Baseline: any collision ends the episode. Stretch: pane collisions penalised by class without ending it**
- Baseline: every collision is penalised and ends the episode. It is the
  simplest rule to train and gives an unambiguous collision rate. Termination
  is a switch in the reward/done config, so the stretch variant is a config
  change, not a rewrite.
- Stretch variant, what: a pane collision costs a class-weighted penalty (human far above car
  and building) and the episode continues. The penalty for a car or building
  must exceed the cost of the longest detour, so driving over one only pays
  when no free path exists.
- Why: the obstacles are paper, and layouts can block every free path. A
  policy that can only stop or fail there is less useful than one that picks
  the least harmful option, and choosing a car or building over a human is
  direct evidence of class-dependent behaviour.
- Alternatives within the stretch: non-terminal for car and building,
  terminal for human (the safer variant, Q23).
- Why it is not the baseline: it adds four pieces of work on top of the
  riskiest part of the project (RL convergence by 2 Dec). Training must include blocked layouts, or the behaviour is
  never learned; evaluation must report avoidable and forced collisions
  separately; the sim needs a rule for what a hit pane does (proposed: it is
  counted once and then lies flat); the Pico e-stop in §7.1 would stop the
  robot before any pane, so it conflicts with this as currently designed.
- Status: baseline proposed; stretch attempted only if the baseline has
  converged before the 2 Dec decision point. Open points in Q23.

**D-017 · 2026-11-13 · Robot model: car-like kinematics on a mocap body, padded collision box**
- What: the sim robot is a mocap body moved each step by a kinematic bicycle
  model (speed and steering angle in; pose out, turning about the rear axle,
  wheelbase 0.15 m). Collisions are checked against a 0.30 × 0.20 m box, the
  measured 0.28 × 0.17 m body plus about 1–1.5 cm on each side. The camera is
  mounted 0.13 m behind the bumper at 0.145 m.
- Why: the robot steers like a car and cannot turn on the spot, so the action
  space is (speed, steering) and the turning radius limits which layouts are
  passable. A kinematic model is fast, needs no wheel or friction tuning, and
  keeps episodes on mocap with no recompile (D-013). The padding absorbs the
  ±1 cm measurement error and gives a small safety margin.
- Alternatives: simulated wheels with actuators and friction (closer to the
  real dynamics, much more tuning and slower); differential-drive kinematics
  (wrong for this robot).
- Inputs so far: front axle 0.08 m behind the bumper, rear axle 0.23 m;
  minimum turning radius about 0.50 m, so about 16.7° maximum steering.
  Wheel positions beyond the wheelbase are not needed: the bicycle model
  merges the two front wheels into one. Still needed: a calibration of servo
  command to turning radius at a few settings each side (Q2). Speed and steering noise are added later as domain
  randomisation, tuned against logged runs on the real robot.
- Status: proposed.

**D-018 · 2026-11-13 · Design B uses a YOLO detector, developed in parallel; one shared sample generator**
- What: Design B's perception is an Ultralytics YOLO detector (boxes for
  human, car, building) instead of the shared encoder's heads. A teammate
  develops it alongside Design A. Rendered samples and their labels come
  from one generator on main that writes a neutral per-image record (class,
  box, distance, bearing per pane, from the segmentation render and the
  scene geometry). Small exporters turn it into the YOLO format and into the
  encoder's auxiliary-head labels.
- Why: running both designs at once removes the wait for the week-3 switch.
  One generator means both designs train on the same scenes and the same
  label rules, so the comparison in the report is fair, and arena fixes reach
  both.
- Repo layout: shared code (arena, generator, reward, evaluation) on main;
  YOLO-specific training code in `perception/yolo/`, also on main. Branches
  are kept short-lived and merged often, rather than one long-lived YOLO
  branch that drifts from arena changes.
- Alternatives: B as originally planned (the encoder's own heads, switched to
  only at week 3); a separate generator per design (duplicated work and
  labels that can disagree).
- Open: YOLO model size and frames per second on the Pi 5; whether the box is
  the whole sheet or only the picture (proposed: whole sheet, since that is
  what the robot must avoid).
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
  dataset/                shared sample generator + exporters (YOLO format, head labels) (D-018)
  yolo/                   Design B detector training and evaluation (D-018)
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
