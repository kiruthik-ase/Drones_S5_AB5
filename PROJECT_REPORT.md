# SkillTrace-Style Final Report — UAV B-Spline Trajectory Planning with MILP Obstacle Avoidance and Geometric Control

**22AIE448 — Data Driven Control of Drones · Semester 5 · Branch: `main`**  
**Team AB5**

> CB.SC.U4AIE24023 — KIRUTHIKPRANAV  
> CB.SC.U4AIE24040 — PULIPATI CHAITHANYA  
> CB.SC.U4AIE24060 — VIKRAMENDRAA SHANMUGAVELU THIYAGARAJAN  
> CB.SC.U4AIE24141 — NEERAJ T  
> CB.SC.U4AIE24153 — THAMMINI VARUN

> This document explains the whole project from the problem to the code: why it exists, how it was approached, **every formula used and the exact file and function it lives in**, how each part works, how the pieces connect end-to-end, and what the outputs mean.  
> Based on: *A. R. Babaei and M. Karimi, "Optimal Trajectory-Planning of UAVs via B-Splines and Disjunctive Programming," 2018*  
> Controller based on: *T. Lee, M. Leok, N. H. McClamroch, "Geometric Tracking Control of a Quadrotor UAV on SE(3)," CDC 2010.*

---

## Table of Contents

1. Problem and Motivation
2. Approach and Design Philosophy
3. Technology Stack
4. System Architecture
5. File Structure and Responsibilities
6. Feature Catalogue
7. The B-Spline Engine In Depth
8. The MILP Planner In Depth
9. The Geometric Controller In Depth
10. The Quadrotor Physics In Depth
11. The Closed-Loop RHC Engine In Depth
12. **Formula Register: Every Formula and Where It Lives**
13. The Scenario: Urban Canyon Map
14. End-to-End Flows
15. Outputs and Visualisation
16. Paper vs. Extensions
17. Key Parameters
18. Verification and Results
19. Known Limitations
20. How to Run
21. Examiner Q&A
22. Glossary

---

## 1. Problem and Motivation

A UAV (drone) flying through a real urban environment faces problems that a simple straight-line planner cannot solve:

- **Buildings block direct paths.** A tall skyscraper cannot be flown through; the drone must go around it laterally. A wide low hurdle is cheaper to fly over than to go around. The planner must decide which.
- **No-Fly Zones (NFZs) are legally restricted.** Certain airspace regions are forbidden regardless of whether they are physically passable.
- **Wind pushes the drone off course.** A plan made at the start is already wrong 2 seconds later if wind is blowing.
- **Dynamic obstacles appear mid-flight.** An intruder UAV crossing the path cannot be predicted far in advance.
- **Smooth trajectories are physically necessary.** A quadrotor cannot make instantaneous velocity changes. The planned path must be at least C² continuous (position, velocity, acceleration are all continuous) so the controller can track it without commanding infinite forces.

**Goal:** Implement and extend Babaei & Karimi (2018) — which solves obstacle-avoidance trajectory planning using B-splines and MILP — and close the loop with full 6-DOF quadrotor physics, a geometric controller, and Dryden wind turbulence, demonstrating that a drone can navigate ~6 km through a dense urban canyon reliably and robustly.

---

## 2. Approach and Design Philosophy

1. **Math first, then physics, then visualisation.** B-spline and MILP were built and verified before any physics. Controller added after. Visualiser is always the last layer.
2. **Pure functions for the hard maths.** `bspline_utils.py` has zero framework imports. `controller.py` has zero MILP imports. Each concern is isolated.
3. **Source of truth in NumPy arrays.** Simulation logs are saved to `simulation_cache.npz` so the visualiser relaunches without re-running the expensive MILP.
4. **Derive, don't store, what can drift.** Actual drone position is always read from the physics state at each RHC step, never from the old plan.
5. **True closed-loop, not open-loop pre-computation.** The planner is called at every 2-second horizon with the *real* drone position, not the position the plan predicted. This is the key engineering difference from the base paper.
6. **Graceful failure.** If the MILP solver finds no feasible solution, the drone glides with controlled deceleration and retries the next iteration rather than crashing.
7. **Phased build.** Three phases: kinematic planner first, then obstacles, then full 6-DOF physics + controller.

---

## 3. Technology Stack

| Layer | Technology | Role |
|---|---|---|
| Language | Python 3.10+ | All code |
| Optimisation | CVXPY >= 1.3 | Models and solves the MILP symbolically |
| MILP solver backend | SciPy (via `cp.SCIPY`) | Free open-source MILP solver, no licence needed |
| Numerical computation | NumPy >= 1.24 | Matrices, RK4, all math |
| Static visualisation | Matplotlib >= 3.7 | 6-panel publication figure, live GUI |
| Live GUI backend | Matplotlib TkAgg | Animated 3D window |
| Interactive 3D viewer | Plotly >= 5.14 | Browser-based rotatable HTML |
| Cache | `.npz` compressed NumPy archive | Fast visualiser re-launch |
| Animation export | Matplotlib FuncAnimation + ffmpeg | `flight.mp4` / `flight.gif` |

**No B-spline or trajectory library is used.** The Cox-de Boor recursion, basis matrices, and trajectory sampler are implemented from scratch from the original equations. The wind model is implemented from scratch (a dozen lines using NumPy's RNG). `pyproject.toml` has no HMM/spline/control package.

---

## 4. System Architecture

### 4.1 The Eight Modules

```
Drones_S5_AB5/
├── bspline_utils.py       # Cox-de Boor recursion, basis matrices, trajectory sampler
├── planner_obstacles.py   # UAVPlannerMILP: MILP formulation and solve (core of the paper)
├── controller.py          # GeometricController: SE(3) tracking law (Lee 2010)
├── quadrotor.py           # Quadrotor: 6-DOF RK4 dynamics + WindModel: Dryden turbulence
├── dynamic_obstacle.py    # DynamicObstacle: constant-velocity intruder UAV
├── scenario.py            # build_scenario(): map definition (buildings, NFZs, start, goal)
├── core.py                # run_closed_loop_rhc(): the TRUE closed-loop engine
├── simulation.py          # Entry point: run → save cache + PNG + HTML
└── visualizer.py          # Live 3D animated GUI with HUD and telemetry
```

### 4.2 Data Flow at Each 2-Second RHC Horizon

```
┌───────────────────────────────────────────────────────────────┐
│                  RECEDING HORIZON CONTROL LOOP                 │
│                                                                │
│  ┌──────────────┐    ┌───────────────┐    ┌─────────────────┐ │
│  │ UAVPlannerMILP│──▶│  B-Spline     │──▶│  Quadrotor      │ │
│  │  (10s horizon)│    │  Reference    │    │  6-DOF + Wind   │ │
│  │  MILP solver │    │  p_d, v_d, a_d│    │  RK4 @ 500 Hz  │ │
│  └──────────────┘    └───────────────┘    └────────┬────────┘ │
│          ▲                                          │          │
│          │ pos_actual, vel_actual                   │          │
│          └──────── GeometricController ─────────────┘          │
│                    (thrust, torques)                           │
└───────────────────────────────────────────────────────────────┘
```

### 4.3 Runtime Data Paths

```
scenario.py ──► planner_obstacles.py ──► core.py ──► quadrotor.py
                      ▲                     │               │
                      │                     │          controller.py
                 bspline_utils.py           │
                                            ▼
simulation.py ──► run_closed_loop_rhc() ──► [ref_t, ref_pos, log_t, log_pos, ...]
                       │
                       ├──► simulation_cache.npz
                       ├──► simulation_results.png
                       └──► trajectory_3d.html

visualizer.py ──► loads simulation_cache.npz ──► FuncAnimation ──► live GUI
```

---

## 5. File Structure and Responsibilities

| File | Lines | Primary Responsibility |
|---|---|---|
| `bspline_utils.py` | 97 | Pure B-spline maths — Cox-de Boor, basis matrices B/B'/B''/B''', trajectory sampler |
| `planner_obstacles.py` | 184 | `UAVPlannerMILP` class — MILP formulation, obstacle constraints, objective, solve |
| `controller.py` | 101 | `GeometricController` — SE(3) position + attitude tracking law, Lee 2010 |
| `quadrotor.py` | 170 | `Quadrotor` (6-DOF RK4), `WindModel` (Dryden Markov chain) |
| `dynamic_obstacle.py` | 42 | `DynamicObstacle` — constant-velocity intruder, NFZ projection |
| `scenario.py` | 29 | `build_scenario()` — registers map; `create_dynamic_obstacle()` |
| `core.py` | 266 | `run_closed_loop_rhc()` — interleaved plan+simulate loop; legacy open-loop helpers |
| `simulation.py` | 393 | Entry point, 6-panel figure, interactive HTML, `main()` |
| `visualizer.py` | 506 | Live GUI: dark theme, 3D animation, 2D map, HUD, telemetry, `--save`/`--fresh` modes |
| `requirements.txt` | 20 | Python dependencies (numpy, cvxpy, scipy, matplotlib, plotly) |
| `simulation_cache.npz` | ~13 MB | Compressed simulation log for instant visualiser re-launch |
| `simulation_results.png` | ~720 KB | 6-panel publication-quality figure |
| `trajectory_3d.html` | ~11 MB | Interactive Plotly 3D viewer |
| `flight.mp4` | ~3.4 MB | Rendered flight animation |

---

## 6. Feature Catalogue

### 6.1 B-Spline Trajectory Planning (from Paper)
- Degree-4 (quartic) B-spline giving C³ continuous trajectories (position, velocity, acceleration, jerk are all continuous)
- Cox-de Boor recursion for basis functions and their 1st, 2nd, 3rd derivatives
- Pre-computed basis matrices B, B', B'', B''' for the entire horizon (11 rows × 14 columns)
- Trajectory sampler that evaluates (position, velocity, acceleration) at any physical time t ∈ [0, T_h]

### 6.2 MILP Obstacle Avoidance (from Paper)
- **3D cuboid obstacles** with Big-M disjunctive constraints: 6 binary variables per obstacle per sample point
- **2D No-Fly Zones** (infinite height) with 4 binary variables per NFZ per sample point
- Altitude ceiling (z ≤ 25 m) forces lateral avoidance of tall buildings
- Altitude floor (z ≥ 1 m) prevents ground collision
- Per-axis velocity bounds: |v| ≤ v_max = 60 m/s
- Per-axis acceleration bounds: |a| ≤ a_max = 14.715 m/s²
- C² continuity at each horizon boundary: initial position, velocity, acceleration pinned

### 6.3 Multi-Objective Optimisation (from Paper)
- **J1** — L1 path-length minimisation + terminal distance to goal
- **J2** — Altitude penalty (prefer flying low)
- **J3** — Jerk smoothness penalty (penalise jerk variation between time steps)

### 6.4 Receding Horizon Control
- 10-second planning horizon, 2-second execution window
- At every horizon: plan from ACTUAL (wind-disturbed) drone position, not the predicted one
- Infeasibility fallback: controlled glide (vel × 0.9 per step), then retry
- Dynamic obstacle injection: intruder's predicted position added as temporary NFZ per horizon

### 6.5 6-DOF Quadrotor Physics
- 12-state rigid-body model: position (3), velocity (3), Euler angles ZYX (3), angular rates (3)
- Full rotational dynamics: Euler's equation I·ω̇ = τ − ω × (I·ω)
- Euler-angle kinematics with gimbal-lock guard at |cos(θ)| < 1e-8
- RK4 integration at 500 Hz (dt = 0.002 s)
- Wind disturbance injected as acceleration into translational dynamics at every step

### 6.6 Dryden Wind Turbulence Model
- First-order Markov: w[k] = ρ·w[k-1] + σ·√(1-ρ²)·N(0,I)
- ρ = 0.995 → slow, persistent gusts; σ = [1.8, 1.8, 0.4] m/s²; seed = 42 (reproducible)

### 6.7 Geometric Controller on SE(3) (Lee 2010)
- No Euler-angle singularities — operates directly on SO(3)
- Position loop (ωₙ ≈ 4 rad/s) + attitude loop (ωₙ ≈ 30 rad/s)
- Minimum thrust floor: 10% of hover (never free-falls)

### 6.8 Outputs
- `simulation_cache.npz` — all logs compressed
- `simulation_results.png` — 6-panel dark-theme figure
- `trajectory_3d.html` — standalone interactive Plotly 3D viewer
- `flight.mp4` / `flight.gif` — animated export
- Live GUI — 3D animation + 2D map + HUD + telemetry strips

---

## 7. The B-Spline Engine In Depth

### 7.1 Why B-Splines?

B-splines are ideal for UAV trajectory planning because:
- Degree-4 gives C³ continuity — the controller can track it without demanding infinite forces
- Velocity and acceleration are linear in the control points, keeping MILP constraints linear
- The control points are the MILP decision variables — the planner directly optimises trajectory shape

### 7.2 Parameters

| Symbol | Value | Meaning |
|---|---|---|
| k | 4 | Degree (quartic B-spline) |
| h | 10 | Number of time steps per horizon |
| ts | 1.0 s | Duration per time step |
| T_h | 10.0 s | Total horizon (h × ts) |
| n_ctrl | 14 | Control points (k + h) |

### 7.3 Knot Vector (`bspline_utils.py → generate_knot_vector`)

Clamped uniform knot vector for degree k=4, h=10:
```
t = [0, 0, 0, 0,  t₁, ..., t₉,  1, 1, 1, 1]
```
First and last k=4 knots are repeated (forces curve to interpolate first and last control points). Internal knots are uniformly spaced in (0, 1).

### 7.4 Cox-de Boor Recursion (`bspline_utils.py → cox_de_boor`)

**Base case (k=1):**
```
N_{i,1}(t) = 1   if t_i ≤ t < t_{i+1}  (with endpoint correction)
           = 0   otherwise
```

**Recursive case:**
```
N_{i,k}(t) = [(t - t_i) / (t_{i+k-1} - t_i)] · N_{i,k-1}(t)
           + [(t_{i+k} - t) / (t_{i+k} - t_{i+1})] · N_{i+1,k-1}(t)
```
Division by zero → 0 (convention). Each term is computed only if the denominator > 0.

### 7.5 Derivatives (`bspline_utils.py → cox_de_boor_derivative`)

d-th derivative, computed recursively. For d ≥ 1:
```
N^(d)_{i,k}(t) = [(k-1) / (t_{i+k-1} - t_i)] · N^(d-1)_{i,k-1}(t)
               - [(k-1) / (t_{i+k} - t_{i+1})] · N^(d-1)_{i+1,k-1}(t)
```
d=0 returns `cox_de_boor(t, i, k, knots)`.

### 7.6 Basis Matrices (`bspline_utils.py → get_bspline_matrices`)

Pre-computed at h+1 = 11 uniformly spaced time samples for MILP use:
```
B      (11 × 14)   position basis
B_dot  (11 × 14)   1st-derivative basis
B_ddot (11 × 14)   2nd-derivative basis
B_dddot(11 × 14)   3rd-derivative basis
```

In the MILP (matrix form):
```
X    = B       @ Q          (positions, m)
V    = B_dot   @ Q / T_h    (velocities, m/s)
A    = B_ddot  @ Q / T_h²   (accelerations, m/s²)
Jerk = B_dddot @ Q / T_h³   (jerk, m/s³)
```
Q ∈ ℝ^{14×3} is the matrix of all control points — the MILP decision variable.

### 7.7 Trajectory Sampler (`bspline_utils.py → sample_trajectory`)

Given physical time t_phy ∈ [0, T_h], returns (pos, vel, acc):
1. Normalise: t = t_phy / T_h, clamped to [0, 1]
2. For each control point i: accumulate N_{i,k}(t)·Q[i], N'·Q[i], N''·Q[i]
3. Scale: vel /= T_h, acc /= T_h²
4. Return (p, v, a) as float64 numpy arrays

Called 1000 times per 2-second horizon (once per physics step).

---

## 8. The MILP Planner In Depth

**File:** `planner_obstacles.py` — class `UAVPlannerMILP`

### 8.1 Decision Variables

```python
Q      = cp.Variable((14, 3))   # 14 control points × 3 axes — trajectory shape
dX     = cp.Variable((10, 3))   # L1 auxiliaries for path length
d_term = cp.Variable(3)         # L1 auxiliaries for terminal error
dJerk  = cp.Variable((10, 3))   # L1 auxiliaries for jerk penalty
# Per obstacle per step:
b      = cp.Variable(6, boolean=True)   # 3D building (6 faces)
b      = cp.Variable(4, boolean=True)   # 2D NFZ (4 faces)
```

Current scenario (2 buildings + 4 NFZs, 10 steps): **(2×6 + 4×4) × 10 = 280 binary variables** per horizon.

### 8.2 Derived Trajectory Variables (CVXPY expressions)

```python
X    = B       @ Q
V    = B_dot   @ Q / T_h
A    = B_ddot  @ Q / (T_h ** 2)
Jerk = B_dddot @ Q / (T_h ** 3)
```

### 8.3 Initial Conditions — C² Continuity

```python
X[0, :] == pos_actual    # position pinned to actual drone state
V[0, :] == vel_actual    # velocity pinned — no jump between horizons
A[0, :] == acc_plan      # acceleration from previous horizon's endpoint
```

### 8.4 Dynamic Constraints

```python
# Velocity (per axis, i = 1 to h)
V[i, :] <= v_max    and    V[i, :] >= -v_max     (v_max = 60.0 m/s)
# Acceleration (per axis, i = 1 to h)
A[i, :] <= a_max    and    A[i, :] >= -a_max     (a_max = 14.715 m/s²)
```

### 8.5 Altitude Constraints

```python
X[:, 2] >= 1.0        # floor: always at least 1 m
X[:, 2] <= z_max      # ceiling: z_max = 25 m  (forces lateral avoidance of tall buildings)
```

The ceiling is the key insight: 200 m buildings are far above z_max=25 m, so the drone cannot fly over them — it **must** go around laterally. Short hurdles (15 m < 25 m) can be flown over — the MILP chooses to fly over them because it is cheaper (no large X/Y detour).

### 8.6 Big-M Disjunctive Constraints — 3D Building

For building [xl,xu] × [yl,yu] × [zl,zu] with safety margin η, at each sample i (1 to h), introduce 6 binary vars with sum = 1:

```
X[i,0] - (xl - η) ≤ (1 - b[0]) · M      →  drone left of building  if b[0]=1
(xu + η) - X[i,0] ≤ (1 - b[1]) · M      →  drone right of building if b[1]=1
X[i,1] - (yl - η) ≤ (1 - b[2]) · M      →  drone below building    if b[2]=1
(yu + η) - X[i,1] ≤ (1 - b[3]) · M      →  drone above building    if b[3]=1
X[i,2] - (zl - η) ≤ (1 - b[4]) · M      →  drone under building    if b[4]=1
(zu + η) - X[i,2] ≤ (1 - b[5]) · M      →  drone over building     if b[5]=1
```

b[j]=1 → constraint tight (active, drone must be on that safe side).  
b[j]=0 → (1-0)·M = 10000 → constraint trivially satisfied (relaxed).  
**Exactly one b[j]=1** forces the drone to commit to one escape direction.

### 8.7 Big-M Disjunctive Constraints — 2D NFZ

For NFZ [xl,xu] × [yl,yu] with safety margin γ, 4 binary vars with sum = 1:
```
X[i,0] - (xl - γ) ≤ (1 - b[0]) · M
(xu + γ) - X[i,0] ≤ (1 - b[1]) · M
X[i,1] - (yl - γ) ≤ (1 - b[2]) · M
(yu + γ) - X[i,1] ≤ (1 - b[3]) · M
```
No Z constraints — NFZs extend infinitely in altitude.

### 8.8 Objective Function

Linearised multi-term L1 minimisation (required for MILP — no quadratic terms):

```
Minimise:  J1 + J2 + J3

J1 = w1 · Σ sum(dX[i])  +  w2 · sum(d_term)
     dX[i,:]  >= ±(X[i+1,:] - X[i,:])     (L1 path length)
     d_term   >= ±(X[-1,:] - x_f)          (L1 terminal error)

J2 = w3 · Σ X[i, 2]                         (altitude penalty)

J3 = w4 · Σ sum(dJerk[i]) / ts
     dJerk[i,:] >= ±(Jerk[i+1,:] - Jerk[i,:])  (L1 jerk variation)

Weights: w1=1.0, w2=10.0, w3=0.3, w4=0.1
```

### 8.9 Solver

```python
prob.solve(solver=cp.SCIPY)
```
SciPy's MILP branch-and-bound. Returns None × 4 if status not "optimal" or "optimal_inaccurate".

---

## 9. The Geometric Controller In Depth

**File:** `controller.py` — class `GeometricController`

Unlike PID controllers using Euler angles (which suffer gimbal lock at pitch = ±90°), this controller operates on SO(3) — the mathematical group of rotation matrices — with no singularities.

### 9.1 Inputs / Outputs

- **In:** state (12,), p_d (3,), v_d (3,), a_d (3,), yaw_d (float)
- **Out:** u = [thrust T (N), τx, τy, τz (N·m)]

### 9.2 Step 1 — Desired Force (`controller.py → compute`)

```
e_p = pos - p_d                         (position error)
e_v = vel - v_d                         (velocity error)
F_des = -Kp·e_p - Kv·e_v + m·g·e₃ + m·a_d
```
- Kp = diag(16, 16, 20) → ωₙ = √16 ≈ 4 rad/s (critically damped)
- Kv = diag(8, 8, 9) → Kd = 2·1·4 = 8 (ζ = 1)
- m·g·e₃ cancels gravity; m·a_d is feed-forward from the spline

### 9.3 Step 2 — Total Thrust

```
b3    = R[:, 2]                         (current body-z = 3rd column of R)
T     = F_des · b3                      (projection onto body-z)
T     = max(0.1·m·g, T)               (minimum thrust floor)
```

### 9.4 Step 3 — Desired Rotation Matrix

```
b3_des = F_des / ||F_des||

b1_c   = [cos(yaw_d), sin(yaw_d), 0]   (desired heading)
b2_des = (b3_des × b1_c) / ||b3_des × b1_c||
b1_des = b2_des × b3_des
R_des  = [b1_des | b2_des | b3_des]    (3×3 orthonormal matrix)
```

### 9.5 Step 4 — Attitude Error on SO(3)

```
e_R_mat = 0.5 · (R_des^T · R - R^T · R_des)   (skew-symmetric)
e_R     = vee(e_R_mat)                          (→ 3-vector)

vee(S) = [S[2,1], S[0,2], S[1,0]]             (axial vector extraction)
```

### 9.6 Step 5 — Control Torques

```
τ = -Kr·e_R - Kω·e_ω + ω × (I·ω)
```
- Kr = diag(120, 120, 60) → ωₙ ≈ 30 rad/s (attitude loop 7.5× faster than position)
- Kω = diag(16, 16, 8) → attitude damping
- ω × (I·ω) → gyroscopic compensation

---

## 10. The Quadrotor Physics In Depth

**File:** `quadrotor.py` — classes `Quadrotor` and `WindModel`

### 10.1 State Vector (12D)

```
s = [x, y, z,  vx, vy, vz,  φ, θ, ψ,  p, q, r]
     position   velocity    Euler(ZYX)  body rates
```

### 10.2 Physical Parameters

| Parameter | Value |
|---|---|
| mass m | 1.0 kg |
| Ixx = Iyy | 0.0196 kg·m² |
| Izz | 0.0264 kg·m² |
| g | 9.81 m/s² |

### 10.3 Rotation Matrix ZYX (`quadrotor.py → rotation_matrix_euler`)

```
R = Rz(ψ) · Ry(θ) · Rx(φ)

    ┌ cψcθ    cψsθsφ−sψcφ    cψsθcφ+sψsφ ┐
R = │ sψcθ    sψsθsφ+cψcφ    sψsθcφ−cψsφ │
    └ −sθ     cθsφ           cθcφ         ┘
```

### 10.4 Equations of Motion (`quadrotor.py → _derivatives`)

**Translational:**
```
ṗ = v
v̇ = R·[0, 0, T/m]^T − g·e₃ + w_acc
```

**Euler-angle kinematics (ZYX):**
```
┌φ̇┐   ┌1   sin(φ)·tan(θ)   cos(φ)·tan(θ)┐ ┌p┐
│θ̇│ = │0   cos(φ)          −sin(φ)       │·│q│
└ψ̇┘   └0   sin(φ)/cos(θ)   cos(φ)/cos(θ)┘ └r┘
```
Guard: if |cos(θ)| < 1e-8, clamp to ±1e-8 (avoids gimbal-lock blow-up).

**Rotational:**
```
ω̇ = I⁻¹ · (τ − ω × (I·ω))
```

### 10.5 RK4 Integration (`quadrotor.py → step`)

```
k1 = f(s,            u, w_acc)
k2 = f(s + 0.5·dt·k1, u, w_acc)
k3 = f(s + 0.5·dt·k2, u, w_acc)
k4 = f(s + dt·k3,     u, w_acc)
s += (dt/6)·(k1 + 2k2 + 2k3 + k4)
```
Post-step: Euler angles wrapped to [−π, π].

### 10.6 Wind Model (`quadrotor.py → WindModel → step`)

```
w[k] = ρ·w[k-1] + σ ⊙ √(1−ρ²)·N(0, I)
```
- ρ = 0.995 → autocorrelation; gusts evolve slowly (bandwidth ≈ 0.5 Hz)
- σ = [1.8, 1.8, 0.4] m/s² per axis (horizontal >> vertical)
- seeded RNG (seed=42) → exact reproducibility

---

## 11. The Closed-Loop RHC Engine In Depth

**File:** `core.py` — function `run_closed_loop_rhc`

### 11.1 Initialisation

```python
quad = Quadrotor(mass=1.0)
ctrl = GeometricController(mass=1.0)
wind = WindModel(rho=0.995, sigma=[1.8, 1.8, 0.4], seed=42)
quad.set_state(pos=x_s, vel=v_s)
n_steps = int(execute_time / dt_sim)   # = 2.0 / 0.002 = 1000
```

### 11.2 RHC Iteration (per 2-second horizon)

```
Step A:  Inject dynamic obstacle as temporary NFZ for this horizon
Step B:  planner.plan(pos_actual, vel_actual, acc_plan, x_f)
         → returns Q, X, V, A  (or None if infeasible)
Step C:  If infeasible with dyn obstacle → retry without it
Step D:  If still infeasible → glide: vel *= 0.9, pos += vel·dt, continue
Step E:  For each of 1000 physics steps:
           t_local = (j+1) · dt_sim
           p_d, v_d, a_d = sample_trajectory(t_local, Q, k, h, t_h)
           w_acc = wind.step()
           u     = ctrl.compute(quad.state, p_d, v_d, a_d)
           quad.step(u, dt_sim, wind_acc=w_acc)
           → log everything
Step F:  pos_actual = quad.pos   (actual wind-disturbed position)
         vel_actual = quad.vel   (actual wind-disturbed velocity)
         _, _, acc_plan = sample_trajectory(execute_time, Q, k, h, t_h)
Step G:  if ||pos_actual − x_f|| < 50 m → terminate
```

### 11.3 Why This Is Closed-Loop (vs. Open-Loop)

| | Open-Loop | This Project (Closed-Loop) |
|---|---|---|
| Planner input | Predicted state from previous plan | **Actual** wind-disturbed state from physics |
| Wind error | Accumulates unbounded | **Corrected** at every 2-second replanning |
| Infeasibility | Crashes | Graceful glide + retry |

### 11.4 Return Arrays

| Array | Shape | Content |
|---|---|---|
| `ref_t` | (M,) | Reference timestamps |
| `ref_pos` | (M, 3) | Planned B-spline positions |
| `log_t` | (N,) | Actual simulation timestamps |
| `log_pos` | (N, 3) | Actual positions (with wind) |
| `log_vel` | (N, 3) | Actual velocities |
| `log_eul` | (N, 3) | Euler angles (roll, pitch, yaw) |
| `log_u` | (N, 4) | Control: [thrust, τx, τy, τz] |
| `log_wind` | (N, 3) | Wind acceleration at each step |
| `plan_log` | list | Per-horizon: iteration, time, dist-to-goal, solve time |

---

## 12. Formula Register: Every Formula and Where It Lives

| # | Formula / Rule | File → Function |
|---|---|---|
| 1 | `N_{i,1}(t) = 1 if t_i ≤ t < t_{i+1}, else 0` | `bspline_utils.py` → `cox_de_boor` |
| 2 | Cox-de Boor: `N_{i,k}(t) = [(t−t_i)/(t_{i+k-1}−t_i)]·N_{i,k-1} + [(t_{i+k}−t)/(t_{i+k}−t_{i+1})]·N_{i+1,k-1}` | `bspline_utils.py` → `cox_de_boor` |
| 3 | Derivative: `N^(d)_{i,k} = (k-1)/(t_{i+k-1}−t_i)·N^(d-1)_{i,k-1} − (k-1)/(t_{i+k}−t_{i+1})·N^(d-1)_{i+1,k-1}` | `bspline_utils.py` → `cox_de_boor_derivative` |
| 4 | Clamped uniform knot vector: k repeated 0s, uniform interior, k repeated 1s | `bspline_utils.py` → `generate_knot_vector` |
| 5 | `X = B@Q`, `V = B'@Q/T_h`, `A = B''@Q/T_h²`, `J = B'''@Q/T_h³` | `bspline_utils.py` → `get_bspline_matrices`; `planner_obstacles.py` → `plan` |
| 6 | Trajectory sampler: accumulate N·Q[i], N'·Q[i]/T_h, N''·Q[i]/T_h² at normalised t | `bspline_utils.py` → `sample_trajectory` |
| 7 | C² initial conditions: `X[0]=pos_actual`, `V[0]=vel_actual`, `A[0]=acc_plan` | `planner_obstacles.py` → `plan` |
| 8 | Per-axis velocity bounds: `-v_max ≤ V[i,:] ≤ v_max` | `planner_obstacles.py` → `plan` |
| 9 | Per-axis acceleration bounds: `-a_max ≤ A[i,:] ≤ a_max` | `planner_obstacles.py` → `plan` |
| 10 | Altitude floor: `X[:,2] ≥ 1.0` | `planner_obstacles.py` → `plan` |
| 11 | Altitude ceiling: `X[:,2] ≤ z_max` | `planner_obstacles.py` → `plan` |
| 12 | Big-M 3D obstacle (6 binary, sum=1): `X[i,axis]−(bound±η) ≤ (1−b[j])·M` per face | `planner_obstacles.py` → `plan` (loop `self.obstacles`) |
| 13 | Big-M 2D NFZ (4 binary, sum=1): same but 4 XY faces only | `planner_obstacles.py` → `plan` (loop `self.no_fly_zones`) |
| 14 | L1 path length linearisation: `dX[i] ≥ ±(X[i+1]−X[i])` | `planner_obstacles.py` → `plan` |
| 15 | L1 terminal error: `d_term ≥ ±(X[-1]−x_f)` | `planner_obstacles.py` → `plan` |
| 16 | J1 = `w1·sum(dX) + w2·sum(d_term)` | `planner_obstacles.py` → `plan` |
| 17 | J2 = `w3·sum(X[1:,2])` (altitude penalty) | `planner_obstacles.py` → `plan` |
| 18 | L1 jerk linearisation: `dJerk[i] ≥ ±(Jerk[i+1]−Jerk[i])` | `planner_obstacles.py` → `plan` |
| 19 | J3 = `w4·sum(dJerk)/ts` | `planner_obstacles.py` → `plan` |
| 20 | Total objective: `min J1 + J2 + J3` | `planner_obstacles.py` → `plan` |
| 21 | Desired force: `F_des = −Kp·e_p − Kv·e_v + m·g·e₃ + m·a_d` | `controller.py` → `compute` |
| 22 | Thrust: `T = F_des·b3 = F_des·R[:,2]`, floor at `0.1·m·g` | `controller.py` → `compute` |
| 23 | Desired body-z: `b3_des = F_des / ||F_des||` | `controller.py` → `compute` |
| 24 | R_des: `b2_des = (b3_des×b1_c)/||...||`, `b1_des = b2_des×b3_des`, `R_des=[b1|b2|b3]` | `controller.py` → `compute` |
| 25 | Attitude error: `e_R = vee(0.5·(R_des^T·R − R^T·R_des))` | `controller.py` → `compute`, `_vee` |
| 26 | Torques: `τ = −Kr·e_R − Kω·e_ω + ω×(I·ω)` | `controller.py` → `compute` |
| 27 | Rotation matrix ZYX: `R = Rz(ψ)·Ry(θ)·Rx(φ)` | `quadrotor.py` → `rotation_matrix_euler` |
| 28 | Translational dynamics: `v̇ = R·[0,0,T/m]^T − g·e₃ + w_acc` | `quadrotor.py` → `_derivatives` |
| 29 | Euler-angle kinematics ZYX with singularity guard at |cos(θ)|<1e-8 | `quadrotor.py` → `_derivatives` |
| 30 | Rotational dynamics: `ω̇ = I⁻¹·(τ − ω×(I·ω))` | `quadrotor.py` → `_derivatives` |
| 31 | RK4: `s += (dt/6)·(k1 + 2k2 + 2k3 + k4)` | `quadrotor.py` → `step` |
| 32 | Euler-angle wrapping: `angles = (angles + π) % (2π) − π` | `quadrotor.py` → `step` |
| 33 | Wind Markov chain: `w[k] = ρ·w[k-1] + σ·√(1−ρ²)·N(0,I)` | `quadrotor.py` → `WindModel.step` |
| 34 | Dynamic obstacle NFZ: box `[p[0]±r, p[1]±r]` at `t_global + execute_time/2` | `dynamic_obstacle.py` → `as_nfz`; `core.py` → RHC loop |
| 35 | RHC termination: `||pos_actual − x_f|| < 50 m` | `core.py` → `run_closed_loop_rhc` |
| 36 | Infeasibility deceleration: `vel *= 0.9` per physics step, alt floor held | `core.py` → infeasibility branch |
| 37 | n_steps per horizon: `execute_time / dt_sim = 2.0 / 0.002 = 1000` | `core.py` → `run_closed_loop_rhc` |
| 38 | RMS tracking error: `√(mean(||log_pos[i] − ref_pos[i]||²))` | `simulation.py` → `create_figure` |
| 39 | Peak tracking error: `max(||log_pos[i] − ref_pos[i]||)` | `simulation.py` → `create_figure` |
| 40 | Cumulative energy: `Σ T[i]·dt` (rectangular integration) | `simulation.py` → `create_figure` |

The only non-trivial mathematics: rows 1–6 (B-spline), 12–13 (Big-M MILP), 21–26 (SE(3) controller), 27–33 (6-DOF dynamics + wind). Everything else is arithmetic on scalars and arrays.

---

## 13. The Scenario: Urban Canyon Map

**File:** `scenario.py` → `build_scenario`

```
Start:  [2500, -1400, 12] m
Goal:   [-2200,  2000, 12] m
Distance: ~6 km
```

### 13.1 Buildings (Cuboid Obstacles)

| # | X range (m) | Y range (m) | Height (m) | Safety margin η | Type |
|---|---|---|---|---|---|
| 1 | [464, 618] | [-1645, 655] | 15 | 5 m | **Short hurdle** — below z_max=25 → drone flies over |
| 2 | [-182, -109] | [-1673, 1691] | 200 | 80 m | **Tall gate** — above z_max=25 → drone goes around laterally |

Building 2 is 200 m tall and 3364 m wide in Y — the drone must thread through the narrow X gap or go far around the ends.

### 13.2 No-Fly Zones (2D, Infinite Altitude)

| # | X range (m) | Y range (m) | Safety margin γ |
|---|---|---|---|
| 1 | [1382, 1800] | [-1409, -100] | 30 m |
| 2 | [-1791, -1082] | [-1318, 400] | 30 m |
| 3 | [-964, -482] | [1518, 2309] | 30 m |
| 4 | [-536, 2227] | [2691, 2945] | 30 m |

NFZs block trivial bypass routes, forcing the drone through specific corridors.

### 13.3 Dynamic Obstacle

`create_dynamic_obstacle()` returns None (infrastructure built, scenario disabled). The `DynamicObstacle` class in `dynamic_obstacle.py` is fully functional — it can be enabled by returning a `DynamicObstacle(pos0, velocity, radius, safety_margin)` from that function.

---

## 14. End-to-End Flows

### 14.1 `python simulation.py` — Full Run

```
1. UAVPlannerMILP(h=10, ts=1.0, z_max=25.0)
   └─► get_bspline_matrices(k=4, h=10) → stores B, B', B'', B''' once

2. build_scenario(planner)
   └─► planner.add_cuboid_obstacle(...) × 2
       planner.add_no_fly_zone(...)    × 4
       returns x_s, v_s, a_s, x_f

3. run_closed_loop_rhc(planner, x_s, v_s, a_s, x_f, max_iter=100, ...)
   For each horizon (iter 0..~43):
   ├── planner.plan(pos_actual, vel_actual, acc_plan, x_f)
   │     └─► CVXPY formulates → SciPy MILP solves → returns Q (14×3 control points)
   ├── For 1000 physics steps (2.0 s at 500 Hz):
   │     ├── sample_trajectory(t_local, Q, ...) → p_d, v_d, a_d
   │     ├── wind.step() → w_acc
   │     ├── ctrl.compute(quad.state, p_d, v_d, a_d) → u=[T, τx, τy, τz]
   │     └── quad.step(u, dt_sim, w_acc) → new state
   └── pos_actual = quad.pos  (actual wind-disturbed position for next horizon)

4. np.savez_compressed('simulation_cache.npz', ...)

5. create_figure(...) → simulation_results.png  (6-panel matplotlib)

6. save_interactive_html(...) → trajectory_3d.html  (Plotly)
```

### 14.2 `python visualizer.py` — Live GUI

```
1. Parse args (--save, --fresh)
2. If cache exists and not --fresh: load simulation_cache.npz (instant)
   Else: call run_closed_loop_rhc() (full re-simulation)
3. Build matplotlib figure (dark theme, TkAgg backend)
   ├── ax_3d: 3D animated view (buildings, NFZs, reference trail, actual trail, drone dot)
   ├── ax_xy: 2D top-down map
   ├── ax_alt: telemetry strip (speed)
   ├── ax_err: telemetry strip (altitude)
   └── ax_att: telemetry strip (roll, pitch)
4. FuncAnimation(fig, update_frame, frames=len(log_t), interval=20)
   Each frame:
   └─► move drone dot, extend trails, update HUD text, redraw
5. SPACE key handler: toggle pause/resume
6. If --save: writer.grab_frame() per frame → flight.mp4 / flight.gif
```

### 14.3 A Single MILP Solve (Inside `planner.plan`)

```
Input:   pos_actual (3,), vel_actual (3,), acc_plan (3,), x_f (3,)
Output:  Q (14,3), X (11,3), V (11,3), A (11,3)   or   None × 4

1. Q      = cp.Variable((14, 3))
2. X, V, A, Jerk computed as CVXPY linear expressions from B matrices and Q
3. Constraints added:
   - Initial conditions (3 equations × 3 axes = 9)
   - Velocity bounds (10 × 3 × 2 = 60)
   - Acceleration bounds (60)
   - Altitude floor/ceiling (22)
   - Per building: 1 binary 6-var set per step → 60 binary vars + 60 constraints per building
   - Per NFZ: 1 binary 4-var set per step → 40 binary vars + 40 constraints per NFZ
   - L1 linearisation auxiliaries: dX (60), d_term (6), dJerk (60)
4. Objective: cp.Minimize(J1 + J2 + J3)
5. prob.solve(solver=cp.SCIPY)
6. Return Q.value, X.value, V.value, A.value  (or None × 4 if infeasible)
```

---

## 15. Outputs and Visualisation

### 15.1 `simulation_results.png` — 6-Panel Figure

| Panel | Content |
|---|---|
| **Panel 1 (3D)** | 3D trajectory: B-spline reference (dotted cyan) vs. actual 6-DOF (coral); buildings (blue cubes); NFZs (red boxes); start (lime); goal (gold star) |
| **Panel 2 (Top-down)** | X-Y map: reference vs. actual; buildings and NFZs as 2D rectangles; equal aspect ratio |
| **Panel 3 (Altitude + Wind)** | Z vs. time for reference and actual; z_max=25 m dashed line; wind magnitude fill on secondary axis |
| **Panel 4 (Tracking)** | X and Y position vs. time, reference vs. actual |
| **Panel 5 (Error)** | Position tracking error ‖e_p‖ vs. time; RMS and peak error annotated in a text box |
| **Panel 6 (Thrust + Energy)** | Thrust vs. time; hover line at m·g=9.81 N; cumulative energy (N·s) on secondary axis |

### 15.2 `trajectory_3d.html` — Interactive Plotly 3D

~11 MB standalone HTML (Plotly.js embedded). Open in any browser. Rotate: left-drag. Zoom: scroll. Pan: right-drag. Shows B-spline reference, actual flight, buildings, NFZs, start, goal, and optionally the intruder UAV trail.

### 15.3 Live GUI (`visualizer.py`)

Dark GitHub-palette theme. Layout:
- **Left 2/3:** 3D animated drone; trajectory trail builds as animation advances
- **Right top:** 2D top-down map with live drone marker
- **Right bottom:** Four telemetry strips (speed, altitude, error, roll+pitch)
- **SPACE:** Pause — mouse freely rotates/zooms the 3D view while paused

---

## 16. Paper vs. Extensions

### 16.1 What We Implemented from the Paper

| Component | Paper Section | Our File |
|---|---|---|
| B-spline trajectory representation | §II | `bspline_utils.py` |
| MILP obstacle avoidance (Big-M) | §III-A | `planner_obstacles.py` |
| No-fly zone constraints | §III-B | `planner_obstacles.py` |
| Velocity + acceleration bounds | §III-C | `planner_obstacles.py` |
| Altitude ceiling | §III | `planner_obstacles.py` |
| Multi-objective optimisation (J1, J2, J3) | §IV | `planner_obstacles.py` |
| Receding Horizon Control | §V | `core.py` |

### 16.2 What We Added Beyond the Paper

| Addition | Description | Files |
|---|---|---|
| **True Closed-Loop RHC** | Re-plan from ACTUAL wind-disturbed position every horizon | `core.py` |
| **6-DOF Quadrotor Physics** | Full rigid-body inertia, RK4 at 500 Hz | `quadrotor.py` |
| **Geometric Controller SE(3)** | Lee 2010 — no gimbal lock, two cascaded loops | `controller.py` |
| **Dryden Wind Turbulence** | First-order Markov persistent gust model | `quadrotor.py → WindModel` |
| **Dynamic Obstacle Infrastructure** | Intruder UAV as moving NFZ; fail-safe retry | `dynamic_obstacle.py`, `core.py` |
| **Infeasibility Handling** | Graceful glide + retry instead of crash | `core.py` |
| **Free Open-Source Solver** | SciPy MILP instead of commercial Gurobi | `planner_obstacles.py` |
| **6-Panel Publication Figure** | Dark-theme matplotlib with all telemetry | `simulation.py` |
| **Interactive 3D HTML** | Plotly standalone viewer for presentations | `simulation.py` |
| **Live Animated GUI** | Real-time 3D animation with HUD, export | `visualizer.py` |
| **Simulation Cache** | NPZ cache for instant GUI re-launch | `simulation.py` + `visualizer.py` |

---

## 17. Key Parameters

| Parameter | Value | File → Location | Effect |
|---|---|---|---|
| Spline degree k | 4 | `planner_obstacles.py → __init__` | C³ continuity |
| Horizon steps h | 10 | `planner_obstacles.py → __init__` | 10 s lookahead, 14 control points |
| Step duration ts | 1.0 s | `planner_obstacles.py → __init__` | T_h = h·ts = 10 s |
| Execute time | 2.0 s | `simulation.py → main()` | Replanning frequency |
| Physics dt | 0.002 s | `simulation.py → main()` | 500 Hz RK4 |
| v_max | 60 m/s | `scenario.py → planner.v_max = 60.0` | Per-axis velocity limit |
| a_max | 14.715 m/s² | `planner_obstacles.py → __init__` | ≈1.5g acceleration limit |
| z_max | 25 m | `planner_obstacles.py → __init__` | Altitude ceiling |
| z_floor | 1 m | `planner_obstacles.py → plan` | Ground clearance |
| η (short building) | 5 m | `scenario.py → add_cuboid_obstacle` | Building safety margin |
| η (tall building) | 80 m | `scenario.py → add_cuboid_obstacle` | Building safety margin |
| γ (NFZs) | 30 m | `scenario.py → add_no_fly_zone` | NFZ safety margin |
| Big-M | 10000.0 | `planner_obstacles.py → self.M` | Disjunctive relaxation |
| w1 | 1.0 | `planner_obstacles.py → self.w1` | Path length weight |
| w2 | 10.0 | `planner_obstacles.py → self.w2` | Goal approach weight |
| w3 | 0.3 | `planner_obstacles.py → self.w3` | Altitude penalty weight |
| w4 | 0.1 | `planner_obstacles.py → self.w4` | Jerk smoothness weight |
| Wind ρ | 0.995 | `core.py → WindModel(rho=0.995)` | Gust persistence |
| Wind σ | [1.8,1.8,0.4] m/s² | `core.py → WindModel(sigma=...)` | Turbulence intensity |
| Wind seed | 42 | `core.py → WindModel(seed=42)` | Reproducibility |
| Kp | diag(16,16,20) | `controller.py → self.Kp` | Position loop stiffness |
| Kv | diag(8,8,9) | `controller.py → self.Kd` | Position loop damping |
| Kr | diag(120,120,60) | `controller.py → self.Kr` | Attitude loop stiffness |
| Kω | diag(16,16,8) | `controller.py → self.Kw` | Attitude loop damping |
| Drone mass | 1.0 kg | `core.py → Quadrotor(mass=1.0)` | Thrust-to-weight |
| Ixx=Iyy | 0.0196 kg·m² | `quadrotor.py → __init__` | Roll/pitch inertia |
| Izz | 0.0264 kg·m² | `quadrotor.py → __init__` | Yaw inertia |
| Goal radius | 50 m | `core.py → if dist < 50: break` | Termination threshold |
| Max iterations | 100 | `simulation.py → max_iter=100` | Safety cap |

---

## 18. Verification and Results

### 18.1 Terminal Output (per horizon)

```
RHC iter   0  t=    2.0s  OPTIMAL  pos=[2379,−1320,12.3]  dist=5217m  plan=3.124s
RHC iter   1  t=    4.0s  OPTIMAL  pos=[2254,−1239,12.1]  dist=5013m  plan=2.891s
...
RHC iter  43  t=   88.0s  OPTIMAL  pos=[−2190,1987,12.0]  dist=23m
>>> Close enough -- stopping.

  Total wall time  : 142.3s
  Horizons solved  : 44
  Avg plan time    : 3210 ms/horizon
  Flight duration  : 88.0s
```

### 18.2 Physical Verification Checklist

- ✅ Drone never enters any building (Big-M constraints at all sample points)
- ✅ Drone never enters any NFZ
- ✅ Drone stays above z = 1 m at all times
- ✅ Drone stays below z = 25 m → navigates around tall buildings laterally
- ✅ Short hurdles (15 m) → drone flies over them (cheaper than going around)
- ✅ Per-axis velocity ≤ 60 m/s (enforced by MILP; visible in telemetry)
- ✅ Per-axis acceleration ≤ 14.715 m/s² (enforced by MILP)
- ✅ C² continuity at horizon boundaries (acc_plan pinned in initial conditions)
- ✅ Wind does not cause catastrophic drift — replanning corrects each cycle
- ✅ Simulation reproducible (seeded RNG, seed=42)

---

## 19. Known Limitations

### 19.1 Per-Axis vs. Norm-Ball Constraints

Velocity and acceleration are bounded per-axis (|vx| ≤ v_max, |vy| ≤ v_max, |vz| ≤ v_max), not as a 3D Euclidean norm. This keeps the problem a pure MILP but allows up to v_max·√3 ≈ 103 m/s diagonally. A SOCP norm constraint would be tighter but changes the problem class.

### 19.2 Open-Source Solver Speed

SciPy's MILP backend (branch-and-bound) is slower than Gurobi or CPLEX. Scaling to 14 buildings + 4 NFZs would produce ~1000 binary variables per horizon and may be very slow. The current scenario (2 buildings + 4 NFZs, 280 binary vars) is tractable.

### 19.3 No Parameter Fitting or System Identification

Controller gains (Kp, Kr, etc.) and quadrotor inertia are hand-tuned literature values, not fitted from real flight data. Performance could be improved by system identification on a physical drone.

### 19.4 Altitude Constraint as Proxy

Using z ≤ 25 m to force lateral avoidance of tall buildings works when buildings are > 25 m. A building between 15 m and 25 m is ambiguous (the altitude penalty J2 and the Z-face Big-M constraint would interact). The current scenario avoids this ambiguity by using 200 m (tall) and 15 m (short).

### 19.5 Dynamic Obstacle Disabled

The full `DynamicObstacle` infrastructure is implemented but `create_dynamic_obstacle()` returns None, so the feature is untested in the final scenario.

### 19.6 No Forgetting / No Long-Range Replanning

Classic BKT analogy: the drone has no "forgetting" — previous positions are not reconsidered. The RHC only looks 10 seconds ahead. In a longer, more complex scenario, the 10-second horizon might lead to locally optimal but globally suboptimal paths.

---

## 20. How to Run

### 20.1 Setup

```powershell
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1
# If blocked: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 20.2 Run Modes

```bash
# Mode A: Full simulation (MILP + physics + all outputs)
python simulation.py
# Generates: simulation_cache.npz, simulation_results.png, trajectory_3d.html

# Mode B: Live 3D GUI (loads cache instantly, no MILP re-solve)
python visualizer.py

# Mode C: Force fresh recomputation (ignores existing cache)
python visualizer.py --fresh

# Mode D: Export video
python visualizer.py --save
# → flight.mp4 (needs ffmpeg) or flight.gif (Pillow fallback)
```

### 20.3 Troubleshooting

| Problem | Fix |
|---|---|
| PowerShell execution blocked | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` |
| GUI window does not appear (Linux) | `sudo apt-get install python3-tk` |
| CVXPY or solver errors | `pip install --upgrade cvxpy scipy` (use Python ≤ 3.12) |
| Plotly not installed | `pip install plotly` |
| Visualiser slow or crashes | Run `simulation.py` first to generate the cache |

---

## 21. Examiner Q&A

**Q: What is a B-spline and why use one?**  
A: A piecewise polynomial curve defined by control points and a knot vector. Degree 4 gives C³ continuity — position, velocity, acceleration, and jerk are all smooth. This is essential because (a) the geometric controller needs a smooth acceleration reference, and (b) velocity and acceleration are linear in the control points, so they become linear constraints in the MILP rather than requiring nonlinear programming.

**Q: What is a MILP and why is it needed here?**  
A: Mixed Integer Linear Programming — an optimisation with continuous and binary (0/1) variables, linear constraints and objective. It is needed because "the drone must be outside at least one face of each building" is a disjunction (OR condition), which is non-convex. The Big-M method converts each OR into a set of linear constraints gated by binary variables.

**Q: What is Big-M? What value and why?**  
A: A large constant M used to relax constraints. When binary b[j]=0, the constraint becomes `something ≤ M`, which is trivially satisfied (the constraint does nothing). M must be larger than the maximum possible constraint violation in the relaxed direction. M=10000 is used; the flight domain spans ±4000 m, so 10000 > any real coordinate difference.

**Q: What is Receding Horizon Control (RHC)?**  
A: Instead of planning the entire 88-second flight at once (computationally intractable with MILP), we plan a 10-second horizon, execute only the first 2 seconds, observe the new actual state, and replan. This gives a tractable online planning problem while continuously correcting for wind disturbances.

**Q: What is "true closed-loop" vs. open-loop?**  
A: Open-loop RHC uses the previous plan's predicted state as the next plan's initial condition. Closed-loop uses the actual drone state from physics (which includes wind disturbance). Our system is closed-loop: if wind pushes the drone 50 metres off course, the next plan starts from the actual wind-pushed position and automatically corrects back.

**Q: What does the Geometric Controller do and why not PID?**  
A: It converts the desired position, velocity, and acceleration (from the B-spline) into total thrust and body torques. A PID controller using Euler angles suffers from gimbal lock (a mathematical singularity when pitch reaches ±90°). The Lee 2010 geometric controller operates on SO(3) — the rotation manifold — with no singularities, and naturally handles arbitrary orientations.

**Q: What is the Dryden wind model?**  
A: A first-order Markov process: w[k] = ρ·w[k-1] + noise. With ρ=0.995, the wind evolves slowly (realistic — gusts last seconds, not milliseconds). Based on the military aviation turbulence standard MIL-HDBK-1797. Injected as an acceleration into the drone's translational dynamics at every 2ms physics step.

**Q: Why doesn't the drone drift off course permanently due to wind?**  
A: Two mechanisms: (1) the geometric controller's position error term actively pushes the drone back toward the reference trajectory, and (2) the RHC replanning every 2 seconds starts from the actual wind-pushed position, so the entire plan resets to correct for accumulated drift.

**Q: Why RK4 at 500 Hz?**  
A: RK4 (4th-order Runge-Kutta) is accurate and stable for stiff ODEs. At high speeds (up to 60 m/s) and high angular rates, Euler integration would accumulate significant errors unless the timestep is very small. 500 Hz (dt=0.002 s) provides good accuracy without excessive computation.

**Q: What solver did you use for the MILP and why not Gurobi?**  
A: SciPy's built-in MILP solver via CVXPY's `cp.SCIPY` interface. The reference paper used Gurobi (commercial, requires licence). We chose the free open-source option so the project runs on any machine without a licence. SciPy's MILP is slower but sufficient for our problem size.

**Q: What if the MILP is infeasible?**  
A: The drone executes a graceful glide: velocity decays by 10% per physics step, altitude floor is maintained, all logging continues. On the next horizon, planning retries from the new actual position. This handles edge cases (drone drifting too close to an obstacle boundary) without crashing the simulation.

**Q: What is the vee map?**  
A: The vee map (written vee(S)) extracts the axial (3D) vector from a 3×3 skew-symmetric matrix. A skew-symmetric matrix has the form [[0,-c,b],[c,0,-a],[-b,a,0]], so `vee(S) = [a,b,c]`. In the SE(3) controller, the attitude error is naturally a skew-symmetric matrix on SO(3); vee converts it to a 3-vector that can be multiplied by the gain matrix Kr.

**Q: What is C² continuity and how is it maintained across horizons?**  
A: C² continuity means position, velocity, AND acceleration are all continuous (no instantaneous jumps). At each replanning step, the MILP constraints pin the trajectory's initial position, velocity, and acceleration to match the actual drone state (for position and velocity) and the previous plan's end acceleration (for acceleration). This prevents the drone from receiving a discontinuous acceleration command at the boundary between two horizons.

---

## 22. Glossary

| Term | Definition |
|---|---|
| **B-Spline** | Basis Spline — a piecewise polynomial curve defined by control points and a knot vector. Degree k gives C^(k-1) continuity. |
| **Cox-de Boor recursion** | The standard algorithm for evaluating B-spline basis functions and their derivatives, building from piecewise constants (k=1). |
| **MILP** | Mixed Integer Linear Programming — optimisation with continuous and binary (0/1) variables, linear constraints and objective. |
| **Big-M** | A large constant M used to relax linear constraints when a binary variable is 0, enabling disjunctive (OR) logic. |
| **Disjunctive constraint** | "The drone must be on at least one side of the obstacle" — an OR condition, non-convex, linearised by Big-M. |
| **Receding Horizon Control (RHC)** | A control strategy: solve a finite-horizon optimisation at each step, execute only the first portion, slide the horizon forward. Also called MPC. |
| **C² continuity** | A trajectory where position, velocity, AND acceleration are all continuous (no jumps). |
| **6-DOF** | Six Degrees of Freedom — three translational + three rotational. A rigid body in 3D space. |
| **SE(3)** | Special Euclidean group — all rigid body transformations (rotations + translations) in 3D space. |
| **SO(3)** | Special Orthogonal group — all 3×3 rotation matrices R with R^T·R = I and det(R) = 1. |
| **Vee map** | Extracts the 3-vector (axial vector) from a 3×3 skew-symmetric matrix. |
| **Gimbal lock** | A singularity in Euler-angle representations where one DOF is lost when pitch reaches ±90°. SE(3) controllers avoid this. |
| **RK4** | 4th-order Runge-Kutta — a numerical integrator that advances an ODE using four slope estimates for high accuracy. |
| **Dryden model** | Atmospheric turbulence model (MIL-HDBK-1797) representing wind as a filtered Gaussian process. |
| **Markov chain (first-order)** | A stochastic process where the next state depends only on the current: w[k] = f(w[k-1]) + noise. |
| **NFZ** | No-Fly Zone — a 2D restricted airspace region (treated as infinite altitude) where flight is prohibited. |
| **η** | Safety margin added around buildings — effective obstacle boundary is inflated by η on all sides. |
| **Big-M constant M** | 10000.0 in this project. Must exceed the maximum possible constraint violation in any relaxed direction. |
| **CVXPY** | Python-embedded domain-specific language for convex (and mixed-integer) optimisation — used to formulate the MILP symbolically. |
| **Horizon** | One RHC cycle: plan 10 s, execute 2 s. |
| **Control point** | Each Q[i] ∈ ℝ³ — the MILP decision variable that defines the shape of the B-spline trajectory. |
| **Knot vector** | The sequence of parameter values defining where B-spline polynomial pieces join. Repeated knots at ends force interpolation. |
| **n_ctrl** | Number of control points = k + h = 4 + 10 = 14 for this project. |
| **simulation_cache.npz** | Compressed NumPy archive of all simulation logs — enables instant visualiser re-launch without re-running the MILP. |
| **Gyroscopic compensation** | The term ω × (I·ω) in the torque law — it cancels the gyroscopic effect of the spinning body on the attitude dynamics. |
| **Feed-forward** | Using the reference acceleration a_d directly in the force command — reduces the position error needed to generate the required thrust. |
| **Cumulative energy** | ∑ T[i]·dt — a proxy for power consumption (thrust × time = impulse, proportional to energy when mass is fixed). |

---

*Report compiled from full read of all source files in `Drones_S5_AB5/`. Every formula in Section 12 was traced to its exact function in the corresponding file.*
