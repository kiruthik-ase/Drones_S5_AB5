"""
core.py  --  shared computation functions (no matplotlib dependency)
====================================================================
Contains the two heavy-lifting routines shared between
simulation.py (static PNG) and visualizer.py (live GUI):

  - precompute_rhc_trajectory()
  - simulate_dynamics()
"""

import numpy as np
import time

from planner_obstacles import UAVPlannerMILP
from controller import GeometricController
from quadrotor import Quadrotor
from bspline_utils import sample_trajectory


# ======================================================================= #
#  1.  PRE-COMPUTE FULL RHC TRAJECTORY                                     #
# ======================================================================= #

def precompute_rhc_trajectory(planner, x_s, v_s, a_s, x_f,
                               max_iter=100, execute_time=2.0, dt_ref=0.01):
    """
    Runs the MILP planner in a Receding-Horizon loop and stitches
    together the reference trajectory at `dt_ref` resolution.

    Returns
    -------
    ref_t         : (N,)     time stamps
    ref_pos       : (N, 3)   reference position
    ref_vel       : (N, 3)   reference velocity
    ref_acc       : (N, 3)   reference acceleration
    plan_segments : list of (Q, t_start, t_end)
    """
    ref_t   = [0.0]
    ref_pos = [x_s.copy()]
    ref_vel = [v_s.copy()]
    ref_acc = [a_s.copy()]
    plan_segments = []

    pos = x_s.copy()
    vel = v_s.copy()
    acc = a_s.copy()
    t_global = 0.0

    k   = planner.k
    h   = planner.h
    t_h = planner.t_h

    for iteration in range(max_iter):
        Q, X, V, A = planner.plan(pos, vel, acc, x_f)

        if Q is None:
            print(f"  RHC iter {iteration:3d}  t={t_global:7.1f}s  INFEASIBLE  "
                  f"pos=[{pos[0]:.0f},{pos[1]:.0f},{pos[2]:.1f}]")
            n_hold = int(execute_time / dt_ref)
            decel  = 0.8
            for j in range(1, n_hold + 1):
                t_global += dt_ref
                vel      *= decel
                pos_new   = pos + vel * dt_ref
                pos_new[2] = max(pos_new[2], 1.0)
                ref_t.append(t_global)
                ref_pos.append(pos_new.copy())
                ref_vel.append(vel.copy())
                ref_acc.append(np.zeros(3))
                pos = pos_new
            acc = np.zeros(3)
            continue

        plan_segments.append((Q.copy(), t_global, t_global + execute_time))

        n_steps = int(execute_time / dt_ref)
        for j in range(1, n_steps + 1):
            t_local = j * dt_ref
            p, v, a = sample_trajectory(t_local, Q, k=k, h=h, t_h=t_h)
            t_global += dt_ref
            ref_t.append(t_global)
            ref_pos.append(p.copy())
            ref_vel.append(v.copy())
            ref_acc.append(a.copy())

        pos, vel, acc = sample_trajectory(execute_time, Q, k=k, h=h, t_h=t_h)

        dist = np.linalg.norm(pos - x_f)
        print(f"  RHC iter {iteration:3d}  t={t_global:7.1f}s  OPTIMAL  "
              f"pos=[{pos[0]:.0f},{pos[1]:.0f},{pos[2]:.1f}]  "
              f"dist={dist:.0f}m")

        if dist < 50:
            print("  >>> Close enough -- stopping RHC.")
            break

    return (np.array(ref_t),
            np.array(ref_pos),
            np.array(ref_vel),
            np.array(ref_acc),
            plan_segments)


# ======================================================================= #
#  2.  SIMULATE QUADROTOR DYNAMICS                                          #
# ======================================================================= #

def simulate_dynamics(ref_t, ref_pos, ref_vel, ref_acc, dt_sim=0.002):
    """
    Run the 6-DOF quadrotor model tracking the reference at 500 Hz.

    Returns
    -------
    log_t, log_pos, log_vel, log_eul, log_u : ndarrays
    """
    quad = Quadrotor(mass=1.0)
    ctrl = GeometricController(mass=1.0)

    quad.set_state(pos=ref_pos[0], vel=ref_vel[0])

    T_total = ref_t[-1]
    n_sim   = int(T_total / dt_sim) + 1
    dt_ref  = ref_t[1] - ref_t[0]

    log_t   = np.zeros(n_sim)
    log_pos = np.zeros((n_sim, 3))
    log_vel = np.zeros((n_sim, 3))
    log_eul = np.zeros((n_sim, 3))
    log_u   = np.zeros((n_sim, 4))

    log_pos[0] = quad.pos
    log_vel[0] = quad.vel

    for i in range(1, n_sim):
        t       = i * dt_sim
        ref_idx = min(int(t / dt_ref), len(ref_t) - 1)
        p_d     = ref_pos[ref_idx]
        v_d     = ref_vel[ref_idx]
        a_d     = ref_acc[ref_idx]

        u = ctrl.compute(quad.state, p_d, v_d, a_d)
        quad.step(u, dt_sim)

        log_t[i]   = t
        log_pos[i] = quad.pos
        log_vel[i] = quad.vel
        log_eul[i] = quad.euler
        log_u[i]   = u

    return log_t, log_pos, log_vel, log_eul, log_u
