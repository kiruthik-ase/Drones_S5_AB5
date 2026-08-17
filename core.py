"""
core.py  --  shared computation (no matplotlib dependency)
==========================================================

run_closed_loop_rhc()
    TRUE closed-loop Receding Horizon Control.
    At every horizon the MILP planner receives the ACTUAL quadrotor
    state from the 6-DOF physics (including wind disturbance effects).
    Planning and simulation are INTERLEAVED -- not pre-computed.

This is the realistic architecture:
  t=0  -> plan 10s horizon -> simulate 2s physics -> actual state
  t=2  -> plan 10s horizon from actual state -> simulate 2s -> actual state
  t=4  -> ...
"""

import numpy as np
import time

from planner_obstacles import UAVPlannerMILP
from controller import GeometricController
from quadrotor import Quadrotor, WindModel
from bspline_utils import sample_trajectory


# ======================================================================= #
#  TRUE CLOSED-LOOP RHC  (planning + simulation interleaved)               #
# ======================================================================= #

def run_closed_loop_rhc(planner, x_s, v_s, a_s, x_f,
                         max_iter=100, execute_time=2.0, dt_sim=0.002,
                         dyn_obs=None, enable_wind=True):
    """
    True closed-loop Receding Horizon Control simulation.

    At every horizon the planner uses:
      - pos_actual, vel_actual  from the 6-DOF physics (with wind)
      - acc_plan                from the previous B-spline (for C2 continuity)

    This means if wind pushes the drone off-course, the next horizon
    planning starts from the REAL deviated position and corrects it.

    Parameters
    ----------
    planner      : UAVPlannerMILP instance (already has obstacles added)
    x_s, v_s, a_s: initial position, velocity, acceleration
    x_f          : goal position
    max_iter     : maximum number of RHC horizons
    execute_time : seconds of the horizon to actually fly before replanning
    dt_sim       : physics integration timestep (s)  -- 500 Hz default
    dyn_obs      : DynamicObstacle or None
    enable_wind  : inject Dryden wind turbulence into the physics

    Returns
    -------
    ref_t    : (M,)    reference timestamps
    ref_pos  : (M, 3)  planned reference positions (concatenated across horizons)
    log_t    : (N,)    actual simulation timestamps
    log_pos  : (N, 3)  actual positions (6-DOF)
    log_vel  : (N, 3)  actual velocities
    log_eul  : (N, 3)  actual Euler angles (roll, pitch, yaw)
    log_u    : (N, 4)  control inputs [thrust, tau_x, tau_y, tau_z]
    log_wind : (N, 3)  wind acceleration disturbance (m/s^2)
    plan_log : list of dicts with per-horizon planning metadata
    """

    # ---------- initialise physics ----------
    quad = Quadrotor(mass=1.0)
    ctrl = GeometricController(mass=1.0)
    wind = WindModel(rho=0.995, sigma=[1.8, 1.8, 0.4], seed=42) if enable_wind else None

    quad.set_state(pos=x_s, vel=v_s)

    k   = planner.k
    h   = planner.h
    t_h = planner.t_h

    # ---------- output buffers ----------
    ref_t   = [0.0]
    ref_pos = [x_s.copy()]

    log_t    = []
    log_pos  = []
    log_vel  = []
    log_eul  = []
    log_u    = []
    log_wind = []
    plan_log = []

    # ---------- state at horizon boundary ----------
    pos_actual = x_s.copy()
    vel_actual = v_s.copy()
    acc_plan   = a_s.copy()   # use planned acc for B-spline C2 continuity
    t_global   = 0.0
    n_steps    = int(execute_time / dt_sim)

    for iteration in range(max_iter):
        t_plan_start = time.time()

        # ---- inject dynamic obstacle ----
        dyn_nfz_added = False
        if dyn_obs is not None:
            t_predict = t_global + execute_time / 2.0
            planner.no_fly_zones.append(dyn_obs.as_nfz(t_predict))
            dyn_nfz_added = True

        # ---- PLAN from actual drone state ----
        Q, X, V, A = planner.plan(pos_actual, vel_actual, acc_plan, x_f)

        if dyn_nfz_added:
            planner.no_fly_zones.pop()

        # fallback: replan without dynamic obstacle if infeasible
        if Q is None and dyn_obs is not None:
            print(f"    [DYN] Infeasible with dynamic obstacle at iter {iteration}; "
                  "retrying without it...")
            Q, X, V, A = planner.plan(pos_actual, vel_actual, acc_plan, x_f)

        plan_dt = time.time() - t_plan_start

        if Q is None:
            print(f"  RHC iter {iteration:3d}  t={t_global:7.1f}s  INFEASIBLE  "
                  f"pos=[{pos_actual[0]:.0f},{pos_actual[1]:.0f},{pos_actual[2]:.1f}]")
            # glide with deceleration
            for j in range(n_steps):
                t_global += dt_sim
                vel_actual *= 0.9
                pos_actual  = pos_actual + vel_actual * dt_sim
                pos_actual[2] = max(pos_actual[2], 1.0)
                w_acc = wind.step() if wind else np.zeros(3)
                # keep ref and log arrays aligned (ref = hold position)
                ref_t.append(t_global)
                ref_pos.append(pos_actual.copy())
                log_t.append(t_global)
                log_pos.append(pos_actual.copy())
                log_vel.append(vel_actual.copy())
                log_eul.append(quad.euler)
                log_u.append(np.zeros(4))
                log_wind.append(w_acc)
            acc_plan = np.zeros(3)
            continue

        # ---- SIMULATE physics for execute_time ----
        for j in range(n_steps):
            t_local = (j + 1) * dt_sim
            t_global += dt_sim

            p_d, v_d, a_d = sample_trajectory(t_local, Q, k=k, h=h, t_h=t_h)

            w_acc = wind.step() if wind else np.zeros(3)
            u = ctrl.compute(quad.state, p_d, v_d, a_d)
            quad.step(u, dt_sim, wind_acc=w_acc)

            # Reference logs
            ref_t.append(t_global)
            ref_pos.append(p_d.copy())

            # Actual logs
            log_t.append(t_global)
            log_pos.append(quad.pos)
            log_vel.append(quad.vel)
            log_eul.append(quad.euler)
            log_u.append(u)
            log_wind.append(w_acc)

        # ---- update state for next horizon from ACTUAL physics ----
        pos_actual = quad.pos       # actual position (with wind error)
        vel_actual = quad.vel       # actual velocity (with wind error)

        # planned acceleration at the execute boundary (for C2 continuity)
        _, _, acc_plan = sample_trajectory(execute_time, Q, k=k, h=h, t_h=t_h)

        dist = np.linalg.norm(pos_actual - x_f)

        # Dynamic obstacle distance tag
        dyn_tag = ""
        if dyn_obs is not None:
            dp = np.linalg.norm(pos_actual[:2] - dyn_obs.position(t_global)[:2])
            if dp < dyn_obs.radius * 2.5:
                dyn_tag = f"  [DYN {dp:.0f}m]"

        plan_log.append({
            'iter': iteration, 't': t_global, 'dist': dist, 'plan_dt': plan_dt
        })

        print(f"  RHC iter {iteration:3d}  t={t_global:7.1f}s  OPTIMAL  "
              f"pos=[{pos_actual[0]:.0f},{pos_actual[1]:.0f},{pos_actual[2]:.1f}]  "
              f"dist={dist:.0f}m  plan={plan_dt:.3f}s{dyn_tag}")

        if dist < 50:
            print("  >>> Close enough -- stopping.")
            break

    return (np.array(ref_t),
            np.array(ref_pos),
            np.array(log_t),
            np.array(log_pos),
            np.array(log_vel),
            np.array(log_eul),
            np.array(log_u),
            np.array(log_wind),
            plan_log)


# ======================================================================= #
#  LEGACY HELPERS (kept for compatibility, not used in main pipeline)       #
# ======================================================================= #

def precompute_rhc_trajectory(planner, x_s, v_s, a_s, x_f,
                               max_iter=100, execute_time=2.0, dt_ref=0.01,
                               dyn_obs=None):
    """Open-loop pre-computation (kept for reference/comparison)."""
    from bspline_utils import sample_trajectory as _st
    ref_t=[0.0]; ref_pos=[x_s.copy()]; ref_vel=[v_s.copy()]
    ref_acc=[a_s.copy()]; plan_segments=[]
    pos=x_s.copy(); vel=v_s.copy(); acc=a_s.copy(); t_global=0.0
    k=planner.k; h=planner.h; t_h=planner.t_h
    for iteration in range(max_iter):
        if dyn_obs is not None:
            planner.no_fly_zones.append(dyn_obs.as_nfz(t_global+execute_time/2))
        Q, X, V, A = planner.plan(pos, vel, acc, x_f)
        if dyn_obs is not None:
            planner.no_fly_zones.pop()
            if Q is None:
                Q, X, V, A = planner.plan(pos, vel, acc, x_f)
        if Q is None:
            n_hold=int(execute_time/dt_ref); decel=0.8
            for j in range(1,n_hold+1):
                t_global+=dt_ref; vel*=decel; pos_new=pos+vel*dt_ref
                pos_new[2]=max(pos_new[2],1.0)
                ref_t.append(t_global); ref_pos.append(pos_new.copy())
                ref_vel.append(vel.copy()); ref_acc.append(np.zeros(3)); pos=pos_new
            acc=np.zeros(3); continue
        plan_segments.append((Q.copy(),t_global,t_global+execute_time))
        n_steps=int(execute_time/dt_ref)
        for j in range(1,n_steps+1):
            t_local=j*dt_ref; p,v,a=_st(t_local,Q,k=k,h=h,t_h=t_h)
            t_global+=dt_ref; ref_t.append(t_global); ref_pos.append(p.copy())
            ref_vel.append(v.copy()); ref_acc.append(a.copy())
        pos,vel,acc=_st(execute_time,Q,k=k,h=h,t_h=t_h)
        dist=np.linalg.norm(pos-x_f)
        print(f"  RHC iter {iteration:3d}  t={t_global:7.1f}s  OPTIMAL  "
              f"pos=[{pos[0]:.0f},{pos[1]:.0f},{pos[2]:.1f}]  dist={dist:.0f}m")
        if dist<50: print("  >>> Close enough."); break
    return np.array(ref_t),np.array(ref_pos),np.array(ref_vel),np.array(ref_acc),plan_segments


def simulate_dynamics(ref_t, ref_pos, ref_vel, ref_acc, dt_sim=0.002, enable_wind=True):
    """Open-loop dynamics simulation (kept for reference/comparison)."""
    quad=Quadrotor(mass=1.0); ctrl=GeometricController(mass=1.0)
    wind=WindModel(rho=0.995,sigma=[1.8,1.8,0.4],seed=42) if enable_wind else None
    quad.set_state(pos=ref_pos[0],vel=ref_vel[0])
    T_total=ref_t[-1]; n_sim=int(T_total/dt_sim)+1; dt_ref=ref_t[1]-ref_t[0]
    log_t=np.zeros(n_sim); log_pos=np.zeros((n_sim,3)); log_vel=np.zeros((n_sim,3))
    log_eul=np.zeros((n_sim,3)); log_u=np.zeros((n_sim,4)); log_wind=np.zeros((n_sim,3))
    log_pos[0]=quad.pos; log_vel[0]=quad.vel
    for i in range(1,n_sim):
        t=i*dt_sim; ref_idx=min(int(t/dt_ref),len(ref_t)-1)
        p_d=ref_pos[ref_idx]; v_d=ref_vel[ref_idx]; a_d=ref_acc[ref_idx]
        w_acc=wind.step() if wind else None
        u=ctrl.compute(quad.state,p_d,v_d,a_d); quad.step(u,dt_sim,wind_acc=w_acc)
        log_t[i]=t; log_pos[i]=quad.pos; log_vel[i]=quad.vel; log_eul[i]=quad.euler
        log_u[i]=u
        if w_acc is not None: log_wind[i]=w_acc
    return log_t,log_pos,log_vel,log_eul,log_u,log_wind
