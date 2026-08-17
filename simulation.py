"""
UAV B-Spline Trajectory Planning & Closed-Loop RHC Simulation
===============================================================
Based on: "Optimal Trajectory-Planning of UAVs via B-Splines
           and Disjunctive Programming" (Babaei & Karimi, 2018)
Controller: Lee et al., "Geometric Tracking Control of a Quadrotor
            UAV on SE(3)", CDC 2010.

TRUE Closed-Loop RHC
---------------------
At every 2-second horizon the MILP planner uses the ACTUAL drone
position/velocity from the physics simulation as initial conditions.
Planning and dynamics are INTERLEAVED, not pre-computed.

Outputs
-------
- simulation_results.png  : 6-panel publication figure
- trajectory_3d.html      : interactive 3D view (open in browser to rotate)

Run:  python simulation.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

from planner_obstacles import UAVPlannerMILP
from scenario import build_scenario, create_dynamic_obstacle
from core import run_closed_loop_rhc


# ======================================================================= #
#  DRAWING HELPERS                                                          #
# ======================================================================= #

def draw_cuboid_3d(ax, xl, xu, yl, yu, zl, zu, color='royalblue', alpha=0.40):
    verts = [
        [[xl,yl,zl],[xu,yl,zl],[xu,yu,zl],[xl,yu,zl]],
        [[xl,yl,zu],[xu,yl,zu],[xu,yu,zu],[xl,yu,zu]],
        [[xl,yl,zl],[xl,yu,zl],[xl,yu,zu],[xl,yl,zu]],
        [[xu,yl,zl],[xu,yu,zl],[xu,yu,zu],[xu,yl,zu]],
        [[xl,yl,zl],[xu,yl,zl],[xu,yl,zu],[xl,yl,zu]],
        [[xl,yu,zl],[xu,yu,zl],[xu,yu,zu],[xl,yu,zu]],
    ]
    pc = Poly3DCollection(verts, facecolors=color, linewidths=0.4,
                          edgecolors='#ffffff33', alpha=alpha)
    ax.add_collection3d(pc)


def draw_rect_2d(ax, xl, xu, yl, yu, color, alpha, ls='-'):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((xl, yl), xu-xl, yu-yl,
                            facecolor=color, alpha=alpha,
                            edgecolor='white', lw=0.8, linestyle=ls))


# ======================================================================= #
#  INTERACTIVE PLOTLY HTML (rotate in browser)                              #
# ======================================================================= #

def save_interactive_html(ref_pos, log_pos, obstacles, no_fly_zones,
                           dyn_obs, ref_t, x_s, x_f, filename='trajectory_3d.html'):
    try:
        import plotly.graph_objects as go

        fig = go.Figure()

        # Planned reference
        fig.add_trace(go.Scatter3d(
            x=ref_pos[:,0], y=ref_pos[:,1], z=ref_pos[:,2],
            mode='lines', name='B-Spline Reference',
            line=dict(color='#00d4ff', width=3, dash='dot'),
            opacity=0.6
        ))

        # Actual trajectory
        fig.add_trace(go.Scatter3d(
            x=log_pos[:,0], y=log_pos[:,1], z=log_pos[:,2],
            mode='lines', name='Actual Flight (6-DOF + Wind)',
            line=dict(color='#ff6b6b', width=4),
            opacity=0.95
        ))

        # Start / Goal
        fig.add_trace(go.Scatter3d(
            x=[x_s[0]], y=[x_s[1]], z=[x_s[2]],
            mode='markers+text', name='Start',
            marker=dict(color='lime', size=8, symbol='circle'),
            text=['START'], textposition='top center'
        ))
        fig.add_trace(go.Scatter3d(
            x=[x_f[0]], y=[x_f[1]], z=[x_f[2]],
            mode='markers+text', name='Goal',
            marker=dict(color='gold', size=10, symbol='diamond'),
            text=['GOAL'], textposition='top center'
        ))

        # Buildings (as coloured boxes)
        def add_box(fig, xl, xu, yl, yu, zl, zu, color, name):
            xs=[xl,xu,xu,xl,xl, xl,xu,xu,xl,xl, xl,xl,xu,xu,xl, xl,xu,xu,xl,xl, xl,xl,xu,xu,xl]
            ys=[yl,yl,yu,yu,yl, yl,yl,yu,yu,yl, yl,yl,yl,yl,yl, yu,yu,yu,yu,yu, yl,yl,yl,yl,yl]
            zs=[zl,zl,zl,zl,zl, zu,zu,zu,zu,zu, zl,zu,zu,zl,zl, zl,zl,zu,zu,zl, zl,zu,zu,zl,zl]
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode='lines', name=name,
                line=dict(color=color, width=2), opacity=0.7,
                showlegend=(name not in [t.name for t in fig.data])
            ))

        for i, obs in enumerate(obstacles):
            add_box(fig, obs['xl'], obs['xu'], obs['yl'], obs['yu'],
                    obs['zl'], obs['zu'], '#4488ff',
                    'Building' if i==0 else 'Building_')

        # NFZ vertical columns
        for i, nfz in enumerate(no_fly_zones):
            add_box(fig, nfz['xl'], nfz['xu'], nfz['yl'], nfz['yu'],
                    0, 30, '#ff3333',
                    'No-Fly Zone' if i==0 else 'NFZ_')

        # Dynamic obstacle trail
        if dyn_obs is not None and ref_t is not None:
            dyn_t   = np.linspace(0, ref_t[-1], 150)
            dyn_pts = np.array([dyn_obs.position(t) for t in dyn_t])
            fig.add_trace(go.Scatter3d(
                x=dyn_pts[:,0], y=dyn_pts[:,1], z=dyn_pts[:,2],
                mode='lines+markers', name='Intruder UAV',
                line=dict(color='#ff9900', width=3, dash='dash'),
                marker=dict(size=3, color='#ff9900'),
                opacity=0.8
            ))

        fig.update_layout(
            title=dict(
                text='UAV Urban Canyon Navigation -- Interactive 3D View<br>'
                     '<sup>Rotate: Left-click drag | Zoom: Scroll | Pan: Right-click drag</sup>',
                font=dict(size=16, color='white')
            ),
            scene=dict(
                xaxis=dict(title='X (m)', backgroundcolor='#111130',
                            gridcolor='#334455', color='white'),
                yaxis=dict(title='Y (m)', backgroundcolor='#111130',
                            gridcolor='#334455', color='white'),
                zaxis=dict(title='Z (m)', backgroundcolor='#111130',
                            gridcolor='#334455', color='white', range=[0, 50]),
                bgcolor='#0a0a1a',
                aspectmode='data',
            ),
            paper_bgcolor='#0a0a1a',
            plot_bgcolor='#0a0a1a',
            font=dict(color='white'),
            legend=dict(bgcolor='#1a1a3a', bordercolor='#334455', borderwidth=1),
        )

        fig.write_html(filename)
        print(f"  Saved -> {filename}  (open in browser to rotate interactively)")
        return True

    except ImportError:
        print("  [plotly not installed] skipping interactive HTML.")
        print("  Install with:  pip install plotly")
        return False


# ======================================================================= #
#  PUBLICATION FIGURE  (6 panels)                                           #
# ======================================================================= #

def create_figure(ref_t, ref_pos,
                  log_t, log_pos, log_vel, log_eul, log_u, log_wind,
                  obstacles, no_fly_zones, x_s, x_f, dyn_obs):

    C_BG   = '#0a0a1a'
    C_REF  = '#00d4ff'
    C_ACT  = '#ff6b6b'
    C_OBS  = '#3a6fd8'
    C_NFZ  = '#cc2233'
    C_DYN  = '#ff9900'
    C_TXT  = '#e0e0e0'
    C_WIND = '#44ff88'
    C_THR  = '#ffcc00'

    fig = plt.figure(figsize=(22, 14), facecolor=C_BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.42, wspace=0.30,
                            left=0.06, right=0.97, top=0.93, bottom=0.07)

    axes = []
    for r in range(2):
        for c in range(3):
            if r == 0 and c == 0:
                ax = fig.add_subplot(gs[r, c], projection='3d')
                ax.set_facecolor('#111130')
            else:
                ax = fig.add_subplot(gs[r, c])
                ax.set_facecolor('#111130')
                ax.tick_params(colors=C_TXT, labelsize=8)
                for sp in ax.spines.values():
                    sp.set_edgecolor('#334455')
                ax.grid(True, alpha=0.12, color='white')
            axes.append(ax)

    ax3d, ax_xy, ax_alt, ax_pos, ax_err, ax_thr = axes

    # ---- Panel 1: 3-D trajectory ----
    ax3d.plot(ref_pos[:,0], ref_pos[:,1], ref_pos[:,2],
              '-', color=C_REF, lw=1.0, alpha=0.4, label='B-spline ref')
    ax3d.plot(log_pos[:,0], log_pos[:,1], log_pos[:,2],
              '-', color=C_ACT, lw=1.5, alpha=0.95, label='Actual (6-DOF+wind)')
    ax3d.scatter(*x_s, color='lime', s=80, zorder=6, label='Start')
    ax3d.scatter(*x_f, color='gold', s=120, zorder=6, marker='*', label='Goal')
    for obs in obstacles:
        draw_cuboid_3d(ax3d, obs['xl'], obs['xu'], obs['yl'], obs['yu'],
                       obs['zl'], obs['zu'], color=C_OBS, alpha=0.45)
    for nfz in no_fly_zones:
        draw_cuboid_3d(ax3d, nfz['xl'], nfz['xu'], nfz['yl'], nfz['yu'],
                       0, 30, color=C_NFZ, alpha=0.10)
    if dyn_obs is not None:
        dyn_t   = np.linspace(0, ref_t[-1], 120)
        dyn_pts = np.array([dyn_obs.position(t) for t in dyn_t])
        ax3d.plot(dyn_pts[:,0], dyn_pts[:,1], dyn_pts[:,2],
                  '--', color=C_DYN, lw=1.2, alpha=0.7, label='Intruder UAV')
    ax3d.set_xlabel('X (m)', color=C_TXT, fontsize=8)
    ax3d.set_ylabel('Y (m)', color=C_TXT, fontsize=8)
    ax3d.set_zlabel('Z (m)', color=C_TXT, fontsize=8)
    ax3d.set_zlim(0, 30)
    ax3d.set_title('3-D Trajectory  [open trajectory_3d.html to rotate]',
                   color=C_TXT, fontsize=10, fontweight='bold')
    ax3d.legend(fontsize=7, loc='upper left', facecolor='#1a1a3a',
                labelcolor=C_TXT, framealpha=0.8)
    ax3d.view_init(elev=35, azim=-55)
    ax3d.tick_params(colors=C_TXT, labelsize=7)

    # ---- Panel 2: Top-down X-Y ----
    ax_xy.plot(ref_pos[:,0], ref_pos[:,1], '-', color=C_REF, lw=1.2, alpha=0.5)
    ax_xy.plot(log_pos[:,0], log_pos[:,1], '-', color=C_ACT, lw=1.2, alpha=0.95)
    ax_xy.plot(x_s[0], x_s[1], 'o', color='lime', ms=9, zorder=8)
    ax_xy.plot(x_f[0], x_f[1], '*', color='gold', ms=12, zorder=8)
    for obs in obstacles:
        draw_rect_2d(ax_xy, obs['xl'], obs['xu'], obs['yl'], obs['yu'], C_OBS, 0.55)
    for nfz in no_fly_zones:
        draw_rect_2d(ax_xy, nfz['xl'], nfz['xu'], nfz['yl'], nfz['yu'], C_NFZ, 0.18, '--')
    if dyn_obs is not None:
        dyn_t   = np.linspace(0, ref_t[-1], 120)
        dyn_pts = np.array([dyn_obs.position(t) for t in dyn_t])
        ax_xy.plot(dyn_pts[:,0], dyn_pts[:,1], '--', color=C_DYN, lw=1.5, alpha=0.8)
        ax_xy.plot(dyn_obs.position(0)[0], dyn_obs.position(0)[1],
                   '^', color=C_DYN, ms=9, zorder=8)
    ax_xy.set_xlabel('X (m)', color=C_TXT, fontsize=9)
    ax_xy.set_ylabel('Y (m)', color=C_TXT, fontsize=9)
    ax_xy.set_title('Top-Down Map (X-Y)', color=C_TXT, fontsize=11, fontweight='bold')
    ax_xy.set_aspect('equal')
    ax_xy.tick_params(colors=C_TXT)

    # ---- Panel 3: Altitude + wind ----
    ax_alt.plot(ref_t, ref_pos[:,2], '--', color=C_REF, lw=1.0, alpha=0.5, label='Z ref')
    ax_alt.plot(log_t, log_pos[:,2], '-', color=C_ACT, lw=1.2, alpha=0.95, label='Z actual')
    ax_alt.axhline(y=25.0, color='yellow', lw=0.8, ls='--', alpha=0.5, label='z_max=25m')
    ax_alt2 = ax_alt.twinx()
    wind_mag = np.linalg.norm(log_wind, axis=1)
    ax_alt2.fill_between(log_t, 0, wind_mag, color=C_WIND, alpha=0.22)
    ax_alt2.plot(log_t, wind_mag, '-', color=C_WIND, lw=0.7, alpha=0.6)
    ax_alt2.set_ylabel('Wind |a| (m/s²)', color=C_WIND, fontsize=8)
    ax_alt2.tick_params(colors=C_WIND, labelsize=7)
    ax_alt.set_xlabel('Time (s)', color=C_TXT, fontsize=9)
    ax_alt.set_ylabel('Altitude Z (m)', color=C_TXT, fontsize=9)
    ax_alt.set_title('Altitude & Wind Disturbance', color=C_TXT, fontsize=11, fontweight='bold')
    ax_alt.legend(fontsize=7, facecolor='#1a1a3a', labelcolor=C_TXT)
    ax_alt.tick_params(colors=C_TXT)

    # ---- Panel 4: X-Y tracking ----
    c_pairs = [('#00d4ff','#ff6b6b'), ('#00ff88','#ff44cc')]
    for i, (cr, ca) in enumerate(c_pairs):
        lbl = ['X','Y'][i]
        ax_pos.plot(ref_t, ref_pos[:,i], '--', color=cr, lw=1.0, alpha=0.5,
                    label=f'{lbl} ref')
        ax_pos.plot(log_t, log_pos[:,i], '-', color=ca, lw=0.9,
                    label=f'{lbl} actual (closed-loop)')
    ax_pos.set_xlabel('Time (s)', color=C_TXT, fontsize=9)
    ax_pos.set_ylabel('Position (m)', color=C_TXT, fontsize=9)
    ax_pos.set_title('Closed-Loop Position Tracking', color=C_TXT,
                     fontsize=11, fontweight='bold')
    ax_pos.legend(fontsize=7, ncol=2, facecolor='#1a1a3a', labelcolor=C_TXT)
    ax_pos.tick_params(colors=C_TXT)

    # ---- Panel 5: Tracking error ----
    # ref_t and log_t are now the same (both come from interleaved simulation)
    err = np.linalg.norm(log_pos - ref_pos[:len(log_pos)], axis=1)
    ax_err.fill_between(log_t, 0, err, color=C_ACT, alpha=0.30)
    ax_err.plot(log_t, err, '-', color=C_ACT, lw=0.9, label='||e_p||')
    rms  = np.sqrt(np.mean(err**2))
    peak = np.max(err)
    ax_err.text(0.98, 0.95,
                f'RMS  = {rms:.2f} m\nPeak = {peak:.2f} m\nClosed-loop + Wind',
                transform=ax_err.transAxes, fontsize=9, color=C_TXT,
                va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='#1a1a3a', alpha=0.85))
    ax_err.set_xlabel('Time (s)', color=C_TXT, fontsize=9)
    ax_err.set_ylabel('Tracking Error (m)', color=C_TXT, fontsize=9)
    ax_err.set_title('Position Error (Closed-Loop + Wind)', color=C_TXT,
                     fontsize=11, fontweight='bold')
    ax_err.legend(fontsize=8, facecolor='#1a1a3a', labelcolor=C_TXT)
    ax_err.tick_params(colors=C_TXT)

    # ---- Panel 6: Thrust + energy ----
    thrust = log_u[:, 0]
    energy = np.cumsum(thrust * (log_t[1] - log_t[0])) if len(log_t) > 1 else np.zeros_like(thrust)
    ax_thr.fill_between(log_t, 0, thrust, color=C_THR, alpha=0.35)
    ax_thr.plot(log_t, thrust, '-', color=C_THR, lw=1.0, label='Thrust (N)')
    ax_thr.axhline(y=1.0*9.81, color='cyan', lw=0.8, ls='--', alpha=0.6, label='Hover')
    ax_thr2 = ax_thr.twinx()
    ax_thr2.plot(log_t, energy, '-', color='#ff88ff', lw=1.2, alpha=0.8,
                 label='Energy (N·s)')
    ax_thr2.set_ylabel('Cumulative Energy (N·s)', color='#ff88ff', fontsize=8)
    ax_thr2.tick_params(colors='#ff88ff', labelsize=7)
    ax_thr.set_xlabel('Time (s)', color=C_TXT, fontsize=9)
    ax_thr.set_ylabel('Thrust (N)', color=C_TXT, fontsize=9)
    ax_thr.set_title('Thrust & Energy Consumption', color=C_TXT, fontsize=11, fontweight='bold')
    ax_thr.legend(fontsize=7, facecolor='#1a1a3a', labelcolor=C_TXT)
    ax_thr.tick_params(colors=C_TXT)

    plt.suptitle(
        'UAV B-Spline Trajectory Planning  |  TRUE Closed-Loop RHC  |  '
        'Urban Chicane + Intruder UAV + Wind Disturbance',
        color='white', fontsize=12, fontweight='bold')
    return fig


# ======================================================================= #
#  MAIN                                                                     #
# ======================================================================= #

def main():
    print("=" * 68)
    print("  UAV B-Spline TRUE Closed-Loop RHC Simulation")
    print("  Planning and physics INTERLEAVED at every 2-second horizon")
    print("=" * 68)

    planner = UAVPlannerMILP(h=10, ts=1.0, z_max=25.0)
    x_s, v_s, a_s, x_f = build_scenario(planner)
    dyn_obs = create_dynamic_obstacle()

    print("\n[Phase 1+2] Running closed-loop RHC (plan + simulate interleaved) ...")
    t0 = time.time()
    (ref_t, ref_pos,
     log_t, log_pos, log_vel, log_eul, log_u, log_wind,
     plan_log) = run_closed_loop_rhc(
        planner, x_s, v_s, a_s, x_f,
        max_iter=100, execute_time=2.0, dt_sim=0.002,
        dyn_obs=dyn_obs, enable_wind=True
    )
    elapsed = time.time() - t0
    n_horizons = len(plan_log)
    avg_plan   = np.mean([p['plan_dt'] for p in plan_log]) if plan_log else 0
    print(f"\n  Total wall time  : {elapsed:.1f}s")
    print(f"  Horizons solved  : {n_horizons}")
    print(f"  Avg plan time    : {avg_plan*1000:.0f} ms/horizon")
    print(f"  Flight duration  : {log_t[-1]:.1f}s")

    # ---- Save cache so visualizer loads instantly next time ----
    CACHE = 'simulation_cache.npz'
    np.savez_compressed(CACHE,
                        ref_t=ref_t, ref_pos=ref_pos,
                        log_t=log_t, log_pos=log_pos,
                        log_vel=log_vel, log_eul=log_eul,
                        log_u=log_u, log_wind=log_wind)
    print(f"\n  Cache saved -> {CACHE}  (visualizer will load this instantly)")

    print("\n[Phase 3] Generating 6-panel publication figure ...")
    fig = create_figure(ref_t, ref_pos,
                        log_t, log_pos, log_vel, log_eul, log_u, log_wind,
                        planner.obstacles, planner.no_fly_zones,
                        x_s, x_f, dyn_obs)
    fig.savefig('simulation_results.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  Saved -> simulation_results.png")

    print("\n[Phase 4] Generating interactive 3D HTML ...")
    save_interactive_html(ref_pos, log_pos, planner.obstacles, planner.no_fly_zones,
                          dyn_obs, ref_t, x_s, x_f)

    print("\nDone!")


if __name__ == "__main__":
    main()
