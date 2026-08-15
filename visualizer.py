"""
Live GUI Visualizer
===================
Runs the full UAV pipeline then plays back the flight as a real-time
animation in a 3-panel matplotlib window.

Layout
------
  Left  (large)  : 3-D flight animation -- drone flies through buildings
  Top-right       : Top-down (X-Y) map with live dot tracking
  Bottom-right    : Live telemetry strip charts (speed, altitude, error)

Usage
-----
  python visualizer.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # interactive window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patheffects as pe
import time

from planner_obstacles import UAVPlannerMILP
from controller import GeometricController
from quadrotor import Quadrotor
from bspline_utils import sample_trajectory
from scenario import build_scenario
from core import precompute_rhc_trajectory, simulate_dynamics

# ======================================================================= #
#  HELPERS                                                                  #
# ======================================================================= #

def draw_cuboid_3d(ax, xl, xu, yl, yu, zl, zu, color, alpha=0.45):
    verts = [
        [[xl,yl,zl],[xu,yl,zl],[xu,yu,zl],[xl,yu,zl]],
        [[xl,yl,zu],[xu,yl,zu],[xu,yu,zu],[xl,yu,zu]],
        [[xl,yl,zl],[xl,yu,zl],[xl,yu,zu],[xl,yl,zu]],
        [[xu,yl,zl],[xu,yu,zl],[xu,yu,zu],[xu,yl,zu]],
        [[xl,yl,zl],[xu,yl,zl],[xu,yl,zu],[xl,yl,zu]],
        [[xl,yu,zl],[xu,yu,zl],[xu,yu,zu],[xl,yu,zu]],
    ]
    pc = Poly3DCollection(verts, facecolors=color, linewidths=0.4,
                          edgecolors='#ffffff44', alpha=alpha)
    ax.add_collection3d(pc)

def draw_cuboid_2d(ax, xl, xu, yl, yu, color, alpha, linestyle='-'):
    rect = Rectangle((xl, yl), xu-xl, yu-yl,
                      facecolor=color, alpha=alpha,
                      edgecolor='white', lw=0.8, linestyle=linestyle)
    ax.add_patch(rect)

# ======================================================================= #
#  MAIN                                                                     #
# ======================================================================= #

def main():
    print("=" * 65)
    print("  UAV Live GUI Visualizer")
    print("  Scenario: Dense Urban Canyon (8 buildings, 2 no-fly zones)")
    print("=" * 65)

    # ------------------------------------------------------------------- #
    #  1. Pre-compute trajectory                                            #
    # ------------------------------------------------------------------- #
    print("\n[Phase 1] Pre-computing Receding-Horizon trajectory ...")
    t0 = time.time()
    planner = UAVPlannerMILP(h=10, ts=1.0, z_max=25.0)
    x_s, v_s, a_s, x_f = build_scenario(planner)
    ref_t, ref_pos, ref_vel, ref_acc, _ = precompute_rhc_trajectory(
        planner, x_s, v_s, a_s, x_f,
        max_iter=100, execute_time=2.0, dt_ref=0.01
    )
    print(f"  Done in {time.time()-t0:.1f}s  -- {ref_t[-1]:.1f}s flight")

    # ------------------------------------------------------------------- #
    #  2. Simulate dynamics                                                 #
    # ------------------------------------------------------------------- #
    print("\n[Phase 2] Simulating 6-DOF quadrotor dynamics ...")
    t0 = time.time()
    log_t, log_pos, log_vel, log_eul, log_u = simulate_dynamics(
        ref_t, ref_pos, ref_vel, ref_acc, dt_sim=0.002
    )
    print(f"  Done in {time.time()-t0:.1f}s  -- {len(log_t)} steps")

    # ------------------------------------------------------------------- #
    #  3. Downsample for animation (aim for ~60 fps visual, 1 s real/sim)  #
    # ------------------------------------------------------------------- #
    # We replay 138 sim-seconds in ~30 real-seconds -> speed x4.6
    ANIM_FPS   = 30                          # frames per second shown
    SPEED_MULT = 5.0                         # sim time per real second
    dt_anim    = SPEED_MULT / ANIM_FPS       # sim-seconds between frames
    dt_sim     = log_t[1] - log_t[0]
    stride     = max(1, int(dt_anim / dt_sim))

    a_t   = log_t[::stride]
    a_pos = log_pos[::stride]
    a_vel = log_vel[::stride]

    # Reference downsampled at same stride (ref is 100 Hz, sim is 500 Hz)
    ref_stride = max(1, int(stride * (dt_sim / (ref_t[1]-ref_t[0]))))
    r_pos = ref_pos[::ref_stride]

    N_frames = len(a_t)
    dt_ref_s = ref_t[1] - ref_t[0]

    # ------------------------------------------------------------------- #
    #  4. Build figure                                                      #
    # ------------------------------------------------------------------- #
    fig = plt.figure(figsize=(19, 10), facecolor='#0a0a1a')
    fig.canvas.manager.set_window_title('UAV Urban Canyon Navigation -- Live Simulation')

    gs = gridspec.GridSpec(2, 2, width_ratios=[1.7, 1], hspace=0.35, wspace=0.3)

    ax3d  = fig.add_subplot(gs[:, 0], projection='3d')   # left (full height)
    ax_xy = fig.add_subplot(gs[0, 1])                    # top-right
    ax_tel= fig.add_subplot(gs[1, 1])                    # bottom-right

    C_BG  = '#0a0a1a'
    C_REF = '#00d4ff'
    C_ACT = '#ff6b6b'
    C_OBS = '#4169E1'
    C_NFZ = '#cc2222'
    C_TXT = '#e8e8e8'
    C_DRN = '#ffcc00'

    for ax in [ax_xy, ax_tel]:
        ax.set_facecolor('#111130')
        ax.tick_params(colors=C_TXT, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#334')

    ax3d.set_facecolor('#0d0d25')

    # ------ 3D: static environment ------
    for obs in planner.obstacles:
        draw_cuboid_3d(ax3d, obs['xl'], obs['xu'], obs['yl'], obs['yu'],
                       obs['zl'], obs['zu'], color=C_OBS)
    for nfz in planner.no_fly_zones:
        draw_cuboid_3d(ax3d, nfz['xl'], nfz['xu'], nfz['yl'], nfz['yu'],
                       0, 30, color=C_NFZ, alpha=0.12)

    # Full reference path (ghost)
    ax3d.plot(ref_pos[:,0], ref_pos[:,1], ref_pos[:,2],
              '-', color=C_REF, lw=0.8, alpha=0.3, label='B-Spline ref')

    # Dynamic elements (will be updated each frame)
    trail_len = int(8.0 / dt_anim)   # ~8 seconds of trail
    trail3d,  = ax3d.plot([], [], [], '-', color=C_ACT, lw=2.0, alpha=0.9)
    drone3d   = ax3d.scatter([], [], [], s=120, color=C_DRN, zorder=10,
                              depthshade=False, marker='o')

    ax3d.scatter(*x_s, color='lime',  s=80, zorder=8, label='Start')
    ax3d.scatter(*x_f, color='gold',  s=120, zorder=8, marker='*', label='Goal')

    # Environment bounds
    all_x = [x_s[0], x_f[0]]
    all_y = [x_s[1], x_f[1]]
    for o in planner.obstacles:
        all_x += [o['xl'], o['xu']]; all_y += [o['yl'], o['yu']]
    xlo, xhi = min(all_x)-200, max(all_x)+200
    ylo, yhi = min(all_y)-200, max(all_y)+200

    ax3d.set_xlim(xlo, xhi); ax3d.set_ylim(ylo, yhi); ax3d.set_zlim(0, 50)
    ax3d.set_xlabel('X (m)', color=C_TXT, fontsize=9)
    ax3d.set_ylabel('Y (m)', color=C_TXT, fontsize=9)
    ax3d.set_zlabel('Z (m)', color=C_TXT, fontsize=9)
    ax3d.set_title('3-D Urban Canyon Navigation', color=C_TXT,
                   fontsize=12, fontweight='bold', pad=6)
    ax3d.legend(fontsize=7, loc='upper left', facecolor='#1a1a3a',
                labelcolor=C_TXT, framealpha=0.8)
    ax3d.view_init(elev=28, azim=-55)
    ax3d.tick_params(colors=C_TXT, labelsize=7)

    # HUD text on 3D
    time_txt = ax3d.text2D(0.02, 0.97, '', transform=ax3d.transAxes,
                            color=C_TXT, fontsize=9,
                            fontfamily='monospace', va='top')

    # ------ Top-right: 2D map ------
    for obs in planner.obstacles:
        draw_cuboid_2d(ax_xy, obs['xl'], obs['xu'], obs['yl'], obs['yu'],
                       C_OBS, 0.55)
    for nfz in planner.no_fly_zones:
        draw_cuboid_2d(ax_xy, nfz['xl'], nfz['xu'], nfz['yl'], nfz['yu'],
                       C_NFZ, 0.15, linestyle='--')

    ax_xy.plot(ref_pos[:,0], ref_pos[:,1], '-', color=C_REF, lw=1.0, alpha=0.4)
    trail2d,  = ax_xy.plot([], [], '-', color=C_ACT, lw=1.8, alpha=0.9)
    dot2d,    = ax_xy.plot([], [], 'o', color=C_DRN, ms=7, zorder=10)
    ax_xy.plot(x_s[0], x_s[1], 'o', color='lime',  ms=8, label='Start', zorder=9)
    ax_xy.plot(x_f[0], x_f[1], '*', color='gold',  ms=12, label='Goal',  zorder=9)
    ax_xy.set_xlim(xlo, xhi); ax_xy.set_ylim(ylo, yhi)
    ax_xy.set_aspect('equal')
    ax_xy.set_xlabel('X (m)', color=C_TXT, fontsize=8)
    ax_xy.set_ylabel('Y (m)', color=C_TXT, fontsize=8)
    ax_xy.set_title('Top-Down Map', color=C_TXT, fontsize=10, fontweight='bold')
    ax_xy.legend(fontsize=7, facecolor='#1a1a3a', labelcolor=C_TXT, framealpha=0.8)
    ax_xy.grid(True, alpha=0.1, color='white')

    # ------ Bottom-right: Telemetry ------
    tel_len = min(300, N_frames)
    tel_t = np.zeros(tel_len)

    speed_data = np.zeros(tel_len)
    alt_data   = np.zeros(tel_len)
    err_data   = np.zeros(tel_len)

    dt_ref_s2 = ref_t[1] - ref_t[0]

    ax_tel.set_xlim(0, a_t[min(tel_len-1, N_frames-1)])
    ax_tel.set_ylim(0, 100)
    ax_tel.set_xlabel('Sim time (s)', color=C_TXT, fontsize=8)
    ax_tel.set_title('Live Telemetry', color=C_TXT, fontsize=10, fontweight='bold')
    ax_tel.grid(True, alpha=0.12, color='white')

    ln_speed, = ax_tel.plot([], [], '-',  color='#00ff88', lw=1.4, label='Speed (m/s)')
    ln_alt,   = ax_tel.plot([], [], '-',  color='#ffaa00', lw=1.4, label='Altitude (m)')
    ln_err,   = ax_tel.plot([], [], '--', color='#ff6b6b', lw=1.1, label='Error x5 (m)')
    ax_tel.legend(fontsize=7, facecolor='#1a1a3a', labelcolor=C_TXT, framealpha=0.8)

    # telemetry window — keep last tel_len points
    buf_t     = []
    buf_spd   = []
    buf_alt   = []
    buf_err   = []

    plt.suptitle('UAV B-Spline Trajectory Planning -- Geometric Tracking Control\n'
                 'Dense Urban Canyon Scenario',
                 color='white', fontsize=13, fontweight='bold', y=1.01)

    # ------------------------------------------------------------------- #
    #  5. Animation                                                         #
    # ------------------------------------------------------------------- #
    def update(frame):
        if frame >= N_frames:
            return trail3d, drone3d, trail2d, dot2d, time_txt, ln_speed, ln_alt, ln_err

        pos  = a_pos[frame]
        t_f  = a_t[frame]
        spd  = np.linalg.norm(a_vel[frame])
        alt  = pos[2]

        # tracking error against reference
        ref_idx = min(int(t_f / dt_ref_s2), len(ref_pos)-1)
        err = np.linalg.norm(pos - ref_pos[ref_idx])

        # --- trail indices ---
        lo = max(0, frame - trail_len)
        hi = frame + 1

        # --- 3D update ---
        trail3d.set_data(a_pos[lo:hi, 0], a_pos[lo:hi, 1])
        trail3d.set_3d_properties(a_pos[lo:hi, 2])

        drone3d._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

        time_txt.set_text(
            f't = {t_f:6.1f} s\n'
            f'pos ({pos[0]:+.0f}, {pos[1]:+.0f}, {pos[2]:.1f}) m\n'
            f'spd {spd:.1f} m/s   alt {alt:.1f} m\n'
            f'err {err:.2f} m'
        )

        # smooth camera rotation
        ax3d.view_init(elev=22 + 8*np.sin(t_f*0.03),
                       azim=-55 + t_f * 0.15)

        # --- 2D update ---
        trail2d.set_data(a_pos[lo:hi, 0], a_pos[lo:hi, 1])
        dot2d.set_data([pos[0]], [pos[1]])

        # --- telemetry update ---
        buf_t.append(t_f)
        buf_spd.append(spd)
        buf_alt.append(alt)
        buf_err.append(err * 5)   # scale for visibility

        if len(buf_t) > tel_len:
            buf_t.pop(0); buf_spd.pop(0); buf_alt.pop(0); buf_err.pop(0)

        ln_speed.set_data(buf_t, buf_spd)
        ln_alt.set_data(buf_t, buf_alt)
        ln_err.set_data(buf_t, buf_err)
        ax_tel.set_xlim(max(0, t_f - tel_len*dt_anim), t_f + 1)
        ax_tel.set_ylim(0, max(80, spd + 10, alt + 10))

        return trail3d, drone3d, trail2d, dot2d, time_txt, ln_speed, ln_alt, ln_err

    interval_ms = int(1000 / ANIM_FPS)
    # Store on fig to prevent garbage collection before plt.show() runs
    fig._anim = FuncAnimation(fig, update, frames=N_frames,
                              interval=interval_ms, blit=False, repeat=False)

    print(f"\n[Phase 3] Opening live GUI ({N_frames} frames @ {ANIM_FPS} fps, "
          f"speed x{SPEED_MULT})  --  close window to exit.\n")
    # Use subplots_adjust instead of tight_layout (avoids 3D axis warning)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.93, bottom=0.08,
                        hspace=0.38, wspace=0.32)
    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
