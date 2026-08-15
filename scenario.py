"""
Scenario Definition
===================
Urban Canyon -- Corridor Navigation Scenario

Key design principles:
  - All buildings TALLER than z_max (25m ceiling) so the drone CANNOT fly over
  - Buildings arranged as a chicane/slalom course forcing lateral weaving
  - Two no-fly zones create additional lateral pressure
  - Start: [2200, -1500, 12]   Goal: [-2000, 2000, 12]

This forces the planner to solve a genuine 2D lateral maze, demonstrating
the disjunctive programming obstacle avoidance clearly.
"""
import numpy as np


# Shared altitude ceiling -- must match planner z_max
Z_CEILING = 25.0   # metres


def build_scenario(planner):
    """
    Add all obstacles and no-fly zones.
    Returns (x_s, v_s, a_s, x_f).
    """

    # ---------- Start / Goal ----------
    x_s = np.array([ 2200.0, -1500.0, 12.0])
    v_s = np.array([-20.0,    15.0,    0.0])
    a_s = np.array([  0.0,     0.0,    0.0])
    x_f = np.array([-2000.0,  2000.0, 12.0])

    bh = 200.0   # building height -- well above z_ceiling (25m)
    e  = 10.0    # safety eta (standoff distance around each building)

    # ---------- Chicane layout ----------
    # The buildings are deliberately arranged in pairs that create narrow
    # lateral gaps (~300-400 m wide) the drone must thread through.
    # All buildings are 400x400 m in footprint.

    # Pair 1 -- forces the drone to pass SOUTH of centre
    planner.add_cuboid_obstacle( 1400,  1800,  -800,  -400, 0, bh, eta=e)   # C1a (north block)
    planner.add_cuboid_obstacle( 1400,  1800,   200,   600, 0, bh, eta=e)   # C1b (south block)

    # Pair 2 -- forces the drone to pass NORTH of centre
    planner.add_cuboid_obstacle(  400,   800, -1200,  -800, 0, bh, eta=e)   # C2a (south block)
    planner.add_cuboid_obstacle(  400,   800,   100,   500, 0, bh, eta=e)   # C2b (north block)

    # Pair 3 -- diagonal chicane, forces diagonal crossing
    planner.add_cuboid_obstacle( -400,     0,  -700,  -300, 0, bh, eta=e)   # C3a
    planner.add_cuboid_obstacle( -400,     0,   400,   800, 0, bh, eta=e)   # C3b

    # Pair 4 -- final approach gate
    planner.add_cuboid_obstacle(-1400, -1000,  -600,  -200, 0, bh, eta=e)   # C4a
    planner.add_cuboid_obstacle(-1400, -1000,   800,  1200, 0, bh, eta=e)   # C4b

    # ---------- No-Fly Zones (restrict upper approach options) ----------
    # NFZ1: blocks the northern shortcut
    planner.add_no_fly_zone(1800, 2400, -200, 1200, gamma=15.0)
    # NFZ2: blocks the southern shortcut in mid-field
    planner.add_no_fly_zone(-800,  600, -1500, -900, gamma=15.0)

    return x_s, v_s, a_s, x_f
