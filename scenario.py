"""
Scenario Definition -- Mixed-Height Urban Canyon
=================================================

A realistic city environment with TWO types of obstacles:

  TALL buildings (200m) -- above altitude ceiling (25m):
    The drone CANNOT fly over. Must navigate laterally (left/right).

  SHORT buildings (15m) -- below altitude ceiling:
    zu + eta = 15 + 5 = 20m, which is feasible since z_max = 25m.
    The drone CAN fly over by climbing to z >= 20m.
    The MILP will choose this if going around costs more (wide walls).

Layout: Alternating TALL chicane gates + SHORT cross-corridor hurdles,
forcing both lateral and vertical manoeuvres.

Start: [2500, -1500, 12]   Goal: [-2200, 2000, 12]
Path line: y = 200 - 0.8 * x  (approximate)
"""
import numpy as np
from dynamic_obstacle import DynamicObstacle


Z_CEILING = 25.0   # metres -- planner hard ceiling


def build_scenario(planner):
    """
    Register all obstacles and no-fly zones.
    Returns (x_s, v_s, a_s, x_f).
    """

    x_s = np.array([ 2500.0, -1400.0, 12.0])   # start between Gate-A buildings
    v_s = np.array([-25.0,    10.0,    0.0])   # initial velocity toward Gate-A gap
    a_s = np.array([  0.0,     0.0,    0.0])
    x_f = np.array([-2200.0,  2000.0, 12.0])

    # path runs y ≈ 200 - 0.8*x
    bh_tall = 200.0   # tall -- drone CANNOT fly over (zu >> z_ceiling)
    bh_low  =  15.0   # short -- drone CAN fly over (zu + eta = 20 <= z_max=25)
    e_tall  =  80.0   # large safety standoff for tall buildings
    e_low   =   5.0   # small standoff for short buildings

    # ================================================================== #
    #  TALL GATE A  -- x ~ 2000                                           #
    #  Lateral chicane: drone threads through 900m N-S gap               #
    # ================================================================== #
    #   path at x=2000: y = 200 - 0.8*2000 = -1400
    planner.add_cuboid_obstacle(1800, 2200, -2100, -1700, 0, bh_tall, eta=e_tall)  # A-south
    planner.add_cuboid_obstacle(1800, 2200,  -800,  -400, 0, bh_tall, eta=e_tall)  # A-north

    # ================================================================== #
    #  SHORT HURDLE 1  -- x ~ 1500                                        #
    #  Wide low wall spanning 2200m in Y -- drone must CLIMB OVER         #
    # ================================================================== #
    #   path at x=1500: y = 200 - 0.8*1500 = -1000
    #   wall yl=-2000, yu=200 → blocks lateral escape → must fly z>=20
    planner.add_cuboid_obstacle(1350, 1650, -2000,  200, 0, bh_low, eta=e_low)     # hurdle-1

    # ================================================================== #
    #  TALL GATE B  -- x ~ 1000                                           #
    #  Narrower lateral gap forces a zigzag                               #
    # ================================================================== #
    #   path at x=1000: y = 200 - 0.8*1000 = -600
    planner.add_cuboid_obstacle( 800, 1200, -1200,  -800, 0, bh_tall, eta=e_tall)  # B-south
    planner.add_cuboid_obstacle( 800, 1200,  -250,   150, 0, bh_tall, eta=e_tall)  # B-north

    # ================================================================== #
    #  SHORT HURDLE 2  -- x ~ 400                                         #
    # ================================================================== #
    #   path at x=400: y = 200 - 0.8*400 = -120
    planner.add_cuboid_obstacle( 250,  550, -1200,  1000, 0, bh_low, eta=e_low)    # hurdle-2

    # ================================================================== #
    #  TALL GATE C  -- x ~ -200                                           #
    # ================================================================== #
    #   path at x=-200: y = 200 - 0.8*(-200) = 360
    planner.add_cuboid_obstacle(-400,    0,   -50,   200, 0, bh_tall, eta=e_tall)  # C-south
    planner.add_cuboid_obstacle(-400,    0,   520,   820, 0, bh_tall, eta=e_tall)  # C-north

    # ================================================================== #
    #  SHORT HURDLE 3  -- x ~ -800                                        #
    # ================================================================== #
    #   path at x=-800: y = 200 - 0.8*(-800) = 840
    planner.add_cuboid_obstacle(-950, -650,  -300,  1700, 0, bh_low, eta=e_low)    # hurdle-3

    # ================================================================== #
    #  TALL GATE D  -- x ~ -1600  (final approach)                        #
    # ================================================================== #
    #   path at x=-1600: y = 200 - 0.8*(-1600) = 1480
    planner.add_cuboid_obstacle(-1800,-1400,  1000,  1300, 0, bh_tall, eta=e_tall) # D-south
    planner.add_cuboid_obstacle(-1800,-1400,  1700,  2000, 0, bh_tall, eta=e_tall) # D-north

    # ================================================================== #
    #  NO-FLY ZONES  (3 zones -- all clear of start/goal positions)       #
    # ================================================================== #
    #  NFZ-A: south of Gate-B, prevents trivial south escape
    planner.add_no_fly_zone(  500,  1300, -2300, -1600, gamma=30.0)   # NFZ-A
    #  NFZ-B: north of Gate-C, forces drone through Gate-C gap
    planner.add_no_fly_zone( -500,   300,  700,  1200, gamma=30.0)   # NFZ-B
    #  NFZ-C: east of final gate, prevents trivial east escape near goal
    planner.add_no_fly_zone(-2100, -1200,  1600,  2100, gamma=30.0)   # NFZ-C

    return x_s, v_s, a_s, x_f


def create_dynamic_obstacle():
    """
    Intruder UAV that crosses the drone's route, forcing online replanning.
    """
    return DynamicObstacle(
        pos0          = [2200.0, -2000.0, 12.0],
        velocity      = [-50.0,   90.0,   0.0],
        radius        = 200.0,
        safety_margin = 50.0,
    )
