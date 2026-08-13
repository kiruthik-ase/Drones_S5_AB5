import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from bspline_utils import get_bspline_matrices
import time

class UAVPlannerMILP:
    def __init__(self, h=10, ts=1.0):
        self.k = 4
        self.h = h
        self.ts = ts
        self.t_h = h * ts
        
        self.B, self.B_dot, self.B_ddot, self.B_dddot = get_bspline_matrices(self.k, self.h, num_points=self.h+1)
        self.n_ctrl = self.B.shape[1]
        
        self.v_max = 60.0
        self.a_max = 14.715
        
        self.w1 = 1.0    # Trajectory length weight
        self.w2 = 10.0   # Target approach weight
        self.w3 = 0.05   # Altitude weight
        self.w4 = 0.1    # Active knots penalty (J3) weight
        
        # Obstacles
        self.obstacles = []
        self.no_fly_zones = []
        self.M = 10000.0 # Big-M
        
    def add_cuboid_obstacle(self, xl, xu, yl, yu, zl, zu, eta=5.0):
        self.obstacles.append({'xl': xl, 'xu': xu, 'yl': yl, 'yu': yu, 'zl': zl, 'zu': zu, 'eta': eta})
        
    def add_no_fly_zone(self, xl, xu, yl, yu, gamma=20.0):
        self.no_fly_zones.append({'xl': xl, 'xu': xu, 'yl': yl, 'yu': yu, 'gamma': gamma})
        
    def plan(self, x_s, v_s, a_s, x_f):
        Q = cp.Variable((self.n_ctrl, 3))
        
        X = self.B @ Q
        V = (self.B_dot @ Q) / self.t_h
        A = (self.B_ddot @ Q) / (self.t_h ** 2)
        Jerk = (self.B_dddot @ Q) / (self.t_h ** 3)
        
        constraints = []
        
        # Initial Conditions
        constraints += [X[0, :] == x_s]
        constraints += [V[0, :] == v_s]
        constraints += [A[0, :] == a_s]
        
        # Dynamic constraints (max velocity, max acceleration) - Linearized for MILP
        for i in range(1, self.h + 1):
            constraints += [V[i, :] <= self.v_max]
            constraints += [V[i, :] >= -self.v_max]
            constraints += [A[i, :] <= self.a_max]
            constraints += [A[i, :] >= -self.a_max]
            
        # Z constraint (flat terrain avoidance, kappa = 1.0)
        constraints += [X[:, 2] >= 1.0]
        
        # Disjunctive constraints for 3D obstacles
        for obs in self.obstacles:
            xl, xu, yl, yu, zl, zu, eta = obs['xl'], obs['xu'], obs['yl'], obs['yu'], obs['zl'], obs['zu'], obs['eta']
            for i in range(1, self.h + 1):
                b = cp.Variable(6, boolean=True)
                constraints += [cp.sum(b) == 1]
                constraints += [X[i, 0] - (xl - eta) <= (1 - b[0]) * self.M]
                constraints += [(xu + eta) - X[i, 0] <= (1 - b[1]) * self.M]
                constraints += [X[i, 1] - (yl - eta) <= (1 - b[2]) * self.M]
                constraints += [(yu + eta) - X[i, 1] <= (1 - b[3]) * self.M]
                constraints += [X[i, 2] - (zl - eta) <= (1 - b[4]) * self.M]
                constraints += [(zu + eta) - X[i, 2] <= (1 - b[5]) * self.M]
                
        # Disjunctive constraints for 2D no-fly zones (extend infinitely in Z)
        for nfz in self.no_fly_zones:
            xl, xu, yl, yu, gamma = nfz['xl'], nfz['xu'], nfz['yl'], nfz['yu'], nfz['gamma']
            for i in range(1, self.h + 1):
                b = cp.Variable(4, boolean=True)
                constraints += [cp.sum(b) == 1]
                constraints += [X[i, 0] - (xl - gamma) <= (1 - b[0]) * self.M]
                constraints += [(xu + gamma) - X[i, 0] <= (1 - b[1]) * self.M]
                constraints += [X[i, 1] - (yl - gamma) <= (1 - b[2]) * self.M]
                constraints += [(yu + gamma) - X[i, 1] <= (1 - b[3]) * self.M]

        # Objectives - explicitly linearized
        dX = cp.Variable((self.h, 3))
        for i in range(self.h):
            constraints += [dX[i, :] >= X[i+1, :] - X[i, :]]
            constraints += [dX[i, :] >= -(X[i+1, :] - X[i, :])]
            
        d_term = cp.Variable(3)
        constraints += [d_term >= X[-1, :] - x_f]
        constraints += [d_term >= -(X[-1, :] - x_f)]
        
        J1 = self.w1 * cp.sum(dX) + self.w2 * cp.sum(d_term)
        
        J2 = self.w3 * cp.sum(X[1:, 2])
        
        dJerk = cp.Variable((self.h, 3))
        for i in range(self.h):
            constraints += [dJerk[i, :] >= Jerk[i+1, :] - Jerk[i, :]]
            constraints += [dJerk[i, :] >= -(Jerk[i+1, :] - Jerk[i, :])]
            
        J3 = self.w4 * cp.sum(dJerk) / self.ts
        
        objective = cp.Minimize(J1 + J2 + J3)
        
        prob = cp.Problem(objective, constraints)
        
        start_time = time.time()
        prob.solve(solver=cp.SCIPY)
        end_time = time.time()
        
        print(f"Optimization finished in {end_time - start_time:.4f}s with status: {prob.status}")
        
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return None, None, None, None
            
        return Q.value, X.value, V.value, A.value

def plot_cuboid(ax, xl, xu, yl, yu, zl, zu, color='b', alpha=0.5):
    # vertices
    x = [xl, xu, xu, xl, xl, xu, xu, xl]
    y = [yl, yl, yu, yu, yl, yl, yu, yu]
    z = [zl, zl, zl, zl, zu, zu, zu, zu]
    
    # plot sides
    faces = [
        [[xl,yl,zl],[xu,yl,zl],[xu,yu,zl],[xl,yu,zl]],
        [[xl,yl,zu],[xu,yl,zu],[xu,yu,zu],[xl,yu,zu]],
        [[xl,yl,zl],[xl,yu,zl],[xl,yu,zu],[xl,yl,zu]],
        [[xu,yl,zl],[xu,yu,zl],[xu,yu,zu],[xu,yl,zu]],
        [[xl,yl,zl],[xu,yl,zl],[xu,yl,zu],[xl,yl,zu]],
        [[xl,yu,zl],[xu,yu,zl],[xu,yu,zu],[xl,yu,zu]]
    ]
    
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='r', alpha=alpha))

if __name__ == "__main__":
    planner = UAVPlannerMILP(h=10, ts=1.0)
    
    # Scenario 2 from the paper
    x_s = np.array([3000.0, -2000.0, 10.0])
    v_s = np.array([-30.0, 30.0, 0.0]) # Direct roughly towards target
    a_s = np.array([0.0, 0.0, 0.0])
    x_f = np.array([-2000.0, 3000.0, 1.0])
    
    # Cuboidal obstacle
    planner.add_cuboid_obstacle(1600, 2400, -1400, -600, 0, 50, eta=5.0)
    
    # Flight prohibited zone
    planner.add_no_fly_zone(-1500, 2400, -600, 2000, gamma=20.0)
    
    print("Planning MILP trajectory...")
    Q, X, V, A = planner.plan(x_s, v_s, a_s, x_f)
    
    if X is not None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        ax.plot(X[:, 0], X[:, 1], X[:, 2], '-o', color='c', label='UAV Trajectory')
        ax.scatter(*x_s, color='g', s=100, label='Start')
        ax.scatter(*x_f, color='r', s=100, label='Target')
        
        # Plot obstacle
        plot_cuboid(ax, 1600, 2400, -1400, -600, 0, 50, color='b')
        # Plot no-fly zone (plot with small height for visualization)
        plot_cuboid(ax, -1500, 2400, -600, 2000, 0, 10, color='r', alpha=0.3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-3000, 4000])
        ax.set_ylim([-3000, 4000])
        ax.set_zlim([0, 100])
        ax.legend()
        plt.title('B-Spline Trajectory (MILP Obstacle Test)')
        plt.savefig('milp_test.png')
        print("Plot saved to milp_test.png")
