import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from bspline_utils import get_bspline_matrices
import time

class UAVPlanner:
    def __init__(self, h=10, ts=1.0):
        self.k = 4
        self.h = h
        self.ts = ts
        self.t_h = h * ts
        
        self.B, self.B_dot, self.B_ddot, self.B_dddot = get_bspline_matrices(self.k, self.h, num_points=self.h+1)
        
        self.n_ctrl = self.B.shape[1]
        
        # Vehicle Params
        self.v_max = 60.0
        self.a_max = 14.715
        
        # Objective weights
        self.w1 = 1.0    # Trajectory length weight
        self.w2 = 10.0   # Target approach weight
        self.w3 = 0.05   # Altitude weight
        self.w4 = 0.1    # Active knots penalty (J3) weight
        
    def plan(self, x_s, v_s, a_s, x_f):
        # Variables
        Q = cp.Variable((self.n_ctrl, 3))
        
        # State trajectories
        X = self.B @ Q
        V = (self.B_dot @ Q) / self.t_h
        A = (self.B_ddot @ Q) / (self.t_h ** 2)
        Jerk = (self.B_dddot @ Q) / (self.t_h ** 3)
        
        constraints = []
        
        # Initial Conditions
        constraints += [X[0, :] == x_s]
        constraints += [V[0, :] == v_s]
        constraints += [A[0, :] == a_s]
        
        # Dynamic constraints (max velocity, max acceleration)
        for i in range(1, self.h + 1):
            constraints += [cp.norm(V[i, :], 2) <= self.v_max]
            constraints += [cp.norm(A[i, :], 2) <= self.a_max]
            
        # Z constraint (don't go below ground)
        constraints += [X[:, 2] >= 0]
            
        # Objectives
        # J1: Path length + terminal distance
        J1 = self.w1 * cp.sum([cp.norm(X[i+1, :] - X[i, :], 2) for i in range(self.h)]) 
        J1 += self.w2 * cp.norm(X[-1, :] - x_f, 2)
        
        # J2: Altitude minimization
        J2 = self.w3 * cp.sum(X[1:, 2])
        
        # J3: Active knots penalty (1-norm of 3rd derivative jumps)
        # J3 = sum( |Jerk_i+1 - Jerk_i| ) / Ts
        J3 = 0
        for i in range(self.h):
            J3 += cp.sum(cp.abs(Jerk[i+1, :] - Jerk[i, :])) / self.ts
        J3 = self.w4 * J3
        
        objective = cp.Minimize(J1 + J2 + J3)
        
        # We start with SOCP (ECOS/SCS) since no binary variables yet.
        # When we add Big-M, we'll switch to a MILP solver (e.g. SCIP or CBC)
        prob = cp.Problem(objective, constraints)
        
        start_time = time.time()
        # ECOS is default for SOCP
        prob.solve() 
        end_time = time.time()
        
        print(f"Optimization finished in {end_time - start_time:.4f}s with status: {prob.status}")
        
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return None, None, None, None
            
        return Q.value, X.value, V.value, A.value

if __name__ == "__main__":
    planner = UAVPlanner(h=10, ts=1.0)
    
    # Simple start to goal
    x_s = np.array([-2000.0, -2000.0, 10.0])
    v_s = np.array([30.0, 30.0, 0.0])
    a_s = np.array([0.0, 0.0, 0.0])
    x_f = np.array([715.0, 1730.0, 4.0])
    
    print("Planning trajectory...")
    Q, X, V, A = planner.plan(x_s, v_s, a_s, x_f)
    
    if X is not None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(X[:, 0], X[:, 1], X[:, 2], '-o', label='UAV Trajectory')
        ax.scatter(*x_s, color='g', s=100, label='Start')
        ax.scatter(*x_f, color='r', s=100, label='Target')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title('B-Spline Trajectory (Kinematic Test)')
        plt.savefig('kinematic_test.png')
        print("Plot saved to kinematic_test.png")
