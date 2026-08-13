import numpy as np

def generate_knot_vector(k, h):
    n_internal = h - 1
    knots = [0.0] * k
    if n_internal > 0:
        internal = np.linspace(0, 1, h + 1)[1:-1].tolist()
        knots.extend(internal)
    knots.extend([1.0] * k)
    return np.array(knots)

def cox_de_boor(t, i, k, knots):
    if k == 1:
        if (knots[i] <= t < knots[i+1]) or (t == knots[-1] and knots[i] <= t <= knots[i+1] and knots[i] < knots[i+1]):
            return 1.0
        return 0.0
    
    denom1 = knots[i+k-1] - knots[i]
    term1 = 0.0
    if denom1 > 0:
        term1 = ((t - knots[i]) / denom1) * cox_de_boor(t, i, k-1, knots)
        
    denom2 = knots[i+k] - knots[i+1]
    term2 = 0.0
    if denom2 > 0:
        term2 = ((knots[i+k] - t) / denom2) * cox_de_boor(t, i+1, k-1, knots)
        
    return term1 + term2

def cox_de_boor_derivative(t, i, k, d, knots):
    if d == 0:
        return cox_de_boor(t, i, k, knots)
    
    denom1 = knots[i+k-1] - knots[i]
    term1 = 0.0
    if denom1 > 0:
        term1 = (k - 1) / denom1 * cox_de_boor_derivative(t, i, k-1, d-1, knots)
        
    denom2 = knots[i+k] - knots[i+1]
    term2 = 0.0
    if denom2 > 0:
        term2 = (k - 1) / denom2 * cox_de_boor_derivative(t, i+1, k-1, d-1, knots)
        
    return term1 - term2

def get_bspline_matrices(k, h, num_points=None):
    knots = generate_knot_vector(k, h)
    n = len(knots) - k - 1
    if num_points is None:
        num_points = h + 1
        
    t_vals = np.linspace(0, 1, num_points)
    
    B = np.zeros((num_points, n + 1))
    B_dot = np.zeros((num_points, n + 1))
    B_ddot = np.zeros((num_points, n + 1))
    B_dddot = np.zeros((num_points, n + 1))
    
    for row, t in enumerate(t_vals):
        for i in range(n + 1):
            B[row, i] = cox_de_boor(t, i, k, knots)
            B_dot[row, i] = cox_de_boor_derivative(t, i, k, 1, knots)
            B_ddot[row, i] = cox_de_boor_derivative(t, i, k, 2, knots)
            B_dddot[row, i] = cox_de_boor_derivative(t, i, k, 3, knots)
            
    return B, B_dot, B_ddot, B_dddot

def sample_trajectory(t_phy, Q, k, h, t_h):
    """
    Samples the B-spline at physical time t_phy.
    t_phy: float in [0, t_h]
    Q: (n+1, 3) control points
    """
    t = t_phy / t_h
    t = np.clip(t, 0.0, 1.0)
    
    knots = generate_knot_vector(k, h)
    n_ctrl = len(Q)
    
    p = np.zeros(3)
    v = np.zeros(3)
    a = np.zeros(3)
    
    for i in range(n_ctrl):
        b = cox_de_boor(t, i, k, knots)
        b_dot = cox_de_boor_derivative(t, i, k, 1, knots)
        b_ddot = cox_de_boor_derivative(t, i, k, 2, knots)
        
        p += Q[i] * b
        v += Q[i] * b_dot
        a += Q[i] * b_ddot
        
    v = v / t_h
    a = a / (t_h ** 2)
    
    return p, v, a
