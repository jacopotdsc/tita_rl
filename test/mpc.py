import mujoco
import mujoco.viewer
import time
import numpy as np
import casadi as ca
from utils.utils_general import export_functions

def build_mpc(N=50, dt=0.002, mass=27.68978, g=9.81,
              Q_pos=None, R_ctrl=None):
 
    nx = 13
    nu = 9
    com_feet_distance = 0.5706
 
    # ════════════════════════════════════════════════════════════════════
    # 1.  f_dynamics:  (x, u) → x_next          ← compilabile con CusADi
    # ════════════════════════════════════════════════════════════════════
    x_sym = ca.SX.sym('x', nx)
    u_sym = ca.SX.sym('u', nu)
 
    p_com = x_sym[0:3]
    v_com = x_sym[3:6]
    c     = x_sym[6:9]
    v_cz  = x_sym[9]
    theta = x_sym[10]
    v     = x_sym[11]
    omega = x_sym[12]
 
    a_    = u_sym[0]
    a_cz  = u_sym[1]
    alpha = u_sym[2]
    fl    = u_sym[3:6]
    fr    = u_sym[6:9]
 
    acc_com      = (1.0 / mass) * (fl + fr) + ca.vertcat(0, 0, -g)
    vel_com_next = v_com + acc_com * dt
    pos_com_next = p_com + v_com * dt
 
    cx_next    = c[0] + v * ca.cos(theta) * dt
    cy_next    = c[1] + v * ca.sin(theta) * dt
    cz_next    = c[2] + v_cz * dt
    
    v_cz_next  = v_cz + a_cz * dt
    theta_next = theta + omega * dt
    v_next     = v + a_ * dt
    omega_next = omega + alpha * dt
 
    x_next = ca.vertcat(
        pos_com_next, 
        vel_com_next,
        cx_next, 
        cy_next,
        cz_next,
        v_cz_next, 
        theta_next, 
        v_next, 
        omega_next
    )
 
    f_dynamics = ca.Function('mpc_f_dynamics',
                             [x_sym, u_sym], [x_next],
                             ['x', 'u'], ['x_next'])
 
    # ════════════════════════════════════════════════════════════════════
    # 2.  f_constraints:  (x, u) → (h_eq, g_ineq)   ← compilabile
    # ════════════════════════════════════════════════════════════════════
    pcom_s  = x_sym[0:3]
    c_s     = x_sym[6:9]
    theta_s = x_sym[10]
    fl_s    = u_sym[3:6]
    fr_s    = u_sym[6:9]
 
    Rz = ca.vertcat(
        ca.horzcat(ca.cos(theta_s), -ca.sin(theta_s), 0),
        ca.horzcat(ca.sin(theta_s),  ca.cos(theta_s), 0),
        ca.horzcat(0,                0,                1)
    )
    left_cp  = c_s + ca.mtimes(Rz, ca.vertcat(0, com_feet_distance / 2, 0))
    right_cp = c_s - ca.mtimes(Rz, ca.vertcat(0, com_feet_distance / 2, 0))
 
    moment_l = ca.cross(left_cp - pcom_s, fl_s)
    moment_r = ca.cross(right_cp - pcom_s, fr_s)
 
    h_eq = ca.vertcat(
        moment_l + moment_r,      # (3,)  momento = 0
        x_sym[9],                 # (1,)  v_cz = 0
        #x_sym[6] - x_sym[0],      # (1,)  cx = px
        #x_sym[7] - x_sym[1],      # (1,)  cy = py
    )

    g_soft = ca.vertcat(
        x_sym[6] - x_sym[0],      # (1,)  abs(cx - px) ≤ ε  →  soft constraint
        x_sym[7] - x_sym[1],      # (1,)  abs(cy - py) ≤ ε  →  soft constraint
    )
 
    g_pos_ineq = ca.vertcat(
        u_sym[5], 
        u_sym[8], 
    )

    g_neg_ineq = ca.vertcat()
 
    f_constraints = ca.Function('mpc_f_constraints',
                                [x_sym, u_sym], [h_eq, g_soft, g_pos_ineq, g_neg_ineq],
                                ['x', 'u'], ['h_eq', 'g_soft', 'g_pos_ineq', 'g_neg_ineq'])
 
    # ════════════════════════════════════════════════════════════════════
    # 3.  NLP:  single-shooting rollout, tutto in SX
    # ════════════════════════════════════════════════════════════════════
 
    # Decision variable: tutti i controlli flat
    U_flat = ca.SX.sym('U', nu * N)
    a_max = 5.0
    acz_max = 5.0
    alpha_max = 5.0
    fx_max = np.inf
    fy_max = np.inf
    fz_max = np.inf
    lbu_a = np.array([-a_max, -acz_max, -alpha_max, -fx_max, -fy_max, -fz_max, -fx_max, -fy_max, -fz_max] )  # a, acz, alpha,
    ubu_a = np.array([ a_max,  acz_max,  alpha_max, fx_max, fy_max, fz_max, fx_max, fy_max, fz_max] )
 
    # Parametri: [x0 (13) | x_ref flattened column-major (3*(N+1))]
    n_params = nx + 3 * (N + 1)
    p_sym    = ca.SX.sym('p', n_params)
    x0_sym   = p_sym[:nx]
 
    # Weights
    Q_pos = ca.DM(Q_pos)
    R_ctrl = ca.DM(R_ctrl)
 
    # ── Rollout + cost + constraints ──
    cost       = 0
    xk         = x0_sym
    ref_u = ca.vertcat(
        0,                  # a
        0,                  # a_cz
        0,                  # alpha
        0,                  # fl_x
        0,                  # fl_y
        mass * g / 2.0,     # fl_z
        0,                  # fr_x
        0,                  # fr_y
        mass * g / 2.0,     # fr_z
    )

    all_h_eq   = []
    all_g_soft = []
    all_g_pos_ineq = []
    all_g_neg_ineq = []
 
    for k in range(N):
        uk      = U_flat[k * nu : (k + 1) * nu]
        ref_k   = p_sym[nx + k * 3 : nx + (k + 1) * 3]
 
        # Cost
        e_pos = xk[0:3] - ref_k
        e_control = uk - ref_u
        cost += ca.mtimes([e_pos.T, Q_pos, e_pos])
        cost += ca.mtimes([e_control.T, R_ctrl, e_control])
 
        # Constraints at step k
        h_k, g_soft_k, g_pos_k, g_neg_k = f_constraints(xk, uk)
        all_h_eq.append(h_k)
        all_g_soft.append(g_soft_k)
        all_g_pos_ineq.append(g_pos_k)
        all_g_neg_ineq.append(g_neg_k)
 
        # Propagate
        xk = f_dynamics(xk, uk)
 
    # Terminal cost
    ref_N = p_sym[nx + N * 3 : nx + (N + 1) * 3]
    e_N   = xk[0:3] - ref_N
    cost += 5.0 * ca.mtimes([e_N.T, Q_pos, e_N])
 
    # Assemble constraints:  equalities first, then inequalities
    g_eq_all   = ca.vertcat(*all_h_eq)
    g_soft = ca.vertcat(*all_g_soft)    
    g_pos_ineq_all = ca.vertcat(*all_g_pos_ineq)
    g_neg_ineq_all = ca.vertcat(*all_g_neg_ineq)

    g_ineq_all = ca.vertcat(g_pos_ineq_all, g_neg_ineq_all)
    g_all      = ca.vertcat(g_eq_all, g_soft, g_ineq_all)
 
    n_eq   = g_eq_all.shape[0]
    n_soft = g_soft.shape[0]
    n_pos_ineq = g_pos_ineq_all.shape[0]
    n_neg_ineq = g_neg_ineq_all.shape[0]
 
    # Constraint bounds
    cnstr_eq = np.ones(n_eq) * 0.1      # h = 0 ( actually modelled as soft )
    cnstr_soft = np.ones(n_soft) * 0.1  # abs(g_soft) ≤ ε

    lbg = np.concatenate([  -cnstr_eq,                          #  0  ≤  h  ≤  0    →  h = 0
                            -cnstr_soft,                       # -ε ≤  g  ≤  ε    →  soft constraint
                            np.zeros(n_pos_ineq),              #  0  ≤  g  ≤  +∞   →  g ≥ 0
                            -np.inf * np.ones(n_neg_ineq)] )   # -∞  ≤  g  ≤  0    →  g ≤ 0

    ubg = np.concatenate([  cnstr_eq,                    #  ↑ uguaglianza
                            cnstr_soft,                         #  ↑ soft constraint
                            np.inf * np.ones(n_pos_ineq),      #  ↑ positivo (≥ 0)
                            np.zeros(n_neg_ineq)])             #  ↑ negativo (≤ 0)
 
    # ════════════════════════════════════════════════════════════════════
    # 4.  Solver (IPOPT su CPU)
    # ════════════════════════════════════════════════════════════════════
    nlp  = {'x': U_flat, 'f': cost, 'g': g_all, 'p': p_sym}
    opts = {
        'ipopt.max_iter':     200,
        'ipopt.print_level':  0,
        'ipopt.sb':           'yes',
        'print_time':         0,
    }
    solver = ca.nlpsol('mpc', 'ipopt', nlp, opts)
 
    # ════════════════════════════════════════════════════════════════════
    # 5.  f_rollout:  (x0, U_flat) → X_flat     ← compilabile
    #     Per ricostruire la traiettoria dopo il solve
    # ════════════════════════════════════════════════════════════════════
    x0_r    = ca.SX.sym('x0r', nx)
    U_flat_r = ca.SX.sym('Ur', nu * N)
    xk_r    = x0_r
    X_cols  = [x0_r]
    for k in range(N):
        uk_r = U_flat_r[k * nu : (k + 1) * nu]
        xk_r = f_dynamics(xk_r, uk_r)
        X_cols.append(xk_r)
    X_full = ca.horzcat(*X_cols)               # (13, N+1)
 
    f_rollout = ca.Function('mpc_f_rollout',
                            [x0_r, U_flat_r], [X_full],
                            ['x0', 'U'], ['X'])
 
    # ════════════════════════════════════════════════════════════════════
    # 6.  Return
    # ════════════════════════════════════════════════════════════════════
    return {
        'solver':        solver,
        'f_dynamics':    f_dynamics,
        'f_constraints': f_constraints,
        'f_rollout':     f_rollout,
        'lbg':           lbg,
        'ubg':           ubg,
        'lbu':           np.tile(lbu_a, N),
        'ubu':           np.tile(ubu_a, N),
        'dims':          {'nx': nx, 'nu': nu, 'N': N,
                          'n_eq': n_eq, 'n_soft': n_soft, 'n_pos_ineq': n_pos_ineq, 'n_neg_ineq': n_neg_ineq},
    }
 
def solve_mpc(mpc, x0_val, xref_val, U_warm=None):
    """
    Parameters
    ----------
    mpc      : dict from build_mpc()
    x0_val   : np.ndarray (13,)       stato corrente
    xref_val : np.ndarray (3, N+1)    traiettoria desiderata [px, py, pz]
    U_warm   : np.ndarray (9, N)      warm start (opzionale)
 
    Returns
    -------
    U_opt    : np.ndarray (9, N)      controlli ottimi
    X_opt    : np.ndarray (13, N+1)   traiettoria risultante
    """
    nu = mpc['dims']['nu']
    N = mpc['dims']['N']
 
    # Pack parameters:  [x0 | x_ref column-major]
    p_val = np.concatenate([x0_val, xref_val.flatten(order='F')])
 
    # Initial guess
    if U_warm is not None:
        u0 = U_warm.flatten(order='F')    # (nu*N,)
    else:
        u0 = np.zeros(nu * N)
 
    # Solve
    try:
        sol = mpc['solver'](x0=u0, p=p_val,
                            lbg=mpc['lbg'], ubg=mpc['ubg'],
                            lbx=mpc['lbu'], ubx=mpc['ubu'])
        U_opt_flat = np.asarray(sol['x']).flatten()
    except RuntimeError as e:
        raise RuntimeError(f"MPC infeasible: {e}")
 
    # Unpack U: (nu*N,) → (nu, N)
    U_opt = U_opt_flat.reshape(N, nu).T
 
    # Rollout X
    X_opt = np.asarray(mpc['f_rollout'](x0_val, U_opt_flat))
 
    return U_opt, X_opt
 
if __name__ == "__main__":
    a_weight     = 1
    acz_weight   = 0.01
    alpha_weight = 0.0001
    fx_weight    = 0.001
    fy_weight    = 0.001
    fz_weight    = 0.0001
    R_ctrl = np.diag([a_weight, acz_weight, alpha_weight, fx_weight, fy_weight, fz_weight, fx_weight, fy_weight, fz_weight])
    Q_pos = np.diag([1, 1, 1])
    mpc = build_mpc(Q_pos=Q_pos, R_ctrl=R_ctrl)
    
    export_functions(mpc['f_dynamics'])
    export_functions(mpc['f_constraints'])
    export_functions(mpc['f_rollout'])
    