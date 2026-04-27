import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cusadi'))

import mujoco
import mujoco.viewer
import time
import numpy as np
import casadi as ca
from utils.utils_general import export_functions

def build_ilqr_cost(f_dynamics, l_stage, l_term, nx, nu, N, dt):
    """
    Compila una singola iterazione iLQR come ca.Function (compilabile con CusADi).

    Parametri
    ---------
    f_dynamics : ca.Function (x[nx], u[nu]) -> x_next[nx]
    l_stage    : ca.Function (x[nx], u[nu], xref_k[3]) -> scalar
    l_term     : ca.Function (x[nx], xref_N[3])        -> scalar
    nx, nu, N  : dimensioni stato, controllo, orizzonte
    dt         : timestep (non usato direttamente, serve per documentazione)

    Input della funzione compilata
    ------------------------------
    x0    : (nx,)      stato iniziale
    U     : (nu, N)    traiettoria controlli iniziale (warm start)
    xref  : (3, N+1)   riferimento posizione CoM

    Output
    ------
    U_new : (nu, N)    traiettoria controlli aggiornata
    """

    # ── simboli input della funzione finale ──
    x0_sym   = ca.SX.sym('x0',   nx)
    U_sym    = ca.SX.sym('U',    nu, N)
    xref_sym = ca.SX.sym('xref', 3,  N + 1)

    # ── simboli per calcolo gradienti costo ──
    x_s    = ca.SX.sym('x',      nx)
    u_s    = ca.SX.sym('u',      nu)
    xref_s = ca.SX.sym('xref_k', 3)

    # gradienti del costo stage calcolati simbolicamente una volta sola
    l_val = l_stage(x_s, u_s, xref_s)
    lx    = ca.jacobian(l_val, x_s).T   # (nx, 1)
    lu    = ca.jacobian(l_val, u_s).T   # (nu, 1)
    lxx   = ca.jacobian(lx,   x_s)     # (nx, nx)
    luu   = ca.jacobian(lu,   u_s)     # (nu, nu)
    lux   = ca.jacobian(lu,   x_s)     # (nu, nx)

    # gradienti del costo terminale
    lf_val = l_term(x_s, xref_s)
    lfx    = ca.jacobian(lf_val, x_s).T  # (nx, 1)
    lfxx   = ca.jacobian(lfx,   x_s)    # (nx, nx)

    # ════════════════════════════════════════════════
    # 1. FORWARD ROLLOUT
    #    dato (x0, U) calcola la traiettoria nominale X_nom
    # ════════════════════════════════════════════════
    X_nom = [x0_sym]
    xk = x0_sym
    for k in range(N):
        xk = f_dynamics(xk, U_sym[:, k])
        X_nom.append(xk)

    # ════════════════════════════════════════════════
    # 2. BACKWARD PASS
    #    calcola i guadagni k_ff (nu,) e K_fb (nu, nx)
    # ════════════════════════════════════════════════

    # value function terminale
    xu_xref_N = ca.vertcat(x_s, xref_s)
    xu_xref_N_nom = ca.vertcat(X_nom[N], xref_sym[:, N])
    Vx  = ca.substitute(lfx,  xu_xref_N, xu_xref_N_nom)
    Vxx = ca.substitute(lfxx, xu_xref_N, xu_xref_N_nom)

    k_ff_list = []
    K_fb_list = []

    # simboli freschi per la linearizzazione della dinamica
    x_lin = ca.SX.sym('x_lin', nx)
    u_lin = ca.SX.sym('u_lin', nu)
    f_eval = f_dynamics(x_lin, u_lin)
    A_sym  = ca.jacobian(f_eval, x_lin)  # (nx, nx) — calcolato una volta
    B_sym  = ca.jacobian(f_eval, u_lin)  # (nx, nu) — calcolato una volta

    for k in range(N - 1, -1, -1):
        xk_nom = X_nom[k]
        uk_nom = U_sym[:, k]
        xref_k = xref_sym[:, k]

        # linearizza dinamiche attorno al punto nominale
        xu_nom = ca.vertcat(xk_nom, uk_nom)
        A = ca.substitute(A_sym, ca.vertcat(x_lin, u_lin), xu_nom)
        B = ca.substitute(B_sym, ca.vertcat(x_lin, u_lin), xu_nom)

        # gradienti costo stage al punto nominale
        xu_xref = ca.vertcat(x_s, u_s, xref_s)
        xu_xref_nom = ca.vertcat(xk_nom, uk_nom, xref_k)
        lx_k  = ca.substitute(lx,  xu_xref, xu_xref_nom)
        lu_k  = ca.substitute(lu,  xu_xref, xu_xref_nom)
        lxx_k = ca.substitute(lxx, xu_xref, xu_xref_nom)
        luu_k = ca.substitute(luu, xu_xref, xu_xref_nom)
        lux_k = ca.substitute(lux, xu_xref, xu_xref_nom)

        # Q-function gradients
        Qx  = lx_k  + ca.mtimes(A.T, Vx)
        Qu  = lu_k  + ca.mtimes(B.T, Vx)
        Qxx = lxx_k + ca.mtimes([A.T, Vxx, A])
        Quu = luu_k + ca.mtimes([B.T, Vxx, B])
        Qux = lux_k + ca.mtimes([B.T, Vxx, A])

        # guadagni ottimi
        Quu_inv = ca.inv(Quu)
        k_ff = -ca.mtimes(Quu_inv, Qu)    # (nu,)
        K_fb = -ca.mtimes(Quu_inv, Qux)   # (nu, nx)

        k_ff_list.insert(0, k_ff)
        K_fb_list.insert(0, K_fb)

        # aggiorna value function (equazioni di Riccati)
        Vx  = Qx  + ca.mtimes(K_fb.T, ca.mtimes(Quu, k_ff)) \
                  + ca.mtimes(K_fb.T, Qu) \
                  + ca.mtimes(Qux.T, k_ff)
        Vxx = Qxx + ca.mtimes(K_fb.T, ca.mtimes(Quu, K_fb)) \
                  + ca.mtimes(K_fb.T, Qux) \
                  + ca.mtimes(Qux.T, K_fb)

    # ════════════════════════════════════════════════
    # 3. FORWARD PASS
    #    aggiorna U con i guadagni calcolati
    # ════════════════════════════════════════════════
    U_new_cols = []
    xk = x0_sym
    for k in range(N):
        dx     = xk - X_nom[k]
        uk_new = U_sym[:, k] + k_ff_list[k] + ca.mtimes(K_fb_list[k], dx)
        U_new_cols.append(uk_new)
        xk = f_dynamics(xk, uk_new)

    U_new = ca.horzcat(*U_new_cols)  # (nu, N)

    # ════════════════════════════════════════════════
    # 4. Compila come ca.Function
    # ════════════════════════════════════════════════
    f_ilqr = ca.Function(
        'mpc_f_ilqr',
        [x0_sym, U_sym, xref_sym],
        [U_new],
        ['x0', 'U', 'xref'],
        ['U_new'],
        {'cse': True, 'post_expand': True},
    )

    return f_ilqr

def build_ilqr(f_dynamics, nx, nu, N, dt,                                           


               Q, R, Qf,
               n_ilqr_iter=3):
    """
    f_dynamics : ca.Function (x, u) -> x_next  (già hai questa)
    nx, nu, N  : dimensioni
    Q, R, Qf   : np.ndarray pesi costo
    n_ilqr_iter: numero iterazioni iLQR (come NQP nel paper)
    """
    n = Q.shape[0]
    Q_full = np.zeros((nx, nx))
    Q_full[:n, :n] = Q

    Q  = ca.DM(Q_full)
    R  = ca.DM(R)
    Qf = ca.DM(Qf)

    # ── simboli input ──
    x0_sym  = ca.SX.sym('x0',  nx)
    U_sym   = ca.SX.sym('U',   nu, N)   # traiettoria controlli iniziale
    xref_sym = ca.SX.sym('xref', 3, N+1)  # traiettoria riferimento
    mass = 27.68978
    g = 9.81
    ref_u = ca.DM([0, 0, 0, 0, 0, mass*g/2, 0, 0, mass*g/2])

    # ════════════════════════════════════════════════
    # 1. FORWARD ROLLOUT
    #    dato x0 e U, calcola X nominale
    # ════════════════════════════════════════════════
    X_nom = [x0_sym]
    xk = x0_sym
    for k in range(N):
        uk = U_sym[:, k]
        xk = f_dynamics(xk, uk)
        X_nom.append(xk)
    # X_nom è lista di N+1 vettori SX

    # ════════════════════════════════════════════════
    # 2. BACKWARD PASS
    #    linearizza le dinamiche attorno a (X_nom, U)
    #    calcola i guadagni k_ff e K_fb
    # ════════════════════════════════════════════════

    # terminal value function
    eN = X_nom[N][0:3] - xref_sym[:, N]
    Vx_pos = 2 * ca.mtimes(Qf, eN)               # (3,)
    Vx  = ca.vertcat(Vx_pos, ca.SX.zeros(nx-3))  # (13,)
    Vxx_small = 2 * Qf                            # (3,3)
    Vxx = ca.blockcat([[Vxx_small,              ca.SX.zeros(3,  nx-3)],
                    [ca.SX.zeros(nx-3, 3),  ca.SX.zeros(nx-3, nx-3)]])  # (13,13)

    k_ff_list = []   # feedforward gains  (nu,)   per ogni step
    K_fb_list = []   # feedback gains     (nu, nx) per ogni step

    for k in range(N-1, -1, -1):
        xk_nom = X_nom[k]
        uk_nom = U_sym[:, k]
        ek_pos  = xk_nom[0:3] - xref_sym[:, k]
        ek_full = ca.vertcat(ek_pos, ca.SX.zeros(nx-3))

        # linearizza dinamiche: A = df/dx, B = df/du
        x_lin = ca.SX.sym('x_lin', nx)
        u_lin = ca.SX.sym('u_lin', nu)
        f_eval = f_dynamics(x_lin, u_lin)
        A = ca.substitute(ca.jacobian(f_eval, x_lin), ca.vertcat(x_lin, u_lin), ca.vertcat(xk_nom, uk_nom))
        B = ca.substitute(ca.jacobian(f_eval, u_lin), ca.vertcat(x_lin, u_lin), ca.vertcat(xk_nom, uk_nom))

        # expanded Q-function gradients
        Qx  = 2*ca.mtimes(Q, ek_full)          + ca.mtimes(A.T, Vx)
        
        Qu  = 2*ca.mtimes(R, uk_nom - ref_u) + ca.mtimes(B.T, Vx)
        Qxx = 2*Q                          + ca.mtimes([A.T, Vxx, A])
        Quu = 2*R                          + ca.mtimes([B.T, Vxx, B])
        Qux =                                ca.mtimes([B.T, Vxx, A])

        # gains
        Quu_inv = ca.inv(Quu)             # (nu, nu) — piccolo, ok
        k_ff = -ca.mtimes(Quu_inv, Qu)   # (nu,)
        K_fb = -ca.mtimes(Quu_inv, Qux)  # (nu, nx)

        k_ff_list.insert(0, k_ff)
        K_fb_list.insert(0, K_fb)

        # aggiorna value function
        Vx  = Qx  + ca.mtimes(K_fb.T, ca.mtimes(Quu, k_ff)) \
                  + ca.mtimes(K_fb.T, Qu) \
                  + ca.mtimes(Qux.T, k_ff)
        Vxx = Qxx + ca.mtimes(K_fb.T, ca.mtimes(Quu, K_fb)) \
                  + ca.mtimes(K_fb.T, Qux) \
                  + ca.mtimes(Qux.T, K_fb)

    # ════════════════════════════════════════════════
    # 3. FORWARD PASS
    #    aggiorna U con i guadagni calcolati
    # ════════════════════════════════════════════════
    alpha = 1.0   # line search fissa come nel paper
    U_new_cols = []
    xk = x0_sym
    for k in range(N):
        uk_nom  = U_sym[:, k]
        k_ff_k  = k_ff_list[k]
        K_fb_k  = K_fb_list[k]

        dx = xk - X_nom[k]   # deviazione dallo stato nominale
        uk_new = uk_nom + alpha * k_ff_k + ca.mtimes(K_fb_k, dx)
        U_new_cols.append(uk_new)

        xk = f_dynamics(xk, uk_new)

    U_new = ca.horzcat(*U_new_cols)   # (nu, N)

    # ════════════════════════════════════════════════
    # 4. compila la funzione  →  compilabile CusADi
    # ════════════════════════════════════════════════
    f_ilqr = ca.Function('mpc_f_ilqr_base',
        [x0_sym, U_sym, xref_sym],
        [U_new],
        ['x0', 'U', 'xref'],
        ['U_new']
    )

    return f_ilqr

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
    ilqr = build_ilqr(mpc['f_dynamics'], nx=13, nu=9, N=50, dt=0.002, Q=Q_pos, R=R_ctrl, Qf=Q_pos*5)
    
    x0_test   = np.zeros(13)
    x0_test[2] = 0.5  # altezza com
    U_test    = np.zeros((9, 50))
    xref_test = np.zeros((3, 51))
    xref_test[2, :] = 0.4  # riferimento a 0.4m di altezza

    U_new = np.array(ilqr(x0_test, U_test, xref_test))
    print("iLQR output shape:", U_new.shape)  # atteso (9, 50)
    print("iLQR primo controllo:", U_new[:, 0])
    
    x_s  = ca.SX.sym('x', 13)
    u_s  = ca.SX.sym('u', 9)
    xr_s = ca.SX.sym('xref_k', 3)

    mass = 27.68978
    g = 9.81
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

    e_pos     = x_s[0:3] - xr_s
    e_ctrl    = u_s - ref_u   # ref_u con forze gravità come in solve_mpc
    Q_ca      = ca.DM(Q_pos)
    R_ca      = ca.DM(R_ctrl)

    h_eq, g_soft, g_pos, _ = mpc['f_constraints'](x_s, u_s)
    fl_z = u_s[5]
    fr_z = u_s[8]
    w_eq   = 1
    w_soft = 1
    w_ineq = 1
    l_cstr = ca.Function('l_cstr', [x_s, u_s, xr_s],
        [w_eq   * ca.dot(h_eq,   h_eq)            # momento = 0
    + w_soft * ca.dot(g_soft, g_soft)         # cx≈px, cy≈py
    + w_ineq * (fl_z - ref_u[5]) ** 2          # fl_z >= 0
    + w_ineq * (fr_z - ref_u[8]) ** 2]) 

    l_traj = ca.Function('l_traj', [x_s, u_s, xr_s],
            [ca.mtimes([e_pos.T, Q_ca, e_pos]) +
            ca.mtimes([e_ctrl.T, R_ca, e_ctrl])])

    l_cstr = ca.Function('l_cstr', [x_s, u_s, xr_s],
        [
        #w_eq   * ca.dot(h_eq,   h_eq)            # momento = 0
        #+ w_soft * ca.dot(g_soft, g_soft)           # cx≈px, cy≈py
        + w_ineq * (fl_z - ref_u[5]) ** 2          # fl_z >= 0
        + w_ineq * (fr_z - ref_u[8]) ** 2
        ])        # fr_z >= 0

    # ── stage totale ──
    l_stage = ca.Function('l_stage', [x_s, u_s, xr_s],
        [l_traj(x_s, u_s, xr_s) + l_cstr(x_s, u_s, xr_s)])

    l_term  = ca.Function('l_term',  [x_s, xr_s],
                        [5 * ca.mtimes([e_pos.T, Q_ca, e_pos])])

    ilqr = build_ilqr_cost(mpc['f_dynamics'], l_stage, l_term, nx=13, nu=9, N=50, dt=0.002)
    nx = 13
    nu = 9
    N = 50
 
    U_new_2 = np.array(ilqr(x0_test, U_test, xref_test))
    print(f"U_ilqr_cost: {U_new_2[:, 0]}")  # stampa primo controllo per verifica

    diff_sol = U_new[:, 0] - U_new_2[:, 0]
    diff_norm = np.linalg.norm(diff_sol)
    print(f"diff solutions: {diff_norm} -> {diff_sol}")  # dovrebbe essere diverso da zero se il costo influenza la soluzione

    ##############

    export_functions(ilqr)
    export_functions(mpc['f_dynamics'])
    export_functions(mpc['f_constraints'])
    export_functions(mpc['f_rollout'])
    