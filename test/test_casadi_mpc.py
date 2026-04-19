import mujoco
import mujoco.viewer
import time
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from trajectory_generator import TrajectoryGenerator
from global_variables import plot_generated_trajectory

from dataclasses import dataclass, field
import numpy as np

@dataclass
class EE3:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acc: np.ndarray = field(default_factory=lambda: np.zeros(3))

@dataclass
class EERot:
    pos: np.ndarray = field(default_factory=lambda: np.eye(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acc: np.ndarray = field(default_factory=lambda: np.zeros(3))

@dataclass
class EE6:
    pos: np.ndarray = field(default_factory=lambda: np.eye(4))     # SE3 → 4x4
    vel: np.ndarray = field(default_factory=lambda: np.zeros(6))
    acc: np.ndarray = field(default_factory=lambda: np.zeros(6))

@dataclass
class DesiredConfiguration:
    position:         np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation:      np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0], dtype=float))  # wxyz
    linear_velocity:  np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    qjnt:             np.ndarray = field(default_factory=lambda: np.zeros(0))
    qjntdot:          np.ndarray = field(default_factory=lambda: np.zeros(0))
    qjntddot:         np.ndarray = field(default_factory=lambda: np.zeros(0))
    com:              EE3   = field(default_factory=EE3)
    lwheel:           EE6   = field(default_factory=EE6)
    rwheel:           EE6   = field(default_factory=EE6)
    base_link:        EERot = field(default_factory=EERot)
    in_contact:       bool  = True

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
 
    f_dynamics = ca.Function('f_dynamics',
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
 
    f_constraints = ca.Function('f_constraints',
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
    '''
    if Q_pos is None:
        Q_pos = np.diag([1, 1, 1])
    if R_ctrl is None:
        a_weight     = 0.01
        acz_weight   = 0.01
        alpha_weight = 0.0001
        fx_weight    = 0.001
        fy_weight    = 0.001
        fz_weight    = 0.0001
        R_ctrl = np.diag([a_weight, acz_weight, alpha_weight, fx_weight, fy_weight, fz_weight, fx_weight, fy_weight, fz_weight])
    '''
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
 
    f_rollout = ca.Function('f_rollout',
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
 
def export_functions(mpc, out_dir='./casadi_functions'):
    """Salva ogni ca.Function come .casadi per CusADi."""
    import os
    os.makedirs(out_dir, exist_ok=True)
 
    for name in ['f_dynamics', 'f_constraints', 'f_rollout']:
        fn   = mpc[name]
        path = os.path.join(out_dir, f'{name}.casadi')
        fn.save(path)
        print(f"Saved: {path}  (n_instructions: {fn.n_instructions()})")

def compute_desired_joint_acc(q_pos, q_vel, q_pos_target, kp=80.0, kd=8.0):
    """PD in joint space → desired q_ddot to feed into WBC."""
    err   = q_pos_target - q_pos
    err   = np.arctan2(np.sin(err), np.cos(err))  # wrap
    return kp * err - kd * q_vel

def plot_mpc_horizon(X_opt, U_opt, dt=0.002):

    xs     = X_opt[0, :];   ys     = X_opt[1, :];   zs     = X_opt[2, :]
    vxs    = X_opt[3, :];   vys    = X_opt[4, :];   vzs    = X_opt[5, :]
    cxs    = X_opt[6, :];   cys    = X_opt[7, :]
    vcz    = X_opt[9, :]
    theta  = X_opt[10, :]
    vs     = X_opt[11, :]
    omegas = X_opt[12, :]

    fl_x = U_opt[3, :];  fl_y = U_opt[4, :];  fl_z = U_opt[5, :]
    fr_x = U_opt[6, :];  fr_y = U_opt[7, :];  fr_z = U_opt[8, :]
    a    = U_opt[0, :];   acz  = U_opt[1, :];  alpha = U_opt[2, :]

    N = U_opt.shape[1]
    t_x = np.arange(X_opt.shape[1]) * dt
    t_u = np.arange(N) * dt

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(f'MPC prediction — horizon={N}  dt={dt}s', fontsize=12)

    # (0,0) traiettoria xy
    ax = axes[0, 0]
    ax.plot(xs, ys, 'b-', lw=1.5, label='CoM')
    ax.plot(cxs, cys, 'g--', lw=1, label='base c')
    ax.plot(xs[0], ys[0], 'go', ms=8, label='start')
    ax.plot(xs[-1], ys[-1], 'ro', ms=8, label='end')
    step = max(1, len(xs) // 10)
    ax.quiver(xs[::step], ys[::step],
              np.cos(theta[::step]), np.sin(theta[::step]),
              scale=20, width=0.004, color='steelblue', alpha=0.6)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('traiettoria xy'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # (0,1) altezza COM
    ax = axes[0, 1]
    ax.plot(t_x, zs, color='darkorchid', label='pcom_z')
    ax.set_xlabel('t (s)'); ax.set_ylabel('z (m)')
    ax.set_title('altezza CoM'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # (0,2) theta
    ax = axes[0, 2]
    ax.plot(t_x, np.degrees(theta), color='steelblue')
    ax.set_xlabel('t (s)'); ax.set_ylabel('theta (deg)')
    ax.set_title('orientazione theta'); ax.grid(True, alpha=0.3)

    # (1,0) velocità lineari
    ax = axes[1, 0]
    ax.plot(t_x, vs,  color='royalblue',      label='v')
    ax.plot(t_x, vxs, color='cornflowerblue', ls='--', lw=1, label='vx')
    ax.plot(t_x, vys, color='tomato',         ls='--', lw=1, label='vy')
    ax.set_xlabel('t (s)'); ax.set_ylabel('m/s')
    ax.set_title('velocità lineare'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # (1,1) omega
    ax = axes[1, 1]
    ax.plot(t_x, omegas, color='mediumseagreen')
    ax.set_xlabel('t (s)'); ax.set_ylabel('rad/s')
    ax.set_title('velocità angolare ω'); ax.grid(True, alpha=0.3)

    # (1,2) vcz
    ax = axes[1, 2]
    ax.plot(t_x, vcz, color='orange')
    ax.set_xlabel('t (s)'); ax.set_ylabel('m/s')
    ax.set_title('velocità verticale vcz'); ax.grid(True, alpha=0.3)

    # (2,0) forze verticali
    ax = axes[2, 0]
    ax.plot(t_u, fl_z, color='steelblue', label='fl_z')
    ax.plot(t_u, fr_z, color='tomato',    label='fr_z')
    ax.set_xlabel('t (s)'); ax.set_ylabel('N')
    ax.set_title('forze verticali contatto'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # (2,1) forze laterali
    ax = axes[2, 1]
    ax.plot(t_u, fl_x, color='steelblue',      lw=1, label='fl_x')
    ax.plot(t_u, fl_y, color='cornflowerblue', lw=1, label='fl_y')
    ax.plot(t_u, fr_x, color='tomato',          lw=1, label='fr_x')
    ax.plot(t_u, fr_y, color='salmon',           lw=1, label='fr_y')
    ax.set_xlabel('t (s)'); ax.set_ylabel('N')
    ax.set_title('forze laterali contatto'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # (2,2) accelerazioni input
    ax = axes[2, 2]
    ax.plot(t_u, a,     color='royalblue',      label='a (lin)')
    ax.plot(t_u, acz,   color='darkorchid',     label='acz (vert)')
    ax.plot(t_u, alpha, color='mediumseagreen', label='alpha (ang)')
    ax.set_xlabel('t (s)'); ax.set_ylabel('m/s² / rad/s²')
    ax.set_title('accelerazioni input'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    plt.tight_layout()
    #plt.show()

def mpc_sol_to_wbc_input(x_curr, u_curr, mass = 27.68978, g=9.81, wheel_radius=0.0925, com_feet_distance=0.5706):

    pcom_curr = x_curr[0:3]
    v_com_curr = x_curr[3:6]
    pl_curr = x_curr[6:9]


    # ── Contact points ──
    c = x_curr[6:9]
    Rz = np.array([[ np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta),  np.cos(theta), 0],
                    [0,                 0,          1]])
    offset = Rz @ np.array([0, com_feet_distance / 2, 0])
    left_contact  = c + offset
    right_contact = c - offset

    # ── Left wheel ──
    des.lwheel.pos = np.eye(4)
    des.lwheel.pos[0:2, 3] = left_contact[0:2]
    des.lwheel.pos[2, 3]   = left_contact[2] + wheel_radius
    des.lwheel.vel = np.zeros(6)
    des.lwheel.vel[0:3] = X_opt[3:6, 0]     # stessa vel del COM
    des.lwheel.acc = np.zeros(6)
    des.lwheel.acc[0:3] = acc_com

    # ── Right wheel ──
    des.rwheel.pos = np.eye(4)
    des.rwheel.pos[0:2, 3] = right_contact[0:2]
    des.rwheel.pos[2, 3]   = right_contact[2] + wheel_radius
    des.rwheel.vel = np.zeros(6)
    des.rwheel.vel[0:3] = X_opt[3:6, 0]
    des.rwheel.acc = np.zeros(6)
    des.rwheel.acc[0:3] = acc_com

    return des

def get_rCP_ca(R_wheel, radius):
    z0 = ca.vertcat(0, 0, 1)
    n  = ca.mtimes(R_wheel, z0)
    I3 = ca.SX.eye(3)
    a  = ca.mtimes(I3 - ca.mtimes(n, n.T), z0)
    s  = a / (ca.norm_2(a) + 1e-9)
    return -s * radius

def get_rCP(R_wheel, radius):
    z0 = np.array([0.0, 0.0, 1.0], dtype=float)
    n = R_wheel @ z0
    a = (np.eye(3) - np.outer(n, n)) @ z0
    s = a / (np.linalg.norm(a) + 1e-9)
    return -s * radius

def get_tita_state( model, data, torso_body_id, 
                    left_wheel_site_id, right_wheel_site_id, 
                    wheel_radius=0.0925):

    # ── COM ──
    mujoco.mj_comPos(model, data)    # aggiorna subtree_com se serve
    pcom = data.subtree_com[torso_body_id].copy()          # (3,)
    vcom = data.cvel[torso_body_id, 3:6].copy()            # (3,)

    # ── Wheel positions + rotations ──
    l_wheel_center = data.site_xpos[left_wheel_site_id].copy()     # (3,)
    r_wheel_center = data.site_xpos[right_wheel_site_id].copy()    # (3,)
    l_wheel_R = data.site_xmat[left_wheel_site_id].reshape(3, 3)  # (3,3)
    r_wheel_R = data.site_xmat[right_wheel_site_id].reshape(3, 3) # (3,3)

    # ── Contact points ──
    left_rCP  = get_rCP(l_wheel_R, wheel_radius)
    right_rCP = get_rCP(r_wheel_R, wheel_radius)
    left_contact  = l_wheel_center + left_rCP
    right_contact = r_wheel_center + right_rCP

    # ── Wheel velocities via Jacobian ──
    jacp_l = np.zeros((3, model.nv))
    jacp_r = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp_l, None, left_wheel_site_id)
    mujoco.mj_jacSite(model, data, jacp_r, None, right_wheel_site_id)
    dpl_world = jacp_l @ data.qvel    # (3,)
    dpr_world = jacp_r @ data.qvel    # (3,)

    # ── Pack ──
    x_IN = np.concatenate([
        pcom,            # [0:3]   p_CoM
        vcom,            # [3:6]   v_CoM
        left_contact,    # [6:9]   left contact point
        right_contact,   # [9:12]  right contact point
        dpl_world,       # [12:15] left wheel velocity
        dpr_world,       # [15:18] right wheel velocity
    ])
    return x_IN

def unwrapNear(theta_wrapped: float, theta_prev: float) -> float:
    # wrapping to pi
    a = theta_wrapped - theta_prev 

    a = (a + np.pi) % (2 * np.pi)
    a = np.where(a < 0, a + 2*np.pi, a)
    a = a - np.pi

    return theta_prev + a

def get_dfip_state(tita_state, theta_prev, com_feet_distance = 0.5706):

    pcom = tita_state[0:3]
    vcom = tita_state[3:6]
    pl_world = tita_state[6:9]
    pr_world = tita_state[9:12]
    dpl_world = tita_state[12:15]
    dpr_world = tita_state[15:18]

    c_world = (pl_world + pr_world) / 2.0
    vc_world = (dpl_world + dpr_world) / 2.0

    diff = pl_world - pr_world
    theta_wrapped = np.arctan2(-diff[0], diff[1])
    theta = unwrapNear(theta_wrapped, theta_prev)

    R = np.array([
        [ np.cos(theta), -np.sin(theta), 0.],
        [ np.sin(theta),  np.cos(theta), 0.],
        [ 0.,              0.,             1.]
    ])

    dpl_body = R.T @ dpl_world
    dpr_body = R.T @ dpr_world

    w = (dpr_body[0] - dpl_body[0]) / com_feet_distance
    v = (dpl_body[0] + dpr_body[0]) / 2.0

    return np.concatenate([
        pcom,
        vcom,
        c_world,
        np.array([vc_world[2], theta, v, w], dtype=float),
    ]).astype(float)


path = "/home/ubuntu/Desktop/repo_rl/TITA-dynamic-obstacle-avoidance/TITA_MJ/tita_mj_description/tita_world.xml"

model = mujoco.MjModel.from_xml_path(path)
data  = mujoco.MjData(model)

# ── Initial joint configuration ──────────────────────────────────────────────
joint_targets = {
    "joint_left_leg_1":  0.0,
    "joint_left_leg_2":  0.5,
    "joint_left_leg_3": -1.0,
    "joint_left_leg_4":  0.0,
    "joint_right_leg_1":  0.0,
    "joint_right_leg_2":  0.5,
    "joint_right_leg_3": -1.0,
    "joint_right_leg_4":  0.0,
}

# Floating‐base pose
data.qpos[0] = 0.0   # x
data.qpos[1] = 0.0   # y
data.qpos[2] = 0.44  # z
data.qpos[3] = 1.0   # quat w
data.qpos[4] = 0.0   # quat x
data.qpos[5] = 0.0   # quat y
data.qpos[6] = 0.0   # quat z

for jname, angle in joint_targets.items():
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if jid != -1:
        data.qpos[model.jnt_qposadr[jid]] = angle
    else:
        print(f"[WARNING] Joint not found: {jname}")

mujoco.mj_forward(model, data)

# ── Actuated joint ordering (follows model.actuator order) ───────────────────
actuated_joint_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0])
    for i in range(model.nu)
]
n_actuated = len(actuated_joint_names)

# Target (standing) positions for the PD
q_pos_target = np.array([
    joint_targets.get(n, 0.0) for n in actuated_joint_names
], dtype=float)

# ── Build CasADi problems once ──────────────────────────────────────────────
print("[CasADi] Building MPC problem …")

a_weight     = 1
acz_weight   = 0.01
alpha_weight = 0.0001
fx_weight    = 0.001
fy_weight    = 0.001
fz_weight    = 0.0001
R_ctrl = np.diag([a_weight, acz_weight, alpha_weight, fx_weight, fy_weight, fz_weight, fx_weight, fy_weight, fz_weight])
Q_pos = np.diag([1, 1, 1])

print("[DEBUG] MPC weights diag: R_ctrl =", np.diag(R_ctrl), "Q_pos =", np.diag(Q_pos))

mpc_build = build_mpc(Q_pos=Q_pos, R_ctrl=R_ctrl)
mpc_solver = mpc_build["solver"]
mpc_f_dynamics = mpc_build["f_dynamics"]
mpc_f_constraints = mpc_build["f_constraints"]
mpc_f_rollout = mpc_build["f_rollout"]
mpc_lbg = mpc_build["lbg"]
mpc_ubg = mpc_build["ubg"]
mpc_dims = mpc_build["dims"]

#print("[CasADi] Building WBC problem …")
#wbc_opti, wbc_tau, wbc_qdd, wbc_qdd_des = build_wbc_problem(n_joints=n_actuated)

# ── Reference state for MPC (stand still at spawn height) ────────────────────
# Trajectory reference: shape (3, N+1) where N=50 from build_mpc
N_mpc = mpc_dims['N']

traj_gen = TrajectoryGenerator()
x_ref, _ = traj_gen.generate_offline_trajectory(
    vel_lin=0.0, 
    vel_ang=0.0, 
    vel_z=-0.0, 
    dt=0.002, 
    pcom=np.array([0.0, 0.0, 0.4])
)

# ── Viewer ───────────────────────────────────────────────────────────────────
viewer = mujoco.viewer.launch_passive(model, data)
viewer.cam.distance  = 15.0
viewer.cam.azimuth   = 30
viewer.cam.elevation = -20

# ── Simulation loop ─────────────────────────────────────────────────────────
MPC_INTERVAL = 10          # solve MPC every N sim steps (0.002 * 10 = 20 ms)
frame_idx = 0
mpc_forces = np.zeros(4)   # cached MPC output

np.set_printoptions(precision=5, suppress=True)

while True:
    try:
        if not viewer.is_running():
            break
        
        if frame_idx > 0: 
            print("----------- JUST TEST ----------")
            break
        
        quat = data.qpos[3:7]  # [w, x, y, z]
        theta_val = 2.0 * np.arctan2(quat[2], quat[0])  #
        
        x0_val = np.array([
            data.qpos[0],      # px
            data.qpos[1],      # py
            data.qpos[2],      # pz
            data.qvel[0],      # vx
            data.qvel[1],      # vy
            data.qvel[2],      # vz
            data.qpos[0],      # cx (assume COM at origin)
            data.qpos[1],      # cy
            data.qpos[2],      # cz
            0.0,               # v_cz (vertical center velocity)
            theta_val,         # theta (yaw)
            0.0,               # v (forward velocity)
            0.0,               # omega (angular velocity)
        ])

        tita_state = get_tita_state(model, data, 
                                    torso_body_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link"),
                                    left_wheel_site_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_leg_4_site"),
                                    right_wheel_site_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_leg_4_site"),
                                    wheel_radius=0.0925)
        dfip_mpc_state = get_dfip_state(tita_state, theta_prev=theta_val)
        print(f"[DEBUG] DFIP state at frame {frame_idx}: \n{dfip_mpc_state}")
        print("[Solving MPC …]")

        x_ref_mpc = x_ref[0:3, frame_idx: frame_idx + N_mpc + 1].numpy()
        mass = 27.68978
        g = 9.81
        if frame_idx == 0:
            t_mpc_start = time.perf_counter()

        U_opt, X_opt = solve_mpc(
            mpc=mpc_build,
            x0_val=dfip_mpc_state, 
            xref_val=x_ref_mpc,
            U_warm=np.array([0.0, 0.0, 0.0, 0.0, 0.0, mass * g / 2.0, 0.0, 0.0, mass * g / 2.0]).reshape(9, 1).repeat(N_mpc, axis=1)
        )

        if frame_idx == 0:
            t_mpc_end = time.perf_counter()
            print(f"[TIMING] MPC solve time (first iteration): {(t_mpc_end - t_mpc_start)*1e3:.3f} ms")
        
        print(f"[DEBUG] MPC solved: U_opt shape {U_opt.shape}, X_opt shape {X_opt.shape}")
        mpc_forces = U_opt[:, 0]   # use first control
        print(f"[MPC] First control output (forces): \na = {mpc_forces[0]}, \nacz = {mpc_forces[1]}, \nalpha = {mpc_forces[2]}, \nfl = {mpc_forces[3:6]}, \nfr = {mpc_forces[6:9]}")
        plot_mpc_horizon(X_opt, U_opt, dt=0.002)

        q_pos_curr = np.array([
            data.qpos[model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]]
            for n in actuated_joint_names
        ])
        q_vel_curr = np.array([
            data.qvel[model.jnt_dofadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]]
            for n in actuated_joint_names
        ])
        q_ddot_des = compute_desired_joint_acc(q_pos_curr, q_vel_curr, q_pos_target)
        #tau = solve_wbc(wbc_opti, wbc_tau, wbc_qdd, wbc_qdd_des, q_ddot_des)
        
        data.ctrl[:] = np.zeros(model.nu) 
        #if not np.isnan(tau).any():
        #    data.ctrl[:] = tau
        #else:
        #    print(f"[WARN] NaN torques at frame {frame_idx}")

        mujoco.mj_step(model, data)
        viewer.sync()
        frame_idx += 1

    except Exception as e:
        print(f"[ERROR] {e}")
        break

try:
    viewer.close()
except Exception:
    pass
