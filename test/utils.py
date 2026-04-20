import mujoco
import mujoco.viewer
import time
import numpy as np
import casadi as ca
import pinocchio as pin
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING
from global_variables import _MPC_DT

if TYPE_CHECKING:
    from trajectory_generator import TrajectoryGenerator


def get_rCP(R_wheel, radius):
    z0 = ca.vertcat(0, 0, 1)
    n  = ca.mtimes(R_wheel, z0)
    I3 = ca.SX.eye(3)
    a  = ca.mtimes(I3 - ca.mtimes(n, n.T), z0)
    s  = a / (ca.norm_2(a) + 1e-9)
    return -s * radius

def get_rCP_np(R_wheel, radius):
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
    left_rCP  = get_rCP_np(l_wheel_R, wheel_radius)
    right_rCP = get_rCP_np(r_wheel_R, wheel_radius)
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

def get_tita_state_pin(mj_data, pin_model, pin_data,
                   left_leg4_idx, right_leg4_idx,
                   wheel_radius=0.0925):

    q    = mj_data.qpos.copy()
    qdot = mj_data.qvel.copy()

    # ── COM ──
    pin.centerOfMass(pin_model, pin_data, q, qdot)
    pcom = pin_data.com[0].copy()
    vcom = pin_data.vcom[0].copy()

    # ── FK ──
    pin.framesForwardKinematics(pin_model, pin_data, q)
    pin.computeJointJacobians(pin_model, pin_data, q)

    l_SE3 = pin_data.oMf[left_leg4_idx]
    r_SE3 = pin_data.oMf[right_leg4_idx]

    left_contact  = l_SE3.translation + get_rCP_np(l_SE3.rotation, wheel_radius)
    right_contact = r_SE3.translation + get_rCP_np(r_SE3.rotation, wheel_radius)

    # ── Jacobians ──
    J_left  = pin.getFrameJacobian(pin_model, pin_data, left_leg4_idx,
                                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    J_right = pin.getFrameJacobian(pin_model, pin_data, right_leg4_idx,
                                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    dpl_world = J_left[:3,  :] @ qdot
    dpr_world = J_right[:3, :] @ qdot

    return np.concatenate([pcom, vcom, left_contact, right_contact,
                           dpl_world, dpr_world])

def compute_desired_joint_acc(q_pos, q_vel, q_pos_target, kp=80.0, kd=8.0):
    """PD in joint space → desired q_ddot to feed into WBC."""
    err   = q_pos_target - q_pos
    err   = np.arctan2(np.sin(err), np.cos(err))  # wrap
    return kp * err - kd * q_vel

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

def mpc_sol_to_wbc_input(x_curr, u_curr, des, mass=27.68978, g=9.81, wheel_radius=0.0925, com_feet_distance=0.5706):

    v_com_curr = x_curr[3:6]
    c = x_curr[6:9]
    theta = x_curr[10]
    acc_com = u_curr[0:3]

    Rz = np.array([[ np.cos(theta), -np.sin(theta), 0],
                    [ np.sin(theta),  np.cos(theta), 0],
                    [ 0,              0,             1]])
    offset = Rz @ np.array([0, com_feet_distance / 2, 0])
    left_contact  = c + offset
    right_contact = c - offset

    des.lwheel.pos = np.eye(4)
    des.lwheel.pos[0:3, 3] = left_contact
    des.lwheel.pos[2, 3]  += wheel_radius
    des.lwheel.vel = np.zeros(6)
    des.lwheel.vel[0:3] = v_com_curr
    des.lwheel.acc = np.zeros(6)
    des.lwheel.acc[0:3] = acc_com

    des.rwheel.pos = np.eye(4)
    des.rwheel.pos[0:3, 3] = right_contact
    des.rwheel.pos[2, 3]  += wheel_radius
    des.rwheel.vel = np.zeros(6)
    des.rwheel.vel[0:3] = v_com_curr
    des.rwheel.acc = np.zeros(6)
    des.rwheel.acc[0:3] = acc_com

    return des

def plot_generated_trajectory(
    traj_gen: 'TrajectoryGenerator',
    vel_lin: float = 0.5, 
    vel_ang: float = 0.1, 
    vel_z: float = -0.05,
    dt: float = _MPC_DT,
) -> None:
    xr, ur = traj_gen.generate_offline_trajectory(
        vel_lin=vel_lin, vel_ang=vel_ang, vel_z=vel_z
    )

    # ── unpack state rows ──────────────────────────────────────────────────────
    xs     = xr[0].cpu().numpy()
    ys     = xr[1].cpu().numpy()
    zs     = xr[2].cpu().numpy()
    vxs    = xr[3].cpu().numpy()
    vys    = xr[4].cpu().numpy()
    vzs    = xr[5].cpu().numpy()
    theta  = xr[10].cpu().numpy()
    vs     = xr[11].cpu().numpy()
    omegas = xr[12].cpu().numpy()

    t = np.arange(xs.shape[0]) * dt

    # ── figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f'x_ref  |  v={vel_lin} m/s   ω={vel_ang} rad/s   vz={vel_z} m/s',
        fontsize=12,
    )

    # 1) xy trajectory
    ax = axes[0, 0]
    sc = ax.scatter(xs, ys, c=t, cmap='viridis', s=2)
    ax.plot(xs[0],  ys[0],  'go', ms=8, label='start')
    ax.plot(xs[-1], ys[-1], 'ro', ms=8, label='end')
    step = max(1, int(0.5 / dt))
    ax.quiver(
        xs[::step], ys[::step],
        np.cos(theta[::step]), np.sin(theta[::step]),
        scale=20, width=0.004, color='steelblue', alpha=0.6,
    )
    plt.colorbar(sc, ax=ax, label='t (s)')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('xy trajectory')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # 2) CoM height z
    ax = axes[0, 1]
    ax.plot(t, zs, color='darkorchid')
    ax.axhline(0.25, ls='--', color='gray', alpha=0.5, label='z_min')
    ax.axhline(0.42, ls='--', color='gray', alpha=0.7, label='z_max')
    ax.set_xlabel('t (s)'); ax.set_ylabel('z (m)')
    ax.set_title('CoM height'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # 3) heading theta
    ax = axes[0, 2]
    ax.plot(t, np.degrees(theta), color='steelblue')
    ax.set_xlabel('t (s)'); ax.set_ylabel('theta (deg)')
    ax.set_title('Theta θ'); ax.grid(True, alpha=0.3)

    # 4) linear velocity
    ax = axes[1, 0]
    ax.plot(t, vs,  color='royalblue',      label='v')
    ax.plot(t, vxs, color='cornflowerblue', ls='--', lw=1, label='vx')
    ax.plot(t, vys, color='tomato',         ls='--', lw=1, label='vy')
    ax.set_xlabel('t (s)'); ax.set_ylabel('m/s')
    ax.set_title('Linear velocity'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # 5) angular velocity omega
    ax = axes[1, 1]
    ax.plot(t, omegas, color='mediumseagreen')
    ax.set_xlabel('t (s)'); ax.set_ylabel('rad/s')
    ax.set_title('Angular velocity ω'); ax.grid(True, alpha=0.3)

    # 6) vertical velocity vz
    ax = axes[1, 2]
    ax.plot(t, vzs, color='orange')
    ax.axhline(0, ls='--', color='gray', alpha=0.5)
    ax.set_xlabel('t (s)'); ax.set_ylabel('m/s')
    ax.set_title('Vertical velocity vz'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dest_dir = os.path.join(current_dir, 'test_results')
    name_fig = f'generated_trajectory_vel{vel_lin}_ang{vel_ang}_vz{vel_z}'.replace('.', 'p')
    
    os.makedirs(dest_dir, exist_ok=True)
    plt.savefig(os.path.join(dest_dir, f'{name_fig}.png'), dpi=300)
    
    