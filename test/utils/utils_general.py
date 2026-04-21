import os
import numpy as np
import matplotlib.pyplot as plt
from .global_variables import _MPC_DT, CASADI_FN_DIR

def export_functions(fn):
    
    fn_name = fn.name() 
    os.makedirs(CASADI_FN_DIR, exist_ok=True)
    path = os.path.join(CASADI_FN_DIR, f'{fn_name}.casadi')
    fn.save(path)
    print(f"Saved: {path}  (n_instructions: {fn.n_instructions()})")

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
    
 