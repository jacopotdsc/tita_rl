import os
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_results')

_MPC_NX      = 13
_MPC_NU      = 9
_MPC_HORIZON = 50
_MPC_DT      = 0.002
_MPC_MASS    = 27.68978
_MPC_GRAV    = 9.81
d_off_       = 0.1

_NV = 14
_NJ = 8

_Q_MIN_LIST   = [-0.5, -1.5, -2.5, -1e9,  -0.5, -1.5, -2.5, -1e9]
_Q_MAX_LIST   = [ 0.5,  0.5,  0.0,  1e9,   0.5,  0.5,  0.0,  1e9]
_QDOT_MAX_VAL = 10.0
_TAU_MAX_VAL  = 120.0

_S_PCOM  = slice(0, 3);  _S_DPCOM = slice(3, 6)
_S_C     = slice(6, 9);  _S_VCZ   = 9
_S_THETA = 10;           _S_V     = 11;  _S_OMEGA = 12
_A_A     = 0;  _A_ACZ   = 1;  _A_ALPHA = 2
_A_FL    = slice(3, 6);  _A_FR    = slice(6, 9)

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trajectory_generator import TrajectoryGenerator

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
    
    