import torch
import math
import time
import os
import numpy as np
from typing import List, Tuple
from typing import NamedTuple, Optional, Tuple, Union
from global_variables import _MPC_NX, _MPC_NU, _MPC_DT, _MPC_MASS, _MPC_GRAV, PLOT_DIR
from global_variables import *

class TrajectoryGenerator:
    
    def generate_offline_trajectory(
        self,
        vel_lin: float = 0.0, 
        vel_ang: float = 0.0, 
        vel_z: float = 0.0,
        pcom:  Tuple[float, float, float] = (0.0, 0.0, 0.4),
        nx: int = _MPC_NX, 
        nu: int = _MPC_NU,
        t_sec: float = 6.0, 
        dt: float = _MPC_DT,
        m: float = _MPC_MASS, 
        grav: float = _MPC_GRAV,
        device: str = 'cpu',
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if not isinstance(pcom, torch.Tensor):
            pcom = torch.tensor(pcom, dtype=torch.float32, device=device)
        else:
            pcom = pcom.to(device)

        N  = int(t_sec / dt);  T = t_sec
        T_const = 2.*T/3.;    T_acc = (T-T_const)/2.
        a_max   = vel_lin  / T_acc if T_acc > 0 else 0.
        alp_max = vel_ang  / T_acc if T_acc > 0 else 0.

        x, y, z   = pcom[0].item(), pcom[1].item(), pcom[2].item()
        theta = 0.; vz = vel_z
        z_min, z_max = 0.25, 0.42

        xr = torch.zeros(N, nx, device=device)
        ur = torch.zeros(N, nu, device=device)

        for s in range(N):
            t  = s * dt; td = t - (T_acc + T_const)
            if   t < T_acc:               a_ = a_max;   v_ = a_max*t;    al_ = alp_max; om_ = alp_max*t
            elif t < T_acc + T_const:     a_ = 0.;      v_ = vel_lin;    al_ = 0.;      om_ = vel_ang
            elif t < T:                   a_ = -a_max;  v_ = vel_lin-a_max*td; al_ = -alp_max; om_ = vel_ang-alp_max*td
            else:                         a_ = v_ = al_ = om_ = 0.
            vx = v_*math.cos(theta); vy = v_*math.sin(theta)
            x += vx*dt; y += vy*dt; theta += om_*dt
            z = max(z_min, min(z_max, z+vz*dt))
            if z <= z_min or z >= z_max: vz = 0.
            xr[s] = torch.tensor([x,y,z,vx,vy,vz,x,y,0.,0.,theta,v_,om_], device=device)
            ur[s] = torch.tensor([0.,0.,0.,0.,0.,m*grav/2.,0.,0.,m*grav/2.], device=device)

        return xr.T.contiguous(), ur.T.contiguous()

def generate_offline_trajectory_batched(
    vel_lin: torch.Tensor,          # (B,)
    vel_ang: torch.Tensor,          # (B,)
    vel_z:   torch.Tensor,          # (B,)
    pcom:    Optional[torch.Tensor] = None,
    nx: int   = _MPC_NX,
    nu: int   = _MPC_NU,
    t_sec: float = 6.0,
    dt:    float = _MPC_DT,
    m:     float = _MPC_MASS,
    grav:  float = _MPC_GRAV,
    device: str  = 'cpu',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fully vectorised batched trajectory generation.
    Returns:
        xr : (B, nx, N)
        ur : (B, nu, N)
    """
    B = vel_lin.shape[0]
    N = int(t_sec / dt)
    T_const = 2. * t_sec / 3.
    T_acc   = (t_sec - T_const) / 2.

    vel_lin = vel_lin.to(device)
    vel_ang = vel_ang.to(device)
    vel_z   = vel_z.to(device)

    # per-batch acceleration limits  (B,)
    a_max   = (vel_lin / T_acc) if T_acc > 0 else torch.zeros_like(vel_lin)
    alp_max = (vel_ang / T_acc) if T_acc > 0 else torch.zeros_like(vel_ang)

    # time axes  (1, N)
    t  = torch.arange(N, device=device, dtype=torch.float32).unsqueeze(0) * dt
    td = t - (T_acc + T_const)

    # broadcast batch dims  (B, 1)
    a_b   = a_max.unsqueeze(1)
    alp_b = alp_max.unsqueeze(1)
    vl_b  = vel_lin.unsqueeze(1)
    va_b  = vel_ang.unsqueeze(1)
    vz_b  = vel_z.unsqueeze(1).expand(B, N)

    zeros = torch.zeros(B, N, device=device)

    # phase masks  (B, N) via broadcast
    p1 = t < T_acc
    p2 = (t >= T_acc) & (t < T_acc + T_const)
    p3 = (t >= T_acc + T_const) & (t < t_sec)

    v_ = torch.where(p1, a_b   * t,
         torch.where(p2, vl_b.expand(B, N),
         torch.where(p3, vl_b  - a_b   * td, zeros)))   # (B, N)

    om_ = torch.where(p1, alp_b * t,
          torch.where(p2, va_b.expand(B, N),
          torch.where(p3, va_b  - alp_b * td, zeros)))  # (B, N)

    # theta[s] = Σ_{i<s} om_[i]*dt  →  right-shifted cumsum
    theta = torch.zeros(B, N, device=device)
    theta[:, 1:] = torch.cumsum(om_[:, :-1] * dt, dim=1)   # (B, N)

    vx = v_ * torch.cos(theta)   # (B, N)
    vy = v_ * torch.sin(theta)   # (B, N)

    # initial positions
    if pcom is None:
        x0 = torch.zeros(B, device=device)
        y0 = torch.zeros(B, device=device)
        z0 = torch.full((B,), 0.4, device=device)
    else:
        pcom = pcom.to(device)
        x0 = pcom[0].expand(B)
        y0 = pcom[1].expand(B)
        z0 = pcom[2].expand(B)

    # positions via right-shifted cumsum
    def _integrate(v_field, p0):
        out = torch.zeros(B, N, device=device)
        out[:, 0]  = p0
        out[:, 1:] = p0.unsqueeze(1) + torch.cumsum(v_field[:, :-1] * dt, dim=1)
        return out

    x = _integrate(vx,  x0)
    y = _integrate(vy,  y0)
    z = _integrate(vz_b, z0).clamp(0.25, 0.42)

    # ── assemble state tensor  (B, nx, N) ────────────────────────────────────
    xr = torch.zeros(B, nx, N, device=device)
    xr[:, 0,  :] = x
    xr[:, 1,  :] = y
    xr[:, 2,  :] = z
    xr[:, 3,  :] = vx
    xr[:, 4,  :] = vy
    xr[:, 5,  :] = vz_b
    xr[:, 6,  :] = x       # foot x  (same as CoM here)
    xr[:, 7,  :] = y       # foot y
    xr[:, 8,  :] = 0.      # roll
    xr[:, 9,  :] = 0.      # pitch
    xr[:, 10, :] = theta
    xr[:, 11, :] = v_
    xr[:, 12, :] = om_

    # ── assemble control tensor  (B, nu, N) ──────────────────────────────────
    ur = torch.zeros(B, nu, N, device=device)
    ur[:, 5, :] = m * grav / 2.
    ur[:, 8, :] = m * grav / 2.

    return xr, ur

def benchmark_trajectory(
    traj_gen,
    batch_sizes: List[int] = [1, 4, 16, 64, 256, 512],
    n_runs:      int  = 100,
    device:      str  = 'cpu',
) -> str:
    """
    Benchmark: for loop over B envs, each calls generate_offline_trajectory once.
    Reports mean and std over n_runs repetitions.
    """
    SEP = '─' * 62
    lines = [
        f'\n{SEP}',
        f'  SEQUENTIAL BENCHMARK  |  device={device}  n_runs={n_runs}',
        SEP,
        f'  {"B":>5}  {"mean ms":>10}  {"std ms":>10}  {"ms/env":>9}',
        SEP,
    ]

    for B in batch_sizes:
        torch.manual_seed(42)
        vel_lins = (torch.rand(B) * 1.0).tolist()
        vel_angs = ((torch.rand(B) - 0.5) * 0.6).tolist()
        vel_zs   = ((torch.rand(B) - 0.5) * 0.2).tolist()

        times = []
        for _ in range(n_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            for i in range(B):
                traj_gen.generate_offline_trajectory(
                    vel_lin=vel_lins[i],
                    vel_ang=vel_angs[i],
                    vel_z=vel_zs[i],
                    device=device,
                )

            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)

        mean_ms = np.mean(times)
        std_ms  = np.std(times)
        lines.append(f"  {B:>5}  {mean_ms:>10.3f}  {std_ms:>10.3f}  {mean_ms/B:>9.4f}")

    lines.append(SEP)
    return '\n'.join(lines)

def benchmark_trajectory_batched(
    batch_sizes: List[int] = [1, 4, 16, 64, 256, 512],
    n_runs:      int  = 100,
    device:      str  = 'cpu',
) -> str:
    """
    Benchmark: single call to generate_offline_trajectory_batched for B envs.
    Reports mean and std over n_runs repetitions.
    """
    SEP = '─' * 62
    lines = [
        f'\n{SEP}',
        f'  BATCHED BENCHMARK  |  device={device}  n_runs={n_runs}',
        SEP,
        f'  {"B":>5}  {"mean ms":>10}  {"std ms":>10}  {"ms/env":>9}',
        SEP,
    ]

    for B in batch_sizes:
        torch.manual_seed(42)
        vl = torch.rand(B, device=device) * 1.0
        va = (torch.rand(B, device=device) - 0.5) * 0.6
        vz = (torch.rand(B, device=device) - 0.5) * 0.2

        times = []
        for _ in range(n_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            generate_offline_trajectory_batched(vl, va, vz, device=device)

            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)

        mean_ms = np.mean(times)
        std_ms  = np.std(times)
        lines.append(f"  {B:>5}  {mean_ms:>10.3f}  {std_ms:>10.3f}  {mean_ms/B:>9.4f}")

    lines.append(SEP)
    return '\n'.join(lines)

if __name__ == '__main__':
    traj_gen = TrajectoryGenerator()
    
    plot_generated_trajectory(traj_gen, vel_lin=0.0, vel_ang=0.0,   vel_z=-0.0)
    plot_generated_trajectory(traj_gen, vel_lin=0.5, vel_ang=0.0,   vel_z=-0.0)
    plot_generated_trajectory(traj_gen, vel_lin=0.0, vel_ang=-0.3,  vel_z=-0.0)
    plot_generated_trajectory(traj_gen, vel_lin=0.5, vel_ang=0.0,   vel_z=0.5)
    plot_generated_trajectory(traj_gen, vel_lin=0.5, vel_ang=0.1,   vel_z=-0.05)

    n_runs = 10
    reports = []
    reports.append(benchmark_trajectory(traj_gen, batch_sizes=[1], n_runs=n_runs, device='cpu'))
    reports.append(benchmark_trajectory(traj_gen, batch_sizes=[1], n_runs=n_runs, device='cuda'))
    reports.append(benchmark_trajectory_batched(batch_sizes=[1, 64, 256], n_runs=n_runs, device='cpu'))
    reports.append(benchmark_trajectory_batched(batch_sizes=[1, 64, 256, 2048], n_runs=n_runs, device='cuda'))

    os.makedirs(PLOT_DIR, exist_ok=True)
    benchmark_path = os.path.join(PLOT_DIR, 'benchmark_report.txt')

    with open(benchmark_path, 'w', encoding='utf-8') as report_file:
        report_file.write('\n\n'.join(reports) + '\n')

