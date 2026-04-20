#!/usr/bin/env python3
"""
test_gpu_functions.py
---------------------
Loads and evaluates the compiled CusADi functions (f_dynamics, f_constraints, f_rollout)
on GPU using CUDA kernels, and compares against CasADi CPU for validation.
"""

import os
import sys
import time
import numpy as np
import torch
import casadi as ca
from global_variables import PLOT_DIR, _MPC_NX, _MPC_NU, _MPC_HORIZON, _MPC_DT, _MPC_MASS, _MPC_GRAV

# ── Paths ─────────────────────────────────────────────────────────────────────
TEST_DIR         = os.path.dirname(os.path.abspath(__file__))
CASADI_FN_DIR    = os.path.join(TEST_DIR, "casadi_functions")
CUSADI_DIR       = os.path.join(TEST_DIR, "cusadi")
sys.path.insert(0, CUSADI_DIR)

from cusadi import CusadiFunction   # noqa: E402  (needs sys.path set first)

# ── Robot / MPC constants (must match mpc.py) ────────────────────────────────
NX   = _MPC_NX
NU   = _MPC_NU
N    = _MPC_HORIZON
MASS = _MPC_MASS
G    = _MPC_GRAV

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 4000
DTYPE      = torch.float64
DEVICE     = "cuda"

def header(msg):
    print(f"\n{'='*65}")
    print(f"  {msg}")
    print(f"{'='*65}\n")

def load_fn(name):
    path = os.path.join(CASADI_FN_DIR, f"{name}.casadi")
    if not os.path.exists(path):
        print(f"  [FAIL] {name}.casadi not found in {CASADI_FN_DIR}")
        sys.exit(1)
    fn = ca.Function.load(path)
    print(f"  [OK]   Loaded  {fn}")
    return fn

# ─────────────────────────────────────────────────────────────────────────────
# 1. f_dynamics  (x[13], u[9]) -> x_next[13]
# ─────────────────────────────────────────────────────────────────────────────
def test_f_dynamics():
    header("TEST: f_dynamics  (x[13], u[9]) -> x_next[13]")

    fn_ca  = load_fn("f_dynamics")
    fn_gpu = CusadiFunction(fn_ca, BATCH_SIZE)

    # Random but physically plausible inputs
    torch.manual_seed(0)
    x_gpu = torch.zeros(BATCH_SIZE, NX, device=DEVICE, dtype=DTYPE)
    x_gpu[:, 2]  =  0.44                                    # pz  (height)
    x_gpu[:, 8]  =  0.44                                    # cz
    x_gpu[:, 3:6] = torch.randn(BATCH_SIZE, 3, device=DEVICE, dtype=DTYPE) * 0.1

    u_gpu = torch.zeros(BATCH_SIZE, NU, device=DEVICE, dtype=DTYPE)
    u_gpu[:, 5] = MASS * G / 2.0    # fl_z  (gravity compensation)
    u_gpu[:, 8] = MASS * G / 2.0    # fr_z

    # ── GPU eval ──
    t0 = time.perf_counter()
    fn_gpu.evaluate([x_gpu, u_gpu])
    torch.cuda.synchronize()
    t_gpu = (time.perf_counter() - t0) * 1e3

    x_next_gpu = fn_gpu.outputs_sparse[0].cpu().numpy()    # (BATCH_SIZE, 13)

    # ── CPU reference (single env) ──
    x0_np = x_gpu[0].cpu().numpy()
    u0_np = u_gpu[0].cpu().numpy()
    x_next_cpu = np.asarray(fn_ca(x0_np, u0_np)).flatten()

    # ── Error ──
    err = np.linalg.norm(x_next_gpu[0] - x_next_cpu)

    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"  GPU eval time   : {t_gpu:.4f} ms")
    print(f"  CPU/GPU error   : {err:.2e}  {'[PASS]' if err < 1e-10 else '[FAIL]'}")
    print(f"\n  x_next (env 0, GPU) : {x_next_gpu[0]}")
    print(f"  x_next (env 0, CPU) : {x_next_cpu}")

    return err < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# 2. f_constraints  (x[13], u[9]) -> (h_eq[4], g_soft[2], g_pos[2], g_neg[0])
# ─────────────────────────────────────────────────────────────────────────────
def test_f_constraints():
    header("TEST: f_constraints  (x[13], u[9]) -> (h_eq, g_soft, g_pos, g_neg)")

    fn_ca  = load_fn("f_constraints")
    fn_gpu = CusadiFunction(fn_ca, BATCH_SIZE)

    torch.manual_seed(1)
    x_gpu = torch.zeros(BATCH_SIZE, NX, device=DEVICE, dtype=DTYPE)
    x_gpu[:, 2] = 0.44
    x_gpu[:, 8] = 0.44

    u_gpu = torch.zeros(BATCH_SIZE, NU, device=DEVICE, dtype=DTYPE)
    u_gpu[:, 5] = MASS * G / 2.0
    u_gpu[:, 8] = MASS * G / 2.0

    # ── GPU eval ──
    t0 = time.perf_counter()
    fn_gpu.evaluate([x_gpu, u_gpu])
    torch.cuda.synchronize()
    t_gpu = (time.perf_counter() - t0) * 1e3

    # ── CPU reference ──
    x0_np = x_gpu[0].cpu().numpy()
    u0_np = u_gpu[0].cpu().numpy()
    out_cpu = fn_ca(x0_np, u0_np)

    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"  GPU eval time   : {t_gpu:.4f} ms")
    print(f"  Outputs         : {fn_ca.n_out()} tensors")
    for i in range(fn_ca.n_out()):
        out_name = fn_ca.name_out(i)
        gpu_val  = fn_gpu.outputs_sparse[i][0].cpu().numpy()
        cpu_val  = np.asarray(out_cpu[i]).flatten()
        err      = np.linalg.norm(gpu_val - cpu_val) if cpu_val.size > 0 else 0.0
        status   = "[PASS]" if err < 1e-10 else "[FAIL]"
        print(f"    {out_name:<18} gpu={gpu_val}  cpu={cpu_val}  err={err:.2e}  {status}")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# 3. f_rollout  (x0[13], U[nu*N]) -> X[13*(N+1)]
# ─────────────────────────────────────────────────────────────────────────────
def test_f_rollout():
    header("TEST: f_rollout  (x0[13], U[9*50]) -> X[13x51]")

    fn_ca  = load_fn("f_rollout")
    fn_gpu = CusadiFunction(fn_ca, BATCH_SIZE)

    NUN = NU * N

    torch.manual_seed(2)
    x0_gpu = torch.zeros(BATCH_SIZE, NX, device=DEVICE, dtype=DTYPE)
    x0_gpu[:, 2] = 0.44
    x0_gpu[:, 8] = 0.44

    U_gpu = torch.zeros(BATCH_SIZE, NUN, device=DEVICE, dtype=DTYPE)
    # Gravity compensation for all N steps
    for k in range(N):
        U_gpu[:, k * NU + 5] = MASS * G / 2.0
        U_gpu[:, k * NU + 8] = MASS * G / 2.0

    # ── GPU eval ──
    t0 = time.perf_counter()
    fn_gpu.evaluate([x0_gpu, U_gpu])
    torch.cuda.synchronize()
    t_gpu = (time.perf_counter() - t0) * 1e3

    # output is flat (NX*(N+1),) in column-major order -> reshape with order='F'
    X_gpu = fn_gpu.outputs_sparse[0][0].cpu().numpy().reshape(NX, N + 1, order='F')

    # ── CPU reference ──
    x0_np  = x0_gpu[0].cpu().numpy()
    U0_np  = U_gpu[0].cpu().numpy()
    X_cpu  = np.asarray(fn_ca(x0_np, U0_np)).reshape(NX, N + 1)

    err = np.linalg.norm(X_gpu - X_cpu)
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"  GPU eval time   : {t_gpu:.4f} ms")
    print(f"  CPU/GPU error   : {err:.2e}  {'[PASS]' if err < 1e-10 else '[FAIL]'}")
    print(f"\n  Rollout X (env 0, GPU) — first 5 steps:")
    print(f"    pz trajectory : {X_gpu[2, :5]}")
    print(f"  Rollout X (env 0, CPU) — first 5 steps:")
    print(f"    pz trajectory : {X_cpu[2, :5]}")

    return err < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# 4. CPU vs GPU benchmark  (all 3 functions)
# ─────────────────────────────────────────────────────────────────────────────
def benchmark_all(n_iters=200):
    header(f"BENCHMARK: CPU vs GPU  —  {n_iters} iters  batch={BATCH_SIZE}")

    # ── shared inputs ────────────────────────────────────────────────────────
    x_np = np.zeros((BATCH_SIZE, NX), dtype=np.float64)
    x_np[:, 2] = 0.44;  x_np[:, 8] = 0.44

    u_np = np.zeros((BATCH_SIZE, NU), dtype=np.float64)
    u_np[:, 5] = MASS * G / 2.0;  u_np[:, 8] = MASS * G / 2.0

    U_flat_np = np.zeros((BATCH_SIZE, NU * N), dtype=np.float64)
    for k in range(N):
        U_flat_np[:, k * NU + 5] = MASS * G / 2.0
        U_flat_np[:, k * NU + 8] = MASS * G / 2.0

    x_gpu      = torch.tensor(x_np,      device=DEVICE, dtype=DTYPE)
    u_gpu      = torch.tensor(u_np,      device=DEVICE, dtype=DTYPE)
    U_flat_gpu = torch.tensor(U_flat_np, device=DEVICE, dtype=DTYPE)

    x0_np = x_np[0];  u0_np = u_np[0];  U0_np = U_flat_np[0]

    specs = [
        ("f_dynamics",    [x_gpu, u_gpu],      [x0_np, u0_np]),
        ("f_constraints", [x_gpu, u_gpu],      [x0_np, u0_np]),
        ("f_rollout",     [x_gpu, U_flat_gpu], [x0_np, U0_np]),
    ]

    stats = {}   # fn_name -> dict

    for fn_name, gpu_inputs, cpu_inputs in specs:
        fn_ca  = load_fn(fn_name)
        fn_gpu = CusadiFunction(fn_ca, BATCH_SIZE)

        # CPU warmup + measure
        for _ in range(5): fn_ca(*cpu_inputs)
        t0 = time.perf_counter()
        for _ in range(n_iters): fn_ca(*cpu_inputs)
        t_cpu_1env = (time.perf_counter() - t0) * 1e3 / n_iters   # ms, 1 env

        # GPU warmup + measure
        for _ in range(10): fn_gpu.evaluate(gpu_inputs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters): fn_gpu.evaluate(gpu_inputs)
        torch.cuda.synchronize()
        t_gpu_total_ms = (time.perf_counter() - t0) * 1e3          # ms, all iters

        t_gpu_mean  = t_gpu_total_ms / n_iters                     # ms per call
        t_cpu_ser   = t_cpu_1env * BATCH_SIZE                      # estimated serial
        throughput  = BATCH_SIZE / (t_gpu_mean * 1e-3)
        speedup     = t_cpu_ser / t_gpu_mean

        stats[fn_name] = {
            "t_gpu_total_ms": t_gpu_total_ms,
            "t_gpu_mean":     t_gpu_mean,
            "throughput":     throughput,
            "t_cpu_1env":     t_cpu_1env,
            "t_cpu_serial":   t_cpu_ser,
            "speedup":        speedup,
        }

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N_ITERS = 200

    print(f"\n  Device : {torch.cuda.get_device_name(0)}")
    print(f"  Batch  : {BATCH_SIZE}  |  dtype: {DTYPE}  |  N={N}")

    results = {}
    for name, fn in [("f_dynamics",    test_f_dynamics),
                     ("f_constraints", test_f_constraints),
                     ("f_rollout",     test_f_rollout)]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            results[name] = False

    bench = benchmark_all(n_iters=N_ITERS)

    # ── Final summary ─────────────────────────────────────────────────────────
    header(f"SUMMARY  —  batch={BATCH_SIZE}  iters={N_ITERS}")

    C0, C1, C2, C3, C4, C5, C6, C7 = 18, 16, 12, 18, 14, 16, 10, 8
    print(
        f"  {'Function':<{C0}}"
        f"  {'Total GPU (ms)':>{C1}}"
        f"  {'Mean (ms)':>{C2}}"
        f"  {'Throughput (env/s)':>{C3}}"
        f"  {'CPU 1env (ms)':>{C4}}"
        f"  {'CPU serial (ms)':>{C5}}"
        f"  {'Speedup':>{C6}}"
        f"  {'Status':>{C7}}"
    )
    sep_len = C0 + C1 + C2 + C3 + C4 + C5 + C6 + C7 + 16
    print("  " + "-" * sep_len)

    for fn_name, s in bench.items():
        status = "PASS" if results.get(fn_name) else "FAIL"
        print(
            f"  {fn_name:<{C0}}"
            f"  {s['t_gpu_total_ms']:>{C1}.2f}"
            f"  {s['t_gpu_mean']:>{C2}.4f}"
            f"  {s['throughput']:>{C3},.0f}"
            f"  {s['t_cpu_1env']:>{C4}.4f}"
            f"  {s['t_cpu_serial']:>{C5}.1f}"
            f"  {s['speedup']:>{C6}.1f}x"
            f"  {status:>{C7}}"
        )
    print(f"\n  * CPU serial = CPU 1env × {BATCH_SIZE} (estimated)")

    if all(results.values()):
        print(f"  All GPU functions validated successfully.")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  Failed: {failed}")
        sys.exit(1)