import os
import mujoco
import mujoco.viewer
import time
import numpy as np
import casadi as ca
import pinocchio as pin
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from trajectory_generator import TrajectoryGenerator

from mpc import build_mpc, solve_mpc
from utils import get_rCP, get_rCP_np, get_tita_state, get_tita_state_pin, get_dfip_state, compute_desired_joint_acc, plot_mpc_horizon, mpc_sol_to_wbc_input
from global_variables import MUJOCO_MODEL_PATH, URDF_PATH


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

path = MUJOCO_MODEL_PATH
model = mujoco.MjModel.from_xml_path(path)
model_pin = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
data_pin  = model_pin.createData()

try:
    model = mujoco.MjModel.from_xml_path(path)
except Exception as e:
    print(f"Error loading model CHECK PATH: {e}")
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
        left_leg4_idx  = model_pin.getFrameId("left_leg_4")
        right_leg4_idx = model_pin.getFrameId("right_leg_4")

        tita_state = get_tita_state_pin(data, model_pin, data_pin, left_leg4_idx, right_leg4_idx, wheel_radius=0.0925)

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
