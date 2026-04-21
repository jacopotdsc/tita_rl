import mujoco
import mujoco.viewer
import time
import torch
import numpy as np
import casadi as ca
import pinocchio as pin
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING
from .global_variables import _MPC_DT

if TYPE_CHECKING:
    from trajectory_generator import TrajectoryGenerator

def get_rCP_np(R_wheel, radius):
    z0 = np.array([0.0, 0.0, 1.0], dtype=float)
    n = R_wheel @ z0
    a = (np.eye(3) - np.outer(n, n)) @ z0
    s = a / (np.linalg.norm(a) + 1e-9)
    return -s * radius

def get_tita_state_np( model, data, torso_body_id, 
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

def compute_desired_joint_acc_np(q_pos, q_vel, q_pos_target, kp=80.0, kd=8.0):
    """PD in joint space → desired q_ddot to feed into WBC."""
    err   = q_pos_target - q_pos
    err   = np.arctan2(np.sin(err), np.cos(err))  # wrap
    return kp * err - kd * q_vel

def unwrapNear_np(theta_wrapped: float, theta_prev: float) -> float:
    a = theta_wrapped - theta_prev 

    a = (a + np.pi) % (2 * np.pi)
    a = np.where(a < 0, a + 2*np.pi, a)
    a = a - np.pi

    return theta_prev + a

def get_dfip_state_np(tita_state, theta_prev, com_feet_distance = 0.5706):

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
    theta = unwrapNear_np(theta_wrapped, theta_prev)
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

def mpc_sol_to_wbc_input_np(x_curr, u_curr, des, mass=27.68978, g=9.81, wheel_radius=0.0925, com_feet_distance=0.5706):

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
