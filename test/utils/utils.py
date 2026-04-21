import mujoco
import mujoco.viewer
import time
import torch
import numpy as np
import casadi as ca
import pinocchio as pin
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING
from utils.global_variables import _MPC_DT

if TYPE_CHECKING:
    from trajectory_generator import TrajectoryGenerator

def get_rCP(R_wheel, radius):
    z0 = ca.vertcat(0, 0, 1)
    n  = ca.mtimes(R_wheel, z0)
    I3 = ca.SX.eye(3)
    a  = ca.mtimes(I3 - ca.mtimes(n, n.T), z0)
    s  = a / (ca.norm_2(a) + 1e-9)
    return -s * radius

def get_tita_state(mj_data, pin_model, pin_data,
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
    err = q_pos_target - q_pos
    err_wrapped = torch.atan2(torch.sin(err), torch.cos(err))
    return kp * err_wrapped - kd * q_vel

def unwrapNear(theta_wrapped: torch.Tensor, theta_prev: torch.Tensor) -> torch.Tensor:
    a = theta_wrapped - theta_prev
    a = (a + torch.pi) % (2 * torch.pi)
    a = torch.where(a < 0, a + 2 * torch.pi, a)
    a = a - torch.pi
    return theta_prev + a

def get_dfip_state(tita_state: torch.Tensor, theta_prev: torch.Tensor, com_feet_distance=0.5706) -> torch.Tensor:

    pcom      = tita_state[0:3]
    vcom      = tita_state[3:6]
    pl_world  = tita_state[6:9]
    pr_world  = tita_state[9:12]
    dpl_world = tita_state[12:15]
    dpr_world = tita_state[15:18]

    c_world  = (pl_world + pr_world) / 2.0 
    vc_world = (dpl_world + dpr_world) / 2.0

    diff          = pl_world - pr_world       
    print(f"[DEBUG] diff: {diff}, type(diff): {type(diff)}")
    theta_wrapped = torch.atan2(-diff[0], diff[1])
    theta         = unwrap_near(theta_wrapped, theta_prev)

    Rz = torch.tensor([
        [ torch.cos(theta), -torch.sin(theta), 0.],
        [ torch.sin(theta),  torch.cos(theta), 0.],
        [ 0.,                0.,               1.]
    ], device=tita_state.device)

    dpl_body = (Rz.transpose(1, 2) @ dpl_world).squeeze(-1)  # (N, 3)
    dpr_body = (Rz.transpose(1, 2) @ dpr_world).squeeze(-1)  # (N, 3)

    w = (dpr_body[:, 0] - dpl_body[:, 0]) / com_feet_distance  # (N,)
    v = (dpl_body[:, 0] + dpr_body[:, 0]) / 2.0                # (N,)

    return torch.cat([
        pcom,                
        vcom,                
        c_world,             
        vc_world,  
        theta,  
        v,      
        w,      
    ], dim=1)  
