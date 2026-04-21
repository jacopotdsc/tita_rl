import mujoco
import mujoco.viewer
import time
import torch
import numpy as np
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING
try:
    from .global_variables import _MPC_DT, URDF_PATH
except ImportError:
    from global_variables import _MPC_DT, URDF_PATH

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cusadi.models import PinocchioModel
from utils.utils_general import export_functions

if TYPE_CHECKING:
    from trajectory_generator import TrajectoryGenerator

def build_rCP_fn(fn_opts={}):
    R_sym      = ca.SX.sym('R', 3, 3)
    radius_sym = ca.SX.sym('radius', 1, 1)

    z0 = ca.vertcat(0, 0, 1)
    n  = R_sym @ z0
    a  = z0 - n * (n.T @ z0)
    s  = a / (ca.norm_2(a) + 1e-9)
    out = -s * radius_sym

    return ca.Function(
        'get_rCP',
        [R_sym, radius_sym], [out],
        ['R', 'radius'], ['rCP'],
        fn_opts
    )

def build_tita_state_fn(pin_model: PinocchioModel,
                        wheel_radius=0.0925,
                        left_frame="left_leg_4", right_frame="right_leg_4"):
    
    cpin_model = pin_model.cpin_model
    cpin_data  = pin_model.cpin_data

    left_idx  = pin_model.pin_model.getFrameId(left_frame)
    right_idx = pin_model.pin_model.getFrameId(right_frame)

    fn_rCP = build_rCP_fn(pin_model.fn_opts)
    q_sym  = ca.SX.sym('q',  cpin_model.nq, 1)
    dq_sym = ca.SX.sym('dq', cpin_model.nv, 1)

    # ── COM ──
    cpin.centerOfMass(cpin_model, cpin_data, q_sym, dq_sym)
    pcom = cpin_data.com[0]
    vcom = cpin_data.vcom[0]

    # ── FK ──
    cpin.forwardKinematics(cpin_model, cpin_data, q_sym, dq_sym)
    cpin.updateFramePlacements(cpin_model, cpin_data)

    l_SE3 = cpin_data.oMf[left_idx]
    r_SE3 = cpin_data.oMf[right_idx]

    left_contact  = l_SE3.translation + fn_rCP(l_SE3.rotation, wheel_radius)
    right_contact = r_SE3.translation + fn_rCP(r_SE3.rotation, wheel_radius)

    # ── velocità frame ──
    l_vel = cpin.getFrameVelocity(cpin_model, cpin_data, left_idx,
                                   cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    r_vel = cpin.getFrameVelocity(cpin_model, cpin_data, right_idx,
                                   cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    dpl_world = l_vel.linear
    dpr_world = r_vel.linear

    out = ca.vertcat(pcom, vcom, left_contact, right_contact, dpl_world, dpr_world)

    return ca.Function(
        'get_tita_state',
        [q_sym, dq_sym], [out],
        ['q', 'dq'], ['tita_state'],
        pin_model.fn_opts
    )

def build_desired_joint_acc_fn(nj, kp=80.0, kd=8.0, fn_opts={}):
    q_pos_sym    = ca.SX.sym('q_pos',    nj, 1)
    q_vel_sym    = ca.SX.sym('q_vel',    nj, 1)
    q_target_sym = ca.SX.sym('q_target', nj, 1)

    err         = q_target_sym - q_pos_sym
    err_wrapped = ca.atan2(ca.sin(err), ca.cos(err))
    q_ddot      = kp * err_wrapped - kd * q_vel_sym

    return ca.Function(
        'desired_joint_acc',
        [q_pos_sym, q_vel_sym, q_target_sym],
        [q_ddot],
        ['q_pos', 'q_vel', 'q_target'],
        ['q_ddot'],
        fn_opts
    )

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

if __name__ == "__main__":
    #export_functions(build_rCP_fn())
    export_functions(build_tita_state_fn(PinocchioModel(URDF_PATH, is_floating=True)))
    export_functions(build_desired_joint_acc_fn(nj=12))