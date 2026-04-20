import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import TYPE_CHECKING

# ── Paths ─────────────────────────────────────────────────────────────────────
TEST_DIR        = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR        = os.path.join(TEST_DIR, "test_results")

# MuJoCo model
MUJOCO_MODEL_PATH = "/home/jacopo/Desktop/repo_rl/TITA-dynamic-obstacle-avoidance/TITA_MJ/tita_mj_description/tita_world.xml"
URDF_PATH = MUJOCO_MODEL_PATH.replace("tita_mj_description/tita_world.xml", "tita_description/tita.urdf")

# CasADi / CusADi
CASADI_FN_DIR   = os.path.join(TEST_DIR, "casadi_functions")
CUSADI_DIR      = os.path.join(TEST_DIR, "cusadi")
CUSADI_DST_DIR  = os.path.join(CUSADI_DIR, "src", "casadi_functions")
CUSADI_BUILD_DIR = os.path.join(CUSADI_DIR, "build")
CODEGEN_SCRIPT  = os.path.join(CUSADI_DIR, "run_codegen.py")
CUSADI_TEST_SCRIPT = os.path.join(CUSADI_DIR, "run_cusadi_function_test.py")

# ── GPU benchmark defaults ────────────────────────────────────────────────────
GPU_BATCH_SIZE     = 4000
GPU_DTYPE          = torch.float64
GPU_DEVICE         = "cuda"
GPU_ERROR_THRESHOLD = 1e-10

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