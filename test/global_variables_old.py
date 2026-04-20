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
