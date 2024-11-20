# -----------------------------------------------------------------
# utils.py 
#
# Jessie Li & Michael Riad Zaky, CS 77/277, Fall 2024
# -----------------------------------------------------------------

import numpy as np

PI = 3.14159265359

def normalize(v):
    d = np.linalg.norm(v)
    return v / d

def floatToVec3(f):
   return np.array([f, f, f])