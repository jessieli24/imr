# 
# utils.py - utility functions for common vector and matrix operations
#
# Jessie Li & Michael Riad Zaky, CS 77/277, Fall 2024
# 

import numpy as np

PI = 3.14159265359

def normalizeVector(v):
    d = np.linalg.norm(v)
    return v / d

def normalizeRows(m):
    d = np.linalg.norm(m, axis=1, keepdims=True)
    
    return m / d

def dotRows(a, b):
    return np.sum(a * b, axis=1, keepdims=True)

def floatToVec3(f):
   return np.array([f, f, f])

def squareError(rgb, rgbHat):
    return np.sum((rgb - rgbHat) ** 2)
