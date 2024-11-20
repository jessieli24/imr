# -----------------------------------------------------------------
# texelSolve.py - optimizer
#
# Jessie Li & Michael Riad Zaky, CS 77/277, Fall 2024
# -----------------------------------------------------------------

from utils import * 
from bsdf import *

import numpy as np
from scipy.optimize import minimize
from multiprocessing import Array

POOL_SIZE = 4
NUM_VAR_DIFFUSE = 3
NUM_VAR_BSDF = 5

resolution = (512, 512)
predictionsSize = resolution[0] * resolution[1] * NUM_VAR_BSDF
sharedArray = Array('d', predictionsSize, lock=False)

def getBoundsAndInitialParameters(bsdf: str):
    w0 = [0.01]
    bounds = [(0, 1)]
    
    match (bsdf):
        case 'diffuse':
            return w0 * 3, bounds * 3
        case 'bsdf':
            return w0 * 5, bounds * 5
        case _:
            print('Error: Unknown BSDF')

def optimize(
    bsdf: str, texelScenes, 
    method='Nelder-Mead', 
    methodOptions={'xatol': 1/256, 'maxiter': 100, 'adaptive': True},
    clampColors = False):
    
    w0, bounds = getBoundsAndInitialParameters(bsdf)
    
    res = minimize(bsdfWithError, 
        x0=w0,
        method=method,
        bounds=bounds,
        args=(bsdf, texelScenes, clampColors),
        options=methodOptions
    )
            
    return res.x, res.fun

def solveTexel(bsdf, v, u, scenes):
    # u = int(u)
    # v = int(v)
    # predictions[u, v] = np.array([-1, 0, 0, 0, 0])
    # return 
    try:    
        uvPrediction, uvError = optimize(bsdf, scenes, clampColors=True)
        # uvPrediction = uvPrediction if uvError > 0 else np.zeros(NUM_VAR_BSDF)
        uvPrediction = uvPrediction if uvError > 0 else np.zeros(NUM_VAR_BSDF)
        uvError = uvError if uvError > 0 else 0
        return uvPrediction, uvError
        # predictions[u, v] = uvPrediction if uvError > 0 else np.zeros(NUM_VAR_BSDF)
        # errors[u, v] = uvError if uvError > 0 else 0
    
    except Exception as err: 
        print(f'Something went wrong. Texel: ({u}, {v}), Error: {err=}')
        return np.zeros(NUM_VAR_BSDF), 0

def solveTexel2(bsdf, v, row):
    # time.sleep(0.01)
    for u, scenes in row.items():
        # predictions[int(u), int(v)] = np.array([-1, 0, 0, 0, 0])
        u = int(u)
        v = int(v)
     
        try:    
            uvPrediction, uvError = optimize(bsdf, scenes)
            # predictions[u, v] = uvPrediction if uvError > 0 else np.zeros(NUM_VAR_BSDF)
            # errors[u, v] = uvError if uvError > 0 else 0
        
        except Exception as err: 
            print(f'Something went wrong. Texel: ({u}, {v}), Error: {err=}')
    
def processChunkSeparately(bsdf, chunk, resolution):
    errors = np.zeros(resolution) 
    predictions = np.zeros((*resolution, NUM_VAR_BSDF))

    for v, row in chunk:
        for u, scenes in row.items():
            u = int(u)
            v = int(v)
            prediction, error = solveTexel(bsdf, v, u, scenes)
            predictions[u, v] = prediction
            errors[u, v] = error
    
    return predictions, errors
        
def processChunkWithSharedArray(bsdf, chunk, sharedPredictions):
    for v, row in chunk:
        for u, scenes in row.items():
            u = int(u)
            v = int(v)
            prediction, error = solveTexel(bsdf, v, u, scenes)
            sharedPredictions[v, u] = prediction

def testProcessChunkWithSharedArray(bsdf, chunk, predictionsSharedArray):
    predictionsNPArray = np.frombuffer(sharedArray, dtype=float).reshape((512, 512, NUM_VAR_BSDF))
    
    for v, row in chunk:
        for u, scenes in row.items():
            u = int(u)
            v = int(v)
            # prediction, error = solveTexel(bsdf, v, u, scenes)
            predictionsNPArray[v, u] = -1

    