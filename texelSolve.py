# -----------------------------------------------------------------
# texelSolve.py - optimizer
#
# Jessie Li & Michael Riad Zaky, CS 77/277, Fall 2024
# -----------------------------------------------------------------

from utils import * 
from bsdf import *

import numpy as np
from scipy.optimize import minimize

NUM_VAR_BSDF = 5

def getDefaultBoundsAndInitialParameters(bsdf: str):
    w0 = [0.01]
    bounds = [(0, 1)]
    
    match (bsdf):
        case 'diffuse':
            return w0 * 3, bounds * 3
        case 'bsdf':
            return w0 * 5, bounds * 5
        case _:
            print('Error: Unknown BSDF')
            

def bsdfEvaluate(bsdf, w0, parameters, args):
    '''
    Parameters:
    - w0: unknown parameters (to optimize)
    - parameters: given scene parameters
    - args: additional arguments
    '''
    
    match(bsdf):
        case 'diffuse':
            return bsdfDiffuse(w0, parameters)
            
        case 'bsdf': 
            return bsdfPrincipled(w0, parameters, args)
            
        case _:
            print('Error: Unknown BSDF')
            
def unpackScene(bsdf, scene):
    '''
    Unpacks and validates scene parameters. 
    
    Returns: 
    - shouldInclude: True if parameters are valid, False otherwise
    - correct: 1 x 3 array (RGB)
    - parameters: p x 3 array
    - lightColor: 1 x 1 scalar
    '''
    
    try:
        if (bsdf == 'bsdf'):
            correct = np.array(scene['correct'])
            ray_d = np.array(scene['ray.d'])
            hit_p = np.array(scene['hit.p'])
            hit_sn = np.array(scene['hit.sn'])
            light = np.array(scene['light'])
            
            if (np.all(correct == 0) or light[3] == 0):
                return False, None, None, None
            
            return True, correct, [ray_d, hit_p, hit_sn, light[:3]], [light[3]]
    
    # parameters missing
    except:
        return False, None, None, None
    
    # unknown bsdf
    return False, None, None, None

def bsdfWithError(w0, bsdf, texelScenes, clampColors, minScenes):
    allCorrect = []
    allParameters = []
    allLightColors = []
    
    for scene, parameters in texelScenes.items():
        isSceneValid, correct, parameters, lightColor = unpackScene(bsdf, parameters)
        
        if (not isSceneValid):
            continue
        
        allCorrect.append(correct)
        allParameters.append(parameters)
        allLightColors.append(lightColor)
    
    # enough scenes? 
    if len(allCorrect) < minScenes:
        return 0
    
    n = len(allCorrect)
    allCorrect = np.array(allCorrect)                       # n x 3
    allParameters = np.stack(allParameters, axis=1)         # p x n x 3
    allLightColors = np.array(allLightColors)               # n x 1
    allw0 = np.tile(w0, (n, 1))                             # n x w
    
    # evaluate BSDF for all valid scenes
    try: 
        allPredicted = bsdfEvaluate(bsdf, allw0, allParameters, allLightColors) # n x 3
    except:
        raise
    
    if clampColors:
        allCorrect = np.clip(allCorrect, 0, 1)
        allPredicted = np.clip(allPredicted, 0, 1)
    
    error = squareError(allCorrect, allPredicted)
    return error / len(allCorrect)

def optimize(
    bsdf, texelScenes, 
    w0=None, 
    bounds=None, 
    method='Nelder-Mead', 
    methodOptions={'xatol': 1/256, 'maxiter': 100, 'adaptive': True},
    clampColors = False,
    minScenes = 1):
    
    if (w0 is None or bounds is None):
        w0, bounds = getDefaultBoundsAndInitialParameters(bsdf)
    
    res = minimize(bsdfWithError, 
        x0=w0,
        method=method,
        bounds=bounds,
        args=(bsdf, texelScenes, clampColors, minScenes),
        options=methodOptions
    )
            
    return res.x, res.fun

def solveTexel(bsdf, v, u, scenes, optimizationParameters):
    try:    
        uvPrediction, uvError = optimize(bsdf, scenes, **optimizationParameters)
        uvPrediction = uvPrediction if uvError > 0 else np.zeros(NUM_VAR_BSDF)
        return uvPrediction, uvError
    
    except Exception as err: 
        print(f'Something went wrong. Texel: ({u}, {v}), Error: {err=}')
        return np.zeros(NUM_VAR_BSDF), 0
    
def processChunk(chunk, bsdf, resolution, optimizationParameters):
    '''
    Processes a chunk of rows.
    '''
    errors = np.zeros(resolution) 
    predictions = np.zeros((*resolution, NUM_VAR_BSDF))

    for v, row in chunk:
        for u, scenes in row.items():
            u = int(u)
            v = int(v)
            prediction, error = solveTexel(bsdf, v, u, scenes, optimizationParameters)
            predictions[u, v] = prediction
            errors[u, v] = error
    
    return predictions, errors

def processRow(v, row, bsdf, resolutionU, optimizationParameters):
    '''
    Processes a single row.
    '''
    
    errors = np.zeros(resolutionU) 
    predictions = np.zeros((resolutionU, NUM_VAR_BSDF))

    v = int(v)
    for u, scenes in row.items():
        u = int(u)
        prediction, error = solveTexel(bsdf, v, u, scenes, optimizationParameters)
        predictions[u] = prediction
        errors[u] = error
    
    return v, predictions, errors