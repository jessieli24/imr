# 
# bsdf.py 
#
# Jessie Li & Michael Riad Zaky, CS 77/277, Fall 2024
# 

import numpy as np
from utils import *

NUM_VAR_DIFFUSE = 3
NUM_VAR_BSDF = 5

def distributionGGX(N, H, roughness):
    '''
    Parameters:
        - N: Vec3f
        - H: Vec3f
        - roughness: float
    
    Returns: float
    '''
    
    a2 = roughness * roughness * roughness * roughness
    NdotH = np.maximum(dotRows(N, H), 0.0)
    denom = (NdotH * NdotH * (a2 - 1.0) + 1.0)
    return a2 / (PI * denom * denom)

def geometrySchlickGGX (NdotV: float, roughness: float):
    '''
    Parameters:
        - NdotV: float
        - roughness: float
    
    Returns: float
    '''
    
    r = roughness + 1.0
    k = (r * r) / 8.0
    return NdotV / (NdotV * (1.0 - k) + k)

def geometrySmith(N, V, L, roughness):
    '''
    Parameters:
        - N: Vec3f
        - V: Vec3f
        - L: Vec3f
        - roughness: float
        
    Returns: float
    '''
    
    return geometrySchlickGGX(np.maximum(dotRows(N, L), 0.0), roughness) * geometrySchlickGGX(np.maximum(dotRows(N, V), 0.0), roughness)

def fresnelSchlick (cosTheta: float, F0):
    '''
    Parameters:
        - cosTheta: float
        - F0: Vec3f
        
    Returns: Vec3f
    '''
    
    return F0 + (1.0 - F0) * np.power(1.0 - cosTheta, 5.0).reshape(-1, 1)

# -----------------------------------------------------------------
# bsdf functions
# -----------------------------------------------------------------

def bsdfDiffuse(w, args):
    radiance = np.array(args['radiance'], dtype=float)
    
    if (np.all(radiance == 0)):
        return False, w
    
    return True, np.array(w) * radiance
    

def bsdfPrincipled(w, sceneParameters, lightColor):
    '''
    Parameters:
        w: n x w artay of parameters (to optimize)
        - albedo: Vec3f
        - metallic: Vec3f
        - roughness: Vec3f
        
        sceneParameters: n x 4 x 3 array
        - ray_d: Vec3f
        - hit_p: Vec3f
        - hit_sn: Vec3f
        - lightPos: Vec3f
        
        lightColors: n x 1 array
        
    Returns: Vec3f
    '''
    
    albedoCol = w[:, :3]
    metallic = w[:, 3].reshape(-1, 1)
    roughness = w[:, 4].reshape(-1, 1)
    
    # n x 3 each
    ray_d = sceneParameters[0, :, :]
    hit_p = sceneParameters[1, :, :]
    hit_sn = sceneParameters[2, :, :]
    lightPos = sceneParameters[3, :, :]
    
    lightColor = lightColor / (4*PI)
    
    gamma = 1.0    
    rd = normalizeRows(ray_d)
    
    worldPos = hit_p
    N = hit_sn
    V = -rd
    L = normalizeRows(lightPos - worldPos)
    H = normalizeRows(V + L)
    
    # Cook-Torrance BRDF
    F0 = 0.04 * (1. - metallic) + np.power(albedoCol, gamma) * metallic
    NDF = distributionGGX(N, H, roughness)
    G = geometrySmith(N, V, L, roughness)
    F = fresnelSchlick(np.maximum(dotRows(H, V), 0.0), F0)
    kD  = 1.0 - F
    kD *= 1.0 - metallic

    numerator = NDF * G * F
    denominator = 4.0 * np.maximum(dotRows(N, V), 0.0) * np.maximum(dotRows(N, L), 0.0)
    specular = numerator / np.maximum(denominator, 0.001)  
        
    NdotL = np.maximum(dotRows(N, L), 0.0)                
    color = lightColor * (kD * np.power(albedoCol, gamma) / PI + specular) * (NdotL / dotRows(lightPos - worldPos, lightPos - worldPos))
    
    return color
