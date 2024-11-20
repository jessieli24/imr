# -----------------------------------------------------------------
# bsdf.py 
#
# Jessie Li & Michael Riad Zaky, CS 77/277, Fall 2024
# -----------------------------------------------------------------
import numpy as np
from utils import *

NUM_VAR_DIFFUSE = 3
NUM_VAR_BSDF = 5

def distributionGGX(N, H, roughness: float):
    '''
    Parameters:
        - N: Vec3f
        - H: Vec3f
        - roughness: float
    
    Returns: float
    '''
    
    a2 = roughness * roughness * roughness * roughness
    NdotH = max(np.dot(N, H), 0.0)
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

def geometrySmith (N, V, L, roughness):
    '''
    Parameters:
        - N: Vec3f
        - V: Vec3f
        - L: Vec3f
        - roughness: float
        
    Returns: float
    '''
    
    return geometrySchlickGGX (max(np.dot(N, L), 0.0), roughness) *  geometrySchlickGGX(max(np.dot(N, V), 0.0), roughness)

def fresnelSchlick (cosTheta: float, F0):
    '''
    Parameters:
        - cosTheta: float
        - F0: Vec3f
        
    Returns: Vec3f
    '''
    
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0)

# -----------------------------------------------------------------
# bsdf functions
# -----------------------------------------------------------------
def bsdfWithError(w0, bsdf, texelScenes, clamped):
    
    error = 0
    nScenes = 0
        
    for scene, parameters in texelScenes.items():
        numScenes = len(texelScenes.items())
        
        try:
            colorTrue = np.array(parameters['correct'], dtype=float)
     
            if (np.all(colorTrue == 0)):
                continue
            
            shouldInclude, colorPredicted = bsdfEvaluate(bsdf, w0, parameters)
            
            if (not shouldInclude):
                continue
            
            # if (clamped):
            #     np.clip(colorTrue, 0., 1.)
            #     np.clip(colorPredicted, 0., 1.)
            
            colorTrue = np.clip(colorTrue, 0., 1.)
            colorPredicted = np.clip(colorPredicted, 0., 1.)
            
            nScenes += 1
            error += colorError(colorTrue, colorPredicted)
        except:
            # print(f'Something went wrong. Parameters: {parameters}')
            raise
            # continue
    
    return (error/nScenes) if nScenes > 3 else 0
    # return error / nScenes if nScenes > 1 else -1
        

def colorError(rgb, rgbHat):
    return np.sum((rgb - rgbHat) ** 2)

def bsdfEvaluate(bsdf, w0, args):
    match(bsdf):
        case 'diffuse':
            return bsdfDiffuse(w0, args)
            
        case 'bsdf': 
            return bsdfPrincipled(w0, args)
            
        case _:
            print('Error: Unknown BSDF')

def bsdfDiffuse(w, args):
    radiance = np.array(args['radiance'], dtype=float)
    
    if (np.all(radiance == 0)):
        return False, w
    
    return True, np.array(w) * radiance
    

def bsdfPrincipled(w, args):
    '''
    Parameters:
        w: 
        - albedo: Vec3f
        - metallic: Vec3f
        - roughness: Vec3f
        
        args:
        - ray_d: Vec3f
        - hit: const HitRecord
        - light: const Vec4f
        - finalColor: Color3f
        
    Returns: Vec3f
    '''

    albedoCol = w[:3]
    metallic = w[3]
    roughness = w[4]
    
    ray_d = np.array(args['ray.d'])
    hit_p = np.array(args['hit.p'])
    hit_sn = np.array(args['hit.sn'])
    light = np.array(args['light'])
    
    if (np.all(light == 0)):
        return False, np.zeros(3)
    
    lightPos = light[:3]
    lightColor = light[3] / (4*PI)
    
    gamma = 1.0    
    rd = normalize(ray_d)
    
    worldPos = hit_p
    N = hit_sn
    V = -rd
    L = normalize(lightPos - worldPos)
    H = normalize(V + L)
    
    # Cook-Torrance BRDF
    F0 = floatToVec3(0.04) * (1. - metallic) + pow(albedoCol, floatToVec3(gamma)) * metallic
    NDF = distributionGGX(N, H, roughness)
    G = geometrySmith(N, V, L, roughness)
    F = fresnelSchlick(max(np.dot(H, V), 0.0), F0)
    kD  = floatToVec3(1.0) - F
    kD *= 1.0 - metallic
    
    numerator = NDF * G * F
    denominator = 4.0 * max(np.dot(N, V), 0.0) * max(np.dot(N, L), 0.0)
    specular = numerator / max(denominator, 0.001)  
        
    NdotL = max(np.dot(N, L), 0.0)                
    color = lightColor * (kD * np.power(albedoCol, floatToVec3(gamma)) / PI + specular) * (NdotL / np.dot(lightPos - worldPos, lightPos - worldPos))
    
    return True, color