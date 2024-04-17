import numpy as np
from scipy.optimize import least_squares


def optimization_function(params, stereo_depth, normalized_depth):
    s, t = params
    adjusted_depth = normalized_depth * s + t
    return np.sum((stereo_depth - adjusted_depth) ** 2)


def de_normalization(stereo_depth_map, normalized_depth_map):
    # Optimize for s and t
    initial_guess = [1, 0]
    result = least_squares(optimization_function, initial_guess, args=(stereo_depth_map, normalized_depth_map))
    s_optimized, t_optimized = result.x
    # Recover true depth map
    recovered = normalized_depth_map * s_optimized + t_optimized
    
    return recovered