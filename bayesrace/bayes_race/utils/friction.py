"""Friction map utilities for variable friction environments."""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'

import numpy as np


def get_friction(x, y):
    """
    Get friction coefficient at position (x, y).
    
    Slippery Split Environment:
    - Right side in plot (plotted x > 0, i.e., actual y < -0.5): μ = 0.3 (Ice)
    - Left side in plot (plotted x <= 0, i.e., actual y >= -0.5): μ = 1.0 (Asphalt)
    
    Note: Plots use -y for x-axis, so we check y < -0.5 for ice (right side of plot).
    Division is at y = -0.5 (not 0).
    
    Args:
        x: x-coordinate (scalar or array, currently unused)
        y: y-coordinate (scalar or array)
    
    Returns:
        μ: Friction coefficient (scalar or array, same shape as y)
    """
    y = np.asarray(y)
    # Ice on right side of plot (plotted x > 0 means actual y < -0.5)
    mu = np.where(y < -0.5, 0.3, 1.0)
    return mu
