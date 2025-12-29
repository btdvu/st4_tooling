"""
Stack-of-Stars Trajectory Generation for MRI

Generate 3D stack-of-stars k-space trajectories for MRI.
Combines radial sampling in the xy-plane with Cartesian sampling along z.
Supports golden-angle sampling and staggered slice acquisition.

Author: Brian-Tinh Vu
Date: 12/29/2025
Dependencies: numpy

Quick Start:
    traj = trajGoldenAngleStackOfStars(N=128, n_spokes_per_slice=360, nz=64)
    traj_staggered = trajGoldenAngleStackOfStars(N=128, n_spokes_per_slice=360, nz=64, staggered=True)
"""

import numpy as np
from lib.radial import trajGoldenAngle


def trajGoldenAngleStackOfStars(N, n_spokes_per_slice, nz, staggered=False):
    """
    Generate 3D stack-of-stars trajectory with golden-angle sampling.
    
    Creates a 3D k-space trajectory by stacking 2D golden-angle radial
    trajectories along the z-axis. Each slice contains a full set of
    radial spokes, and slices can be staggered for improved coverage.
    
    Parameters
    ----------
    N : int
        Number of samples along each radial spoke.
    n_spokes_per_slice : int
        Number of radial spokes per slice.
    nz : int
        Number of slices along the z-axis.
    staggered : bool, optional
        If True, stagger starting angles between slices for improved coverage.
        Default: False.
    
    Returns
    -------
    traj : np.ndarray
        3D k-space coordinates shape: (nz*n_spokes_per_slice, N, 3).
        Contains (kx, ky, kz) coordinates for each spoke.
    
    Notes
    -----
    - Uses 111.25° golden-angle increment for optimal xy-plane coverage
    - z-coordinates are centered around zero (symmetric sampling)
    - When staggered=True, each slice starts at a different angle to
      improve 3D coverage and reduce coherent aliasing
    """
    # Generate z-coordinates for symmetric slice coverage
    pez_lines = np.arange(-nz//2, -nz//2+nz)

    # Initialize list to store trajectory for each slice
    slice_trajectories = []

    # Generate radial trajectory for each slice
    for i, pez in enumerate(pez_lines):
        # Determine starting angle for this slice
        if staggered:
            # Stagger angles between slices using golden-angle offset
            starting_angle = 111.25*i*n_spokes_per_slice
        else:
            # All slices start at same angle
            starting_angle = 0

        # Generate 2D radial trajectory for this slice
        tmp_slice_traj = trajGoldenAngle(N, n_spokes_per_slice, starting_angle=starting_angle)
        
        # Add z-coordinate to create 3D trajectory
        # Broadcast z-coordinate to match trajectory shape
        z_coords = np.ones(list(tmp_slice_traj.shape[:2])+[1]) * pez
        tmp_slice_traj_3d = np.concatenate([tmp_slice_traj, z_coords], axis=-1)
        
        slice_trajectories.append(tmp_slice_traj_3d)

    # Concatenate all slices into single 3D trajectory
    slice_trajectories = np.concatenate(slice_trajectories, axis=0)

    return slice_trajectories