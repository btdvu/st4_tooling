"""
cartesian_undersampling.py

Functions to perform Cartesian undersampling of phase-encode lines
in 2D and 3D MRI acquisitions.

Author: Brian-Tinh Vu
Date: 11/08/2025
Requirements: numpy

Overview
--------
- regular_undersampling: Generates a binary mask for regular or golden-angle
  undersampling in 2D or 3D Cartesian k-space. Includes calibration region.
- compute_accel: Utility to compute actual acceleration from the mask (undersampling factor).

Usage
-----
mask = regular_undersampling(img_shape, accel, calib=24, mode="2D")
accel_measured = compute_accel(mask)
"""

import numpy as np
import copy

def regular_undersampling(img_shape, accel, calib=24, mode="2D", tol=0.1, max_attempts=100):
    """
    Generate undersampling masks for 2D or 3D MRI Cartesian acquisitions.

    Parameters
    ----------
    img_shape : tuple
        Shape of image (for 2D: (Nx, Ny), for 3D: (Ny, Nz)).
    accel : float
        Desired acceleration factor (total lines / scanned lines).
    calib : int, optional
        Size of calibration region in phase encode direction(s). Default is 24.
    mode : str, optional
        '2D' or '3D'. Determines the undersampling scheme. Default is '2D'.
    tol : float, optional
        Acceleration tolerance. Mask will be within accel ± tol. Default is 0.1.
    max_attempts : int, optional
        Maximum number of attempts to reach desired acceleration. Default is 100.

    Returns
    -------
    mask : ndarray
        Binary mask with sampled locations set to 1 and others 0.
        Shape: (Nx, Ny) for 2D, (Ny, Nz) for 3D.

    Raises
    ------
    ValueError
        If mode is invalid/not supported.

    Examples
    --------
    >>> mask_2d = regular_undersampling((128, 128), accel=4, calib=24, mode="2D")
    >>> mask_3d = regular_undersampling((64, 64), accel=8, calib=16, mode="3D")
    """
    if mode == "2D":
        # Get spatial dimensions
        Nx = img_shape[0]
        Ny = img_shape[1]

        # Initialize a temporary mask for iterations
        tmp_mask = np.zeros(Ny, dtype=np.complex64)
        actual_accel = -1

        # Start increment estimate for phase-encode lines sampling
        test_increment = accel
        inc_upper_bound = test_increment * 1.5
        inc_lower_bound = test_increment * 0.5
        n_attempts = 0
        isMetTolerance = False

        # Search for increment giving closest acceleration to requested value
        while (not isMetTolerance) and (n_attempts < max_attempts):
            tmp_mask = np.zeros(Ny, dtype=np.complex64)
            n_attempts += 1

            # Define calibration region (centered)
            calib_start = Ny // 2 - calib // 2
            calib_end = Ny // 2 + calib // 2
            tmp_mask[calib_start:calib_end] = 1

            # Sample phase-encode lines outside calibration region at test_increment intervals
            sampled_idxs = (np.round(np.arange(0, Ny - 1/2, test_increment))).astype(int)
            tmp_mask[sampled_idxs] = 1

            # Compute actual acceleration for this mask
            actual_accel = compute_accel(tmp_mask)

            # Adjust increment search bounds to approach desired acceleration
            if actual_accel < accel:
                inc_lower_bound = test_increment
                test_increment = 0.5 * (inc_lower_bound + inc_upper_bound)
            else:
                inc_upper_bound = test_increment
                test_increment = 0.5 * (inc_lower_bound + inc_upper_bound)

            isMetTolerance = np.abs(accel - compute_accel(tmp_mask)) < tol

        # Stack mask across readout direction to produce 2D sampling mask
        mask = tmp_mask
        mask = np.stack([mask] * Nx, axis=0)
        return mask

    elif mode == "3D":
        # For 3D sampling, use two phase-encode directions Ny and Nz
        Ny = img_shape[0]
        Nz = img_shape[1]
        tmp_mask = np.zeros((Ny, Nz), dtype=np.complex64)

        # Calibration region (centered square in PE plane)
        calib_start_y = Ny // 2 - calib // 2
        calib_end_y = Ny // 2 + calib // 2
        calib_start_z = Nz // 2 - calib // 2
        calib_end_z = Nz // 2 + calib // 2
        tmp_mask[calib_start_y:calib_end_y, calib_start_z:calib_end_z] = 1

        # Define golden means for phase encoding directions
        GA_y = (1 + np.sqrt(5)) / 2  # Golden ratio for y-direction
        GA_z = np.sqrt(3)            # Irrational value for z-direction

        # Estimate number of additional PE locations needed to achieve acceleration
        num_additional_PE = np.max([int((Ny*Nz) / accel - np.count_nonzero(tmp_mask)), 0])
        num_added_PE = 0
        n_attempts = 0

        # Iteratively fill mask until acceleration/tolerance met or attempts exhausted
        while (compute_accel(tmp_mask) > accel - tol) and (n_attempts < max_attempts) and (num_additional_PE > 0):
            golden_angle_idxs = np.arange(num_added_PE, num_added_PE + num_additional_PE)
            golden_angles = [(GA_y * i, GA_z * i) for i in golden_angle_idxs]
            fractional_part, integral_part = np.modf(golden_angles)
            sampled_idxs = np.array([Ny * fractional_part[:, 0], Nz * fractional_part[:, 1]]).astype(int)

            # Update mask with sampled indices
            tmp_mask[sampled_idxs[0], sampled_idxs[1]] = 1
            num_added_PE += num_additional_PE
            n_attempts += 1

            # Recalculate number of PE locations needed
            num_additional_PE = np.max([int((Ny*Nz)/accel - np.count_nonzero(tmp_mask)), 0])

        mask = tmp_mask
        return mask

    else:
        raise ValueError(f"Invalid mode: {mode}. Must be '2D' or '3D'.")

def compute_accel(mask):
    """
    Compute actual acceleration based on a binary mask.

    Parameters
    ----------
    mask : ndarray
        Binary mask (sampled=1, unsampled=0).

    Returns
    -------
    accel : float
        Acceleration factor (total locations / sampled locations).

    Examples
    --------
    >>> mask = np.ones((64, 64))
    >>> accel = compute_accel(mask)
    """
    return np.prod(mask.shape) / np.count_nonzero(mask)
