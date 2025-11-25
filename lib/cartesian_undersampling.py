"""
Cartesian Undersampling for MRI

Generate Cartesian sampling masks for accelerated MRI acquisitions with ACS regions.
Supports 2D regular spacing and 3D golden-angle sampling patterns.

Author: Brian-Tinh Vu
Date: 11/08/2025
Dependencies: numpy

Quick Start:
    mask_2d = regular_undersampling((256, 256), accel=4.0, mode="2D")
    mask_3d = regular_undersampling((128, 128), accel=8.0, mode="3D")
    actual_accel = compute_accel(mask_2d)
"""

import numpy as np

def regular_undersampling(img_shape, accel, calib=24, mode="2D", tol=0.1, max_attempts=100):
    """
    Generate Cartesian undersampling masks for 2D/3D MRI acquisitions.
    
    Parameters
    ----------
    img_shape : tuple
        (Nx, Ny) for 2D or (Ny, Nz) for 3D image dimensions.
    accel : float
        Target acceleration factor R = total_lines / sampled_lines (2.0-8.0 typical).
    calib : int, optional
        ACS region size in phase-encode direction(s). Default: 24.
    mode : {'2D', '3D'}, optional
        '2D': regular spacing, '3D': golden-angle sampling. Default: '2D'.
    tol : float, optional
        Acceleration tolerance |R_actual - R_target| < tol. Default: 0.1.
    max_attempts : int, optional
        Maximum iterations for convergence. Default: 100.
    
    Returns
    -------
    mask : np.ndarray
        Binary sampling mask (1=sampled, 0=skipped). Shape matches img_shape.
    
    Raises
    ------
    ValueError
        If mode is not '2D' or '3D'.
    
    Examples
    --------
    >>> mask_2d = regular_undersampling((256, 256), accel=4.0, mode="2D")
    >>> mask_3d = regular_undersampling((128, 128), accel=8.0, mode="3D")
    """
    if mode == "2D":
        # Extract dimensions for 2D Cartesian trajectory
        Nx = img_shape[0]  # Readout direction
        Ny = img_shape[1]  # Phase encoding direction

        # Initialize 1D mask for phase-encode line selection
        tmp_mask = np.zeros(Ny, dtype=np.complex64)
        actual_accel = -1

        # Binary search parameters for optimal PE line spacing
        test_increment = accel  # Initial guess: skip every (accel-1) lines
        inc_upper_bound = test_increment * 1.5
        inc_lower_bound = test_increment * 0.5
        n_attempts = 0
        isMetTolerance = False

        # Binary search to achieve target acceleration within tolerance
        while (not isMetTolerance) and (n_attempts < max_attempts):
            tmp_mask = np.zeros(Ny, dtype=np.complex64)
            n_attempts += 1

            # Define ACS region - centered in k-space
            calib_start = Ny // 2 - calib // 2
            calib_end = Ny // 2 + calib // 2
            tmp_mask[calib_start:calib_end] = 1

            # Sample phase-encode lines outside ACS at regular intervals
            sampled_idxs = (np.round(np.arange(0, Ny - 1/2, test_increment))).astype(int)
            tmp_mask[sampled_idxs] = 1

            # Calculate actual acceleration achieved
            actual_accel = compute_accel(tmp_mask)

            # Binary search: adjust spacing based on acceleration achieved
            if actual_accel < accel:
                # Under-accelerated: need larger spacing
                inc_lower_bound = test_increment
                test_increment = 0.5 * (inc_lower_bound + inc_upper_bound)
            else:
                # Over-accelerated: need smaller spacing
                inc_upper_bound = test_increment
                test_increment = 0.5 * (inc_lower_bound + inc_upper_bound)

            # Check if acceleration meets tolerance
            isMetTolerance = np.abs(accel - compute_accel(tmp_mask)) < tol

        # Expand 1D PE mask to full 2D k-space mask
        mask = tmp_mask
        mask = np.stack([mask] * Nx, axis=0)
        return mask

    elif mode == "3D":
        # 3D Cartesian: undersample in phase-encode plane (Ny×Nz)
        Ny = img_shape[0]  # Fast phase-encode direction
        Nz = img_shape[1]  # Slow phase-encode direction
        tmp_mask = np.zeros((Ny, Nz), dtype=np.complex64)

        # Define square ACS region in center of PE plane
        calib_start_y = Ny // 2 - calib // 2
        calib_end_y = Ny // 2 + calib // 2
        calib_start_z = Nz // 2 - calib // 2
        calib_end_z = Nz // 2 + calib // 2
        tmp_mask[calib_start_y:calib_end_y, calib_start_z:calib_end_z] = 1

        # Golden-angle sampling parameters
        GA_y = (1 + np.sqrt(5)) / 2  # Golden ratio for y-direction
        GA_z = np.sqrt(3)            # Irrational number for z-direction

        # Calculate additional PE points needed for target acceleration
        total_points = Ny * Nz
        current_points = np.count_nonzero(tmp_mask)
        num_additional_PE = np.max([int(total_points / accel - current_points), 0])
        num_added_PE = 0
        n_attempts = 0

        # Add golden-angle sampled points until acceleration target met
        while (compute_accel(tmp_mask) > accel - tol) and (n_attempts < max_attempts) and (num_additional_PE > 0):
            # Generate golden-angle sequence
            golden_angle_idxs = np.arange(num_added_PE, num_added_PE + num_additional_PE)
            golden_angles = [(GA_y * i, GA_z * i) for i in golden_angle_idxs]
            
            # Extract fractional parts for uniform distribution
            fractional_part, integral_part = np.modf(golden_angles)
            
            # Convert to integer PE indices
            sampled_idxs = np.array([Ny * fractional_part[:, 0], Nz * fractional_part[:, 1]]).astype(int)

            # Update mask with new PE locations
            tmp_mask[sampled_idxs[0], sampled_idxs[1]] = 1
            num_added_PE += num_additional_PE
            n_attempts += 1

            # Recalculate points needed
            current_points = np.count_nonzero(tmp_mask)
            num_additional_PE = np.max([int(total_points / accel - current_points), 0])

        mask = tmp_mask
        return mask

    else:
        raise ValueError(f"Invalid mode: {mode}. Must be '2D' or '3D'.")

def compute_accel(mask):
    """
    Calculate acceleration factor from sampling mask.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask (1=sampled, 0=skipped).
    
    Returns
    -------
    accel : float
        Acceleration factor R = total_points / sampled_points.
    
    Examples
    --------
    >>> mask = np.ones((64, 64))
    >>> compute_accel(mask)
    1.0
    """
    return np.prod(mask.shape) / np.count_nonzero(mask)
