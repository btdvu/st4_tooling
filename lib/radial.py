"""
Radial Trajectory Generation for MRI

Generate radial k-space trajectories and gradient delay correction for MRI.
Supports golden-angle sampling, calibration trajectories, and delay estimation.

Author: Brian-Tinh Vu
Date: 12/05/2025
Dependencies: numpy, sigpy, scipy

Citations:
    Gradient delay methods based on:
    1) Block KT, Uecker M. Simple method for adaptive gradient-delay compensation in radial MRI.
       Proceedings of the 19th Annual Meeting of ISMRM, Montreal, Canada, 2011. p. 2816.
    2) Rosenzweig S, Holme HCM, Uecker M. Simple auto-calibrated gradient delay estimation from 
       few spokes using Radial Intersections (RING). Magn Reson Med. 2019;81(3):1898-1906.
       doi: 10.1002/mrm.27506

Quick Start:
    traj = trajGoldenAngle(N=128, n_spokes=360)
    calib_traj = trajCalibration(N=128, n_calib_pairs=32)
    shift_matrix = estGradDelayMultiCh(ksp_calib, calib_traj)
    corrected_traj = shiftTrajectory(traj, shift_matrix)
"""

import numpy as np
import sigpy as sp
from scipy.optimize import least_squares


def trajGoldenAngle(N, n_spokes, starting_angle=0, negative_angle=False):
    """
    Generate golden-angle radial k-space trajectory.
    
    Parameters
    ----------
    N : int
        Number of samples along each radial spoke.
    n_spokes : int
        Number of radial spokes (projections).
    starting_angle: float
        Angle of the first golden-angle spoke.
    negative_angle : bool, optional
        If True, invert trajectory direction. Default: False.
    
    Returns
    -------
    traj : np.ndarray
        K-space coordinates shape: (n_spokes, N, 2).
        Contains (kx, ky) coordinates for each spoke.
    
    Notes
    -----
    Uses 111.25° golden-angle increment for optimal coverage.
    """
    # Initialize trajectory array
    traj = np.zeros((n_spokes, N, 2), dtype=float)

    # Golden-angle increment (degrees)
    PI = 3.141592
    angle_inc = 111.25

    # Compute radial positions along spoke
    k_max = N//2 - 0.5
    line = np.linspace(-k_max, k_max, N)

    # Generate each radial spoke
    for i_projection in range(n_spokes):
        # Calculate spoke angle
        angle_in_deg = i_projection * angle_inc + starting_angle
        angle = angle_in_deg * PI / 180.0

        # Direction vector for this spoke
        kx_dir = np.cos(angle)
        ky_dir = np.sin(angle)

        if negative_angle:
            kx_dir *= -1
            ky_dir *= -1

        # Scale radial positions by direction
        line_x = kx_dir * line
        line_y = ky_dir * line

        # Store coordinates
        traj[i_projection, :, 0] = line_x
        traj[i_projection, :, 1] = line_y

    return traj


def trajCalibration(N, n_calib_pairs):
    """
    Generate calibration trajectory with parallel/antiparallel spokes.
    
    Parameters
    ----------
    N : int
        Number of samples along each radial spoke.
    n_calib_pairs : int
        Number of parallel/antiparallel spoke pairs.
    
    Returns
    -------
    traj : np.ndarray
        K-space coordinates shape: (2*n_calib_pairs, N, 2).
        Alternating parallel and antiparallel spokes.
    
    Notes
    -----
    Creates interleaved spoke pairs for calibration data.
    Useful for gradient delay artifact correction.
    """
    # Create parallel and antiparallel spoke sets
    traj_par = trajGoldenAngle(N, n_calib_pairs)
    traj_antipar = trajGoldenAngle(N, n_calib_pairs, negative_angle=True)

    # Interleave parallel and antiparallel spokes
    traj = np.zeros((2*n_calib_pairs, N, 2))
    traj[0::2] = traj_par
    traj[1::2] = traj_antipar

    return traj


def estGradDelayMultiCh(ksp_calib, traj_calib):
    """
    Estimate gradient delays from multi-channel calibration data.
    
    Parameters
    ----------
    ksp_calib : np.ndarray
        Multi-channel calibration k-space data shape: (n_coils, n_spokes, n_ro).
    traj_calib : np.ndarray
        Calibration trajectory shape: (n_spokes, n_ro, 2).
    
    Returns
    -------
    shift_matrix : np.ndarray
        2x2 gradient delay shift matrix.
    
    Notes
    -----
    Uses L2-norm weighting to combine coil-specific delay estimates.
    """
    # Extract parameters
    n_coils, n_spokes, n_ro = ksp_calib.shape
    n_spoke_pairs = n_spokes//2

    # Compute coil weights by L2-norm
    coil_weights = np.zeros(n_coils)
    for i_coil in range(n_coils):
        coil_weights[i_coil] = np.linalg.norm(ksp_calib[i_coil])
    
    coil_weights = coil_weights/np.sum(coil_weights)

    # Compute shift matrices for each coil
    shift_matrices = np.zeros((n_coils, 2, 2))
    for i_coil in range(n_coils):
        shift_matrices[i_coil] = estGradDelay(ksp_calib[i_coil], traj_calib)

    # Weighted sum of shift matrices
    shift_matrix = np.zeros((2,2))
    for i_coil in range(n_coils):
        shift_matrix += coil_weights[i_coil] * shift_matrices[i_coil]
    
    return shift_matrix


def estGradDelay(ksp_calib, traj_calib, thresh=0.1):
    """
    Estimate gradient delays from calibration data.
    
    Based on Block & Uecker (ISMRM 2011) cross-correlation method for
    parallel/antiparallel spoke analysis with ellipse model fitting.
    
    Parameters
    ----------
    ksp_calib : np.ndarray
        Single-channel calibration k-space data shape: (n_spokes, n_ro).
    traj_calib : np.ndarray
        Calibration trajectory shape: (n_spokes, n_ro, 2).
    thresh : float, optional
        Threshold for spoke support detection. Default: 0.1.
    
    Returns
    -------
    shift_matrix : np.ndarray
        2x2 gradient delay shift matrix.
    
    Notes
    -----
    Uses cross-correlation of parallel/antiparallel spokes and least-squares fitting.
    Related to RING method (Rosenzweig et al., MRM 2019) for ellipse parameter estimation.
    """
    n_ro = ksp_calib.shape[-1]
    n_calib_pairs = ksp_calib.shape[0]//2

    # Separate parallel and antiparallel spokes
    par_spokes = ksp_calib[0::2]
    antipar_spokes = ksp_calib[1::2]

    # Flip antiparallel spokes along readout
    antipar_spokes_flip = np.flip(antipar_spokes, axis=-1)

    # Apply IFFT along readout direction
    par_spokes_f = sp.ifft(par_spokes, axes=(-1,))
    antipar_spokes_flip_f = sp.ifft(antipar_spokes_flip, axes=(-1,))

    # Complex conjugate of antiparallel spokes
    antipar_spokes_flip_f_conj = np.conj(antipar_spokes_flip_f)

    # Cross-correlation function in Fourier domain
    g_r = par_spokes_f * antipar_spokes_flip_f_conj

    # Initialize mask for each spoke pair
    mask = np.zeros((n_calib_pairs, n_ro), dtype=np.int32)

    for i_spoke in range(n_calib_pairs):
        tmp_spoke = ksp_calib[2*i_spoke-1]
        mask[i_spoke] = _findSpokeSupport(tmp_spoke, thresh=thresh)

    # Apply mask to cross-correlation
    g_r = mask * g_r

    # Compute phase slopes using linear regression
    slope_collector = np.zeros(n_calib_pairs)

    for i_spoke in range(0, n_calib_pairs):
        g_r_cropped = np.squeeze(g_r[i_spoke, np.nonzero(mask[i_spoke])])
        poly_coeffs = np.polyfit(np.arange(0, len(g_r_cropped)), np.unwrap(np.angle(g_r_cropped)), 1)
        slope_collector[i_spoke] = poly_coeffs[0]

    # Compute k-space shifts
    delta_k = (1/2)*slope_collector*n_ro/(2*np.pi)

    delta_k_dupl = np.zeros(2*len(delta_k))
    delta_k_dupl[0::2] = delta_k
    delta_k_dupl[1::2] = delta_k

    # Compute projection angles
    theta_proj = _trajToAngles(traj_calib)

    # Initial guess for [Sx, Sy, Sxy]
    x0 = np.array([0, 0, 0])

    result = least_squares(
        _residuals,
        x0,
        args=(theta_proj, delta_k_dupl),
        bounds=([-n_ro/2, -n_ro/2, -n_ro/2], [n_ro/2, n_ro/2, n_ro/2])
    )

    Sx, Sy, Sxy = result.x

    return np.array([[Sx, Sxy], [Sxy, Sy]])


def shiftTrajectory(traj, shift_matrix):
    """
    Apply gradient delay correction to radial trajectory.
    
    Implements trajectory correction from Block & Uecker (ISMRM 2011) and
    Rosenzweig et al. (MRM 2019) gradient delay compensation methods.
    
    Parameters
    ----------
    traj : np.ndarray
        Input trajectory shape: (n_spokes, n_ro, n_dim).
    shift_matrix : np.ndarray
        2x2 gradient delay shift matrix.
    
    Returns
    -------
    shifted_traj : np.ndarray
        Corrected trajectory shape: (n_spokes, n_ro, n_dim).
    
    Notes
    -----
    Computes angle-dependent k-space shifts and applies to nominal trajectory.
    """
    # Extract parameters
    n_spokes = traj.shape[0]
    ndim = traj.shape[-1]

    # Compute projection angles for all spokes
    proj_angles = _trajToAngles(traj)

    # Construct projection direction unit vectors
    proj_unit_vectors = np.array([np.cos(proj_angles), np.sin(proj_angles)])

    # Compute k-space shifts for each projection angle
    k_shifts = shift_matrix@proj_unit_vectors

    # Reshape shifts for broadcasting and to match the trajectory dimensions
    k_shifts_tobeAdded = np.reshape(k_shifts.T, (n_spokes, 1, 2))
    if ndim >= 2:
        zero_pad = np.zeros(list(k_shifts_tobeAdded.shape[:2])+[ndim-2])
        k_shifts_tobeAdded = np.concatenate([k_shifts_tobeAdded, zero_pad], axis=-1)

    # Apply shifts to nominal trajectory
    shifted_traj = traj - k_shifts_tobeAdded

    return shifted_traj




def _gradientDelayModel(theta, shifts):
    """
    Gradient delay model for k-space shifts.
    
    Parameters
    ----------
    theta : np.ndarray
        Projection angles.
    shifts : tuple
        (Sx, Sy, Sxy) gradient delay parameters.
    
    Returns
    -------
    np.ndarray
        Predicted k-space shifts.
    """
    Sx, Sy, Sxy = shifts
    term1 = Sx*np.cos(theta)**2
    term2 = Sxy*2*np.cos(theta)*np.sin(theta)
    term3 = Sy*np.sin(theta)**2

    return term1 + term2 + term3



def _residuals(shifts, theta, delta_k):
    """
    Compute residuals for gradient delay optimization.
    
    Parameters
    ----------
    shifts : tuple
        (Sx, Sy, Sxy) gradient delay parameters.
    theta : np.ndarray
        Projection angles.
    delta_k : np.ndarray
        Measured k-space shifts.
    
    Returns
    -------
    np.ndarray
        Residuals between model and measurements.
    """
    return _gradientDelayModel(theta, shifts) - delta_k

def _findSpokeSupport(spoke, thresh=0.1):
    """
    Find support region of radial spoke using region growing.
    
    Parameters
    ----------
    spoke : np.ndarray
        Radial spoke k-space data.
    thresh : float, optional
        Threshold for support detection. Default: 0.1.
    
    Returns
    -------
    mask : np.ndarray
        Binary mask indicating support region.
    
    Notes
    -----
    Grows region from maximum signal point using threshold criterion.
    """
    # Get readout length
    n_ro = len(spoke)

    # Compute absolute value in image domain
    spoke_f = sp.ifft(spoke)
    spoke_f_abs = np.abs(spoke_f)

    # Find maximum signal and location
    max_signal = np.max(spoke_f_abs)
    idx_max_signal = np.argmax(spoke_f_abs)

    # Determine threshold
    signal_thresh = thresh * max_signal

    # Initialize region growing parameters
    keep_growing_L = True
    keep_growing_R = True
    left_radius = 0
    right_radius = 0

    # Initialize mask
    mask = np.zeros_like(spoke_f_abs)
    mask[idx_max_signal] = 1

    # Region growing loop
    while keep_growing_L or keep_growing_R:
        left_idx = idx_max_signal - left_radius
        right_idx = idx_max_signal + right_radius

        # Check array bounds
        is_left_done = left_idx < 0
        is_right_done = right_idx > n_ro-1

        # Grow left border
        if (not is_left_done) and (spoke_f_abs[left_idx] > signal_thresh):
            mask[left_idx] = 1
        else:
            keep_growing_L = False
        
        # Grow right border
        if (not is_right_done) and (spoke_f_abs[right_idx] > signal_thresh):
            mask[right_idx] = 1
        else:
            keep_growing_R = False

        # Expand radius
        right_radius += 1
        left_radius += 1

    return mask


def _trajToAngles(traj):
    """
    Extract projection angles from radial trajectory.
    
    Parameters
    ----------
    traj : np.ndarray
        Radial trajectory shape: (n_spokes, n_ro, 2).
    
    Returns
    -------
    proj_angles : np.ndarray
        Projection angles in radians.
    
    Notes
    -----
    Uses endpoint of each spoke to compute angle with arctan2.
    """
    # Get readout length
    n_ro = traj.shape[1]

    # Get endpoint of each spoke
    traj_endpt_x = np.squeeze(traj[:, n_ro-1, 0])
    traj_endpt_y = np.squeeze(traj[:, n_ro-1, 1])

    # Compute projection angles
    proj_angles = np.arctan2(traj_endpt_y, traj_endpt_x)

    # Modulo 2*pi
    proj_angles = np.mod(proj_angles, 2*np.pi)

    return proj_angles