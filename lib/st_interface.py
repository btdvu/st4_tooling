"""
SequenceTree Interface for MRI

Interface functions between SequenceTree and Python for MRI pulse sequence
programming and reconstruction workflows.

Author: Brian-Tinh Vu
Date: 11/08/2025
Dependencies: numpy, sigpy

Quick Start:
    pe_indices = get_pe_idxs_from_mask(mask, mode="2D")
    save_idxs_to_file('pe_lines.txt', pe_indices)
"""

import numpy as np
import sigpy.plot as pl

def get_pe_idxs_from_mask(mask, mode="2D"):
    """
    Extract phase encoding indices from binary mask.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask indicating sampled PE locations.
        Shape: (Nx, Ny) for 2D, (Ny, Nz) for 3D.
    mode : {'2D', '3D'}, optional
        Extraction mode. '2D' along columns, '3D' flattened. Default: '2D'.
    
    Returns
    -------
    idxs : np.ndarray
        PE indices where mask is non-zero.
    
    Raises
    ------
    ValueError
        If mode is invalid.
    
    Examples
    --------
    >>> mask = np.array([[0, 1], [1, 0]])
    >>> get_pe_idxs_from_mask(mask, mode="2D")
    array([0, 1])
    """
    if mode == "2D":
        Ny = mask.shape[1]
        # Extract PE indices from summed mask columns
        tmp_mask = np.sum(mask, axis=0) > 0
        idxs = np.squeeze(np.nonzero(tmp_mask))
        return idxs

    elif mode == "3D":
        Ny = mask.shape[0]  # Fast PE dimension
        Nz = mask.shape[1]  # Slow PE dimension
        idxs = np.nonzero(mask)
        idxs_flat = np.sort(idxs[0] + Ny * idxs[1])
        return idxs_flat

    else:
        raise ValueError(f"Invalid mode: {mode}. Must be '2D' or '3D'.")

def save_idxs_to_file(file_path, int_array):
    """
    Save integer indices to text file.
    
    Parameters
    ----------
    file_path : str
        Output text file path.
    int_array : array-like
        Integer indices to save.
    
    Returns
    -------
    None
    
    Examples
    --------
    >>> save_idxs_to_file("indices.txt", [1, 2, 3])
    """
    with open(file_path, 'w') as f:
        for num in int_array:
            f.write(f"{num}\n")
