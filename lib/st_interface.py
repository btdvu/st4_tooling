# Functions to interface between SequenceTree and Python.

import numpy as np
import sigpy.plot as pl

def get_pe_idxs_from_mask(mask, mode="2D"):
    """
    Extract phase encoding (PE) indices from a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        A binary mask array indicating valid PE locations.
        For "2D" mode, expected shape: (Nx, Ny).
        For "3D" mode, expected shape: (Ny, Nz).
    mode : str, optional
        Mode of extraction. "2D" performs extraction along columns.
        "3D" performs flattening across two dimensions. 
        Must be "2D" or "3D". Default is "2D".

    Returns
    -------
    idxs : np.ndarray
        Array of PE indices where the mask is non-zero.

    Raises
    ------
    ValueError
        If an invalid mode is supplied.

    Examples
    --------
    >>> mask = np.array([[0, 1], [1, 0]])
    >>> get_pe_idxs_from_mask(mask, mode="2D")
    array([0, 1])
    """
    if mode == "2D":
        Ny = mask.shape[1]
        # flatten the mask, get PE indices where the mask is nonzero
        tmp_mask = np.sum(mask, axis=0) > 0
        # get PE indices
        idxs = np.squeeze(np.nonzero(tmp_mask))
        return idxs

    elif mode == "3D":
        Ny = mask.shape[0]  # "fast" PE dimension
        Nz = mask.shape[1]  # "slow" PE dimension
        idxs = np.nonzero(mask)
        idxs_flat = np.sort(idxs[0] + Ny * idxs[1])
        return idxs_flat

    else:
        raise ValueError(f"Invalid mode: {mode}. Must be '2D' or '3D'.")

def save_idxs_to_file(file_path, int_array):
    """
    Saves a list of indices to a text file, one per line.

    Parameters
    ----------
    file_path : str
        Path to the output text file.
    int_array : array-like
        List or array of integer indices to be saved.

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
