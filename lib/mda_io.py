"""
MDA File I/O for MRI Data

Read/write MDA (multi-dimensional array) files for MRI k-space and image data.
Supports complex, float, and integer data types with Fortran ordering.

Author: Brian-Tinh Vu
Date: 11/08/2025
Dependencies: numpy

Quick Start:
    data = readmda('kspace.mda')
    writemda('image.mda', recon_data)
"""

import os
import numpy as np

def readmda(fname, folder=os.getcwd(), dtype=np.double):
    """
    Read MDA file into NumPy array.
    
    Parameters
    ----------
    fname : str
        MDA filename to read.
    folder : str, optional
        Directory containing the file. Default: current directory.
    dtype : np.dtype, optional
        Output data type. Default: np.double.
    
    Returns
    -------
    raw_data : np.ndarray
        N-dimensional array from file.
    
    Raises
    ------
    ValueError
        If file format is invalid.
    
    Examples
    --------
    >>> data = readmda('array.mda')
    >>> print(data.shape)
    (128, 128)
    """
    # Construct file path and read header
    file_name = os.path.join(folder, fname)

    fid = open(file_name, 'rb')
    tmp = np.fromfile(fid, np.intc)
    fid.close()

    # Parse header for data type and dimensions
    dtype_code = tmp[0]
    header_bit_depth = 4

    if dtype_code > 0:
        # Old MDA format: dtype_code = number of dimensions
        num_matrix_dims = dtype_code
        matrix_dims = tmp[1 : 1 + num_matrix_dims]
        total_num_elements = np.int64(np.prod(matrix_dims))
        dtype_code = -1  # Default to complex handling
        bit_offset = (1 + num_matrix_dims) * header_bit_depth
    else:
        # New MDA format: negative dtype codes
        num_matrix_dims = tmp[2]
        matrix_dims = tmp[3 : 3 + num_matrix_dims]
        total_num_elements = np.int64(np.prod(matrix_dims))
        bit_offset = (3 + num_matrix_dims) * header_bit_depth

    if dtype_code == -1:
        # Complex or floating point data
        print('\n\n READING FILE NOW . . . \n\n')
        fid = open(file_name, 'rb')
        fid.seek(bit_offset, os.SEEK_SET)
        data_stream = np.fromfile(fid, np.float32)
        fid.close()
        length_data_stream = data_stream.size

        # Extract interleaved real/imaginary parts
        inds_to_keep = np.arange(2 * total_num_elements, dtype='int64')
        data_stream = data_stream[inds_to_keep]
        real_part = data_stream[0:length_data_stream:2].copy().reshape(matrix_dims, order='F')
        imag_part = data_stream[1:length_data_stream:2].copy().reshape(matrix_dims, order='F')

        if np.count_nonzero(imag_part.flatten()) > 0:
            # Complex data
            dtype = np.complex128
            raw_data = real_part + 1j * imag_part
        else:
            # Real data
            raw_data = real_part.astype(dtype=dtype)

    elif dtype_code == -4:
        # Integer data (int16)
        fid = open(file_name, 'rb')
        fid.seek(bit_offset, os.SEEK_SET)
        data_stream = np.fromfile(fid, np.int16)
        fid.close()
        length_data_stream = data_stream.size
        raw_data = data_stream.copy().reshape(matrix_dims, order='F')

    return raw_data


def writemda(fname, mat):
    """
    Write NumPy array to MDA file.
    
    Parameters
    ----------
    fname : str
        Output MDA filename.
    mat : np.ndarray
        N-dimensional array to write.
    
    Returns
    -------
    None
    
    Examples
    --------
    >>> arr = np.random.randint(0, 10, (64, 64), dtype='int16')
    >>> writemda('output.mda', arr)
    """
    fid = open(fname, 'wb')
    is_int = np.issubdtype(mat.dtype, np.integer)
    mat_size = mat.shape
    num_dims = len(mat_size)
    total_num_elements = np.prod(mat.shape)

    if is_int:
        # Write integer matrix (int16 format)
        mat = mat.flatten(order='F')
        fid.write((-4).to_bytes(4, byteorder=sys.byteorder, signed=1))  # Type code -4
        fid.write((2).to_bytes(4, byteorder=sys.byteorder, signed=1))   # Version/format
        fid.write(num_dims.to_bytes(4, byteorder=sys.byteorder, signed=1))
        for iter_dim in range(num_dims):
            fid.write(mat_size[iter_dim].to_bytes(4, byteorder=sys.byteorder, signed=1))
        for iter_element in range(total_num_elements):
            fid.write(mat[iter_element].tolist().to_bytes(2, byteorder=sys.byteorder, signed=1))
    else:
        # Write floating point or complex matrix
        fid.write(num_dims.to_bytes(4, byteorder=sys.byteorder, signed=1))
        for iter_dim in range(num_dims):
            fid.write(mat_size[iter_dim].to_bytes(4, byteorder=sys.byteorder, signed=1))
        flatten_order = 'F'
        mat_real = np.real(mat).flatten(order=flatten_order).astype(np.single)
        mat_imag = np.imag(mat).flatten(order=flatten_order).astype(np.single)
        for iter_element in range(total_num_elements):
            fid.write(mat_real[iter_element].tobytes())
            fid.write(mat_imag[iter_element].tobytes())
    fid.close()
    return
