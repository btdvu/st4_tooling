import os
import numpy as np

def readmda(fname, folder=os.getcwd(), dtype=np.double):
    """
    Reads an .mda (multi-dimensional array) file and loads the data as a NumPy array.

    Parameters
    ----------
    fname : str
        Filename of the MDA file to read.
    folder : str, optional
        Folder path where the MDA file is located. Defaults to current directory.
    dtype : np.dtype, optional
        Desired NumPy dtype for output. Defaults to np.double.

    Returns
    -------
    raw_data : np.ndarray
        N-dimensional array read from the file.

    Raises
    ------
    ValueError
        If the file format or data type is invalid.

    Examples
    --------
    >>> data = readmda('array.mda')
    >>> print(data.shape)
    (128, 128)
    """
    # Construct the full file path
    file_name = os.path.join(folder, fname)

    # Open the file and read the header to determine type and dimensions
    fid = open(file_name, 'rb')
    tmp = np.fromfile(fid, np.intc)
    fid.close()

    # The first integer is a code indicating the data type
    dtype_code = tmp[0]
    header_bit_depth = 4  # Each header value uses 4 bytes

    if dtype_code > 0:
        # Old format: dtype_code contains the number of dimensions
        num_matrix_dims = dtype_code
        matrix_dims = tmp[1 : 1 + num_matrix_dims]
        total_num_elements = np.int64(np.prod(matrix_dims))
        dtype_code = -1  # Default to "complex" handling
        bit_offset = (1 + num_matrix_dims) * header_bit_depth
    else:
        # Newer format: dtype_code is negative, -4 for int types
        num_matrix_dims = tmp[2]
        matrix_dims = tmp[3 : 3 + num_matrix_dims]
        total_num_elements = np.int64(np.prod(matrix_dims))
        bit_offset = (3 + num_matrix_dims) * header_bit_depth

    if dtype_code == -1:
        # Complex, float32, or float64 matrix
        print('\n\n READING FILE NOW . . . \n\n')
        fid = open(file_name, 'rb')
        fid.seek(bit_offset, os.SEEK_SET)
        data_stream = np.fromfile(fid, np.float32)
        fid.close()
        length_data_stream = data_stream.size

        # Interleaved real and imaginary parts
        inds_to_keep = np.arange(2 * total_num_elements, dtype='int64')
        data_stream = data_stream[inds_to_keep]
        real_part = data_stream[0:length_data_stream:2].copy().reshape(matrix_dims, order='F')
        imag_part = data_stream[1:length_data_stream:2].copy().reshape(matrix_dims, order='F')

        if np.count_nonzero(imag_part.flatten()) > 0:
            # Complex data
            dtype = np.complex128
            raw_data = real_part + 1j * imag_part
        else:
            # Purely real data
            raw_data = real_part.astype(dtype=dtype)

    elif dtype_code == -4:
        # Integer matrix (int16)
        fid = open(file_name, 'rb')
        fid.seek(bit_offset, os.SEEK_SET)
        data_stream = np.fromfile(fid, np.int16)
        fid.close()
        length_data_stream = data_stream.size
        raw_data = data_stream.copy().reshape(matrix_dims, order='F')

    return raw_data


def writemda(fname, mat):
    """
    Writes a NumPy array to an .mda (multi-dimensional array) file in binary format.

    Parameters
    ----------
    fname : str
        Target filename to write the MDA data.
    mat : np.ndarray
        The n-dimensional NumPy array to write.

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
        # Write integer matrix in MDA int16 format
        mat = mat.flatten(order='F')
        fid.write((-4).to_bytes(4, byteorder=sys.byteorder, signed=1))  # Type code -4 for integer types
        fid.write((2).to_bytes(4, byteorder=sys.byteorder, signed=1))   # Possibly indicating version or format
        fid.write(num_dims.to_bytes(4, byteorder=sys.byteorder, signed=1))
        for iter_dim in range(num_dims):
            fid.write(mat_size[iter_dim].to_bytes(4, byteorder=sys.byteorder, signed=1))
        for iter_element in range(total_num_elements):
            fid.write(mat[iter_element].tolist().to_bytes(2, byteorder=sys.byteorder, signed=1))
    else:
        # Write floating point (or complex) matrix
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
