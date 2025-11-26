"""
Siemens TWIX Data I/O for MRI

Read Siemens TWIX .dat files for MRI raw k-space data processing.
Supports XA format with Cartesian undersampling masks.

Author: Brian-Tinh Vu
Date: 11/26/2025
Dependencies: numpy

Quick Start:
    data = read_twix_siemens_XA('measurement.dat', nviews=128)
    ksp = read_twix_from_cartesian_mask('measurement.dat', mask, n_prep=100)
"""
import numpy as np
from st_interface import get_pe_idxs_from_mask

# TODO: add a safety check to see if there are any additional ADCs that were acquired that we did not read in
def read_twix_siemens_XA(datfile, nviews):
    """
    Read Siemens TWIX .dat file (XA format) for k-space data.
    
    Parameters
    ----------
    datfile : str
        Path to Siemens TWIX .dat file (XA format).
    nviews : int
        Number of phase encoding lines to read.
    
    Returns
    -------
    np.ndarray
        Complex k-space data shape: (num_readouts, nviews, num_channels).
    
    Notes
    -----
    Skips PMUDATA blocks and extracts imaging data from XA30 format.
    Handles interleaved real/imaginary data conversion.
    """
    with open(datfile, "rb") as f:
        # Skip header and calculate measurement length
        _ = np.fromfile(f, dtype=np.uint32, count=1)
        nMeas = int(np.fromfile(f, dtype=np.uint32, count=1)[0])

        measLen = 0

        # Calculate total measurement length from all measurements except last
        for _ in range(nMeas - 1):
            f.seek(16, 1)
            tmp = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
            measLen = measLen + int(np.ceil(tmp / 512.0) * 512)
            f.seek(152 - 24, 1)

        # Seek to data section and read header
        offset = 2 * 4 + 152 * 64 + 126 * 4 + measLen
        f.seek(offset, 0)
        headerSize = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
        _ = np.fromfile(f, dtype=np.uint8, count=headerSize - 4)

        # Skip PMUDATA blocks and find first imaging data
        num_readouts = 0
        cnt = 0

        while num_readouts == 0:
            _ = np.fromfile(f, dtype=np.uint8, count=48)
            nr_arr = np.fromfile(f, dtype=np.uint16, count=1)
            if nr_arr.size == 0:
                raise EOFError("Unexpected end of file while searching for first readout count.")
            num_readouts = int(nr_arr[0])

            if num_readouts == 0:
                # Skip PMUDATA block
                f.seek(-50, 1)
                ulDMALength = 184
                mdhStart = -ulDMALength
                data_u8 = np.fromfile(f, dtype=np.uint8, count=ulDMALength)
                data_u8 = data_u8[mdhStart:]
                data_u8[3] = (data_u8[3] & 1)
                ulDMALength = int(np.frombuffer(data_u8[0:4].tobytes(), dtype=np.uint32)[0])
                _ = np.fromfile(f, dtype=np.uint8, count=ulDMALength - 184)
                cnt += 1

        # Get number of channels and prepare data array
        num_channels = int(np.fromfile(f, dtype=np.uint16, count=1)[0])
        f.seek(-(48 + 4), 1)

        print(f"Reading: {datfile}\nProgress...")
        data = np.zeros((num_readouts, nviews, num_channels), dtype=np.complex64)

        # Read k-space data for all phase encoding lines
        iPE = 0
        percentFinished = 0

        while True:
            iPE += 1

            # Read header for current phase encoding line
            _ = np.fromfile(f, dtype=np.uint8, count=48)
            nr_arr = np.fromfile(f, dtype=np.uint16, count=1)
            if nr_arr.size == 0:
                raise EOFError("Unexpected end of file while reading readout count.")
            num_readouts_now = int(nr_arr[0])

            if num_readouts_now == 0:
                # Skip PMUDATA block
                f.seek(-50, 1)
                ulDMALength = 184
                mdhStart = -ulDMALength
                data_u8 = np.fromfile(f, dtype=np.uint8, count=ulDMALength)
                data_u8 = data_u8[mdhStart:]
                data_u8[3] = (data_u8[3] & 1)
                ulDMALength = int(np.frombuffer(data_u8[0:4].tobytes(), dtype=np.uint32)[0])
                _ = np.fromfile(f, dtype=np.uint8, count=ulDMALength - 184)
                iPE -= 1
                continue

            # Skip remaining header bytes
            _ = np.fromfile(f, dtype=np.uint8, count=192 - 50)

            # Read data for each channel
            for iCh in range(num_channels):
                _ = np.fromfile(f, dtype=np.uint8, count=32)
                tmp = np.fromfile(f, dtype=np.float32, count=num_readouts_now * 2)
                
                # Convert interleaved real/imag to complex
                real_part = tmp[0::2]
                imag_part = tmp[1::2]
                tmp_complex = real_part + 1j * imag_part
                
                data[:num_readouts_now, iPE - 1, iCh] = tmp_complex

            # Update progress indicator
            new_percent = int((100 * iPE) / nviews)
            if new_percent > percentFinished + 9:
                percentFinished = new_percent
                print(f"{percentFinished:3d} %")

            if iPE == nviews:
                break

    return data


def read_twix_from_cartesian_mask(datfile, mask, n_prep, mode='2D'):
    """
    Read TWIX data using Cartesian undersampling mask.
    
    Parameters
    ----------
    datfile : str
        Path to Siemens TWIX .dat file.
    mask : np.ndarray
        Binary sampling mask (1=sampled, 0=skipped).
        Shape: (Nx, Ny) for 2D, (Ny, Nz) for 3D.
    n_prep : int
        Number of preparation lines to skip.
    mode : {'2D', '3D'}, optional
        Acquisition mode. Default: '2D'.
    
    Returns
    -------
    ksp : np.ndarray
        K-space data with channel-first ordering.
        Shape: (num_channels, Nx, Ny) for 2D,
        (num_channels, Nx, Ny, Nz) for 3D.
    
    Raises
    ------
    ValueError
        If mode is not '2D' or '3D'.
    
    Notes
    -----
    For 3D Cartesian, assumes fast PE dimension is along Ny.
    Handles SequenceTree to Python index conversion.
    """
    if mode == '2D':
        pe_idxs = get_pe_idxs_from_mask(mask, mode=mode)

        # Read raw data from TWIX
        n_lines_acquired = len(pe_idxs) + n_prep

        data = read_twix_siemens_XA(datfile, n_lines_acquired)
        data = data[:,n_prep:,:]

        n_ro = data.shape[0]
        n_ch = data.shape[-1]
        n_y = mask.shape[1]

        # Initialize complex k-space matrix
        ksp = np.zeros((n_ro, n_y, n_ch), dtype=np.complex64)

        # Place data in proper k-space locations
        ksp[:, pe_idxs, :] = data

        ksp = np.moveaxis(ksp, -1, 0)

        return ksp

    elif mode == '3D':
        pe_idxs_flat = get_pe_idxs_from_mask(mask, mode=mode)

        n_y = mask.shape[0]
        n_z = mask.shape[1]
        
        # Get PE indices from SequenceTree (-N/2 to N/2 convention)
        pe1 = np.mod(pe_idxs_flat, n_y) - int(n_y/2)
        pe2 = pe_idxs_flat//n_y - int(n_z/2)

        # Shift PE indices to Python mask coordinates
        pe1 = pe1 + int(n_y/2)
        pe2 = pe2 + int(n_z/2)

        # Read raw data from TWIX
        n_lines_acquired = np.count_nonzero(mask) + n_prep
        data = read_twix_siemens_XA(datfile, n_lines_acquired)
        data = data[:,n_prep:,:]

        n_ro = data.shape[0]
        n_ch = data.shape[-1]
        # Initialize complex k-space matrix
        ksp = np.zeros((n_ro, n_y, n_z, n_ch), dtype=np.complex64)

        # Place data in proper k-space locations
        ksp[:, pe1, pe2, :] = data  

        ksp = np.moveaxis(ksp, -1, 0)

        return ksp

    else:
        raise ValueError(f"Invalid mode: {mode}. Must be '2D' or '3D'.")