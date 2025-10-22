"""
Utilities module for directivity analysis.

This module contains common utility functions and the main data generation
function for processing balloon measurements.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Optional
from PyQt5.QtWidgets import QProgressBar

from audio.processing import (
    calculate_calibration_offset,
    calculate_recording_spl,
    get_third_octave_edges,
    get_octave_edges
)
from config import (
    FREQUENCY_BANDS,
    CALIBRATION_FREQUENCY_LOWCUT,
    CALIBRATION_FREQUENCY_HIGHCUT
)


def generate_balloon_data(
    measurements_path: str,
    filter_type: str,
    azimuths: List[int],
    elevations: List[int],
    progress_bar: QProgressBar,
    interpolation_step: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate balloon data from measurements directory.
    
    Args:
        measurements_path: Path to measurements directory
        filter_type: Type of filtering ("Tercios de octava" or "Octava")
        azimuths: List of azimuth angles
        elevations: List of elevation angles
        progress_bar: Progress bar widget for updates
        interpolation_step: Optional interpolation step size
        
    Returns:
        DataFrame with balloon data
    """
    # Get frequency bands and edge calculation function
    frequency_bands = FREQUENCY_BANDS[filter_type]
    edge_function = get_third_octave_edges if filter_type == "Tercios de octava" else get_octave_edges
    
    # Calculate calibration offsets
    offsets = {}
    for i in range(1, 18):
        calibration_path = os.path.join(measurements_path, "Calibracion", f"Calibracion-{i:03}.wav")
        offset = calculate_calibration_offset(
            calibration_path, 
            CALIBRATION_FREQUENCY_LOWCUT, 
            CALIBRATION_FREQUENCY_HIGHCUT
        )
        offsets[i] = offset

    # Process measurements
    data = []
    for idx, azimuth in enumerate(azimuths):
        folder = os.path.join(measurements_path, f"{azimuth}°")
        
        for elev_idx, elevation in [(i, e) for i, e in enumerate(elevations, start=1) if i in offsets]:
            wav_filename = f"{azimuth}°v1-{elev_idx:03}.wav"
            wav_path = os.path.join(folder, wav_filename)

            offset = offsets[elev_idx]
            entry = {"azim": azimuth, "elev": elevation}

            # Process each frequency band
            for center_freq in frequency_bands:
                lowcut, highcut = edge_function(center_freq)
                spl = calculate_recording_spl(wav_path, offset, lowcut, highcut)
                entry[f"{center_freq} Hz"] = spl

            data.append(entry)
        
        # Update progress
        total = len(azimuths)
        current = (idx + 1) * 100 / total
        progress_bar.setValue(int(current))

    df = pd.DataFrame(data)
    
    # Apply interpolation if requested
    if interpolation_step:
        df = interpolate_data(df, interpolation_step)
    
    return df


def interpolate_data(df: pd.DataFrame, step: int) -> pd.DataFrame:
    """
    Interpolate data to create higher resolution grid.
    
    Args:
        df: Original DataFrame
        step: Interpolation step size
        
    Returns:
        Interpolated DataFrame
    """
    from scipy.interpolate import griddata
    
    azimuth = df['azim'].values
    elevation = df['elev'].values
    angle_points = np.column_stack((azimuth, elevation))

    # Detect SPL columns
    spl_columns = [col for col in df.columns if col.endswith('Hz')]

    # Create new grid
    azimuth_new = np.arange(azimuth[0], azimuth[-1] + 1, step)
    elevation_range = np.arange(elevation[0], elevation[-1] + 1, step)
    elevation_new = elevation_range[np.isin(elevation_range, elevation)]
    
    AZ, EL = np.meshgrid(azimuth_new, elevation_new)
    azimuth_flat = AZ.flatten()
    elevation_flat = EL.flatten()
    interp_points = (AZ, EL)

    # Create DataFrame with interpolated values
    df_interp = pd.DataFrame({
        'azim': azimuth_flat,
        'elev': elevation_flat
    })

    # Interpolate each frequency band
    for band in spl_columns:
        spl_values = df[band].values
        spl_interp = griddata(angle_points, spl_values, interp_points, method='cubic')
        df_interp[band] = spl_interp.flatten()
    
    return df_interp
