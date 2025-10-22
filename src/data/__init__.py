"""
Data processing module for directivity analysis.

This module contains functions for statistical analysis, data normalization,
and DataFrame operations for directivity measurements.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def energy_std_spl(spl_values: np.ndarray) -> float:
    """
    Compute the standard deviation of SPL values based on energy (linear scale),
    and return the result in decibels (dB).
    
    Args:
        spl_values: Array of SPL values in dB
        
    Returns:
        Standard deviation in dB
    """
    spl_values = np.array(spl_values, dtype=float)
    linear_values = 2 * 10**(-5) * 10 ** (spl_values / 20)

    linear_std = np.std(linear_values)
    linear_mean = np.mean(linear_values)
    
    # Avoid log of zero or negative
    if linear_std <= 0:
        return -np.inf  # or 0 dB, depending on context
    
    std_db = 20 * np.log10(linear_mean + linear_std) - 20 * np.log10(linear_mean)
    return std_db


def canonical_direction(azim: float, elev: float) -> Tuple[float, float]:
    """
    Return a canonical (azim, elev) pair such that two directions pointing to
    the same point in space yield the same canonical representation.

    Uses (azim, elev) and (azim+180, 180-elev), and keeps the one with elev in [0, 90].
    
    Args:
        azim: Azimuth angle in degrees
        elev: Elevation angle in degrees
        
    Returns:
        Tuple of canonical (azimuth, elevation) angles
    """
    # Special case: vertical direction
    if elev == 90 or elev == 270:
        return (0, 90)
    
    # Ensure elevation is in real degrees, do not use mod
    mirror_azim = (azim + 180) % 360
    mirror_elev = 180 - elev

    # Choose canonical based on smaller elevation
    if abs(mirror_elev) < abs(elev):
        canonical_azim = mirror_azim
        canonical_elev = mirror_elev
    else:
        canonical_azim = azim % 360
        canonical_elev = elev

    return canonical_azim, canonical_elev


def redundancy_average_spl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average redundant measurements that point to the same spatial direction.
    
    Args:
        df: DataFrame with columns 'azim', 'elev', and frequency columns
        
    Returns:
        DataFrame with averaged redundant measurements
    """
    df = df.copy()
    spl_cols = [col for col in df.columns if col not in ["azim", "elev"]]

    # Step 1: Canonical key per row
    df["canonical"] = df.apply(lambda row: canonical_direction(row["azim"], row["elev"]), axis=1)

    # Step 2: Compute energy average per canonical group
    def energy_average(group):
        result = {}
        for col in spl_cols:
            values = group[col].dropna().values
            if len(values) > 0:
                linear = 10 ** (values / 10)
                result[col] = 10 * np.log10(linear.mean())
            else:
                result[col] = np.nan
        return pd.Series(result)

    averaged = df.groupby("canonical")[spl_cols].apply(energy_average)

    # Step 3: Map back to all original rows
    for col in spl_cols:
        df[col] = df["canonical"].map(averaged[col])
    
    return df.drop(columns=["canonical"])


def normalize_by_90deg(df: pd.DataFrame, df_90: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize directivity data using 90-degree elevation reference measurements.
    
    Args:
        df: Main DataFrame with all measurements
        df_90: DataFrame containing only 90-degree elevation measurements
        
    Returns:
        Normalized DataFrame
    """
    df = df.copy()
    spl_cols = [col for col in df.columns if col not in ["azim", "elev"]]

    # Step 2: Get reference SPL at azim = 0, elev = 90
    ref_row = df_90[df_90["azim"] == 0]
    if ref_row.empty:
        raise ValueError("No data found for azim=0, elev=90")
    ref_spl = ref_row.iloc[0][spl_cols]

    # Step 3: Compute azim-based correction values (difference from ref)
    correction = {}
    for azim in df_90["azim"].unique():
        row = df_90[df_90["azim"] == azim]
        if not row.empty:
            spl = row.iloc[0][spl_cols]
            delta = spl - ref_spl
            correction[azim] = delta

    # Step 4: Apply correction to all rows by azim
    df["azim"] = df["azim"].astype(int)  # Ensure int keys match
    def apply_correction(row):
        az = row["azim"]
        if az in correction:
            return row[spl_cols] - correction[az]
        else:
            return row[spl_cols]  # No correction if missing

    df_norm = df.copy()
    df_norm[spl_cols] = df.apply(apply_correction, axis=1)

    return df_norm
