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

    Ensures elevation is in [-90, 90] range and azimuth is in [0, 360].
    Special handling: treats 0° and 360° as the same for averaging purposes.
    
    Args:
        azim: Azimuth angle in degrees
        elev: Elevation angle in degrees
        
    Returns:
        Tuple of canonical (azimuth, elevation) angles
    """
    # Handle azimuth: treat 0° and 360° as the same for averaging
    canonical_azim = azim % 360
    if canonical_azim == 0:
        canonical_azim = 0  # Use 0 as canonical for both 0° and 360°
    
    # Normalize elevation to [-90, 90]
    canonical_elev = elev % 360
    
    # Handle elevation outside [-90, 90] range
    if canonical_elev > 270:  # [270, 360) -> [-90, 0)
        canonical_elev = canonical_elev - 360
    elif canonical_elev > 90:  # (90, 270] -> flip to other hemisphere
        # For elevations > 90, flip to the other hemisphere to preserve the same point in space
        canonical_azim = (canonical_azim + 180) % 360
        canonical_elev = 180 - canonical_elev
    
    return canonical_azim, canonical_elev


def redundancy_average_spl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average redundant measurements that point to the same spatial direction.
    
    Args:
        df: DataFrame with columns 'azim', 'elev', and frequency columns
        
    Returns:
        DataFrame with averaged measurements (no canonical filtering)
    """
    df = df.copy()
    spl_cols = [col for col in df.columns if col not in ["azim", "elev"]]

    # Step 1: Group by exact coordinates and average
    coordinate_groups = df.groupby(["azim", "elev"])
    result_rows = []
    
    for (azim, elev), group in coordinate_groups:
        # Compute averaged SPL values
        averaged_spl = {}
        for col in spl_cols:
            values = group[col].dropna().values
            if len(values) > 0:
                linear = 10 ** (values / 10)
                averaged_spl[col] = 10 * np.log10(linear.mean())
            else:
                averaged_spl[col] = np.nan
        
        # Create row with original coordinates
        row = {
            "azim": azim,
            "elev": elev,
            **averaged_spl
        }
        result_rows.append(row)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(result_rows)
    
    # Step 2: Ensure both 0° and 360° azimuth values exist for continuity
    continuity_rows = []
    for _, row in result_df.iterrows():
        azim, elev = row["azim"], row["elev"]
        
        # If we have azim=0, also add azim=360 (same data)
        if azim == 0:
            continuity_row = row.copy()
            continuity_row["azim"] = 360
            continuity_rows.append(continuity_row)
        # If we have azim=360, also add azim=0 (same data)
        elif azim == 360:
            continuity_row = row.copy()
            continuity_row["azim"] = 0
            continuity_rows.append(continuity_row)
    
    # Add continuity rows
    if continuity_rows:
        continuity_df = pd.DataFrame(continuity_rows)
        result_df = pd.concat([result_df, continuity_df], ignore_index=True)
    
    # Remove duplicates
    result_df = result_df.drop_duplicates(subset=["azim", "elev"])
    
    return result_df


def apply_canonical_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply canonical conversion to constrain coordinates to [-90°, 90°] elevation range.
    This is used only for balloon plots to ensure proper mesh generation.
    
    Args:
        df: DataFrame with columns 'azim', 'elev', and frequency columns
        
    Returns:
        DataFrame with canonical coordinates (elevations constrained to [-90°, 90°])
    """
    df = df.copy()
    spl_cols = [col for col in df.columns if col not in ["azim", "elev"]]

    # Step 1: Canonical key per row
    df["canonical"] = df.apply(lambda row: canonical_direction(row["azim"], row["elev"]), axis=1)

    # Step 2: Group by canonical coordinates and average
    canonical_groups = df.groupby("canonical")
    result_rows = []
    
    for canonical_coords, group in canonical_groups:
        canonical_azim, canonical_elev = canonical_coords
        
        # Compute averaged SPL values
        averaged_spl = {}
        for col in spl_cols:
            values = group[col].dropna().values
            if len(values) > 0:
                linear = 10 ** (values / 10)
                averaged_spl[col] = 10 * np.log10(linear.mean())
            else:
                averaged_spl[col] = np.nan
        
        # Create row with canonical coordinates
        row = {
            "azim": canonical_azim,
            "elev": canonical_elev,
            **averaged_spl
        }
        result_rows.append(row)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(result_rows)
    
    # Step 3: Ensure both 0° and 360° azimuth values exist for continuity
    continuity_rows = []
    for _, row in result_df.iterrows():
        azim, elev = row["azim"], row["elev"]
        
        # If we have azim=0, also add azim=360 (same data)
        if azim == 0:
            continuity_row = row.copy()
            continuity_row["azim"] = 360
            continuity_rows.append(continuity_row)
        # If we have azim=360, also add azim=0 (same data)
        elif azim == 360:
            continuity_row = row.copy()
            continuity_row["azim"] = 0
            continuity_rows.append(continuity_row)
    
    # Add continuity rows
    if continuity_rows:
        continuity_df = pd.DataFrame(continuity_rows)
        result_df = pd.concat([result_df, continuity_df], ignore_index=True)
    
    # Remove duplicates
    result_df = result_df.drop_duplicates(subset=["azim", "elev"])
    
    return result_df


def apply_canonical_filtering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply canonical filtering to constrain coordinates to canonical form.
    This is used only for balloon plots to ensure proper mesh generation.
    
    Args:
        df: DataFrame with columns 'azim', 'elev', and frequency columns
        
    Returns:
        DataFrame with canonical coordinates and reconstructed elevations above 90°
    """
    df = df.copy()
    spl_cols = [col for col in df.columns if col not in ["azim", "elev"]]

    # Step 1: Canonical key per row
    df["canonical"] = df.apply(lambda row: canonical_direction(row["azim"], row["elev"]), axis=1)

    # Step 2: Group by canonical coordinates and average
    canonical_groups = df.groupby("canonical")
    result_rows = []
    
    for canonical_coords, group in canonical_groups:
        canonical_azim, canonical_elev = canonical_coords
        
        # Compute averaged SPL values
        averaged_spl = {}
        for col in spl_cols:
            values = group[col].dropna().values
            if len(values) > 0:
                linear = 10 ** (values / 10)
                averaged_spl[col] = 10 * np.log10(linear.mean())
            else:
                averaged_spl[col] = np.nan
        
        # Create row with canonical coordinates
        row = {
            "azim": canonical_azim,
            "elev": canonical_elev,
            **averaged_spl
        }
        result_rows.append(row)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(result_rows)
    
    # Step 3: Reconstruct elevations above 90° by mirroring canonical coordinates
    reconstruction_rows = []
    for _, row in result_df.iterrows():
        azim, elev = row["azim"], row["elev"]
        
        # If elevation is in [-90, 90], also add the mirrored version for elevations above 90°
        if -90 <= elev <= 90:
            # Add the mirrored version: elev_high = 180 - elev, azim_high = azim + 180
            mirrored_azim = (azim + 180) % 360
            mirrored_elev = 180 - elev
            
            # Only add if the mirrored elevation is above 90° and not already present
            if mirrored_elev > 90:
                mirrored_row = row.copy()
                mirrored_row["azim"] = mirrored_azim
                mirrored_row["elev"] = mirrored_elev
                reconstruction_rows.append(mirrored_row)
    
    # Add reconstructed rows
    if reconstruction_rows:
        reconstruction_df = pd.DataFrame(reconstruction_rows)
        result_df = pd.concat([result_df, reconstruction_df], ignore_index=True)
    
    # Step 4: Ensure both 0° and 360° azimuth values exist for continuity
    continuity_rows = []
    for _, row in result_df.iterrows():
        azim, elev = row["azim"], row["elev"]
        
        # If we have azim=0, also add azim=360 (same data)
        if azim == 0:
            continuity_row = row.copy()
            continuity_row["azim"] = 360
            continuity_rows.append(continuity_row)
        # If we have azim=360, also add azim=0 (same data)
        elif azim == 360:
            continuity_row = row.copy()
            continuity_row["azim"] = 0
            continuity_rows.append(continuity_row)
    
    # Add continuity rows
    if continuity_rows:
        continuity_df = pd.DataFrame(continuity_rows)
        result_df = pd.concat([result_df, continuity_df], ignore_index=True)
    
    # Remove duplicates
    result_df = result_df.drop_duplicates(subset=["azim", "elev"])
    
    return result_df


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
