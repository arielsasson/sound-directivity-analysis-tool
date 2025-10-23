"""
Visualization module for directivity analysis.

This module contains functions for creating 3D balloon plots and polar plots
for visualizing directivity patterns.
"""

import numpy as np
import pandas as pd
from vedo import show, Grid, Mesh, Text3D, Line, Axes
from matplotlib.figure import Figure
import seaborn as sns

from config import (
    BALLOON_PLOT_COLOR,
    REFERENCE_LINE_COLOR,
    REFERENCE_LINE_WIDTH,
    POLAR_PLOT_COLOR,
    POLAR_PLOT_FILL_COLOR,
    POLAR_PLOT_ALPHA
)


def spherical_to_cartesian(azim_deg: np.ndarray, elev_deg: np.ndarray, r: float = 1) -> np.ndarray:
    """
    Convert spherical coordinates to Cartesian coordinates.
    
    Args:
        azim_deg: Azimuth angles in degrees
        elev_deg: Elevation angles in degrees
        r: Radius (default 1)
        
    Returns:
        Array of Cartesian coordinates (x, y, z)
    """
    azim = np.radians(azim_deg)
    elev = np.radians(elev_deg)
    x = r * np.cos(elev) * np.cos(azim)
    y = r * np.cos(elev) * np.sin(azim)
    z = r * np.sin(elev)
    return np.vstack((x, y, z)).T


def create_surface_mesh(df: pd.DataFrame, spl_column: str, normalized: bool = False) -> Mesh:
    """
    Create a 3D surface mesh from directivity data.
    
    Args:
        df: DataFrame with directivity data
        spl_column: Column name containing SPL values
        
    Returns:
        Vedo Mesh object representing the surface
    """
    df['az'] = np.radians(-df['azim'])
    df['el'] = np.radians(df['elev'])
    
    # For normalized data with negative values, invert the distance so 0 dB is farthest and most negative is closest
    spl_values = df[spl_column].copy()
    if normalized:
        # For normalized data: 0 dB should be farthest, negative values closer to center
        # We need to invert the relationship: use current_value - min as distance
        # This way: 0 dB (max) gets max distance, most negative gets 0 distance
        min_spl = spl_values.min()  # This should be the most negative value
        distance_values = spl_values - min_spl
    else:
        distance_values = spl_values
    
    df['x'] = distance_values * np.cos(df['el']) * np.cos(df['az'])
    df['y'] = distance_values * np.cos(df['el']) * np.sin(df['az'])
    df['z'] = distance_values * np.sin(df['el'])
    
    # Sort by elevation and azimuth (assuming grid-like order)
    df_sorted = df.sort_values(by=['elev', 'azim'])
    
    # Get the unique counts
    n_el = df['elev'].nunique()
    n_az = df['azim'].nunique()
    
    # Reshape Cartesian coords to grid
    xgrid = df_sorted['x'].values.reshape((n_el, n_az))
    ygrid = df_sorted['y'].values.reshape((n_el, n_az))
    zgrid = df_sorted['z'].values.reshape((n_el, n_az))
    
    # Stack into points
    points = np.stack([xgrid, ygrid, zgrid], axis=-1)  # shape: (n_el, n_az, 3)
    
    # Flatten to 2D points array
    flat_points = points.reshape(-1, 3)
    
    # Generate triangle indices for the grid manually
    faces = []
    for i in range(n_el - 1):
        for j in range(n_az):
            jp = (j + 1) % n_az  # wrap horizontally
            
            p0 = i * n_az + j
            p1 = i * n_az + jp
            p2 = (i + 1) * n_az + j
            p3 = (i + 1) * n_az + jp
            
            faces.append([p0, p2, p1])
            faces.append([p1, p2, p3])
            
    faces = np.array(faces)
    
    # Create mesh with original SPL values for coloring
    mesh = Mesh([flat_points, faces]).lw(0)
    
    # Add original SPL values as point data for coloring
    spl_grid = df_sorted[spl_column].values.reshape((n_el, n_az))
    flat_spl = spl_grid.reshape(-1)
    mesh.pointdata["spl"] = flat_spl
    
    # Apply colormap based on SPL values
    mesh.cmap("viridis", flat_spl)
    
    return mesh


def create_reference_lines(max_distance: float) -> tuple:
    """
    Create reference lines for azimuth and elevation axes.
    
    Args:
        max_distance: Maximum distance for reference lines
        
    Returns:
        Tuple of (azimuth_line, elevation_line, text_labels)
    """
    # Line for Azimuth = 0°, elevations from -30° to +210°
    elevations = np.linspace(-30, 210, 100)
    azimuth_zero = np.zeros_like(elevations)
    line_coords = spherical_to_cartesian(azimuth_zero, elevations, r=max_distance + 5)
    azimuth_line = Line(line_coords, c=REFERENCE_LINE_COLOR, lw=REFERENCE_LINE_WIDTH)
    
    # Line for Elevation = 0°, Azimuth from -180° to +180°
    azimuth = np.linspace(-180, 180, 100)
    elevation_zero = np.zeros_like(azimuth)
    line_coords_elev0 = spherical_to_cartesian(azimuth, elevation_zero, r=max_distance + 5)
    elevation_line = Line(line_coords_elev0, c=REFERENCE_LINE_COLOR, lw=REFERENCE_LINE_WIDTH)
    
    # Text labels for cardinal directions
    text_labels = []
    positions = [
        (max_distance + 15, 0, 0, "0°"),
        (0, max_distance + 15, 0, "270°"),
        (-max_distance - 9, 0, 0, "180°"),
        (0, -max_distance - 8, 0, "90°")
    ]
    
    for x, y, z, text in positions:
        text_obj = Text3D(text, pos=(x, y, z), s=5, c=REFERENCE_LINE_COLOR)
        text_obj.rotate(180, axis=(1, 0, 0), point=(x, y, z), rad=False)
        text_obj.rotate(180, axis=(0, 1, 0), point=(x, y, z), rad=False)
        text_labels.append(text_obj)
    
    return azimuth_line, elevation_line, text_labels


def create_balloon_plot(df: pd.DataFrame, frequency: str, normalized: bool = False) -> tuple:
    """
    Create a complete balloon plot visualization.
    
    Args:
        df: DataFrame with directivity data
        frequency: Frequency column name
        normalized: Whether the data is normalized
        
    Returns:
        Tuple of (mesh, azimuth_line, elevation_line, text_labels)
    """
    spls = df[frequency]
    max_spl = max(spls)
    
    # Create surface mesh
    mesh = create_surface_mesh(df, frequency, normalized)
    
    # Create reference lines and labels
    azimuth_line, elevation_line, text_labels = create_reference_lines(max_spl)
    
    # Colormap is now applied in create_surface_mesh function
    
    # Add scalar bar
    if normalized:
        mesh.add_scalarbar(title='dB')
    else:
        mesh.add_scalarbar(title='dB SPL')
    
    return mesh, azimuth_line, elevation_line, text_labels


def create_polar_plot(ax, azimuth: np.ndarray, spl: np.ndarray, title: str = '') -> None:
    """
    Create a polar plot on the given matplotlib axis.
    
    Args:
        ax: Matplotlib polar axis
        azimuth: Azimuth angles in degrees
        spl: SPL values in dB
        title: Plot title
    """
    ax.clear()

    ax.set_facecolor('#f9f9f9')
    ax.set_theta_zero_location('W')
    ax.set_theta_direction(-1)
    
    ax.grid(True, linestyle='--', color='gray', linewidth=0.6)

    sns.set_style("whitegrid")
    ax.plot(np.radians(azimuth), spl, color=POLAR_PLOT_COLOR, linewidth=2.5, alpha=0.9)
    ax.fill(np.radians(azimuth), spl, color=POLAR_PLOT_FILL_COLOR, alpha=POLAR_PLOT_ALPHA)
    ax.set_rmax(max(spl) + 2)
    ax.set_title(title, va='bottom', fontsize=10)
    ax.set_rticks([20, 40, 60, 80])  # adjust depending on your SPL range
    ax.set_rlabel_position(135)
