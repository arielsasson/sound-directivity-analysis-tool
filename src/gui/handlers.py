"""
Event handlers module for directivity analysis application.

This module contains the business logic and event handling for the GUI.
"""

import os
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import Qt

if TYPE_CHECKING:
    from .main_window import MainWindow

from utils.data_generation import generate_balloon_data
from data.processing import energy_std_spl, redundancy_average_spl, normalize_by_90deg, apply_canonical_conversion
from visualization.plots import create_balloon_plot, create_polar_plot
from .mouse_follower import MouseFollower
from config import (
    ELEVATION_RANGE, AZIMUTH_RANGE, MIN_ELEVATION_STEP, MAX_ELEVATION_STEP,
    MIN_AZIMUTH_STEP, MAX_AZIMUTH_STEP, REQUIRED_ELEVATION_REFERENCE,
    REQUIRED_AZIMUTH_REFERENCES
)


class EventHandlers:
    """Handles all application events and business logic."""
    
    def __init__(self, main_window: 'MainWindow'):
        self.main_window = main_window
        self.spl_mouse_follower = None
        self.norm_mouse_follower = None
    
    def validate_angles(self) -> bool:
        """Validate angle input parameters."""
        try:
            # Parse inputs
            elev_start = int(self.main_window.control_panel.elev_start_input.text())
            elev_stop = int(self.main_window.control_panel.elev_stop_input.text())
            elev_step = int(self.main_window.control_panel.elev_step_input.text())
            azim_start = int(self.main_window.control_panel.azim_start_input.text())
            azim_stop = int(self.main_window.control_panel.azim_stop_input.text())
            azim_step = int(self.main_window.control_panel.azim_step_input.text())
            
            # Interpolation step check
            if self.main_window.control_panel.interpolation_checkbox.isChecked():
                inter_step = int(self.main_window.control_panel.interpolation_step_input.text())
                if inter_step >= elev_step and inter_step >= azim_step:
                    raise ValueError("No se puede interpolar con ese paso para su conjunto de mediciones.")
            
            # Elevation validation
            if not (elev_start <= REQUIRED_ELEVATION_REFERENCE <= elev_stop and 
                   (REQUIRED_ELEVATION_REFERENCE - elev_start) % elev_step == 0):
                raise ValueError("La elevación de 90° debe estar incluida en el rango.")
            
            # Azimuth validation
            azim_values = list(range(azim_start, azim_stop + 1, azim_step))
            if not any(ref in azim_values for ref in REQUIRED_AZIMUTH_REFERENCES):
                raise ValueError("Debe incluirse el azimut 0° o 360° en el rango.")
            
            # Range checks
            if not (ELEVATION_RANGE[0] <= elev_start <= ELEVATION_RANGE[1]):
                raise ValueError("Elevación inicio debe estar entre -90° y 270°")
            if not (ELEVATION_RANGE[0] <= elev_stop <= ELEVATION_RANGE[1]):
                raise ValueError("Elevación fin debe estar entre -90° y 270°")
            if not (elev_start + 4 * elev_step <= elev_stop):
                raise ValueError("Cantidad de elevaciones insuficientes")
            if not (MIN_ELEVATION_STEP < elev_step < MAX_ELEVATION_STEP):
                raise ValueError("Paso de elevación debe ser mayor que 0 y menor que 45")
            
            if not (AZIMUTH_RANGE[0] <= azim_start <= AZIMUTH_RANGE[1]):
                raise ValueError("Azimut inicio debe estar entre 0° y 360°")
            if not (AZIMUTH_RANGE[0] <= azim_stop <= AZIMUTH_RANGE[1]):
                raise ValueError("Azimut fin debe estar entre 0° y 360°")
            if not (azim_start + 4 * azim_step <= azim_stop):
                raise ValueError("Cantidad de azimutales insuficientes")
            if not (MIN_AZIMUTH_STEP < azim_step < MAX_AZIMUTH_STEP):
                raise ValueError("Paso de azimut debe ser mayor que 0 y menor que 45")
            
            return True
            
        except ValueError as e:
            msg = str(e)
            if "invalid literal for int()" in msg:
                self.main_window.show_error("Error de validación", "Ingrese valores enteros")
            else:
                self.main_window.show_error("Error de validación", msg)
            return False
    
    def calculate(self):
        """Calculate balloon data from measurements."""
        if not self.validate_angles():
            return
        
        try:
            # Get file path for saving
            filepath = self.get_save_path()
            
            # Generate angle lists
            elev_start = int(self.main_window.control_panel.elev_start_input.text())
            elev_stop = int(self.main_window.control_panel.elev_stop_input.text())
            elev_step = int(self.main_window.control_panel.elev_step_input.text())
            azim_start = int(self.main_window.control_panel.azim_start_input.text())
            azim_stop = int(self.main_window.control_panel.azim_stop_input.text())
            azim_step = int(self.main_window.control_panel.azim_step_input.text())
            
            elevations = list(range(elev_start, elev_stop + 1, elev_step))
            azimuths = list(range(azim_start, azim_stop + 1, azim_step))
            
            # Show progress bar
            self.main_window.control_panel.progress_bar.setVisible(True)
            
            # Get interpolation step
            interpolation_step = None
            if self.main_window.control_panel.interpolation_checkbox.isChecked():
                interpolation_step = int(self.main_window.control_panel.interpolation_step_input.text())
            
            # Generate data
            self.main_window.df = generate_balloon_data(
                self.main_window.mediciones_path,
                self.main_window.control_panel.filtrado_dropdown.currentText(),
                azimuths,
                elevations,
                self.main_window.control_panel.progress_bar,
                interpolation_step
            )
            
            # Hide progress bar
            self.main_window.control_panel.progress_bar.setVisible(False)
            
            # Save data
            if filepath:
                self.main_window.df.to_csv(filepath, index=False, float_format="%.10f")
                self.main_window.control_panel.calc_path_lbl.setText(filepath)
            
            # Update band dropdown
            selected_band = self.main_window.control_panel.band_dropdown.currentText()
            self.main_window.control_panel.band_dropdown.clear()
            self.main_window.control_panel.band_dropdown.addItems([col for col in self.main_window.df.columns if "Hz" in col])
            self.main_window.control_panel.band_dropdown.setEnabled(True)
            
            # Plot data
            self.plot_data(selected_band)
            
        except Exception as e:
            self.main_window.show_error("Error en cálculo", str(e))
    
    def get_save_path(self) -> str:
        """Get file path for saving CSV data."""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        
        filename = f"datos_balloon-{'interpolado-' + self.main_window.control_panel.interpolation_step_input.text() + '-' if self.main_window.control_panel.interpolation_checkbox.isChecked() else ''}{self.main_window.control_panel.filtrado_dropdown.currentText()}.csv"
        
        filepath, _ = QFileDialog.getSaveFileName(
            self.main_window,
            "Guardar archivo CSV",
            os.path.join(self.main_window.mediciones_path, filename),
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )
        
        return filepath
    
    def cleanup_data(self, frequency: str):
        """Clean and process data for visualization."""
        # Calculate standard deviation at 90°
        spls_at_90 = self.main_window.df[self.main_window.df["elev"] == REQUIRED_ELEVATION_REFERENCE][frequency]
        std_db = energy_std_spl(spls_at_90.values)
        self.main_window.control_panel.std_label.setText(f"Desvío estándar en la posición de referencia (90°): {std_db:.2f} dB")
        
        # Average redundant measurements (keep this step)
        self.main_window.df_clean = redundancy_average_spl(self.main_window.df)
        
        # Save reference data from averaged data
        df_90 = self.main_window.df_clean[self.main_window.df_clean["elev"] == REQUIRED_ELEVATION_REFERENCE].copy()
        
        # Step 1: Normalize data using 90-degree reference
        df_norm_temp = normalize_by_90deg(self.main_window.df_clean, df_90)
        
        # Step 2: Apply additional normalization - make highest value 0 dB
        spl_cols = [col for col in df_norm_temp.columns if col not in ["azim", "elev"]]
        max_values = {}
        for col in spl_cols:
            max_values[col] = df_norm_temp[col].max()
        
        # Make highest value 0 dB by subtracting max from each value
        for col in spl_cols:
            df_norm_temp[col] = df_norm_temp[col] - max_values[col]
        
        self.main_window.df_norm = df_norm_temp
    
    def plot_data(self, frequency: str):
        """Plot all visualizations for the given frequency."""
        self.cleanup_data(frequency)
        
        # Get distance scale factor from slider
        distance_scale_factor = self.main_window.control_panel.distance_scale_slider.value()
        
        # Update distance scale factor in mouse followers
        if self.spl_mouse_follower:
            self.spl_mouse_follower.update_distance_scale_factor(distance_scale_factor)
        if self.norm_mouse_follower:
            self.norm_mouse_follower.update_distance_scale_factor(distance_scale_factor)
        
        # Apply canonical conversion only for balloon plots
        df_clean_canonical = apply_canonical_conversion(self.main_window.df_clean)
        df_norm_canonical = apply_canonical_conversion(self.main_window.df_norm)
        
        
        # Create balloon plots using canonical data
        mesh_spl, line_azim1, line_elev1, *text_labels1 = create_balloon_plot(
            df_clean_canonical, frequency, False, distance_scale_factor
        )
        
        mesh_norm, line_azim2, line_elev2, *text_labels2 = create_balloon_plot(
            df_norm_canonical, frequency, True, distance_scale_factor
        )
        
        # Update VTK plotters
        self.update_vtk_plotter(self.main_window.plotter_spl, mesh_spl, line_azim1, line_elev1, text_labels1, frequency, df_clean_canonical)
        self.update_vtk_plotter(self.main_window.plotter_norm, mesh_norm, line_azim2, line_elev2, text_labels2, frequency, df_norm_canonical)
        
        # Update polar plots
        self.update_polar_plots(frequency)
    
    def update_vtk_plotter(self, plotter, mesh, azimuth_line, elevation_line, text_labels, frequency, df_canonical):
        """Update VTK plotter with new data."""
        # Remove old elements but keep callbacks
        plotter.clear()
        plotter.reset_camera()
        plotter.renderer.ResetCamera()
        plotter.renderer.ResetCameraClippingRange()
        
        # Use show() function to properly set up interactive mode
        plotter.show(
            mesh, azimuth_line, elevation_line, *text_labels,
            interactive=True,
            viewup='z',
            title=frequency,
            axes=0,
            size=(600, 600),
            mode=8,  # Terrain mode
            resetcam=True
        )
        
        # Create or update MouseFollower AFTER show() call
        if plotter == self.main_window.plotter_spl:
            # SPL plot
            if self.spl_mouse_follower:
                # Update existing follower with new data and mesh
                self.spl_mouse_follower.update_data(df_canonical, frequency)
                self.spl_mouse_follower.mesh = mesh
                # Reconnect callbacks after show() call
                self.spl_mouse_follower._reconnect_callbacks()
            else:
                # Create new follower
                self.spl_mouse_follower = MouseFollower(
                    plotter, mesh, df_canonical, frequency, "SPL"
                )
                self.spl_mouse_follower.values_updated.connect(self.update_mouse_values)
                self.spl_mouse_follower.cursor_info_updated.connect(
                    lambda info_type, info: self._update_cursor_info(self.main_window.top_left, info_type, info)
                )
        
        elif plotter == self.main_window.plotter_norm:
            # Normalized plot
            if self.norm_mouse_follower:
                # Update existing follower with new data and mesh
                self.norm_mouse_follower.update_data(df_canonical, frequency)
                self.norm_mouse_follower.mesh = mesh
                # Reconnect callbacks after show() call
                self.norm_mouse_follower._reconnect_callbacks()
            else:
                # Create new follower
                self.norm_mouse_follower = MouseFollower(
                    plotter, mesh, df_canonical, frequency, "normalized"
                )
                self.norm_mouse_follower.values_updated.connect(self.update_mouse_values)
                self.norm_mouse_follower.cursor_info_updated.connect(
                    lambda info_type, info: self._update_cursor_info(self.main_window.top_right, info_type, info)
                )
    
    def update_polar_plots(self, frequency: str):
        """Update polar plots with new data."""
        # Check if user wants to use normalized data for 2D plots
        use_normalized = self.main_window.control_panel.normalized_2d_plots_checkbox.isChecked()
        # Use canonical conversion for 2D plots to get the same data as balloon plots
        df_raw = self.main_window.df_norm if use_normalized else self.main_window.df_clean
        df = apply_canonical_conversion(df_raw)
        
        # Vista superior: Get data at exactly elev = 0
        elev_0_data = df[df['elev'] == 0]
        if not elev_0_data.empty:
            spls_superior = elev_0_data.sort_values('azim')[frequency].to_numpy()
            azim_superior = elev_0_data.sort_values('azim')['azim'].to_numpy()
        else:
            # If no data at elev=0, create empty arrays
            spls_superior = np.array([])
            azim_superior = np.array([])
        
        # Vista frontal: Get data from azim=90 (elevations -90° to 90°) and azim=270 (elevations 90° to 270°)
        azim_90_data = df[df['azim'] == 90]
        azim_270_data = df[df['azim'] == 270]
        
        frontal_data = []
        if not azim_90_data.empty:
            # Add data from azim=90 with original elevations (-90° to 90°)
            for _, row in azim_90_data.iterrows():
                frontal_data.append((row['elev'], row[frequency]))
        
        if not azim_270_data.empty:
            # Add data from azim=270, but convert elevations to 90° to 270° range
            for _, row in azim_270_data.iterrows():
                # Convert elevation: if canonical elev is in [-90, 90], map it to [90, 270]
                original_elev = row['elev']
                if original_elev <= 90:  # This should be the case for canonical data
                    converted_elev = 180 - original_elev  # Maps [-90, 90] to [270, 90] -> reverse to [90, 270]
                    frontal_data.append((converted_elev, row[frequency]))
        
        if frontal_data:
            frontal_data.sort(key=lambda x: x[0])  # Sort by elevation
            elev_frontal = np.array([x[0] for x in frontal_data])
            spls_frontal = np.array([x[1] for x in frontal_data])
        else:
            spls_frontal = np.array([])
            elev_frontal = np.array([])
        
        # Vista sagital: Get data from azim=0 (elevations -90° to 90°) and azim=180 (elevations 90° to 270°)
        azim_0_data = df[df['azim'] == 0]
        azim_180_data = df[df['azim'] == 180]
        
        sagital_data = []
        if not azim_0_data.empty:
            # Add data from azim=0 with original elevations (-90° to 90°)
            for _, row in azim_0_data.iterrows():
                sagital_data.append((row['elev'], row[frequency]))
        
        if not azim_180_data.empty:
            # Add data from azim=180, but convert elevations to 90° to 270° range
            for _, row in azim_180_data.iterrows():
                # Convert elevation: if canonical elev is in [-90, 90], map it to [90, 270]
                original_elev = row['elev']
                if original_elev <= 90:  # This should be the case for canonical data
                    converted_elev = 180 - original_elev  # Maps [-90, 90] to [270, 90] -> reverse to [90, 270]
                    sagital_data.append((converted_elev, row[frequency]))
        
        if sagital_data:
            sagital_data.sort(key=lambda x: x[0])  # Sort by elevation
            elev_sagital = np.array([x[0] for x in sagital_data])
            spls_sagital = np.array([x[1] for x in sagital_data])
        else:
            spls_sagital = np.array([])
            elev_sagital = np.array([])
        
        # Update polar plots
        self.main_window.polar_canvases[0].update_plot(azim_superior, spls_superior)
        self.main_window.polar_canvases[1].update_plot(elev_frontal, spls_frontal)
        self.main_window.polar_canvases[2].update_plot(elev_sagital, spls_sagital)
    
    def update_mouse_values(self, azim, elev, spl, azim_diff, spl_diff):
        """Update mouse interaction values - information now displayed in plot headers."""
        # Mouse interaction values are now displayed in plot headers via cursor_info_updated signal
        pass
    
    def _update_cursor_info(self, container, info_type, info):
        """Update cursor information in the appropriate label."""
        if info_type == "current":
            container.update_current_cursor(info)
        elif info_type == "delta":
            container.update_delta_cursor(info)
