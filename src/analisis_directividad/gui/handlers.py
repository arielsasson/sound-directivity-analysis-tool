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

from ..utils.data_generation import generate_balloon_data
from ..data.processing import energy_std_spl, redundancy_average_spl, normalize_by_90deg
from ..visualization.plots import create_balloon_plot, create_polar_plot
from ..config import (
    ELEVATION_RANGE, AZIMUTH_RANGE, MIN_ELEVATION_STEP, MAX_ELEVATION_STEP,
    MIN_AZIMUTH_STEP, MAX_AZIMUTH_STEP, REQUIRED_ELEVATION_REFERENCE,
    REQUIRED_AZIMUTH_REFERENCES
)


class EventHandlers:
    """Handles all application events and business logic."""
    
    def __init__(self, main_window: 'MainWindow'):
        self.main_window = main_window
    
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
        
        # Save reference data before modification
        df_90 = self.main_window.df[self.main_window.df["elev"] == REQUIRED_ELEVATION_REFERENCE].copy()
        
        # Average redundant measurements
        self.main_window.df_clean = redundancy_average_spl(self.main_window.df)
        
        # Normalize data
        self.main_window.df_norm = normalize_by_90deg(self.main_window.df_clean, df_90)
    
    def plot_data(self, frequency: str):
        """Plot all visualizations for the given frequency."""
        self.cleanup_data(frequency)
        
        # Create balloon plots
        mesh_spl, line_azim1, line_elev1, *text_labels1 = create_balloon_plot(
            self.main_window.df_clean, frequency, False
        )
        
        mesh_norm, line_azim2, line_elev2, *text_labels2 = create_balloon_plot(
            self.main_window.df_norm, frequency, True
        )
        
        # Update VTK plotters
        self.update_vtk_plotter(self.main_window.plotter_spl, mesh_spl, line_azim1, line_elev1, text_labels1, frequency)
        self.update_vtk_plotter(self.main_window.plotter_norm, mesh_norm, line_azim2, line_elev2, text_labels2, frequency)
        
        # Update polar plots
        self.update_polar_plots(frequency)
    
    def update_vtk_plotter(self, plotter, mesh, azimuth_line, elevation_line, text_labels, frequency):
        """Update VTK plotter with new data."""
        plotter.clear()
        plotter.reset_camera()
        plotter.renderer.ResetCamera()
        plotter.renderer.ResetCameraClippingRange()
        plotter.render()
        
        plotter.show(
            mesh, azimuth_line, elevation_line, *text_labels,
            interactive=True,
            viewup='z',
            title=frequency,
            axes=0,
            size=(600, 600),
            mode=8,
            resetcam=True
        )
    
    def update_polar_plots(self, frequency: str):
        """Update polar plots with new data."""
        azimuth_angles = self.main_window.df_clean['azim'].unique()
        elevation_angles = self.main_window.df_clean['elev'].unique()
        
        # Get data for different views
        spls_superior = self.main_window.df_clean.loc[self.main_window.df_clean['elev'] == 0, frequency].to_numpy()
        spls_frontal = self.main_window.df_clean.loc[self.main_window.df_clean['azim'] == 90, frequency].to_numpy()
        spls_sagital = self.main_window.df_clean.loc[self.main_window.df_clean['azim'] == 0, frequency].to_numpy()
        
        # Update polar plots
        self.main_window.polar_canvases[0].update_plot(azimuth_angles, spls_superior)
        self.main_window.polar_canvases[1].update_plot(elevation_angles, spls_frontal)
        self.main_window.polar_canvases[2].update_plot(elevation_angles, spls_sagital)
