"""
Main window module for directivity analysis application.

This module contains the main application window and its layout management.
"""

import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt
from vedo import Plotter

from .widgets import ControlPanel, GraphContainer, SimpleGraphContainer, VTKWidget, PolarPlotWidget
from .handlers import EventHandlers
from config import WINDOW_SIZE, GRAPH_CONTAINER_SIZE, TOP_GRAPH_SIZE, BOTTOM_GRAPH_SIZE


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Análisis de directividad")
        self.setFixedSize(*WINDOW_SIZE)
        
        # Initialize data attributes
        self.mediciones_path = None
        self.calculos_path = None
        self.df = None
        self.df_clean = None
        self.df_norm = None
        
        # Initialize event handlers first
        self.handlers = EventHandlers(self)
        
        # Setup UI
        self.setup_ui()
        self.connect_signals()
    
    def setup_ui(self):
        """Setup the main window UI."""
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        central_widget.setLayout(self.main_layout)
        
        # Create control panel
        self.control_panel = ControlPanel()
        self.main_layout.addWidget(self.control_panel)
        
        # Create graph containers
        self.create_graph_containers()
        
        # Embed VTK widgets
        self.embed_vtk_widgets()
        
        # Setup polar plots
        self.setup_polar_plots()
    
    def create_graph_containers(self):
        """Create the graph container layout."""
        # Right side frame
        self.right_frame = QWidget()
        self.right_frame.setFixedSize(*GRAPH_CONTAINER_SIZE)
        self.right_layout = QVBoxLayout(self.right_frame)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(0)
        
        # Top row (2x balloon plots)
        self.top_frame = QWidget()
        self.top_frame.setFixedSize(GRAPH_CONTAINER_SIZE[0], TOP_GRAPH_SIZE[1])
        self.top_layout = QHBoxLayout(self.top_frame)
        self.top_layout.setContentsMargins(0, 0, 0, 0)
        self.top_layout.setSpacing(0)
        
        self.top_left = GraphContainer("SPL", *TOP_GRAPH_SIZE, "lightblue")
        self.top_right = GraphContainer("Normalizado", *TOP_GRAPH_SIZE, "lightblue")
        
        self.top_layout.addWidget(self.top_left)
        self.top_layout.addWidget(self.top_right)
        
        # Bottom row (3x polar plots)
        self.bottom_frame = QWidget()
        self.bottom_frame.setFixedSize(GRAPH_CONTAINER_SIZE[0], BOTTOM_GRAPH_SIZE[1])
        self.bottom_layout = QHBoxLayout(self.bottom_frame)
        self.bottom_layout.setContentsMargins(0, 0, 0, 0)
        self.bottom_layout.setSpacing(0)
        
        self.bottom_left = SimpleGraphContainer("Vista superior", *BOTTOM_GRAPH_SIZE, "lightgreen")
        self.bottom_center = SimpleGraphContainer("Vista frontal", *BOTTOM_GRAPH_SIZE, "lightgreen")
        self.bottom_right = SimpleGraphContainer("Vista sagital", *BOTTOM_GRAPH_SIZE, "lightgreen")
        
        self.bottom_layout.addWidget(self.bottom_left)
        self.bottom_layout.addWidget(self.bottom_center)
        self.bottom_layout.addWidget(self.bottom_right)
        
        # Assemble
        self.right_layout.addWidget(self.top_frame)
        self.right_layout.addWidget(self.bottom_frame)
        
        self.main_layout.addWidget(self.right_frame)
    
    def embed_vtk_widgets(self):
        """Embed VTK widgets for 3D visualization."""
        # Create VTK widgets
        self.vtk_widget_spl = VTKWidget(self.top_left.canvas_frame)
        self.vtk_widget_norm = VTKWidget(self.top_right.canvas_frame)
        
        # Add to canvas frames
        spl_layout = QVBoxLayout(self.top_left.canvas_frame)
        spl_layout.setContentsMargins(0, 0, 0, 0)
        spl_layout.addWidget(self.vtk_widget_spl)
        
        norm_layout = QVBoxLayout(self.top_right.canvas_frame)
        norm_layout.setContentsMargins(0, 0, 0, 0)
        norm_layout.addWidget(self.vtk_widget_norm)
        
        # Create plotters - use offscreen=False for proper callbacks
        self.plotter_spl = Plotter(qt_widget=self.vtk_widget_spl, offscreen=False, interactive=True)
        self.plotter_norm = Plotter(qt_widget=self.vtk_widget_norm, offscreen=False, interactive=True)
    
    def setup_polar_plots(self):
        """Setup polar plot widgets."""
        self.polar_canvases = []
        
        for canvas_host in [self.bottom_left.canvas_frame, 
                           self.bottom_center.canvas_frame, 
                           self.bottom_right.canvas_frame]:
            polar_widget = PolarPlotWidget()
            
            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(polar_widget)
            canvas_host.setLayout(layout)
            
            self.polar_canvases.append(polar_widget)
    
    def connect_signals(self):
        """Connect widget signals to handlers."""
        # Import buttons
        self.control_panel.import_btn.clicked.connect(self.import_measurements)
        self.control_panel.import_calc_btn.clicked.connect(self.import_calculations)
        
        # Calculate button
        self.control_panel.calcular_btn.clicked.connect(self.handlers.calculate)
        
        # Band dropdown
        self.control_panel.band_dropdown.currentIndexChanged.connect(self.on_band_change)
        
        # Interpolation checkbox
        self.control_panel.interpolation_checkbox.stateChanged.connect(self.on_interpolation_toggle)
    
    def import_measurements(self):
        """Import measurements directory."""
        folderpath = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de mediciones")
        if folderpath:
            self.mediciones_path = folderpath
            self.control_panel.mediciones_path_lbl.setText(folderpath)
            self.control_panel.enable_calculation_config()
    
    def import_calculations(self):
        """Import calculations CSV file."""
        filepath, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo de cálculos")
        if filepath:
            self.calculos_path = filepath
            self.control_panel.calc_path_lbl.setText(filepath)
            
            import pandas as pd
            self.df = pd.read_csv(filepath)
            
            # Update band dropdown
            selected_band = self.control_panel.band_dropdown.currentText()
            self.control_panel.band_dropdown.clear()
            self.control_panel.band_dropdown.addItems([col for col in self.df.columns if "Hz" in col])
            self.control_panel.band_dropdown.setCurrentText("1000 Hz")
            self.control_panel.band_dropdown.setEnabled(True)
    
    def on_band_change(self, index):
        """Handle frequency band change."""
        selected_band = self.control_panel.band_dropdown.currentText()
        if selected_band != '' and self.df is not None:
            self.handlers.plot_data(selected_band)
    
    def on_interpolation_toggle(self, state):
        """Handle interpolation toggle."""
        enabled = state == Qt.Checked
        self.control_panel.interpolation_step_input.setEnabled(enabled)
    
    def show_error(self, title: str, message: str):
        """Show error message dialog."""
        QMessageBox.critical(self, title, message)
