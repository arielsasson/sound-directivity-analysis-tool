"""
GUI widgets module for directivity analysis.

This module contains custom widgets and UI components used in the main application.
"""

from PyQt5.QtWidgets import (
    QFrame, QLabel, QPushButton, QProgressBar, QComboBox, QLineEdit, 
    QGroupBox, QHBoxLayout, QVBoxLayout, QCheckBox, QSizePolicy
)
from PyQt5.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

from ..config import (
    SIDEBAR_WIDTH,
    TOP_GRAPH_SIZE,
    BOTTOM_GRAPH_SIZE,
    FREQUENCY_BANDS,
    DEFAULT_FREQUENCY_BAND
)


class ControlPanel(QFrame):
    """Control panel widget containing all input controls."""
    
    def __init__(self):
        super().__init__()
        self.setFixedWidth(SIDEBAR_WIDTH)
        self.setStyleSheet("background-color: lightgray;")
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the control panel UI."""
        layout = QVBoxLayout(self)
        
        # Import measurements section
        self.import_btn = QPushButton("Importar mediciones")
        self.mediciones_path_lbl = QLabel("")
        self.mediciones_path_lbl.setWordWrap(True)
        self.mediciones_path_lbl.setStyleSheet("color: gray;")
        
        layout.addWidget(self.import_btn)
        layout.addWidget(self.mediciones_path_lbl)
        
        # Filtering dropdown
        filtrado_row = QHBoxLayout()
        filtrado_label = QLabel("Filtrado:")
        filtrado_row.addWidget(filtrado_label)
        
        self.filtrado_dropdown = QComboBox()
        self.filtrado_dropdown.addItems(list(FREQUENCY_BANDS.keys()))
        self.filtrado_dropdown.setCurrentText(DEFAULT_FREQUENCY_BAND)
        self.filtrado_dropdown.setEnabled(False)
        self.filtrado_dropdown.setStyleSheet("background-color: white;")
        filtrado_row.addWidget(self.filtrado_dropdown)
        layout.addLayout(filtrado_row)
        
        # Elevation controls
        self.elev_frame = self.create_angle_group("Elevaciones [-90° - 270°]", "elev")
        layout.addWidget(self.elev_frame)
        
        # Azimuth controls
        self.azim_frame = self.create_angle_group("Azimutales [0° - 360°]", "azim")
        layout.addWidget(self.azim_frame)
        
        # Interpolation controls
        self.interpolation_frame = self.create_interpolation_group()
        layout.addWidget(self.interpolation_frame)
        
        # Calculate button
        self.calcular_btn = QPushButton("Calcular")
        self.calcular_btn.setEnabled(False)
        layout.addWidget(self.calcular_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Import calculations section
        self.import_calc_btn = QPushButton("Importar cálculos")
        self.calc_path_lbl = QLabel("")
        self.calc_path_lbl.setWordWrap(True)
        self.calc_path_lbl.setStyleSheet("color: gray;")
        
        layout.addWidget(self.import_calc_btn)
        layout.addWidget(self.calc_path_lbl)
        
        # Frequency band dropdown
        self.band_dropdown = QComboBox()
        self.band_dropdown.addItems([
            "31.5 Hz", "63 Hz", "125 Hz", "250 Hz", "500 Hz",
            "1000 Hz", "2000 Hz", "4000 Hz", "8000 Hz", "16000 Hz"
        ])
        self.band_dropdown.setCurrentText("1000 Hz")
        self.band_dropdown.setEnabled(False)
        self.band_dropdown.setStyleSheet("background-color: white;")
        layout.addWidget(self.band_dropdown)
        
        # Standard deviation label
        self.std_label = QLabel("Desvío estándar en la posición de referencia (90°): —")
        self.std_label.setWordWrap(True)
        self.std_label.setStyleSheet("color: gray;")
        layout.addWidget(self.std_label)
        
        # Stretch to push widgets to top
        layout.addStretch()
    
    def create_angle_group(self, title: str, prefix: str) -> QGroupBox:
        """Create angle input group."""
        frame = QGroupBox(title)
        layout = QVBoxLayout(frame)
        
        # Create input fields
        start_input = QLineEdit()
        stop_input = QLineEdit()
        step_input = QLineEdit()
        
        # Store references
        setattr(self, f"{prefix}_start_input", start_input)
        setattr(self, f"{prefix}_stop_input", stop_input)
        setattr(self, f"{prefix}_step_input", step_input)
        
        for label, widget in [("Inicio:", start_input),
                              ("Fin:", stop_input),
                              ("Paso:", step_input)]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            widget.setFixedWidth(50)
            widget.setEnabled(False)
            widget.setStyleSheet("background-color: white;")
            row.addWidget(widget)
            row.addWidget(QLabel("°"))
            layout.addLayout(row)
        
        return frame
    
    def create_interpolation_group(self) -> QGroupBox:
        """Create interpolation controls group."""
        frame = QGroupBox("Interpolación")
        layout = QHBoxLayout(frame)
        
        self.interpolation_checkbox = QCheckBox("Activar")
        self.interpolation_checkbox.setChecked(False)
        self.interpolation_checkbox.setEnabled(False)
        
        self.interpolation_step_input = QLineEdit()
        self.interpolation_step_input.setText("5")
        self.interpolation_step_input.setFixedWidth(50)
        self.interpolation_step_input.setEnabled(False)
        self.interpolation_step_input.setStyleSheet("background-color: white")
        
        layout.addWidget(self.interpolation_checkbox)
        layout.addWidget(QLabel("Paso:"))
        layout.addWidget(self.interpolation_step_input)
        layout.addWidget(QLabel("°"))
        
        return frame
    
    def enable_calculation_config(self):
        """Enable controls after measurements are imported."""
        self.filtrado_dropdown.setEnabled(True)
        self.calcular_btn.setEnabled(True)
        self.interpolation_checkbox.setEnabled(True)
        
        # Enable angle input widgets
        for prefix in ["elev", "azim"]:
            for suffix in ["start", "stop", "step"]:
                widget = getattr(self, f"{prefix}_{suffix}_input")
                widget.setEnabled(True)


class GraphContainer(QFrame):
    """Container for graph widgets."""
    
    def __init__(self, title: str, width: int, height: int, color: str):
        super().__init__()
        self.setFixedSize(width, height)
        self.setup_ui(title, width, height, color)
    
    def setup_ui(self, title: str, width: int, height: int, color: str):
        """Setup the graph container UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Title label
        label = QLabel(title)
        label.setStyleSheet(f"background-color: {color}; padding: 4px;")
        label.setFixedHeight(25)
        label.setAlignment(Qt.AlignCenter)
        
        # Canvas frame
        self.canvas_frame = QFrame()
        self.canvas_frame.setFixedSize(width, height - 25)
        self.canvas_frame.setStyleSheet("background-color: white; border: 1px solid gray;")
        
        layout.addWidget(label)
        layout.addWidget(self.canvas_frame)


class VTKWidget(QVTKRenderWindowInteractor):
    """Custom VTK widget for 3D visualizations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(600, 600)


class PolarPlotWidget(FigureCanvas):
    """Custom polar plot widget."""
    
    def __init__(self, parent=None):
        fig = Figure(figsize=(4, 4), dpi=100, tight_layout=True)
        super().__init__(fig)
        self.setFixedSize(400, 400)
        
        self.ax = fig.add_subplot(111, polar=True)
        self.ax.set_facecolor('#f9f9f9')
        self.ax.set_theta_zero_location('W')
        self.ax.set_theta_direction(-1)
        self.ax.grid(True, linestyle='--', linewidth=0.6, color='gray')
    
    def update_plot(self, azimuth: list, spl: list, title: str = ''):
        """Update the polar plot with new data."""
        import numpy as np
        
        self.ax.clear()
        self.ax.set_facecolor('#f9f9f9')
        self.ax.set_theta_zero_location('W')
        self.ax.set_theta_direction(-1)
        self.ax.grid(True, linestyle='--', color='gray', linewidth=0.6)
        
        sns.set_style("whitegrid")
        self.ax.plot(np.radians(azimuth), spl, color='darkblue', linewidth=2.5, alpha=0.9)
        self.ax.fill(np.radians(azimuth), spl, color='lightblue', alpha=0.3)
        self.ax.set_rmax(max(spl) + 2)
        self.ax.set_title(title, va='bottom', fontsize=10)
        self.ax.set_rticks([20, 40, 60, 80])
        self.ax.set_rlabel_position(135)
        
        self.draw()
