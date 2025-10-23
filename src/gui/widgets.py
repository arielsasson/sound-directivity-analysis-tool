"""
GUI widgets module for directivity analysis.

This module contains custom widgets and UI components used in the main application.
"""

from PyQt5.QtWidgets import (
    QFrame, QLabel, QPushButton, QProgressBar, QComboBox, QLineEdit, 
    QGroupBox, QHBoxLayout, QVBoxLayout, QCheckBox, QSizePolicy, QSlider
)
from PyQt5.QtCore import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

from config import (
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
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
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
        
        # Distance scale factor slider
        self.create_distance_scale_slider(layout)
        
        # Normalized 2D plots checkbox
        self.create_normalized_2d_plots_checkbox(layout)
        
        # Standard deviation label
        self.std_label = QLabel("Desvío estándar en la posición de referencia (90°): —")
        self.std_label.setWordWrap(True)
        self.std_label.setStyleSheet("color: gray; font-weight: bold;")
        layout.addWidget(self.std_label)
        
        # Mouse interaction display section (simplified - main info now in plot headers)
        self.create_mouse_display_section(layout)
        
        # Stretch to push widgets to top
        layout.addStretch()
    
    def create_distance_scale_slider(self, layout):
        """Create distance scale factor slider."""
        # Distance scale factor group
        scale_group = QGroupBox("Factor de escala de distancias")
        scale_layout = QVBoxLayout(scale_group)
        
        # Slider
        self.distance_scale_slider = QSlider(Qt.Horizontal)
        self.distance_scale_slider.setMinimum(0)
        self.distance_scale_slider.setMaximum(50)
        self.distance_scale_slider.setValue(30)  # Default to 30
        self.distance_scale_slider.setEnabled(False)
        self.distance_scale_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #d4d4d4, stop:1 #afafaf);
            }
        """)
        
        # Value label
        self.distance_scale_label = QLabel("30")
        self.distance_scale_label.setAlignment(Qt.AlignCenter)
        self.distance_scale_label.setStyleSheet("font-weight: bold; color: blue;")
        
        # Range labels
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("0"))
        range_layout.addStretch()
        range_layout.addWidget(QLabel("50"))
        
        scale_layout.addWidget(self.distance_scale_label)
        scale_layout.addWidget(self.distance_scale_slider)
        scale_layout.addLayout(range_layout)
        
        layout.addWidget(scale_group)
    
    def create_normalized_2d_plots_checkbox(self, layout):
        """Create checkbox for normalized 2D plots option."""
        # Normalized 2D plots group
        normalized_group = QGroupBox("Opciones de gráficos 2D")
        normalized_layout = QVBoxLayout(normalized_group)
        
        # Checkbox for using normalized data in 2D plots
        self.normalized_2d_plots_checkbox = QCheckBox("Usar datos normalizados\nen gráficos 2D")
        self.normalized_2d_plots_checkbox.setChecked(False)  # Default unchecked
        self.normalized_2d_plots_checkbox.setEnabled(False)
        self.normalized_2d_plots_checkbox.setStyleSheet("""
            QCheckBox {
                font-weight: bold;
                color: #333;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #999;
                background-color: white;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #0078d4;
                background-color: #0078d4;
                border-radius: 3px;
            }
        """)
        
        normalized_layout.addWidget(self.normalized_2d_plots_checkbox)
        layout.addWidget(normalized_group)
    
    def create_mouse_display_section(self, layout):
        """Create mouse interaction display section."""
        # Mouse interaction group
        mouse_group = QGroupBox("Interacción con gráficos")
        mouse_layout = QVBoxLayout(mouse_group)
        
        # Instructions
        instructions = QLabel("Instrucciones:\n• Mover mouse: seguir cursor\n• Click izquierdo: fijar punto\n• Click derecho: borrar punto\n• Información del cursor se muestra en los títulos de los gráficos")
        instructions.setStyleSheet("color: gray; font-size: 9px;")
        instructions.setWordWrap(True)
        mouse_layout.addWidget(instructions)
        
        layout.addWidget(mouse_group)
    
    # Mouse interaction methods removed - information now displayed in plot headers
    
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
        
        # Header with three labels
        header_frame = QFrame()
        header_frame.setFixedHeight(25)
        header_frame.setStyleSheet(f"background-color: {color}; border: 1px solid gray;")
        
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(4, 2, 4, 2)
        header_layout.setSpacing(8)
        
        # Left label - current cursor info (fixed to left)
        self.current_label = QLabel("Cursor: --")
        self.current_label.setStyleSheet("font-weight: bold; color: red; font-size: 12px; background-color: rgba(255, 255, 255, 0.8); padding: 2px 4px; border-radius: 3px;")
        self.current_label.setFixedWidth(250)
        self.current_label.setAlignment(Qt.AlignLeft)
        
        # Center label - plot title (more space)
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setFixedWidth(80)
        
        # Right label - delta cursor info (fixed to right)
        self.delta_label = QLabel("Δ --")
        self.delta_label.setStyleSheet("font-weight: bold; color: #D2691E; font-size: 12px; background-color: rgba(255, 255, 255, 0.8); padding: 2px 4px; border-radius: 3px;")
        self.delta_label.setFixedWidth(250)
        self.delta_label.setAlignment(Qt.AlignRight)
        
        # Add labels with proper alignment
        header_layout.addWidget(self.current_label, 0, Qt.AlignLeft)
        header_layout.addStretch(1)  # Add stretch to push center label
        header_layout.addWidget(self.title_label, 0, Qt.AlignCenter)
        header_layout.addStretch(1)  # Add stretch to push right label
        header_layout.addWidget(self.delta_label, 0, Qt.AlignRight)
        
        # Canvas frame
        self.canvas_frame = QFrame()
        self.canvas_frame.setFixedSize(width, height - 25)
        self.canvas_frame.setStyleSheet("background-color: white; border: 1px solid gray;")
        
        layout.addWidget(header_frame)
        layout.addWidget(self.canvas_frame)
    
    def update_title(self, base_title: str, cursor_info: str = ""):
        """Update the title with optional cursor information."""
        self.title_label.setText(base_title)
    
    def update_current_cursor(self, cursor_info: str):
        """Update the current cursor information."""
        self.current_label.setText(f"Cursor: {cursor_info}")
    
    def update_delta_cursor(self, delta_info: str):
        """Update the delta cursor information."""
        self.delta_label.setText(f"Δ: {delta_info}")
    
    def clear_cursors(self):
        """Clear both cursor labels."""
        self.current_label.setText("Cursor: --")
        self.delta_label.setText("Δ: --")


class SimpleGraphContainer(QFrame):
    """Simple container for 2D polar plots without cursor labels."""
    
    def __init__(self, title: str, width: int, height: int, color: str):
        super().__init__()
        self.setFixedSize(width, height)
        self.setup_ui(title, width, height, color)
    
    def setup_ui(self, title: str, width: int, height: int, color: str):
        """Setup the simple graph container UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Simple title label only
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"background-color: {color}; padding: 4px; font-weight: bold;")
        self.title_label.setFixedHeight(25)
        self.title_label.setAlignment(Qt.AlignCenter)
        
        # Canvas frame
        self.canvas_frame = QFrame()
        self.canvas_frame.setFixedSize(width, height - 25)
        self.canvas_frame.setStyleSheet("background-color: white; border: 1px solid gray;")
        
        layout.addWidget(self.title_label)
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
    
    def calculate_polar_scale(self, spl):
        """Calculate optimal scale and tick intervals for polar plots."""
        import numpy as np
        
        if len(spl) == 0:
            return 40, 80, [40, 50, 60, 70, 80]
        
        # Remove NaN values
        spl_clean = spl[~np.isnan(spl)]
        
        if len(spl_clean) == 0:
            return 40, 80, [40, 50, 60, 70, 80]
        
        # Find data range
        min_spl = np.min(spl_clean)
        max_spl = np.max(spl_clean)
        
        # Calculate range and determine appropriate tick interval
        data_range = max_spl - min_spl
        
        # Determine tick interval based on data range
        if data_range <= 8:
            tick_interval = 1
        elif data_range <= 20:
            tick_interval = 2.5
        elif data_range <= 40:
            tick_interval = 5
        elif data_range <= 80:
            tick_interval = 10
        else:
            tick_interval = 20
        
        # Calculate bounds (next/previous multiple of 5)
        rmin = int(min_spl // 5) * 5  # Previous multiple of 5
        rmax = int(max_spl // 5) * 5 + 5  # Next multiple of 5
        
        # Generate ticks
        rticks = []
        current = rmin
        while current <= rmax and len(rticks) <= 5:
            rticks.append(current)
            current += tick_interval
        
        # Ensure we don't exceed 5 ticks
        if len(rticks) > 5:
            # Increase interval to reduce number of ticks
            if tick_interval == 1:
                tick_interval = 2.5
            elif tick_interval == 2.5:
                tick_interval = 5
            elif tick_interval == 5:
                tick_interval = 10
            elif tick_interval == 10:
                tick_interval = 20
            
            rticks = []
            current = rmin
            while current <= rmax and len(rticks) <= 5:
                rticks.append(current)
                current += tick_interval
        
        return rmin, rmax, rticks

    def update_plot(self, azimuth: list, spl: list, title: str = ''):
        """Update the polar plot with new data."""
        import numpy as np
        
        self.ax.clear()
        self.ax.set_facecolor('#f9f9f9')
        self.ax.set_theta_zero_location('W')
        self.ax.set_theta_direction(-1)
        self.ax.grid(True, linestyle='--', color='gray', linewidth=0.6)
        
        # Calculate dynamic scale
        rmin, rmax, rticks = self.calculate_polar_scale(spl)
        
        sns.set_style("whitegrid")
        self.ax.plot(np.radians(azimuth), spl, color='darkblue', linewidth=2.5, alpha=0.9)
        self.ax.fill(np.radians(azimuth), spl, color='lightblue', alpha=0.3)
        self.ax.set_rlim(rmin, rmax)
        self.ax.set_rticks(rticks)
        
        # Set azimuth ticks at 30° intervals
        self.ax.set_thetagrids(range(0, 360, 30), labels=[f'{i}°' for i in range(0, 360, 30)])
        
        self.ax.set_title(title, va='bottom', fontsize=10)
        self.ax.set_rlabel_position(135)
        
        self.draw()
