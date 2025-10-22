"""
Configuration constants and settings for the directivity analysis application.
"""

# Audio processing constants
CALIBRATION_FREQUENCY_LOWCUT = 707
CALIBRATION_FREQUENCY_HIGHCUT = 1414
CALIBRATION_SPL_LEVEL = 94  # dB SPL

# Frequency bands for analysis
FREQUENCY_BANDS = {
    "Tercios de octava": [
        20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400,
        500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
        5000, 6300, 8000, 10000, 12500, 16000, 20000
    ],
    "Octava": [
        31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000
    ]
}

# Default frequency band
DEFAULT_FREQUENCY_BAND = "Octava"

# Angle validation ranges
ELEVATION_RANGE = (-90, 270)
AZIMUTH_RANGE = (0, 360)
MIN_ELEVATION_STEP = 1
MAX_ELEVATION_STEP = 44
MIN_AZIMUTH_STEP = 1
MAX_AZIMUTH_STEP = 44

# Required reference angles
REQUIRED_ELEVATION_REFERENCE = 90
REQUIRED_AZIMUTH_REFERENCES = [0, 360]

# GUI constants
WINDOW_SIZE = (1450, 1050)
SIDEBAR_WIDTH = 250
GRAPH_CONTAINER_SIZE = (1200, 1050)
TOP_GRAPH_SIZE = (600, 625)
BOTTOM_GRAPH_SIZE = (400, 425)

# File extensions
SUPPORTED_AUDIO_FORMATS = [".wav"]
SUPPORTED_DATA_FORMATS = [".csv"]

# Visualization settings
BALLOON_PLOT_COLOR = "red"
REFERENCE_LINE_COLOR = "black"
REFERENCE_LINE_WIDTH = 4
POLAR_PLOT_COLOR = "darkblue"
POLAR_PLOT_FILL_COLOR = "lightblue"
POLAR_PLOT_ALPHA = 0.3
