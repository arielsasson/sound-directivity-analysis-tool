# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-10-14

### Added - Initial Release
- **Professional project structure** with modular architecture
- **Configuration module** (`config.py`) for centralized settings
- **Audio processing module** for WAV file handling and SPL calculations
- **Data processing module** for statistical analysis and normalization
- **Visualization module** for 3D balloon plots and polar graphs
- **GUI module** split into widgets, main window, and event handlers
- **Utilities module** for data generation and interpolation
- **Comprehensive README** with usage instructions
- **Requirements file** with all dependencies
- **Main entry point** for launching the application

### Changed - Refactoring
- Refactored monolithic files into modular structure:
  - `datos_balloon.py` → `utils/data_generation.py`
  - `graphs.py` → `visualization/plots.py`
  - `gui.py` → `gui/` (split into 3 files)
  - `process_audio.py` → `data/processing.py`
- Improved code organization with separation of concerns
- Added type hints for better code documentation
- Enhanced docstrings for all functions and classes

### Fixed
- Layout spacing issues in GUI
- Circular import problems between modules
- Proper margin and spacing configuration

### Project Structure
```
analisis-directividad/
├── .gitignore
├── CHANGELOG.md
├── README.md
├── requirements.txt
├── main.py
└── src/
    └── analisis_directividad/
        ├── __init__.py
        ├── config.py
        ├── audio/
        │   ├── __init__.py
        │   └── processing.py
        ├── data/
        │   ├── __init__.py
        │   └── processing.py
        ├── visualization/
        │   ├── __init__.py
        │   └── plots.py
        ├── gui/
        │   ├── __init__.py
        │   ├── widgets.py
        │   ├── main_window.py
        │   └── handlers.py
        └── utils/
            ├── __init__.py
            └── data_generation.py
```

### Technical Details
- **Total lines of code**: ~2,592
- **Number of modules**: 8
- **Python version**: 3.8+
- **Main dependencies**: PyQt5, numpy, pandas, scipy, vedo, matplotlib, seaborn
