#!/usr/bin/env python3
"""
Main entry point for the Directivity Analysis application.

This script launches the GUI application for analyzing audio directivity patterns.
"""

import sys
import os
from PyQt5.QtWidgets import QApplication

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from gui.main_window import MainWindow


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Análisis de Directividad")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("UNTreF")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
