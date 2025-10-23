"""
Mouse follower module for balloon plot interaction.

This module contains the MouseFollower class that handles mouse interactions
with the 3D balloon plots, including cursor following, clicking, and value display.
"""

import time
import numpy as np
from vedo import Sphere
from PyQt5.QtCore import QObject, pyqtSignal


class MouseFollower(QObject):
    """Mouse follower for 3D balloon plot interaction."""
    
    # Signals for updating the UI
    values_updated = pyqtSignal(float, float, float, float, float)  # azim, elev, spl, azim_diff, spl_diff
    cursor_info_updated = pyqtSignal(str)  # compact cursor info string
    
    def __init__(self, plotter, mesh, df, frequency_column, plot_type="SPL"):
        super().__init__()
        self.plotter = plotter
        self.mesh = mesh
        self.df = df
        self.frequency_column = frequency_column
        self.plot_type = plot_type
        
        self.cursor = None
        self.locked = False
        self.fixed_azim = None
        self.fixed_elev = None
        self.fixed_spl = None
        
        self._last_move_time = 0
        self._move_interval = 1/60  # 60 FPS
        self._last_click_time = 0
        self._click_debounce_interval = 0.1  # 100ms debounce
        
        # Connect mouse events - simple vedo callbacks
        print(f"Setting up MouseFollower for {plot_type} plot")
        
        # Add callbacks - use click press instead of release
        self.plotter.add_callback("mouse move", self.on_mouse_move)
        self.plotter.add_callback("mouse left click", self.on_left_click_press)
        self.plotter.add_callback("mouse right click", self.on_right_click_press)
        print("Callbacks added successfully")
        
        # Store mesh vertices for snapping
        self.mesh_vertices = None
        self.mesh_vertex_data = None
        self._update_mesh_vertices()
    
    def _update_mesh_vertices(self):
        """Update mesh vertices by reconstructing them from the DataFrame."""
        if self.df is not None and hasattr(self, 'frequency_column'):
            # Get unique angles from the dataframe
            elevations = sorted(self.df['elev'].unique())
            azimuths = sorted(self.df['azim'].unique())
            
            vertex_data = []
            vertex_positions = []
            
            for elev in elevations:
                for azim in azimuths:
                    # Find the corresponding SPL value
                    row = self.df[(self.df['elev'] == elev) & (self.df['azim'] == azim)]
                    if not row.empty:
                        spl = row[self.frequency_column].iloc[0]
                        vertex_data.append([azim, elev, spl])
                        
                        # Reconstruct 3D position from spherical coordinates
                        # This matches the logic from create_surface_mesh in plots.py
                        az_rad = np.radians(-azim)  # Note: negative for azimuth
                        el_rad = np.radians(elev)
                        
                        # For normalized data, use distance based on SPL values
                        if self.plot_type == "normalized":
                            min_spl = self.df[self.frequency_column].min()
                            distance = spl - min_spl
                        else:
                            distance = spl
                        
                        x = distance * np.cos(el_rad) * np.cos(az_rad)
                        y = distance * np.cos(el_rad) * np.sin(az_rad)
                        z = distance * np.sin(el_rad)
                        
                        vertex_positions.append([x, y, z])
            
            self.mesh_vertex_data = np.array(vertex_data)
            self.mesh_vertices = np.array(vertex_positions)
        else:
            self.mesh_vertices = None
            self.mesh_vertex_data = None
    
    def _find_closest_vertex(self, picked_pos):
        """Find the closest mesh vertex to the picked position."""
        if self.mesh_vertices is None:
            return picked_pos, None
        
        # Calculate distances to all vertices
        distances = np.linalg.norm(self.mesh_vertices - picked_pos, axis=1)
        closest_idx = np.argmin(distances)
        closest_vertex = self.mesh_vertices[closest_idx]
        
        # Get corresponding data if available
        closest_data = None
        if self.mesh_vertex_data is not None and closest_idx < len(self.mesh_vertex_data):
            closest_data = self.mesh_vertex_data[closest_idx]
        
        return closest_vertex, closest_data
    
    def on_mouse_move(self, evt):
        """Handle mouse move events."""
        if self.locked:
            # Only print this occasionally to avoid spam
            if not hasattr(self, '_last_ignore_time') or time.time() - self._last_ignore_time > 2:
                print("Mouse move ignored - cursor is locked (click to unlock)")
                self._last_ignore_time = time.time()
            return
        
        now = time.time()
        if (now - self._last_move_time < self._move_interval):
            return
        self._last_move_time = now

        picked = evt.picked3d
        actor = evt.actor

        if picked is None or actor != self.mesh:
            return

        # Snap to closest vertex
        snapped_pos, vertex_data = self._find_closest_vertex(picked)
        
        if not self.cursor:
            # Create cursor if it doesn't exist
            self.cursor = Sphere(pos=snapped_pos, r=1.0, c='red', alpha=1.0)
            self.cursor.lighting('off')  # Disable lighting for consistent color
            self.plotter += self.cursor
            self.locked = False
            print(f"Red cursor created on mouse move! Cursor locked: {self.locked}")
        else:
            # Move existing cursor
            self.cursor.pos(snapped_pos)
            
            # Use vertex data if available, otherwise calculate from position
            if vertex_data is not None:
                azim, elev, spl = vertex_data
            else:
                azim, elev, spl = self._position_to_angles_spl(snapped_pos)
            
            # Calculate differences if we have a fixed point
            azim_diff = azim - self.fixed_azim if self.fixed_azim is not None else 0.0
            spl_diff = spl - self.fixed_spl if self.fixed_spl is not None else 0.0
            
            # Emit signal with values
            self.values_updated.emit(azim, elev, spl, azim_diff, spl_diff)
            
            # Emit compact cursor info for plot header
            cursor_info = f"{azim:.0f}° azim, {elev:.0f}° elev, {spl:.1f} dB"
            self.cursor_info_updated.emit(cursor_info)
            
            self.plotter.render()
    
    def on_left_click_press(self, evt):
        """Handle left click press events."""
        # Debounce to prevent double-triggering
        now = time.time()
        if now - self._last_click_time < self._click_debounce_interval:
            print("Click ignored - too soon after last click")
            return
        self._last_click_time = now
        
        print("Left click press triggered!")
        picked = evt.picked3d
        actor = evt.actor
        print(f"Picked: {picked}, Actor: {actor}")

        if picked is None:
            print("Click not on any object, ignoring")
            return
            
        # Allow clicks on mesh or cursor
        if actor != self.mesh and actor != self.cursor:
            print(f"Click not on mesh or cursor, ignoring. Actor: {actor}, Mesh: {self.mesh}, Cursor: {self.cursor}")
            return

        # Snap to closest vertex
        snapped_pos, vertex_data = self._find_closest_vertex(picked)

        # Only handle locking/unlocking - cursor creation is handled in mouse move
        print(f"Checking cursor: {self.cursor is not None}")
        if self.cursor:
            self.locked = not self.locked
            print(f"Cursor lock toggled! Cursor locked: {self.locked}")
            
            # If locking, save the fixed values
            if self.locked:
                if vertex_data is not None:
                    self.fixed_azim, self.fixed_elev, self.fixed_spl = vertex_data
                else:
                    self.fixed_azim, self.fixed_elev, self.fixed_spl = self._position_to_angles_spl(snapped_pos)
                print(f"Fixed values saved: azim={self.fixed_azim}, elev={self.fixed_elev}, spl={self.fixed_spl}")
            else:
                print("Cursor unlocked - will follow mouse movement")
        else:
            print("No cursor exists yet - move mouse over balloon to create cursor")

        self.plotter.render()
    
    def on_right_click_press(self, evt):
        """Handle right click press events."""
        if self.cursor:
            self.plotter.remove(self.cursor)
            self.cursor = None
            self.locked = False
            self.fixed_azim = None
            self.fixed_elev = None
            self.fixed_spl = None
            # Clear cursor info
            self.cursor_info_updated.emit("")
            self.plotter.render()
    
    def _position_to_angles_spl(self, pos):
        """Convert 3D position to azimuth, elevation, and SPL values."""
        x, y, z = pos
        
        # Calculate distance from center
        distance = np.sqrt(x**2 + y**2 + z**2)
        
        # Calculate azimuth (in degrees)
        azim = np.degrees(np.arctan2(y, x)) % 360
        
        # Calculate elevation (in degrees)
        elev = np.degrees(np.arcsin(z / distance))
        
        # For normalized plots, we need to convert distance back to SPL
        if self.plot_type == "normalized":
            # Distance was calculated as (spl - min_spl), so spl = distance + min_spl
            min_spl = self.df[self.frequency_column].min()
            spl = distance + min_spl
        else:
            # For SPL plots, distance is directly the SPL value
            spl = distance
        
        return azim, elev, spl
    
    def update_data(self, df, frequency_column):
        """Update the data used for calculations."""
        self.df = df
        self.frequency_column = frequency_column
        # Update mesh vertices when data changes
        self._update_mesh_vertices()
    
    def _reconnect_callbacks(self):
        """Reconnect callbacks after plotter.show() call."""
        # Clear any existing callbacks
        if hasattr(self.plotter, '_callbacks'):
            self.plotter._callbacks.clear()
        
        # Re-add the callbacks
        self.plotter.add_callback("mouse move", self.on_mouse_move)
        self.plotter.add_callback("mouse left click", self.on_left_click_press)
        self.plotter.add_callback("mouse right click", self.on_right_click_press)
        print("Callbacks reconnected successfully")
    
