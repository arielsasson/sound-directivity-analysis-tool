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
    cursor_info_updated = pyqtSignal(str, str)  # (type, info) where type is "current" or "delta"
    
    def __init__(self, plotter, mesh, df, frequency_column, plot_type="SPL"):
        super().__init__()
        self.plotter = plotter
        self.mesh = mesh
        self.df = df
        self.frequency_column = frequency_column
        self.plot_type = plot_type
        
        self.red_cursor = None  # Fixed cursor (red)
        self.brown_cursor = None  # Following cursor (brown)
        self.red_cursor_locked = False
        self.fixed_azim = None
        self.fixed_elev = None
        self.fixed_spl = None
        
        # Store scaling parameters for distance calculations
        self.min_spl = None
        self.max_spl = None
        self.distance_scale_factor = 30  # Default value
        
        self._last_move_time = 0
        self._move_interval = 1/60  # 60 FPS
        self._last_click_time = 0
        self._click_debounce_interval = 0.1  # 100ms debounce
        
        # Connect mouse events - simple vedo callbacks
        # Add callbacks - use click press instead of release
        self.plotter.add_callback("mouse move", self.on_mouse_move)
        self.plotter.add_callback("mouse left click", self.on_left_click_press)
        self.plotter.add_callback("mouse right click", self.on_right_click_press)
        
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
                        
                        # Use the same distance scaling as the mesh
                        min_spl = self.df[self.frequency_column].min()
                        max_spl = self.df[self.frequency_column].max()
                        distance = self.distance_scale_factor + (spl - min_spl) * (50 - self.distance_scale_factor) / (max_spl - min_spl)
                        
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
        """Find the closest mesh vertex to the picked position and return the properly scaled position."""
        if self.mesh_vertices is None:
            return picked_pos, None
        
        # Calculate distances to all vertices
        distances = np.linalg.norm(self.mesh_vertices - picked_pos, axis=1)
        closest_idx = np.argmin(distances)
        
        # Get corresponding data if available
        closest_data = None
        if self.mesh_vertex_data is not None and closest_idx < len(self.mesh_vertex_data):
            closest_data = self.mesh_vertex_data[closest_idx]
        
        if closest_data is not None and self.min_spl is not None and self.max_spl is not None:
            azim, elev, spl = closest_data
            
            # Calculate the properly scaled distance for this SPL value
            scaled_distance = self.distance_scale_factor + (spl - self.min_spl) * (50 - self.distance_scale_factor) / (self.max_spl - self.min_spl)
            
            # Convert back to 3D coordinates using the scaled distance
            x = scaled_distance * np.cos(np.radians(elev)) * np.cos(np.radians(-azim))
            y = scaled_distance * np.cos(np.radians(elev)) * np.sin(np.radians(-azim))
            z = scaled_distance * np.sin(np.radians(elev))
            
            scaled_position = np.array([x, y, z])
            return scaled_position, closest_data
        else:
            # Fallback to original position if no vertex data
            return picked_pos, closest_data
    
    def on_mouse_move(self, evt):
        """Handle mouse move events."""
        # Always allow mouse movement - either red cursor follows or brown cursor follows
        
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
        
        if not self.red_cursor:
            # Create red cursor if it doesn't exist
            self.red_cursor = Sphere(pos=snapped_pos, r=1.0, c='red', alpha=1.0)
            self.red_cursor.lighting('off')
            self.plotter += self.red_cursor
            self.red_cursor_locked = False
            self.plotter.render()
            
            # Update current cursor info
            if vertex_data is not None:
                azim, elev, spl = vertex_data
            else:
                azim, elev, spl = self._position_to_angles_spl(snapped_pos)
            cursor_info = f"{azim:.0f}° azim, {elev:.0f}° elev, {spl:.1f} dB"
            self.cursor_info_updated.emit("current", cursor_info)
            
        elif not self.red_cursor_locked:
            # Red cursor exists and is unlocked - move it
            
            # Try different approach to update cursor position
            # Remove old cursor and create new one at new position
            self.plotter.remove(self.red_cursor)
            self.red_cursor = Sphere(pos=snapped_pos, r=1.0, c='red', alpha=1.0)
            self.red_cursor.lighting('off')
            self.plotter += self.red_cursor
            
            self.plotter.render()
            
            # Update current cursor info
            if vertex_data is not None:
                azim, elev, spl = vertex_data
            else:
                azim, elev, spl = self._position_to_angles_spl(snapped_pos)
            cursor_info = f"{azim:.0f}° azim, {elev:.0f}° elev, {spl:.1f} dB"
            self.cursor_info_updated.emit("current", cursor_info)
            
        elif self.red_cursor_locked and not self.brown_cursor:
            # Red cursor is locked, create brown cursor to follow mouse
            self.brown_cursor = Sphere(pos=snapped_pos, r=1.0, c='tan', alpha=1.0)
            self.brown_cursor.lighting('off')
            self.plotter += self.brown_cursor
            
        elif self.brown_cursor:
            # Brown cursor exists - move it and calculate delta
            self.brown_cursor.pos(snapped_pos)
            
            # Calculate delta values between brown cursor and locked red cursor
            if vertex_data is not None:
                azim, elev, spl = vertex_data
            else:
                azim, elev, spl = self._position_to_angles_spl(snapped_pos)
            
            # Calculate differences with respect to locked red cursor
            azim_diff = azim - self.fixed_azim if self.fixed_azim is not None else 0.0
            elev_diff = elev - self.fixed_elev if self.fixed_elev is not None else 0.0
            spl_diff = spl - self.fixed_spl if self.fixed_spl is not None else 0.0
            
            # Emit signal with values
            self.values_updated.emit(azim, elev, spl, azim_diff, spl_diff)
            
            # Emit delta cursor info for plot header
            delta_info = f"azim:{azim_diff:+.0f}° elev:{elev_diff:+.0f}° spl:{spl_diff:+.1f}dB"
            self.cursor_info_updated.emit("delta", delta_info)
            
            self.plotter.render()
    
    def on_left_click_press(self, evt):
        """Handle left click press events."""
        # Debounce to prevent double-triggering
        now = time.time()
        if now - self._last_click_time < self._click_debounce_interval:
            return
        self._last_click_time = now
        
        picked = evt.picked3d
        actor = evt.actor

        if picked is None:
            return
            
        # Allow clicks on mesh or cursor
        if actor != self.mesh and actor != self.cursor:
            return

        # Snap to closest vertex
        snapped_pos, vertex_data = self._find_closest_vertex(picked)

        # Handle red cursor locking
        if self.red_cursor and not self.red_cursor_locked:
            # Lock the red cursor
            self.red_cursor_locked = True
            if vertex_data is not None:
                self.fixed_azim, self.fixed_elev, self.fixed_spl = vertex_data
            else:
                self.fixed_azim, self.fixed_elev, self.fixed_spl = self._position_to_angles_spl(snapped_pos)

        self.plotter.render()
    
    def on_right_click_press(self, evt):
        """Handle right click press events."""
        # Unlock red cursor and remove brown cursor
        if self.red_cursor_locked:
            self.red_cursor_locked = False
        
        if self.brown_cursor:
            self.plotter.remove(self.brown_cursor)
            self.brown_cursor = None
            
            # Clear delta values
            self.cursor_info_updated.emit("delta", "--")
        
        # Reset fixed values
        self.fixed_azim = None
        self.fixed_elev = None
        self.fixed_spl = None
        
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
        
        # Convert distance back to SPL using proper linear interpolation
        # The mesh uses: distance = distance_scale_factor + (spl - min_spl) * (50 - distance_scale_factor) / (max_spl - min_spl)
        # So to reverse: spl = min_spl + (distance - distance_scale_factor) * (max_spl - min_spl) / (50 - distance_scale_factor)
        
        # Update distance scale factor from slider
        self.distance_scale_factor = self.main_window.control_panel.distance_scale_slider.value()
        
        if self.distance_scale_factor < 50 and self.max_spl is not None and self.min_spl is not None:
            # Linear interpolation: distance maps to [distance_scale_factor, 50] -> spl maps to [min_spl, max_spl]
            spl = self.min_spl + (distance - self.distance_scale_factor) * (self.max_spl - self.min_spl) / (50 - self.distance_scale_factor)
        else:
            # Fallback to min_spl if no scaling possible
            spl = self.min_spl if self.min_spl is not None else 0
        
        return azim, elev, spl
    
    def update_data(self, df, frequency_column):
        """Update the data used for calculations."""
        self.df = df
        self.frequency_column = frequency_column
        # Store min/max values for distance calculations
        self.min_spl = df[frequency_column].min()
        self.max_spl = df[frequency_column].max()
        # Update mesh vertices when data changes
        self._update_mesh_vertices()
    
    def update_distance_scale_factor(self, scale_factor):
        """Update the distance scale factor and recalculate vertices."""
        self.distance_scale_factor = scale_factor
        # Recalculate mesh vertices with new scaling
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
    
