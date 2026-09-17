"""
Zone Manager Module

Manages the three monitoring zones (Input, Processing, Output).
Handles zone configuration loading, point-in-polygon testing,
and drawing zone overlays on video frames.
"""

import json
import os
import cv2
import numpy as np


class Zone:
    """Represents a single monitoring zone."""

    def __init__(self, zone_id, name, points, color=(0, 255, 0)):
        self.zone_id = zone_id
        self.name = name
        self.points = np.array(points, dtype=np.int32)
        self.color = tuple(color) if isinstance(color, list) else color
        self._contour = self.points.reshape((-1, 1, 2))

    def contains_point(self, x, y):
        """Check if a point (x, y) is inside this zone."""
        result = cv2.pointPolygonTest(self._contour, (float(x), float(y)), False)
        return result >= 0  # >= 0 means inside or on edge

    def contains_detection(self, detection):
        """Check if a detection's center point is inside this zone."""
        return self.contains_point(detection.center_x, detection.center_y)

    def update_points(self, new_points):
        """Update the zone's polygon points."""
        self.points = np.array(new_points, dtype=np.int32)
        self._contour = self.points.reshape((-1, 1, 2))

    @property
    def bounding_rect(self):
        """Get the bounding rectangle of the zone polygon."""
        return cv2.boundingRect(self.points)


class ZoneManager:
    """
    Manages all three monitoring zones.

    Loads zone configurations from JSON, determines which zone
    a detection falls in, and draws zone overlays on frames.
    """

    def __init__(self, config_path=None, zone_padding=20):
        self.zones = {}
        self.config_path = config_path
        self.zone_padding = zone_padding  # Pixels to expand zones outward for detection
        self.reference_resolution = None  # (width, height) zones were drawn for
        self._current_scale = (1.0, 1.0)  # Current scale factors applied
        if config_path and os.path.exists(config_path):
            self.load_zones(config_path)

    def load_zones(self, config_path):
        """Load zone definitions from a JSON config file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Read reference resolution if stored
            if "reference_resolution" in config:
                ref = config["reference_resolution"]
                self.reference_resolution = (int(ref[0]), int(ref[1]))

            self.zones = {}
            for zone_key, zone_data in config.items():
                # Skip non-zone keys like reference_resolution
                if not isinstance(zone_data, dict) or "points" not in zone_data:
                    continue
                zone_id = zone_key  # e.g., "zone_1", "zone_2", "zone_3"
                zone = Zone(
                    zone_id=zone_id,
                    name=zone_data.get("name", zone_id),
                    points=zone_data.get("points", []),
                    color=zone_data.get("color", [0, 255, 0])
                )
                self.zones[zone_id] = zone

            # Build buffer zone between zone_2 and zone_3
            self._build_buffer_zone()

            self.config_path = config_path
            self._current_scale = (1.0, 1.0)
        except Exception as e:
            raise RuntimeError(f"Failed to load zone config from '{config_path}': {e}")

    def _build_buffer_zone(self):
        """Create a buffer zone between zone_2 and zone_3."""
        if "zone_2" not in self.zones or "zone_3" not in self.zones:
            return
            
        z2_pts = self.zones["zone_2"].points
        z3_pts = self.zones["zone_3"].points
        if len(z2_pts) == 0 or len(z3_pts) == 0:
            return
            
        z2_x, z2_y, z2_w, z2_h = cv2.boundingRect(z2_pts)
        z3_x, z3_y, z3_w, z3_h = cv2.boundingRect(z3_pts)
        
        z2_max_y = z2_y + z2_h
        z3_max_y = z3_y + z3_h
        
        if z2_y > z3_max_y:
            y_top = z3_max_y
            y_bottom = z2_y
        else:
            y_top = z2_max_y
            y_bottom = z3_y
            
        if y_bottom - y_top < 5:
            mid_y = (y_top + y_bottom) // 2
            y_top = mid_y - 10
            y_bottom = mid_y + 10
            
        x_min = min(z2_x, z3_x)
        x_max = max(z2_x + z2_w, z3_x + z3_w)
        
        buffer_points = [
            [int(x_min), int(y_top)],
            [int(x_max), int(y_top)],
            [int(x_max), int(y_bottom)],
            [int(x_min), int(y_bottom)]
        ]
        
        self.zones["buffer_zone"] = Zone(
            zone_id="buffer_zone",
            name="Buffer Zone",
            points=buffer_points,
            color=(200, 100, 200)
        )

    def scale_zones_to_resolution(self, target_width, target_height):
        """
        Scale all zone coordinates to match a different video resolution.
        
        Uses the stored reference_resolution to compute scale factors.
        If no reference_resolution is stored, does nothing.
        
        Args:
            target_width: Width of the video being processed
            target_height: Height of the video being processed
        """
        if not self.reference_resolution:
            return
            
        ref_w, ref_h = self.reference_resolution
        
        # If resolution matches, no scaling needed
        if target_width == ref_w and target_height == ref_h:
            self._current_scale = (1.0, 1.0)
            return
            
        scale_x = target_width / ref_w
        scale_y = target_height / ref_h
        self._current_scale = (scale_x, scale_y)
        
        # Reload original zones first (to avoid cumulative scaling)
        if self.config_path:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            for zone_id, zone in self.zones.items():
                if zone_id == "buffer_zone":
                    continue
                if zone_id in config and isinstance(config[zone_id], dict):
                    original_pts = config[zone_id].get("points", [])
                    scaled_pts = [
                        [int(pt[0] * scale_x), int(pt[1] * scale_y)]
                        for pt in original_pts
                    ]
                    zone.update_points(scaled_pts)
            
            # Rebuild buffer zone with scaled coordinates
            if "buffer_zone" in self.zones:
                del self.zones["buffer_zone"]
            self._build_buffer_zone()

    def save_zones(self, config_path=None):
        """Save current zone definitions to a JSON config file."""
        path = config_path or self.config_path
        if not path:
            return

        config = {}
        for zone_id, zone in self.zones.items():
            if zone_id == "buffer_zone":
                continue
            config[zone_id] = {
                "name": zone.name,
                "color": list(zone.color),
                "points": zone.points.tolist()
            }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)

    def update_zone(self, zone_id, points=None, name=None, color=None):
        """Update a specific zone's configuration."""
        if zone_id in self.zones:
            if points is not None:
                self.zones[zone_id].update_points(points)
            if name is not None:
                self.zones[zone_id].name = name
            if color is not None:
                self.zones[zone_id].color = tuple(color) if isinstance(color, list) else color
        else:
            # Create new zone
            self.zones[zone_id] = Zone(
                zone_id=zone_id,
                name=name or zone_id,
                points=points or [],
                color=color or (0, 255, 0)
            )

    def _expand_polygon(self, points, padding):
        """
        Expand a convex polygon outward by `padding` pixels.
        Uses centroid-based expansion: each vertex is pushed away from the center.
        """
        if len(points) < 3 or padding <= 0:
            return points
            
        pts = np.array(points, dtype=np.float64)
        centroid = pts.mean(axis=0)
        
        expanded = []
        for pt in pts:
            direction = pt - centroid
            length = np.linalg.norm(direction)
            if length > 0:
                unit = direction / length
                expanded.append(pt + unit * padding)
            else:
                expanded.append(pt)
        
        return np.array(expanded, dtype=np.int32)

    def get_zone_for_point(self, x, y):
        """
        Determine which zone a point belongs to.
        
        Uses zone_padding to expand zones outward before testing,
        making detection more forgiving without changing the drawn boundaries.

        Returns:
            Zone object if point is in a zone, None otherwise
        """
        # Prioritize buffer zone if there is overlap
        if "buffer_zone" in self.zones:
            bz = self.zones["buffer_zone"]
            if len(bz.points) >= 3 and bz.contains_point(x, y):
                return bz
        
        # Check each zone with padding expansion
        for zone_id, zone in self.zones.items():
            if zone_id == "buffer_zone":
                continue
            if len(zone.points) < 3:
                continue
                
            # First try the exact zone boundary
            if zone.contains_point(x, y):
                return zone
            
            # Then try the padded (expanded) zone boundary
            if self.zone_padding > 0:
                expanded_pts = self._expand_polygon(zone.points, self.zone_padding)
                expanded_contour = expanded_pts.reshape((-1, 1, 2))
                result = cv2.pointPolygonTest(expanded_contour, (float(x), float(y)), False)
                if result >= 0:
                    return zone
                    
        return None

    def get_zone_for_detection(self, detection):
        """
        Determine which zone a detection belongs to.

        Returns:
            Zone object if detection center is in a zone, None otherwise
        """
        return self.get_zone_for_point(detection.center_x, detection.center_y)

    def classify_detections(self, detections):
        """
        Classify a list of detections by zone.

        Returns:
            dict mapping zone_id -> list of detections in that zone
            Also includes "unassigned" for detections not in any zone
        """
        classified = {zone_id: [] for zone_id in self.zones}
        classified["unassigned"] = []

        for detection in detections:
            zone = self.get_zone_for_detection(detection)
            if zone:
                classified[zone.zone_id].append(detection)
            else:
                classified["unassigned"].append(detection)

        return classified

    def draw_zones(self, frame, alpha=0.25, show_labels=True, show_borders=True):
        """
        Draw semi-transparent zone overlays on a frame.

        Args:
            frame: BGR numpy array
            alpha: Transparency (0=invisible, 1=opaque)
            show_labels: Whether to show zone name labels
            show_borders: Whether to show zone borders

        Returns:
            Annotated frame (modified in-place and returned)
        """
        overlay = frame.copy()

        for zone_id, zone in self.zones.items():
            if len(zone.points) < 3:
                continue

            # Draw filled polygon
            cv2.fillPoly(overlay, [zone.points], zone.color)

            if show_borders:
                # Draw border
                cv2.polylines(frame, [zone.points], True, zone.color, 2)

        # Blend overlay
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        if show_labels:
            for zone_id, zone in self.zones.items():
                if len(zone.points) < 3:
                    continue

                # Calculate label position (top of zone bounding rect)
                x, y, w, h = zone.bounding_rect

                # Zone label with background
                label = zone.name
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                label_x = x + 5
                label_y = y + 20

                # Background rectangle for text
                cv2.rectangle(frame,
                              (label_x - 2, label_y - th - 4),
                              (label_x + tw + 4, label_y + 4),
                              zone.color, -1)
                # Text
                cv2.putText(frame, label, (label_x, label_y),
                            font, font_scale, (0, 0, 0), thickness)

        return frame

    def get_zone_names(self):
        """Return a dict of zone_id -> zone_name."""
        return {zid: z.name for zid, z in self.zones.items()}

    def get_zone_by_id(self, zone_id):
        """Get a zone by its ID."""
        return self.zones.get(zone_id)
