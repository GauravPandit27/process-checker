"""
Video Processor Module

Handles video input (file upload or webcam), frame-by-frame processing,
frame annotation, and violation evidence snapshot saving.
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime


class VideoProcessor:
    """
    Handles video I/O and frame annotation for the surveillance system.
    """

    def __init__(self, snapshot_dir=None):
        """
        Args:
            snapshot_dir: Directory to save violation evidence snapshots
        """
        self.snapshot_dir = snapshot_dir or os.path.join("data", "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)

        self.cap = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 30.0
        self.total_frames = 0
        self.current_frame_num = 0

    def open_video(self, video_path):
        """
        Open a video file for processing.

        Args:
            video_path: Path to the video file

        Returns:
            True if successfully opened
        """
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            return False

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_num = 0

        return True

    def read_frame(self):
        """
        Read the next frame from the video.

        Returns:
            (success, frame) tuple
        """
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if ret:
            self.current_frame_num += 1
        return ret, frame

    def get_progress(self):
        """Get video processing progress (0.0 to 1.0)."""
        if self.total_frames > 0:
            return self.current_frame_num / self.total_frames
        return 0.0

    def annotate_frame(self, frame, detections, zone_manager, state_machine,
                       alarm_manager, process_status="OK"):
        """
        Annotate a frame with detection boxes, IDs, zones, and status.

        Args:
            frame: BGR numpy array
            detections: List of Detection objects
            zone_manager: ZoneManager instance
            state_machine: ProcessStateMachine instance
            alarm_manager: AlarmManager instance
            process_status: Current process status string

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # 1. Draw zones
        zone_manager.draw_zones(annotated, alpha=0.15)

        # 2. Draw detection boxes and IDs
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = self._get_detection_color(det)

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label with class and ID
            label_parts = [det.class_name]
            if det.track_id is not None:
                label_parts.append(f"#{det.track_id}")
            label_parts.append(f"{det.confidence:.0%}")
            label = " ".join(label_parts)

            # Label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            cv2.rectangle(annotated,
                          (x1, y1 - th - 8),
                          (x1 + tw + 4, y1),
                          color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        font, font_scale, (255, 255, 255), thickness)

            # Draw center point
            cx, cy = int(det.center_x), int(det.center_y)
            cv2.circle(annotated, (cx, cy), 4, color, -1)

        # 3. Draw status overlay (top of frame)
        self._draw_status_overlay(annotated, state_machine, alarm_manager,
                                  process_status)

        return annotated

    def _get_detection_color(self, detection):
        """Get color for a detection based on its class."""
        if detection.class_name == "person":
            return (255, 200, 0)   # Cyan-ish for persons
        else:
            return (0, 255, 255)   # Yellow for objects/parts

    def _draw_status_overlay(self, frame, state_machine, alarm_manager,
                             process_status):
        """Draw status info overlay on the frame."""
        h, w = frame.shape[:2]

        # Status bar at top
        bar_height = 40
        bar_color = (0, 150, 0) if process_status == "OK" else (0, 0, 200)
        if process_status == "WARNING":
            bar_color = (0, 180, 255)

        # Semi-transparent bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_height), bar_color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Status text
        status_text = f"STATUS: {process_status}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, status_text, (10, 28),
                     font, 0.7, (255, 255, 255), 2)

        # State text
        state_text = f"State: {state_machine.current_state.value}"
        cv2.putText(frame, state_text, (w - 250, 28),
                     font, 0.5, (255, 255, 255), 1)

        # Active alarm indicator
        if alarm_manager.has_active_alarms():
            # Flashing alarm indicator
            if int(time.time() * 2) % 2 == 0:
                cv2.rectangle(frame, (0, bar_height), (w, bar_height + 30),
                              (0, 0, 255), -1)
                alarms = alarm_manager.get_active_alarms()
                alarm_text = f"🚨 ALARM: {alarms[0].alarm_type}" if alarms else ""
                cv2.putText(frame, alarm_text, (10, bar_height + 22),
                             font, 0.55, (255, 255, 255), 2)

        # Part info at bottom
        if state_machine.current_part_id is not None:
            cycle = state_machine.get_active_cycle()
            if cycle:
                info_y = h - 20
                info_text = (f"Part #{cycle.part_id} | "
                             f"Processing: {cycle.current_processing_time:.1f}s")
                cv2.putText(frame, info_text, (10, info_y),
                             font, 0.5, (255, 255, 255), 1)

    def save_snapshot(self, frame, alarm_type, part_id=None):
        """
        Save a frame as violation evidence.

        Args:
            frame: BGR numpy array
            alarm_type: Type of alarm/violation
            part_id: Associated part ID

        Returns:
            Path to saved snapshot
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            part_str = f"_part{part_id}" if part_id else ""
            filename = f"{timestamp}_{alarm_type}{part_str}.jpg"
            filepath = os.path.join(self.snapshot_dir, filename)

            cv2.imwrite(filepath, frame)
            return filepath
        except Exception as e:
            print(f"Error saving snapshot: {e}")
            return None

    def release(self):
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def video_info(self):
        """Get video metadata."""
        return {
            "width": self.frame_width,
            "height": self.frame_height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_sec": self.total_frames / self.fps if self.fps > 0 else 0,
        }

    def __del__(self):
        self.release()
