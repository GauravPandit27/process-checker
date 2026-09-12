"""
Pose Detector Module

Wraps YOLOv8-Pose model for operator hand and body keypoint tracking.
Extracts wrist positions (left/right), applies EMA smoothing,
computes movement velocity vectors, and determines hand-zone occupancy.

COCO Pose Keypoints (17 points):
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
"""

import time
import numpy as np
from collections import deque
from ultralytics import YOLO


# COCO keypoint indices
KP_NOSE = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10

# Minimum confidence to consider a keypoint valid
MIN_KP_CONFIDENCE = 0.45

# EMA smoothing factor (0 = no smoothing, 1 = no memory)
EMA_ALPHA = 0.4

# Trail history length (frames)
TRAIL_MAX_LEN = 60


class HandState:
    """Tracks the state of a single hand (left or right)."""

    def __init__(self, side="left"):
        self.side = side
        self.raw_position = None       # (x, y) raw from detector
        self.smoothed_position = None  # (x, y) EMA-smoothed
        self.confidence = 0.0
        self.velocity = (0.0, 0.0)     # (vx, vy) pixels per frame
        self.speed = 0.0               # |velocity| pixels per frame
        self.current_zone = None       # zone_id string or None
        self.previous_zone = None      # previous zone_id
        self.zone_enter_time = None    # when hand entered current zone
        self.trail = deque(maxlen=TRAIL_MAX_LEN)  # position history
        self.last_update_time = None

    def update(self, x, y, conf, zone_id=None):
        """Update hand state with new detection."""
        self.confidence = conf

        if conf < MIN_KP_CONFIDENCE:
            # Low confidence — don't update position
            return

        old_pos = self.smoothed_position
        self.raw_position = (x, y)

        # EMA smoothing
        if self.smoothed_position is None:
            self.smoothed_position = (x, y)
        else:
            sx = EMA_ALPHA * x + (1 - EMA_ALPHA) * self.smoothed_position[0]
            sy = EMA_ALPHA * y + (1 - EMA_ALPHA) * self.smoothed_position[1]
            self.smoothed_position = (sx, sy)

        # Compute velocity
        if old_pos is not None:
            self.velocity = (
                self.smoothed_position[0] - old_pos[0],
                self.smoothed_position[1] - old_pos[1]
            )
            self.speed = (self.velocity[0] ** 2 + self.velocity[1] ** 2) ** 0.5
        else:
            self.velocity = (0.0, 0.0)
            self.speed = 0.0

        # Trail
        self.trail.append(self.smoothed_position)

        # Zone transition
        if zone_id != self.current_zone:
            self.previous_zone = self.current_zone
            self.current_zone = zone_id
            self.zone_enter_time = time.time()

        self.last_update_time = time.time()

    @property
    def is_valid(self):
        """Whether this hand has a valid, recent position."""
        return (self.smoothed_position is not None and
                self.confidence >= MIN_KP_CONFIDENCE)

    @property
    def time_in_zone(self):
        """Seconds the hand has been in its current zone."""
        if self.zone_enter_time is None:
            return 0.0
        return time.time() - self.zone_enter_time

    def get_trail_points(self):
        """Return list of trail positions for drawing."""
        return list(self.trail)


class OperatorPose:
    """
    Represents the full pose of the operator for the current frame.
    Combines both hands and body keypoints.
    """

    def __init__(self):
        self.left_hand = HandState("left")
        self.right_hand = HandState("right")
        self.detected = False
        self.person_bbox = None        # [x1, y1, x2, y2]
        self.person_confidence = 0.0

        # All raw keypoints for skeleton drawing
        self.keypoints = None          # (17, 3) array: x, y, conf
        self.timestamp = None

    @property
    def active_hand(self):
        """
        Return the hand most likely carrying a part.
        Prefers the hand with the higher confidence that is inside a zone.
        Falls back to the hand closer to zone 1/2/3.
        """
        lh = self.left_hand
        rh = self.right_hand

        # If only one hand is valid, use that
        if lh.is_valid and not rh.is_valid:
            return lh
        if rh.is_valid and not lh.is_valid:
            return rh
        if not lh.is_valid and not rh.is_valid:
            return None

        # Prefer hand that is in a zone
        if lh.current_zone and not rh.current_zone:
            return lh
        if rh.current_zone and not lh.current_zone:
            return rh

        # Both valid and in zones (or both not) — prefer higher confidence
        if lh.confidence >= rh.confidence:
            return lh
        return rh

    @property
    def dominant_hand_zone(self):
        """Get the zone of the active/dominant hand."""
        hand = self.active_hand
        return hand.current_zone if hand else None

    def get_any_hand_in_zone(self, zone_id):
        """Check if either hand is in a specific zone."""
        if self.left_hand.current_zone == zone_id:
            return self.left_hand
        if self.right_hand.current_zone == zone_id:
            return self.right_hand
        return None


class PoseDetector:
    """
    YOLOv8-Pose based operator pose detector.

    Detects the operator's pose keypoints, extracts wrist positions,
    and determines which zone their hands are operating in.
    """

    # COCO skeleton connections for drawing
    SKELETON_CONNECTIONS = [
        (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER),
        (KP_LEFT_SHOULDER, KP_LEFT_ELBOW),
        (KP_LEFT_ELBOW, KP_LEFT_WRIST),
        (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW),
        (KP_RIGHT_ELBOW, KP_RIGHT_WRIST),
        (KP_LEFT_SHOULDER, 11),   # left hip
        (KP_RIGHT_SHOULDER, 12),  # right hip
        (11, 12),                 # hip to hip
    ]

    def __init__(self, model_path="yolov8n-pose.pt", confidence=0.35):
        """
        Args:
            model_path: Path to YOLOv8-Pose model weights
            confidence: Minimum person detection confidence
        """
        self.model_path = model_path
        self.confidence = confidence
        self.model = None
        self.operator = OperatorPose()

        # Debounce: per-zone consecutive frame counts for each hand
        self._zone_frame_counts = {
            "left": {},   # zone_id -> consecutive_frames
            "right": {},
        }
        self.debounce_frames = 3

        self._load_model()

    def _load_model(self):
        """Load the YOLOv8-Pose model."""
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load pose model '{self.model_path}': {e}")

    def update_settings(self, confidence=None, model_path=None):
        """Update detection settings."""
        if confidence is not None:
            self.confidence = confidence
        if model_path is not None and model_path != self.model_path:
            self.model_path = model_path
            self._load_model()

    def detect(self, frame, zone_manager=None):
        """
        Run pose detection on a single frame.

        Args:
            frame: BGR numpy array (OpenCV format)
            zone_manager: ZoneManager instance for zone assignment

        Returns:
            OperatorPose with updated hand states
        """
        if self.model is None:
            return self.operator

        results = self.model(
            frame,
            conf=self.confidence,
            verbose=False
        )

        self._parse_results(results, zone_manager)
        return self.operator

    def _parse_results(self, results, zone_manager=None):
        """Parse YOLO-Pose results and update operator state."""
        self.operator.detected = False
        self.operator.timestamp = time.time()

        if not results or len(results) == 0:
            return

        result = results[0]
        if result.keypoints is None or len(result.keypoints.data) == 0:
            return

        # Get the primary person (highest confidence or closest to center)
        boxes = result.boxes
        kpts_data = result.keypoints.data.cpu().numpy()

        if len(kpts_data) == 0:
            return

        # Use the first (highest confidence) person
        best_idx = 0
        if boxes is not None and len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(confs))
            self.operator.person_confidence = float(confs[best_idx])
            bbox = boxes.xyxy[best_idx].cpu().numpy().astype(int).tolist()
            self.operator.person_bbox = bbox

        keypoints = kpts_data[best_idx]  # shape: (17, 3) -> x, y, conf
        self.operator.keypoints = keypoints
        self.operator.detected = True

        # Extract wrist positions
        lw = keypoints[KP_LEFT_WRIST]   # [x, y, conf]
        rw = keypoints[KP_RIGHT_WRIST]  # [x, y, conf]

        # Determine zone for each wrist
        lw_current_zone_id = None
        rw_current_zone_id = None
        lw_zone = None
        rw_zone = None
        
        if zone_manager and lw[2] >= MIN_KP_CONFIDENCE:
            zone = zone_manager.get_zone_for_point(lw[0], lw[1])
            if zone:
                lw_current_zone_id = zone.zone_id
                lw_zone = self._debounce_zone("left", zone.zone_id)
                
        if zone_manager and rw[2] >= MIN_KP_CONFIDENCE:
            zone = zone_manager.get_zone_for_point(rw[0], rw[1])
            if zone:
                rw_current_zone_id = zone.zone_id
                rw_zone = self._debounce_zone("right", zone.zone_id)

        # Reset debounce for zones no longer occupied (hand is not in ANY zone)
        if lw_current_zone_id is None:
            self._reset_debounce("left")
        if rw_current_zone_id is None:
            self._reset_debounce("right")

        # Update hand states
        self.operator.left_hand.update(lw[0], lw[1], lw[2], lw_zone)
        self.operator.right_hand.update(rw[0], rw[1], rw[2], rw_zone)

    def _debounce_zone(self, hand_side, zone_id):
        """
        Debounce zone assignment: requires N consecutive frames in the same zone.

        Returns:
            zone_id if confirmed, None if still debouncing
        """
        counts = self._zone_frame_counts[hand_side]

        # Increment count for this zone
        counts[zone_id] = counts.get(zone_id, 0) + 1

        # Reset counts for other zones
        for zid in list(counts.keys()):
            if zid != zone_id:
                counts[zid] = 0

        # Check if debounce threshold met
        if counts[zone_id] >= self.debounce_frames:
            return zone_id
        return None

    def _reset_debounce(self, hand_side):
        """Reset all debounce counters for a hand."""
        self._zone_frame_counts[hand_side].clear()

    def reset(self):
        """Reset all pose tracking state."""
        self.operator = OperatorPose()
        self._zone_frame_counts = {"left": {}, "right": {}}

    def get_skeleton_points(self):
        """
        Get skeleton line segments for drawing.

        Returns:
            List of ((x1,y1), (x2,y2), confidence) tuples
        """
        if self.operator.keypoints is None:
            return []

        segments = []
        kps = self.operator.keypoints

        for (i, j) in self.SKELETON_CONNECTIONS:
            if i >= len(kps) or j >= len(kps):
                continue
            x1, y1, c1 = kps[i]
            x2, y2, c2 = kps[j]
            min_conf = min(c1, c2)
            if min_conf >= MIN_KP_CONFIDENCE:
                segments.append(
                    ((int(x1), int(y1)), (int(x2), int(y2)), min_conf)
                )

        return segments
