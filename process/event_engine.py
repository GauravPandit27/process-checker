"""
Event Engine Module

Converts raw hand/wrist zone detections from the PoseDetector
into debounced process events. This bridges the vision layer
(what we see) and the process layer (what it means).

Key responsibilities:
- Temporal debouncing (requires N consecutive frames to trigger)
- Hand entry/exit zone event generation
- Preventing duplicate/redundant events per cycle
- Supporting both pose-based and legacy object-based detection
"""

import time
from collections import defaultdict


class EventType:
    """Event type constants."""
    # Hand-based events (pose tracking)
    HAND_ENTERED_ZONE_1 = "HAND_ENTERED_ZONE_1"
    HAND_ENTERED_ZONE_2 = "HAND_ENTERED_ZONE_2"
    HAND_LEFT_ZONE_2 = "HAND_LEFT_ZONE_2"
    HAND_ENTERED_BUFFER = "HAND_ENTERED_BUFFER"
    HAND_ENTERED_ZONE_3 = "HAND_ENTERED_ZONE_3"
    DOUBLE_PROCESSING_DETECTED = "DOUBLE_PROCESSING_DETECTED"

    # Legacy object-based events (kept for compatibility)
    PART_PICKED = "PART_PICKED"
    INPUT_DETECTED = "INPUT_DETECTED"
    PROCESSING_START = "PROCESSING_START"
    PROCESSING_COMPLETE = "PROCESSING_COMPLETE"
    OUTPUT_CONFIRMED = "OUTPUT_CONFIRMED"
    PROCESSING_TIMEOUT = "PROCESSING_TIMEOUT"
    ZONE_3_BUFFER_TIMEOUT = "ZONE_3_BUFFER_TIMEOUT"
    PART_ENTERED_ZONE = "PART_ENTERED_ZONE"
    PART_LEFT_ZONE = "PART_LEFT_ZONE"
    PERSON_ENTERED_ZONE = "PERSON_ENTERED_ZONE"


class ProcessEvent:
    """Represents a debounced, validated process event."""

    def __init__(self, event_type, part_id=None, zone_id=None,
                 class_name=None, timestamp=None, details=None):
        self.event_type = event_type
        self.part_id = part_id
        self.zone_id = zone_id
        self.class_name = class_name
        self.timestamp = timestamp or time.time()
        self.details = details or {}

    def __repr__(self):
        return (f"ProcessEvent({self.event_type} "
                f"part={self.part_id} zone={self.zone_id})")


class EventEngine:
    """
    Generates debounced process events from hand zone detection data.

    Takes hand positions and zones and produces meaningful
    process events suitable for the state machine.
    """

    def __init__(self, debounce_frames=5, person_classes=None):
        """
        Args:
            debounce_frames: Consecutive frames needed to confirm an event
            person_classes: List of class names considered as "person"
        """
        self.debounce_frames = debounce_frames
        self.person_classes = person_classes or ["person"]

        # Debounce state per (track_id, zone_id, event_type)
        self._pending_events = defaultdict(int)  # key -> consecutive_frame_count

        # Track which events have already been fired for each part
        self._fired_events = defaultdict(set)  # part_id -> set of event_types

        # Zone presence tracking
        self._zone_presence = defaultdict(set)  # zone_id -> set of track_ids

        # Hand zone tracking for pose mode
        self._last_hand_zone = None   # Last confirmed zone for active hand
        self._hand_zone_frames = defaultdict(int)  # zone_id -> consecutive frames
        self._fired_hand_events = set()  # Set of event_types fired this cycle
        self._cycle_id = None

        # Last event timestamps
        self.last_input_time = None
        self.last_event_time = None

        # Frame counter
        self._frame_count = 0

    def reset(self):
        """Reset the event engine state."""
        self._pending_events.clear()
        self._fired_events.clear()
        self._zone_presence.clear()
        self._last_hand_zone = None
        self._hand_zone_frames.clear()
        self._fired_hand_events.clear()
        self._cycle_id = None
        self.last_input_time = None
        self.last_event_time = None
        self._frame_count = 0

    def process_pose_frame(self, operator_pose, state_machine):
        """
        Process a single frame of pose data and generate hand events.

        Args:
            operator_pose: OperatorPose object from PoseDetector
            state_machine: ProcessStateMachine for context

        Returns:
            List of ProcessEvent objects generated this frame
        """
        self._frame_count += 1
        events = []

        if not operator_pose.detected:
            return events

        # Determine which zone the active hand is in
        hand = operator_pose.active_hand
        if hand is None:
            # Try either hand
            for h in [operator_pose.left_hand, operator_pose.right_hand]:
                if h.is_valid and h.current_zone:
                    hand = h
                    break

        current_zone = hand.current_zone if hand else None
        
        self.latest_y = hand.smoothed_position[1] if hand and hand.smoothed_position else None

        # ---------------------------------------------------------
        # Debounce logic: The zone must be consistent for N frames
        # ---------------------------------------------------------
        if not hasattr(self, '_candidate_zone'):
            self._candidate_zone = None
            self._candidate_frames = 0
            
        if current_zone == self._candidate_zone:
            self._candidate_frames += 1
        else:
            self._candidate_zone = current_zone
            self._candidate_frames = 1

        # Only process events if the candidate zone is confirmed (debounced)
        required_frames = 1 if self._candidate_zone == "buffer_zone" else self.debounce_frames
        if self._candidate_frames >= required_frames:
            confirmed_zone = self._candidate_zone
            
            # Check if hand entered a NEW confirmed zone
            if confirmed_zone and confirmed_zone != self._last_hand_zone:
                # Hand transitioned to a new zone
                event = self._generate_hand_zone_event(
                    confirmed_zone, self._last_hand_zone, state_machine
                )
                if event:
                    events.append(event)

                # Also check: if leaving Zone 2 and entering anything else
                if (self._last_hand_zone == "zone_2" and
                        confirmed_zone != "zone_2" and
                        "HAND_LEFT_ZONE_2" not in self._fired_hand_events):
                    leave_event = ProcessEvent(
                        event_type=EventType.HAND_LEFT_ZONE_2,
                        part_id=self._cycle_id,
                        zone_id="zone_2",
                        class_name="operator_hand",
                        details={"frame": self._frame_count, "to_zone": confirmed_zone}
                    )
                    self._fired_hand_events.add("HAND_LEFT_ZONE_2")
                    self._fired_hand_events.discard("HAND_ENTERED_ZONE_2") # Allow re-entry!
                    events.append(leave_event)

                self._last_hand_zone = confirmed_zone

            elif confirmed_zone is None and self._last_hand_zone == "zone_2":
                # Hand left Zone 2 to no-zone area (confirmed)
                if "HAND_LEFT_ZONE_2" not in self._fired_hand_events:
                    leave_event = ProcessEvent(
                        event_type=EventType.HAND_LEFT_ZONE_2,
                        part_id=self._cycle_id,
                        zone_id="zone_2",
                        class_name="operator_hand",
                        details={"frame": self._frame_count}
                    )
                    self._fired_hand_events.add("HAND_LEFT_ZONE_2")
                    self._fired_hand_events.discard("HAND_ENTERED_ZONE_2") # Allow re-entry!
                    events.append(leave_event)
                self._last_hand_zone = confirmed_zone

        # Update timestamps
        if events:
            self.last_event_time = time.time()
            for e in events:
                if e.event_type == EventType.HAND_ENTERED_ZONE_1:
                    self.last_input_time = time.time()

        return events

    def _generate_hand_zone_event(self, current_zone, previous_zone, state_machine):
        """
        Generate the appropriate event when hand enters a new zone.

        Returns:
            ProcessEvent or None
        """
        from process.state_machine import ProcessState

        current_state = state_machine.current_state

        if current_zone == "zone_1":
            if "HAND_ENTERED_ZONE_1" not in self._fired_hand_events:
                self._fired_hand_events.add("HAND_ENTERED_ZONE_1")
                # If cycle is None, start_new_cycle will be called by state machine itself now
                # Or we can just let state machine handle cycle creation
                return ProcessEvent(
                    event_type=EventType.HAND_ENTERED_ZONE_1,
                    part_id=self._cycle_id,
                    zone_id="zone_1",
                    class_name="operator_hand",
                    details={"frame": self._frame_count}
                )

        elif current_zone == "zone_2":
            if "HAND_ENTERED_ZONE_2" not in self._fired_hand_events:
                self._fired_hand_events.add("HAND_ENTERED_ZONE_2")
                return ProcessEvent(
                    event_type=EventType.HAND_ENTERED_ZONE_2,
                    part_id=self._cycle_id,
                    zone_id="zone_2",
                    class_name="operator_hand",
                    details={"frame": self._frame_count}
                )

        elif current_zone == "zone_3":
            if "HAND_ENTERED_ZONE_3" not in self._fired_hand_events:
                self._fired_hand_events.add("HAND_ENTERED_ZONE_3")
                
                event = ProcessEvent(
                    event_type=EventType.HAND_ENTERED_ZONE_3,
                    part_id=self._cycle_id,
                    zone_id="zone_3",
                    class_name="operator_hand",
                    details={"frame": self._frame_count, "y": self.latest_y}
                )
                
                # Critical: Clear memory so the next cycle can begin fresh!
                self._fired_hand_events.clear()
                self._last_hand_zone = None
                
                return event
                
        elif current_zone == "buffer_zone":
            # Allow buffer zone to fire multiple times so we catch the exit after picking!
            return ProcessEvent(
                event_type=EventType.HAND_ENTERED_BUFFER,
                part_id=self._cycle_id,
                zone_id="buffer_zone",
                class_name="operator_hand",
                details={"frame": self._frame_count, "y": self.latest_y}
            )

        return None

