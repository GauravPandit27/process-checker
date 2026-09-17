"""
Event Engine Module

Converts raw hand/wrist zone detections from the PoseDetector
into debounced process events. This is the SINGLE layer responsible
for temporal filtering (debouncing).

Key responsibilities:
- Temporal debouncing (requires N consecutive frames to trigger)
- Sticky-zone grace period (tolerates brief YOLO dropouts)
- Hand locking (tracks the same hand for an entire cycle)
- State-aware event generation (only fires events the state machine can accept)
- Preventing duplicate/redundant events per cycle
"""

import time
from collections import defaultdict


class EventType:
    """Event type constants."""
    HAND_ENTERED_ZONE_1 = "HAND_ENTERED_ZONE_1"
    HAND_ENTERED_ZONE_2 = "HAND_ENTERED_ZONE_2"
    HAND_LEFT_ZONE_2 = "HAND_LEFT_ZONE_2"
    HAND_ENTERED_BUFFER = "HAND_ENTERED_BUFFER"
    HAND_ENTERED_ZONE_3 = "HAND_ENTERED_ZONE_3"
    DOUBLE_PROCESSING_DETECTED = "DOUBLE_PROCESSING_DETECTED"

    # Legacy (kept for compatibility)
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

    This is the SINGLE source of truth for temporal filtering.
    The PoseDetector reports raw zones every frame; this engine
    applies debouncing, sticky-zone grace, and hand locking.
    
    IMPORTANT: Events are STATE-AWARE. The engine checks the state 
    machine's current state before generating events, so events that 
    would be rejected are never created in the first place.
    """

    def __init__(self, debounce_frames=3, grace_frames=2, person_classes=None):
        """
        Args:
            debounce_frames: Consecutive frames needed to confirm a zone entry
            grace_frames: Frames to tolerate a brief zone dropout before resetting
            person_classes: List of class names considered as "person"
        """
        self.debounce_frames = debounce_frames
        self.grace_frames = grace_frames
        self.person_classes = person_classes or ["person"]

        # --- Debounce state ---
        self._candidate_zone = None
        self._candidate_frames = 0
        self._grace_counter = 0

        # --- Cycle tracking ---
        self._last_confirmed_zone = None
        self._cycle_id = None
        self._locked_hand_side = None    # "left" or "right" — locked for entire cycle
        self._in_cycle = False           # True when a cycle is active

        # --- Timestamps ---
        self.last_input_time = None
        self.last_event_time = None

        # --- Frame counter ---
        self._frame_count = 0

        # --- Latest Y for buffer zone direction ---
        self.latest_y = None

    def reset(self):
        """Reset the event engine state."""
        self._candidate_zone = None
        self._candidate_frames = 0
        self._grace_counter = 0
        self._last_confirmed_zone = None
        self._cycle_id = None
        self._locked_hand_side = None
        self._in_cycle = False
        self.last_input_time = None
        self.last_event_time = None
        self._frame_count = 0
        self.latest_y = None

    def _get_tracking_hand(self, operator_pose):
        """
        Get the hand to track for this frame.
        If a hand is locked (mid-cycle), always use that hand.
        Otherwise, pick the best available hand.
        """
        if self._locked_hand_side == "left":
            hand = operator_pose.left_hand
            if hand.is_valid:
                return hand
            return hand
            
        elif self._locked_hand_side == "right":
            hand = operator_pose.right_hand
            if hand.is_valid:
                return hand
            return hand

        # No hand locked yet — use the best available
        hand = operator_pose.active_hand
        if hand is None:
            for h in [operator_pose.left_hand, operator_pose.right_hand]:
                if h.is_valid and h.current_zone:
                    hand = h
                    break
        return hand

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
            self._grace_counter += 1
            if self._grace_counter > self.grace_frames:
                self._candidate_zone = None
                self._candidate_frames = 0
            return events

        # Get the hand we should be tracking
        hand = self._get_tracking_hand(operator_pose)
        current_zone = hand.current_zone if hand else None
        self.latest_y = hand.smoothed_position[1] if hand and hand.smoothed_position else None

        # --- Sync cycle state with state machine ---
        from process.state_machine import ProcessState
        sm_state = state_machine.current_state
        
        # Detect cycle completion: state machine went back to IDLE after we were in a cycle
        if self._in_cycle and sm_state == ProcessState.IDLE:
            self._in_cycle = False
            self._locked_hand_side = None
            self._last_confirmed_zone = None

        # ---------------------------------------------------------
        # Debounce logic with sticky-zone grace period
        # ---------------------------------------------------------
        if current_zone == self._candidate_zone:
            self._candidate_frames += 1
            self._grace_counter = 0
        elif current_zone is None and self._candidate_zone is not None:
            # Zone disappeared — use grace period
            self._grace_counter += 1
            if self._grace_counter > self.grace_frames:
                self._candidate_zone = None
                self._candidate_frames = 0
        else:
            # Different zone — switch candidate
            self._candidate_zone = current_zone
            self._candidate_frames = 1
            self._grace_counter = 0

        # Only process events if the candidate zone is confirmed (debounced)
        required_frames = 1 if self._candidate_zone == "buffer_zone" else self.debounce_frames
        if self._candidate_frames >= required_frames:
            confirmed_zone = self._candidate_zone
            
            # Check if hand entered a NEW confirmed zone
            if confirmed_zone and confirmed_zone != self._last_confirmed_zone:
                
                # Generate event only if state machine can accept it
                event = self._generate_state_aware_event(
                    confirmed_zone, self._last_confirmed_zone, state_machine
                )
                if event:
                    events.append(event)
                    
                    # Lock hand on Zone 1 entry
                    if event.event_type == EventType.HAND_ENTERED_ZONE_1 and hand:
                        self._locked_hand_side = hand.side
                        self._in_cycle = True

                # Handle leaving Zone 2
                if (self._last_confirmed_zone == "zone_2" and
                        confirmed_zone != "zone_2"):
                    leave_event = ProcessEvent(
                        event_type=EventType.HAND_LEFT_ZONE_2,
                        part_id=self._cycle_id,
                        zone_id="zone_2",
                        class_name="operator_hand",
                        details={"frame": self._frame_count, "to_zone": confirmed_zone}
                    )
                    events.append(leave_event)

                self._last_confirmed_zone = confirmed_zone

            elif confirmed_zone is None and self._last_confirmed_zone is not None:
                # Hand left a zone to no-zone area
                if self._last_confirmed_zone == "zone_2":
                    leave_event = ProcessEvent(
                        event_type=EventType.HAND_LEFT_ZONE_2,
                        part_id=self._cycle_id,
                        zone_id="zone_2",
                        class_name="operator_hand",
                        details={"frame": self._frame_count}
                    )
                    events.append(leave_event)
                        
                self._last_confirmed_zone = confirmed_zone

        # Update timestamps
        if events:
            self.last_event_time = time.time()
            for e in events:
                if e.event_type == EventType.HAND_ENTERED_ZONE_1:
                    self.last_input_time = time.time()

        return events

    def _generate_state_aware_event(self, current_zone, previous_zone, state_machine):
        """
        Generate events ONLY if the state machine is in a state that can accept them.
        
        This prevents:
        - Zone 3 firing when no cycle is active (infinite loop bug)
        - Zone 2 firing when cycle hasn't started
        - Duplicate events within a cycle
        
        Returns:
            ProcessEvent or None
        """
        from process.state_machine import ProcessState
        sm_state = state_machine.current_state

        if current_zone == "zone_1":
            # Zone 1 always fires — it starts new cycles or triggers errors
            return ProcessEvent(
                event_type=EventType.HAND_ENTERED_ZONE_1,
                part_id=self._cycle_id,
                zone_id="zone_1",
                class_name="operator_hand",
                details={"frame": self._frame_count}
            )

        elif current_zone == "zone_2":
            # Zone 2 only fires if we're in a cycle that expects it
            if sm_state in [ProcessState.ZONE_1_STARTED, ProcessState.ZONE_2_PLACED]:
                return ProcessEvent(
                    event_type=EventType.HAND_ENTERED_ZONE_2,
                    part_id=self._cycle_id,
                    zone_id="zone_2",
                    class_name="operator_hand",
                    details={"frame": self._frame_count}
                )

        elif current_zone == "zone_3":
            # Zone 3 only fires if we're in a state that can complete
            if sm_state in [ProcessState.ZONE_2_PLACED, ProcessState.ZONE_2_PICKED]:
                return ProcessEvent(
                    event_type=EventType.HAND_ENTERED_ZONE_3,
                    part_id=self._cycle_id,
                    zone_id="zone_3",
                    class_name="operator_hand",
                    details={"frame": self._frame_count, "y": self.latest_y}
                )
                
        elif current_zone == "buffer_zone":
            # Buffer zone fires if we're in a relevant state
            if sm_state in [ProcessState.ZONE_2_PICKED, ProcessState.ZONE_2_PLACED]:
                return ProcessEvent(
                    event_type=EventType.HAND_ENTERED_BUFFER,
                    part_id=self._cycle_id,
                    zone_id="buffer_zone",
                    class_name="operator_hand",
                    details={"frame": self._frame_count, "y": self.latest_y}
                )

        return None
