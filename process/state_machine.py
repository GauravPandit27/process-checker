"""
Process State Machine - Simplified Sequence

Enforces a strict, linear flow:
1. IDLE
2. ZONE_1_STARTED
3. ZONE_2_PLACED / ZONE_2_PICKED
4. COMPLETED (Triggers when entering Zone 3)

Includes timeout recovery: if a cycle is stuck for too long,
transitions to TIMEOUT_WARNING state.
"""

from enum import Enum
import time

class ProcessState(Enum):
    IDLE = "IDLE"
    ZONE_1_STARTED = "ZONE_1_STARTED"
    ZONE_2_PLACED = "ZONE_2_PLACED"
    ZONE_2_PICKED = "ZONE_2_PICKED"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    TIMEOUT_WARNING = "TIMEOUT_WARNING"

class PartCycle:
    def __init__(self, part_id):
        self.part_id = part_id
        self.state = ProcessState.IDLE
        self.start_time = time.time()
        self.end_time = None
        self.is_complete = False

class ProcessStateMachine:
    def __init__(self, cycle_timeout=30):
        """
        Args:
            cycle_timeout: Seconds before a stuck cycle triggers a timeout warning
        """
        self.current_state = ProcessState.IDLE
        self._cycle_counter = 0
        self.active_cycle = None
        self.total_completed = 0
        self.cycle_timeout = cycle_timeout
        self.zone_2_placed_time = 0
        self.exited_zone_2 = False
        self.buffer_entry_y = None
        
    def reset(self):
        self.current_state = ProcessState.IDLE
        self.active_cycle = None
        self.zone_2_placed_time = 0
        self.exited_zone_2 = False
        self.buffer_entry_y = None
        
    def start_new_cycle(self):
        self._cycle_counter += 1
        self.active_cycle = PartCycle(self._cycle_counter)
        self.current_state = ProcessState.IDLE
        self.zone_2_placed_time = 0
        self.exited_zone_2 = False
        self.buffer_entry_y = None
        return self._cycle_counter
        
    def get_active_cycle(self):
        return self.active_cycle

    def handle_event(self, event_type, part_id=None, class_name=None, zone_id=None, details=None):
        """
        Handle a zone entry event and transition linearly.
        """
        result = {
            "transition": False,
            "new_state": self.current_state,
        }
        
        if not self.active_cycle:
            # Start a cycle automatically if we hit zone 1
            if event_type == "HAND_ENTERED_ZONE_1":
                self.start_new_cycle()
            else:
                return result

        cycle = self.active_cycle
        
        # If we're in TIMEOUT_WARNING, only allow reset via zone 1 (new cycle)
        if cycle.state == ProcessState.TIMEOUT_WARNING:
            if event_type == "HAND_ENTERED_ZONE_1":
                self.start_new_cycle()
                cycle = self.active_cycle
                cycle.state = ProcessState.ZONE_1_STARTED
                result["transition"] = True
                result["new_state"] = ProcessState.ZONE_1_STARTED
            return result
        
        if event_type == "HAND_ENTERED_ZONE_1":
            if cycle.state == ProcessState.IDLE or cycle.is_complete:
                if cycle.is_complete:
                    self.start_new_cycle()
                    cycle = self.active_cycle
                    
                cycle.state = ProcessState.ZONE_1_STARTED
                result["transition"] = True
                result["new_state"] = ProcessState.ZONE_1_STARTED
            elif cycle.state in [ProcessState.ZONE_2_PLACED, ProcessState.ZONE_2_PICKED]:
                # Flow broken! Hand returned to Zone 1 without completing the cycle
                cycle.state = ProcessState.ERROR
                result["transition"] = True
                result["new_state"] = ProcessState.ERROR
                
        elif event_type == "HAND_ENTERED_ZONE_2":
            if cycle.state == ProcessState.ZONE_1_STARTED:
                # First time hand enters Zone 2
                cycle.state = ProcessState.ZONE_2_PLACED
                self.zone_2_placed_time = time.time()
                result["transition"] = True
                result["new_state"] = ProcessState.ZONE_2_PLACED
            elif cycle.state == ProcessState.ZONE_2_PLACED:
                # Second time hand enters Zone 2
                # Add a cooldown to prevent hand adjustments during placement from being counted as a Pick
                if self.zone_2_placed_time > 0 and time.time() - self.zone_2_placed_time > 2.5:
                    cycle.state = ProcessState.ZONE_2_PICKED
                    result["transition"] = True
                    result["new_state"] = ProcessState.ZONE_2_PICKED
                
        elif event_type == "HAND_ENTERED_BUFFER":
            if cycle.state == ProcessState.ZONE_2_PICKED:
                self.exited_zone_2 = True
                if details and "y" in details and details["y"] is not None:
                    self.buffer_entry_y = details["y"]
                
        elif event_type == "HAND_ENTERED_ZONE_3":
            # Only allow completion if hand has come from Zone 2
            if cycle.state in [ProcessState.ZONE_2_PLACED, ProcessState.ZONE_2_PICKED]:
                cycle.state = ProcessState.COMPLETED
                cycle.is_complete = True
                cycle.end_time = time.time()
                self.exited_zone_2 = False
                self.total_completed += 1
                
                result["transition"] = True
                result["new_state"] = ProcessState.COMPLETED
                
                # Reset global state to wait for next cycle
                self.current_state = ProcessState.IDLE
                
        if result["transition"] and not cycle.is_complete:
            self.current_state = result["new_state"]
            
        return result

    def update(self, current_zone):
        """
        Handle time-based transitions:
        1. Auto-advance from ZONE_2_PLACED to ZONE_2_PICKED after cooldown
        2. Timeout recovery: if cycle is stuck too long, show warning
        """
        if not self.active_cycle:
            return
            
        cycle = self.active_cycle
        
        # Skip timeout checks for terminal states
        if cycle.state in [ProcessState.COMPLETED, ProcessState.ERROR, ProcessState.TIMEOUT_WARNING]:
            return
        
        # --- Auto-advance: operator rests hand in Zone 2 ---
        if cycle.state == ProcessState.ZONE_2_PLACED:
            if current_zone == "zone_2":
                if self.zone_2_placed_time > 0 and time.time() - self.zone_2_placed_time > 2.5:
                    cycle.state = ProcessState.ZONE_2_PICKED
                    self.current_state = ProcessState.ZONE_2_PICKED

        # --- Timeout recovery ---
        if cycle.state != ProcessState.IDLE:
            elapsed = time.time() - cycle.start_time
            if elapsed > self.cycle_timeout:
                cycle.state = ProcessState.TIMEOUT_WARNING
                self.current_state = ProcessState.TIMEOUT_WARNING
