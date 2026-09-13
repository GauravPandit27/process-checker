"""
Process State Machine - Simplified Sequence

Enforces a strict, linear flow:
1. IDLE
2. ZONE_1_STARTED
3. ZONE_2_COMPLETED
4. COMPLETED (Triggers when entering Zone 3)
"""

from enum import Enum
import time

class ProcessState(Enum):
    IDLE = "IDLE"
    ZONE_1_STARTED = "ZONE_1_STARTED"
    ZONE_2_PLACED = "ZONE_2_PLACED"
    ZONE_2_PICKED = "ZONE_2_PICKED"
    COMPLETED = "COMPLETED"

class PartCycle:
    def __init__(self, part_id):
        self.part_id = part_id
        self.state = ProcessState.IDLE
        self.start_time = time.time()
        self.end_time = None
        self.is_complete = False

class ProcessStateMachine:
    def __init__(self):
        self.current_state = ProcessState.IDLE
        self._cycle_counter = 0
        self.active_cycle = None
        
        self.total_completed = 0
        
    def reset(self):
        self.current_state = ProcessState.IDLE
        self.active_cycle = None
        self.zone_2_placed_time = 0
        
    def start_new_cycle(self):
        self._cycle_counter += 1
        self.active_cycle = PartCycle(self._cycle_counter)
        self.current_state = ProcessState.IDLE
        self.zone_2_placed_time = 0
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
        
        if event_type == "HAND_ENTERED_ZONE_1":
            if cycle.state == ProcessState.IDLE or cycle.is_complete:
                if cycle.is_complete:
                    self.start_new_cycle()
                    cycle = self.active_cycle
                    
                cycle.state = ProcessState.ZONE_1_STARTED
                result["transition"] = True
                result["new_state"] = ProcessState.ZONE_1_STARTED
                
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
                if hasattr(self, 'zone_2_placed_time') and time.time() - self.zone_2_placed_time > 2.5:
                    cycle.state = ProcessState.ZONE_2_PICKED
                    result["transition"] = True
                    result["new_state"] = ProcessState.ZONE_2_PICKED
                
        elif event_type == "HAND_ENTERED_BUFFER":
            if cycle.state == ProcessState.ZONE_2_PICKED:
                self.exited_zone_2 = True
                if details and "y" in details and details["y"] is not None:
                    self.buffer_entry_y = details["y"]
                
        elif event_type == "HAND_ENTERED_ZONE_3":
            # Only allow completion if hand has picked up from Zone 2
            if cycle.state == ProcessState.ZONE_2_PICKED:
                cycle.state = ProcessState.COMPLETED
                cycle.is_complete = True
                cycle.end_time = time.time()
                self.exited_zone_2 = False # Reset for next cycle
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
        Handle time-based transitions for cases where no events fire
        (e.g., operator rests hand inside a zone without leaving it).
        """
        if not self.active_cycle:
            return
            
        cycle = self.active_cycle
        
        # If operator rests hand in Zone 2 (never leaves it) for more than 2.5 seconds,
        # we can assume the processing is done and they are now picking it up.
        if cycle.state == ProcessState.ZONE_2_PLACED:
            if current_zone == "zone_2":
                if hasattr(self, 'zone_2_placed_time') and time.time() - self.zone_2_placed_time > 2.5:
                    cycle.state = ProcessState.ZONE_2_PICKED
                    self.current_state = ProcessState.ZONE_2_PICKED
