from process.state_machine import ProcessStateMachine, ProcessState
sm = ProcessStateMachine()
print("Init state:", sm.current_state)

sm.handle_event("HAND_ENTERED_ZONE_1", sm.active_cycle.part_id, "hand", "zone_1")
print("After Z1:", sm.current_state)

sm.handle_event("HAND_ENTERED_ZONE_2", sm.active_cycle.part_id, "hand", "zone_2")
print("After Z2:", sm.current_state)

sm.handle_event("HAND_ENTERED_ZONE_1", sm.active_cycle.part_id, "hand", "zone_1")
print("After Z1 again:", sm.current_state)
