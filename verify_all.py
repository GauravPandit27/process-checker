"""
Verification script: processes all demo videos through the hardened pipeline
and prints the event sequence and final state for each.
"""
import cv2
import os
import sys
import time
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vision.zone_manager import ZoneManager
from vision.pose_detector import PoseDetector
from process.event_engine import EventEngine
from process.state_machine import ProcessStateMachine, ProcessState

# We will use this to override time.time() so it reflects video time
current_video_time = 0.0

def mock_time():
    global current_video_time
    return current_video_time

def analyze_video(video_path, label):
    global current_video_time
    
    print(f"\n{'='*60}")
    print(f"  {label}: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    
    if not os.path.exists(video_path):
        print(f"  ❌ File not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  FPS: {fps:.1f}, Total Frames: {total_frames}")
    
    zone_manager = ZoneManager(os.path.join("config", "zones.json"), zone_padding=20)
    pose_detector = PoseDetector(model_path="yolov8n-pose.pt", confidence=0.35)
    
    # Scale zones to match video resolution
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    zone_manager.scale_zones_to_resolution(vid_w, vid_h)
    print(f"  Resolution: {vid_w}x{vid_h}, Scale: {zone_manager._current_scale}")
    
    # Print scaled zone bounds
    for zid, z in zone_manager.zones.items():
        if zid == "buffer_zone": continue
        bx, by, bw, bh = cv2.boundingRect(z.points)
        print(f"  {zid}: x=[{bx},{bx+bw}] y=[{by},{by+bh}]")
    
    # We patch time.time() so that the State Machine and Event Engine use video time
    with patch('time.time', side_effect=mock_time):
        state_machine = ProcessStateMachine(cycle_timeout=30)
        event_engine = EventEngine(debounce_frames=3, grace_frames=2)
        
        frame_count = 0
        current_video_time = 0.0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            current_video_time = frame_count / fps
            
            operator_pose = pose_detector.detect(frame, zone_manager)
            events = event_engine.process_pose_frame(operator_pose, state_machine)
            
            for e in events:
                # Override timestamp to use video time
                e.timestamp = current_video_time
                state_machine.handle_event(
                    e.event_type,
                    e.part_id,
                    e.class_name,
                    e.zone_id,
                    e.details
                )
                print(f"  Frame {frame_count:4d} ({current_video_time:.1f}s): {e.event_type:25s} → State: {state_machine.current_state.value}")
            
            hand = operator_pose.active_hand
            if hand is None:
                for h in [operator_pose.left_hand, operator_pose.right_hand]:
                    if h.is_valid and h.current_zone:
                        hand = h
                        break
            active_zone = hand.current_zone if hand else None
            state_machine.update(active_zone)
                
    cap.release()
    
    final = state_machine.current_state
    completed = state_machine.total_completed
    print(f"\n  ✅ Final State: {final.value}")
    print(f"  ✅ Cycles Completed: {completed}")
    
    return final, completed


if __name__ == "__main__":
    videos = [
        ("../demo.mp4", "GOOD FLOW (expect COMPLETED)"),
        ("../wrong_demo.mp4", "BAD FLOW 1 (expect ERROR)"),
        ("../wrong_demo2.mp4", "BAD FLOW 2 (expect ERROR)"),
        ("../wrond_demo3.mp4", "BAD FLOW 3 (expect ERROR)"),
    ]
    
    results = {}
    for path, label in videos:
        result = analyze_video(path, label)
        if result:
            results[label] = result
    
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for label, (state, completed) in results.items():
        status = "✅ PASS" if (("GOOD" in label and completed > 0) or ("BAD" in label and state == ProcessState.ERROR)) else "❌ FAIL"
        print(f"  {status} {label}: state={state.value}, completed={completed}")
