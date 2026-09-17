"""
Industrial Process Surveillance System
Main Streamlit Application - Drag & Drop UI

A computer-vision surveillance application for monitoring
an industrial manufacturing process using YOLOv8-Pose hand tracking.
"""

# pyrefly: ignore [missing-import]
import streamlit as st
import cv2
import numpy as np
import json
import os
import time
from PIL import Image

# pyrefly: ignore [missing-import]
from streamlit_drawable_canvas import st_canvas

from vision.pose_detector import PoseDetector, KP_LEFT_WRIST, KP_RIGHT_WRIST, MIN_KP_CONFIDENCE
from vision.zone_manager import ZoneManager
from process.state_machine import ProcessStateMachine, ProcessState
from process.event_engine import EventEngine

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Process Monitor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown("""
<style>
    .status-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        border: 2px solid #555;
        background-color: #333;
        color: #aaa;
        transition: all 0.3s ease;
    }
    
    .status-done {
        background-color: #27ae60;
        border-color: #2ecc71;
        color: white;
        box-shadow: 0 0 15px rgba(46, 204, 113, 0.5);
    }
    
    .status-active {
        background-color: #f39c12;
        border-color: #f1c40f;
        color: white;
        animation: pulse 1s infinite alternate;
    }
    
    @keyframes pulse {
        from { box-shadow: 0 0 5px rgba(243, 156, 18, 0.5); }
        to { box-shadow: 0 0 20px rgba(243, 156, 18, 0.8); }
    }
    
    .cycle-header {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
        text-align: center;
        color: #3498db;
    }
    
    .warning-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        border: 3px solid #e74c3c;
        background-color: #c0392b;
        color: white;
        animation: warning-pulse 0.5s infinite alternate;
    }
    
    @keyframes warning-pulse {
        from { box-shadow: 0 0 10px rgba(231, 76, 60, 0.4); }
        to { box-shadow: 0 0 30px rgba(231, 76, 60, 0.8); }
    }
    
    .status-warning {
        background-color: #e74c3c;
        border-color: #c0392b;
        color: white;
        animation: warning-pulse 0.5s infinite alternate;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Initialization
# -------------------------------------------------------------------
@st.cache_resource
def init_system():
    zone_manager = ZoneManager(os.path.join("config", "zones.json"))
    pose_detector = PoseDetector(model_path="yolov8n-pose.pt", confidence=0.35)
    state_machine = ProcessStateMachine()
    event_engine = EventEngine(debounce_frames=7)
    return zone_manager, pose_detector, state_machine, event_engine

zone_manager, pose_detector, state_machine, event_engine = init_system()

if "running" not in st.session_state:
    st.session_state.running = False
if "video_path" not in st.session_state:
    st.session_state.video_path = "WhatsApp Video 2026-09-10 at 4.04.34 PM.mp4"

# -------------------------------------------------------------------
# Helper: Get First Frame
# -------------------------------------------------------------------
def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.zeros((720, 1280, 3), dtype=np.uint8)

# -------------------------------------------------------------------
# Layout
# -------------------------------------------------------------------
st.title("🏭 Live Process Monitor")

video_options = {
    "Default Video (WhatsApp)": "WhatsApp Video 2026-09-10 at 4.04.34 PM.mp4",
    "Mistake Test Video (Sep 15)": "WhatsApp Video 2026-09-15 at 2.23.34 PM.mp4",
    "Demo Video": "../demo.mp4"
}
selected_video = st.selectbox("Select Camera Feed", list(video_options.keys()))
st.session_state.video_path = video_options[selected_video]

tab_monitor, tab_draw = st.tabs(["📊 Monitoring Dashboard", "📐 Draw Zones"])

# ==========================================
# TAB 2: DRAW ZONES
# ==========================================
with tab_draw:
    st.markdown("### Drag & Drop Zone Editor")
    st.write("Draw exactly **3 rectangles** on the frame below. The system will auto-assign them by size or you can redraw them.")
    
    bg_image = get_first_frame(st.session_state.video_path)
    pil_image = Image.fromarray(bg_image)
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=3,
        stroke_color="#ff0000",
        background_image=pil_image,
        update_streamlit=True,
        height=bg_image.shape[0] // 2,
        width=bg_image.shape[1] // 2,
        drawing_mode="rect",
        key="canvas",
    )
    
    if st.button("💾 Save Drawn Zones"):
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if len(objects) == 3:
                # Assuming the user draws them in order: Zone 1, Zone 2, Zone 3
                # Or we just save them in the order drawn
                scale_x = bg_image.shape[1] / (bg_image.shape[1] // 2)
                scale_y = bg_image.shape[0] / (bg_image.shape[0] // 2)
                
                zone_config_path = os.path.join("config", "zones.json")
                try:
                    with open(zone_config_path, 'r') as f:
                        zone_config = json.load(f)
                except:
                    zone_config = {}

                for i, obj in enumerate(objects):
                    if i > 2: break
                    left = int(obj["left"] * scale_x)
                    top = int(obj["top"] * scale_y)
                    width = int(obj["width"] * scale_x)
                    height = int(obj["height"] * scale_y)
                    
                    pts = [
                        [left, top],
                        [left + width, top],
                        [left + width, top + height],
                        [left, top + height]
                    ]
                    
                    zone_key = f"zone_{i+1}"
                    if zone_key not in zone_config:
                        zone_config[zone_key] = {"color": [0,255,0], "name": f"Zone {i+1}"}
                    zone_config[zone_key]["points"] = pts

                with open(zone_config_path, 'w') as f:
                    json.dump(zone_config, f, indent=4)
                
                zone_manager.load_zones(zone_config_path)
                st.success("Successfully saved 3 zones!")
            else:
                st.error(f"Please draw exactly 3 zones. You drew {len(objects)}.")

# ==========================================
# TAB 1: MONITOR
# ==========================================
with tab_monitor:
    col_video, col_status = st.columns([2.5, 1])

    with col_status:
        st.markdown("### 📊 Status Dashboard")
        cycle_header = st.empty()
        st_input = st.empty()
        st_process = st.empty()
        st_output = st.empty()
        
        # Warning area
        st.markdown("---")
        warning_placeholder = st.empty()
        
        # Controls
        st.markdown("### 🎮 Controls")
        
        col_btn1, col_btn2 = st.columns(2)
        if col_btn1.button("▶️ Start Stream", use_container_width=True, type="primary"):
            st.session_state.running = True
            state_machine.reset()
            event_engine.reset()
            st.rerun()
            
        if col_btn2.button("⏹️ Stop Stream", use_container_width=True):
            st.session_state.running = False
            st.rerun()
        
        # Manual Reset button — clears warnings and restarts process
        if st.button("🔄 Reset & Continue", use_container_width=True):
            state_machine.manual_reset()
            event_engine.reset()
            st.session_state.running = True
            st.rerun()

    with col_video:
        video_placeholder = st.empty()

    # -------------------------------------------------------------------
    # Drawing Helpers
    # -------------------------------------------------------------------
    def draw_overlay(frame, operator_pose, zone_manager):
        annotated = frame.copy()
        zone_manager.draw_zones(annotated, alpha=0.2)
        
        active_zone = "None"
        
        if operator_pose.detected and operator_pose.keypoints is not None:
            skeleton_segments = pose_detector.get_skeleton_points()
            for (pt1, pt2, conf) in skeleton_segments:
                cv2.line(annotated, pt1, pt2, (200, 200, 200), 2)
                
            for i, kp in enumerate(operator_pose.keypoints):
                x, y, c = kp
                if c >= MIN_KP_CONFIDENCE:
                    if i == KP_LEFT_WRIST:
                        cv2.circle(annotated, (int(x), int(y)), 12, (255, 100, 255), -1)
                        cv2.circle(annotated, (int(x), int(y)), 6, (255, 255, 255), -1)
                    elif i == KP_RIGHT_WRIST:
                        cv2.circle(annotated, (int(x), int(y)), 12, (255, 200, 0), -1)
                        cv2.circle(annotated, (int(x), int(y)), 6, (255, 255, 255), -1)
                    else:
                        cv2.circle(annotated, (int(x), int(y)), 4, (0, 255, 0), -1)
            
            # Identify active zone for display
            hand = operator_pose.active_hand
            if hand is None:
                for h in [operator_pose.left_hand, operator_pose.right_hand]:
                    if h.is_valid and h.current_zone:
                        hand = h
                        break
            if hand and hand.current_zone:
                active_zone = hand.current_zone
                
        # Draw status text at the top
        cv2.putText(annotated, f"Active Zone: {active_zone}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
        cv2.putText(annotated, f"Active Zone: {active_zone}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                        
        return annotated

    def update_ui(cycle):
        # Handle warning state
        if state_machine.is_warning:
            cycle_header.markdown('<div class="cycle-header" style="color:#e74c3c;">🚨 PROCESS WARNING</div>', unsafe_allow_html=True)
            warning_placeholder.markdown(
                f'<div class="warning-box">{state_machine.warning_reason}<br><br>'
                f'Press <b>🔄 Reset &amp; Continue</b> to resume</div>',
                unsafe_allow_html=True
            )
            st_input.markdown('<div class="status-box status-warning">📥 Zone 1 — HALTED</div>', unsafe_allow_html=True)
            st_process.markdown('<div class="status-box status-warning">⚙️ Zone 2 — HALTED</div>', unsafe_allow_html=True)
            st_output.markdown('<div class="status-box status-warning">📤 Zone 3 — HALTED</div>', unsafe_allow_html=True)
            return
        else:
            warning_placeholder.empty()

        if not cycle:
            cycle_header.markdown('<div class="cycle-header">Waiting for Hand in Zone 1...</div>', unsafe_allow_html=True)
            st_input.markdown('<div class="status-box">📥 Zone 1 Input</div>', unsafe_allow_html=True)
            st_process.markdown('<div class="status-box">⚙️ Zone 2 Processing</div>', unsafe_allow_html=True)
            st_output.markdown('<div class="status-box">📤 Zone 3 Output</div>', unsafe_allow_html=True)
            return

        current = cycle.state
        
        # Cycle Header
        if cycle.is_complete:
            cycle_header.markdown(f'<div class="cycle-header" style="color:#27ae60;">Cycle {cycle.part_id} Complete!</div>', unsafe_allow_html=True)
        else:
            cycle_header.markdown(f'<div class="cycle-header">Cycle {cycle.part_id} Active</div>', unsafe_allow_html=True)
        
        # Input Status
        if current in (ProcessState.ZONE_1_STARTED, ProcessState.ZONE_2_PLACED, ProcessState.ZONE_2_PICKED, ProcessState.COMPLETED) or cycle.is_complete:
            st_input.markdown('<div class="status-box status-done">📥 Zone 1 Picked</div>', unsafe_allow_html=True)
        else:
            st_input.markdown('<div class="status-box">📥 Zone 1 Input</div>', unsafe_allow_html=True)
            
        # Processing Status
        if current in (ProcessState.ZONE_2_PICKED, ProcessState.COMPLETED) or cycle.is_complete:
            st_process.markdown('<div class="status-box status-done">⚙️ Zone 2 Picked Up</div>', unsafe_allow_html=True)
        elif current == ProcessState.ZONE_2_PLACED:
            st_process.markdown('<div class="status-box status-active">⚙️ Placed (Processing...)</div>', unsafe_allow_html=True)
        elif current == ProcessState.ZONE_1_STARTED:
            st_process.markdown('<div class="status-box status-active">⚙️ Waiting for Zone 2...</div>', unsafe_allow_html=True)
        else:
            st_process.markdown('<div class="status-box">⚙️ Zone 2 Processing</div>', unsafe_allow_html=True)
            
        # Output Status
        if cycle.is_complete:
            st_output.markdown('<div class="status-box status-done">📤 Completed</div>', unsafe_allow_html=True)
        elif current == ProcessState.ZONE_2_PICKED:
            st_output.markdown('<div class="status-box status-active">📤 Waiting for Zone 3...</div>', unsafe_allow_html=True)
        else:
            st_output.markdown('<div class="status-box">📤 Zone 3 Output</div>', unsafe_allow_html=True)

    # Show warning state on page load if one is active
    if state_machine.is_warning:
        update_ui(state_machine.get_active_cycle())

    # -------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------
    if st.session_state.running:
        video_path = st.session_state.video_path
        
        if not os.path.exists(video_path):
            st.error(f"Video not found: {video_path}")
            st.session_state.running = False
            st.rerun()
            
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30
        frame_time = 1.0 / fps
        
        update_ui(None)
        
        while st.session_state.running and cap.isOpened():
            loop_start = time.time()
            ret, frame = cap.read()
            
            if not ret:
                st.info("End of stream. Stopping.")
                st.session_state.running = False
                break
                
            # 1. Detect
            operator_pose = pose_detector.detect(frame, zone_manager)
            
            # 2. Events
            events = event_engine.process_pose_frame(operator_pose, state_machine)
            
            # 3. State Machine
            for event in events:
                state_machine.handle_event(
                    event.event_type,
                    event.part_id,
                    event.class_name,
                    event.zone_id,
                    event.details
                )
                
            hand = operator_pose.active_hand
            if hand is None:
                for h in [operator_pose.left_hand, operator_pose.right_hand]:
                    if h.is_valid and h.current_zone:
                        hand = h
                        break
            active_zone = hand.current_zone if hand else None
            state_machine.update(active_zone)
                
            # 4. Display
            annotated = draw_overlay(frame, operator_pose, zone_manager)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_rgb, use_container_width=True)
            
            # 5. UI Update
            cycle = state_machine.get_active_cycle()
            if not cycle and state_machine.total_completed > 0:
                # Just show the last completed text
                pass 
            
            update_ui(cycle)
            
            # 6. Check for warning — pause the stream
            if state_machine.is_warning:
                st.session_state.running = False
                cap.release()
                st.rerun()
                
            # 6. Throttle to normal playback speed
            process_time = time.time() - loop_start
            if process_time < frame_time:
                time.sleep(frame_time - process_time)
                
        cap.release()
