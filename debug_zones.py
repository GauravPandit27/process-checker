"""Debug: Print raw hand zones on every frame to understand detection gaps."""
import cv2, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision.zone_manager import ZoneManager
from vision.pose_detector import PoseDetector, KP_LEFT_WRIST, KP_RIGHT_WRIST, MIN_KP_CONFIDENCE

def debug_video(video_path):
    print(f"\nDebugging: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    zm = ZoneManager(os.path.join("config", "zones.json"), zone_padding=20)
    pd = PoseDetector(model_path="yolov8n-pose.pt", confidence=0.35)
    
    # Print zone boundaries
    for zid, z in zm.zones.items():
        if zid == "buffer_zone": continue
        x, y, w, h = cv2.boundingRect(z.points)
        print(f"  {zid}: x=[{x},{x+w}] y=[{y},{y+h}]")
    
    frame_count = 0
    last_zone = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        op = pd.detect(frame, zm)
        if not op.detected: continue
        
        # Check both hands
        for hand_name, hand in [("L", op.left_hand), ("R", op.right_hand)]:
            if hand.is_valid and hand.current_zone:
                zone = hand.current_zone
                if zone != last_zone or frame_count % 30 == 0:
                    t = frame_count / fps if fps > 0 else 0
                    pos = hand.smoothed_position
                    print(f"  Frame {frame_count:4d} ({t:5.1f}s): {hand_name} hand in {zone:12s} at ({pos[0]:.0f}, {pos[1]:.0f})")
                    last_zone = zone
        
        # Also check raw wrist positions vs zones (without padding)
        if frame_count <= 5 or frame_count % 100 == 0:
            kps = op.keypoints
            lw = kps[KP_LEFT_WRIST]
            rw = kps[KP_RIGHT_WRIST]
            t = frame_count / fps if fps > 0 else 0
            if lw[2] >= MIN_KP_CONFIDENCE:
                print(f"    [raw] Frame {frame_count} ({t:.1f}s): L wrist at ({lw[0]:.0f}, {lw[1]:.0f}), conf={lw[2]:.2f}")
            if rw[2] >= MIN_KP_CONFIDENCE:
                print(f"    [raw] Frame {frame_count} ({t:.1f}s): R wrist at ({rw[0]:.0f}, {rw[1]:.0f}), conf={rw[2]:.2f}")
    
    cap.release()
    print(f"  Total frames: {frame_count}")

if __name__ == "__main__":
    debug_video("../demo.mp4")
