import cv2
import time
import os
import requests
from datetime import datetime
from collections import deque
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
RTSP_URL = os.getenv("RTSP_URL")
# Using the requested newest model path
DETECTOR_MODEL = "models/yolo26s.v2/best.pt"
ESP8266_IP = os.getenv("ESP8266_IP") 
DETECTIONS_DIR = "detections"

CONF_THRESHOLD = 0.7
ALERT_COOLDOWN = 60 
VIDEO_BUFFER_SECONDS = 5 

# Deterrent Logic
DETERRENT_THRESHOLD = 15
HISTORY_WINDOW = 30
identity_history = deque(maxlen=HISTORY_WINDOW)
last_deterrent_time = 0

# Identity Mapping (based on cat_config.yaml classes)
RESIDENTS = ["orange", "squaky"]
STRAY_LABEL = "kalaban"

os.makedirs(DETECTIONS_DIR, exist_ok=True)

# Load the single consolidated model
model = YOLO(DETECTOR_MODEL)

def trigger_deterrent():
    if not ESP8266_IP:
        print("⚠️ ESP8266_IP not set in environment. Skipping deterrent trigger.")
        return False
    
    try:
        url = f"http://{ESP8266_IP}/trigger"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print("✅ Horn triggered successfully via ESP8266.")
            return True
        else:
            print(f"❌ Failed to trigger horn. Status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error communicating with ESP8266: {e}")
        return False

def run_monitor():
    global last_deterrent_time
    cap = cv2.VideoCapture(RTSP_URL)
    
    # Original stream dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Crop parameters from ffmpeg -vf "crop=in_w-200:in_h-220:0:220"
    crop_x, crop_y = 0, 220
    crop_w = frame_width - 200
    crop_h = frame_height - 220
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20 
    
    video_writer = None
    recording_until = 0
    
    print(f"--- Garden Monitoring Active (Single Model: {DETECTOR_MODEL}) ---")
    print(f"--- Detection Zone: {crop_w}x{crop_h} at offset ({crop_x}, {crop_y}) ---")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Create the detection slice for YOLO (matching training data perspective)
        crop_frame = frame[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

        # Single inference pass handles both detection and identification on the CROP
        results = model(crop_frame, verbose=False, device='mps')
        
        current_frame_identity = None
        current_frame_conf = 0.0

        for r in results:
            for box in r.boxes:
                conf = box.conf[0].item()
                if conf > CONF_THRESHOLD:
                    # Get label directly from the detection result
                    class_id = int(box.cls[0].item())
                    label = r.names[class_id].lower()
                    
                    # Track the most confident identity in the frame
                    if conf > current_frame_conf:
                        current_frame_conf = conf
                        current_frame_identity = label

        # --- 1. DRAW TOP-RIGHT LABEL (If cat present) ---
        if current_frame_identity:
            color = (0, 0, 255) if current_frame_identity == STRAY_LABEL else (0, 255, 0)
            label_str = f"{current_frame_identity.upper()} {current_frame_conf:.2f}"
            
            font_scale = 1.0
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            text_x = frame_width - text_w - 20
            text_y = 40
            
            cv2.rectangle(frame, (text_x - 10, text_y - text_h - 10), (frame_width - 10, text_y + 10), color, -1)
            cv2.putText(frame, label_str, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            # Update history
            identity_history.append(current_frame_identity)
            if current_frame_identity in RESIDENTS:
                identity_history.clear()

        # --- 2. DETERRENT LOGIC & DRAWING ---
        stray_count = identity_history.count(STRAY_LABEL)
        if stray_count >= DETERRENT_THRESHOLD:
            current_time = time.time()
            # Visual indicator that deterrent is ready/active
            cv2.putText(frame, "!!! STRAY DETECTED - DETERRENT READY !!!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            if current_time - last_deterrent_time < 5: # Show "TRIGGERED" for 5 seconds
                cv2.putText(frame, "!!! DETERRENT TRIGGERED !!!", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

            if current_time - last_deterrent_time > ALERT_COOLDOWN:
                print(f"🚨 DETERRENT TRIGGERED! (Stray seen {stray_count}/{HISTORY_WINDOW} times)")
                #Reset the stray count and history to avoid multiple triggers in a short time
                stray_count = 0
                #trigger_deterrent() # Uncomment to enable ESP8266 trigger
                last_deterrent_time = current_time

        # --- VIDEO SAVING LOGIC ---
        if current_frame_identity:
            recording_until = time.time() + VIDEO_BUFFER_SECONDS
            
            if video_writer is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                prob_int = int(current_frame_conf * 100)
                filename = f"{current_frame_identity}_p{prob_int}_{timestamp}.mp4"
                save_path = os.path.join(DETECTIONS_DIR, filename)
                
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                video_writer = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))
                print(f"📹 Recording: {filename}")

        if video_writer is not None:
            video_writer.write(frame)
            if time.time() > recording_until:
                video_writer.release()
                video_writer = None
                print("🏁 Saved.")

        # --- 3. DRAW DETECTION ZONE (WINDOW ONLY) ---
        # Drawn after video_writer.write so it's not in the saved clip
        cv2.rectangle(frame, (crop_x, crop_y), (crop_x + crop_w, crop_y + crop_h), (150, 150, 150), 1)
        cv2.putText(frame, "YOLO DETECTION ZONE", (crop_x + 5, crop_y + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        cv2.imshow("Garden Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if video_writer: video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_monitor()
