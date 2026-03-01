# GEMINI.md - Cat Detector v2

## Project Overview
`cat-detector-v2` is a computer vision system designed to monitor, identify, and deter stray cats from a garden environment. It utilizes an RTSP stream for real-time video input and a custom-trained YOLOv11 (specifically `yolo26n`) model optimized for Apple Silicon (Mac Mini M4) via Metal Performance Shaders (`mps`).

### Core Functionality
- **Resident Identification:** Identifies specific cats: "Orange", "Squaky" (Residents), and "Kalaban" (Stray/Target).
- **Deterrent Mechanism:** Triggers an ESP8266-controlled horn when a stray cat ("Kalaban") is detected consistently within a history window.
- **Automated Recording:** Saves video clips of detections with identity and confidence overlays.
- **Data Lifecycle:** Includes tools for frame extraction, manual auditing, dataset splitting, and model training.

## Architecture & Technology Stack
- **AI Framework:** Ultralytics YOLOv11.
- **Model:** `yolo26n.pt` (Nano architecture).
- **Hardware Acceleration:** MPS (Metal Performance Shaders) for M4 GPU acceleration.
- **Inference/Monitoring:** `cat_monitor.py` (Main monitoring application).
- **Video Capture:** OpenCV (cv2) for RTSP stream handling.
- **Configuration:** `cat_config.yaml` for YOLO dataset paths and class names.
- **Environment:** Python 3.x with `.env` for sensitive configurations (RTSP URL, ESP8266 IP).

## Directory Structure Highlights
- `/detections`: Saved mp4 clips of cat detections.
- `/clips`: Raw video data (positives/negatives) and labeling tools.
- `/yolo_ready`: Final processed dataset for training (train/val).
- `/runs/detect`: YOLO training outputs and weights (e.g., `best.pt`).
- `/models`: Staging area for trained model weights.

## Key Operations

### Monitoring & Detection
To start the real-time monitor:
```bash
python cat_monitor.py
```
*Note: Ensure `.env` is configured with `RTSP_URL` and `ESP8266_IP`.*

### Data Collection & Auditing
- **Extract Frames:** `bash extract_frames.sh` (converts video clips to JPGs at 1fps).
- **Manual Audit:** `python fast_audit.py` (GUI to quickly re-classify or trash images).
- **Split Dataset:** `python split_data.py` (80/20 train/val split into `yolo_ready`).

### Training
To train the model on the Mac Mini M4 GPU:
```bash
python train_cats.py
```

## Development Conventions
- **Model Paths:** Always verify `DETECTOR_MODEL` path in `cat_monitor.py` matches the latest successful run in `runs/detect/`.
- **Labeling:**
  - `0`: Orange
  - `1`: Squaky
  - `2`: Kalaban
- **Testing:** New model weights should be validated against the `val` set in `yolo_ready` before deployment.
- **UI/Overlays:** Monitoring clips should display identification and confidence in the upper right corner. Bounding boxes are generally suppressed for saved clips but used during audit.

## TODO / Future Improvements
- [ ] Integrate automated upload of detections to cloud storage.
- [ ] Implement a web-based dashboard for real-time viewing.
- [ ] Refine deterrent logic to include time-of-day constraints.
