"""
cat_recorder_v2.py — Optimized for Custom YOLO11n Model
======================================================
"""

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
RTSP_URL        = os.getenv("RTSP_URL", "")
OUTPUT_DIR      = Path("recordings")
# Pointing to your new best.pt from the M4 training run
MODEL_PATH      = "models/detector/yolo26n_vanilla.pt" 
DEVICE          = "mps" 

CONF_THRESHOLD  = 0.70 
# Using standard COCO classes from the yolo26n model
# COCO class ids: bird=14, cat=15
BIRD_CLASS_ID   = 14
CAT_CLASS_ID    = 15
MIN_DETECTION_DURATION = 1.0
ABSENCE_TIMEOUT = 4.0
RECONNECT_DELAY = 5
CODEC           = "avc1"
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

def connect_stream(url: str):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {url}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 15.0
    log.info("Stream opened — %dx%d @ %.1f fps", width, height, fps)
    return cap, width, height, fps

def open_writer(width: int, height: int, fps: float, info: tuple[str, float, float]):
    # info: (label, min_conf, max_conf)
    label, cmin, cmax = info
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    conf_tag   = f"c{int(cmin*100)}-{int(cmax*100)}"
    filepath   = OUTPUT_DIR / f"{label}_{timestamp}_{conf_tag}.mp4"

    for codec in (CODEC, "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
        if writer.isOpened():
            log.info("Recording started → %s", filepath.name)
            return writer, filepath
        writer.release()
    raise RuntimeError("No compatible video codec found")

def detect_animal(results) -> Optional[tuple[str, float, float]]:
    """
    Look for COCO 'cat' or 'bird' classes and return (label, min_conf, max_conf)
    """
    confs = {"cat": [], "bird": []}

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            conf = float(box.conf)

            if conf < CONF_THRESHOLD:
                continue

            if class_id == CAT_CLASS_ID:
                confs["cat"].append(conf)
            elif class_id == BIRD_CLASS_ID:
                confs["bird"].append(conf)

    # Prefer cat over bird if both present
    if confs["cat"]:
        return ("cat", min(confs["cat"]), max(confs["cat"]))
    if confs["bird"]:
        return ("bird", min(confs["bird"]), max(confs["bird"]))

    return None

def main() -> None:
    if not RTSP_URL:
        log.error("RTSP_URL not set in .env")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading model: %s", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    while True:
        cap, writer, clip_path = None, None, None
        is_recording, last_seen_ts = False, 0.0
        pending_since: Optional[float] = None
        pending_info: Optional[tuple[str, float, float]] = None

        try:
            cap, width, height, fps = connect_stream(RTSP_URL)

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Request detections for cat and bird classes from the standard model
                results = model(frame, device=DEVICE, classes=[BIRD_CLASS_ID, CAT_CLASS_ID], verbose=False)
                detected = detect_animal(results)
                animal_present = detected is not None
                now = time.monotonic()

                if animal_present:
                    label, cmin, cmax = detected
                    last_seen_ts = now

                    if pending_info is None or pending_info[0] != label:
                        pending_since = now
                        pending_info = (label, cmin, cmax)
                    else:
                        pending_label, pending_cmin, pending_cmax = pending_info
                        pending_info = (
                            pending_label,
                            min(pending_cmin, cmin),
                            max(pending_cmax, cmax),
                        )

                    if not is_recording:
                        assert pending_since is not None
                        assert pending_info is not None

                        if (now - pending_since) >= MIN_DETECTION_DURATION:
                            log.info(
                                "%s detected for %.1fs, starting recording",
                                pending_info[0].capitalize(),
                                MIN_DETECTION_DURATION,
                            )
                            writer, clip_path = open_writer(width, height, fps, pending_info)
                            is_recording = True
                            pending_since = None
                            pending_info = None
                elif not is_recording:
                    pending_since = None
                    pending_info = None

                if is_recording:
                    writer.write(frame)
                    if not animal_present and (now - last_seen_ts) > ABSENCE_TIMEOUT:
                        writer.release()
                        log.info("Recording saved → %s", clip_path.name)
                        writer, clip_path = None, None
                        is_recording = False

        except KeyboardInterrupt:
            break
        except Exception as exc:
            log.error("Error: %s", exc)
        finally:
            if writer: writer.release()
            if cap: cap.release()
        
        time.sleep(RECONNECT_DELAY)

if __name__ == "__main__":
    main()
