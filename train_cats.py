from ultralytics import YOLO

# Initialize the 26n/Nano architecture
# If you have a specific .pt file for yolo26n, replace 'yolo11n.pt' with its path
model = YOLO('yolo26n.pt') 

# Start the training process
results = model.train(
    data='/Volumes/external-nvme256gb/cat-detector-v2/cat_config.yaml',
    epochs=100,      # 100 is a good "sweet spot" for 460 images
    imgsz=640,       # Standard resolution for cat detection
    batch=16,        # High enough for M4's unified memory
    device='mps',    # CRITICAL: Uses your Mac Mini M4 GPU
    name='cat_detector_y26n',
    patience=20,     # Stop early if the model stops improving to save electricity
    save=True
)