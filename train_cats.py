from ultralytics import YOLO

# Initialize the 26n/Nano architecture
# If you have a specific .pt file for yolo26n, replace 'yolo11n.pt' with its path
model = YOLO('yolo26n.pt') 

# Start the training process
results = model.train(
    data='/Volumes/external-nvme256gb/cat-detector-v2/cat_config.yaml',
    epochs=300,      # 100 is a good "sweet spot" for 460 images
    imgsz=640,       # Standard resolution for cat detection
    batch=-1,        # High enough for M4's unified memory
    device='mps',    # CRITICAL: Uses your Mac Mini M4 GPU
    name='cat_detector_y26s.v3',
    patience=50,     # Stop early if the model stops improving to save electricity
    save=True,
     # --- OPTIMIZATION ---
    lr0=0.01,           # Initial learning rate
    lrf=0.01,           # Final learning rate (lr0 * lrf)
    optimizer='MuSGD',  # YOLO26's native optimizer for better convergence
    # --- AUGMENTATION SETTINGS ---
    hsv_h=0.015,      # Image hue (fraction)
    hsv_s=0.7,        # Image saturation (fraction)
    hsv_v=0.4,        # Image value (brightness) (fraction)
    degrees=10.0,     # Rotate +/- 10 degrees
    translate=0.1,    # Translate +/- 10%
    scale=0.5,        # Scale +/- 50%
    shear=2.0,        # Shear +/- 2 degrees
    perspective=0.0,  # Perspective (0.0 - 0.001)
    flipud=0.0,       # Vertical flip (probability)
    fliplr=0.5,       # Horizontal flip (probability)
    mosaic=1.0,       # Mosaic (1.0 = always on). Great for cats in various scenes.
    mixup=0.0,        # Mixup (usually 0.0 for Nano models, can be 0.1 for larger)
    bgr=0.1           # BGR channel swap probability (as requested in your link)
)