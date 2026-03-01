import os
from ultralytics import YOLO

# 1. Load the model (YOLO26n is great for this)
#model = YOLO('yolo26s.pt') # Note: Ultralytics uses .pt for current 2026 models
model = YOLO('../models/yolo26s.v1/best.pt')

# 2. Define your paths
POS_DIR = "dataset/positives"
NEG_DIR = "dataset/negatives"
LABEL_DIR = "dataset/labels"

os.makedirs(LABEL_DIR, exist_ok=True)

def generate_labels(directory, is_positive=True):
    images = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_name in images:
        img_path = os.path.join(directory, img_name)
        label_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")
        
        if is_positive:
            # Predict only 'cat' (COCO class 15 is cat)
            results = model.predict(img_path, classes=[0,1,2], conf=0.3, save_txt=False)
            
            with open(label_path, 'w') as f:
                for result in results:
                    for box in result.boxes.xywhn: # Normalized coordinates
                        # We use '0' as our class index for "Cat" in our custom model
                        coords = box.tolist()
                        f.write(f"0 {' '.join(map(str, coords))}\n")
        else:
            # For negatives, we just create an empty file
            open(label_path, 'a').close()

print("Processing Positives...")
generate_labels(POS_DIR, is_positive=True)
print("Processing Negatives...")
generate_labels(NEG_DIR, is_positive=False)
print("Done! Labels are in 'dataset/labels'")