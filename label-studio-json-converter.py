import os
import json

# Paths from your image_5677f7.png
LABEL_DIR = "/Volumes/external-nvme256gb/cat-detector-v2/clips/dataset/labels"
IMAGE_DIR = "/Volumes/external-nvme256gb/cat-detector-v2/clips/dataset/positives"
OUT_FILE = "ls_final_fix.json"

tasks = []
# Get only image files
images = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

for img_name in images:
    label_file = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, label_file)
    
    # We point directly to the local serving path
    task = {
        "data": {"image": f"/data/local-files/?d=dataset/positives/{img_name}"},
        "annotations": [{
            "result": []
        }]
    }
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    cls, x, y, w, h = map(float, parts)
                    label = "Resident" if cls == 0 else "Kalaban"
                    
                    # YOLO to Label Studio percentage conversion
                    task["annotations"][0]["result"].append({
                        "from_name": "label",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "value": {
                            "rectanglelabels": [label],
                            "x": (x - w/2) * 100,
                            "y": (y - h/2) * 100,
                            "width": w * 100,
                            "height": h * 100
                        }
                    })
    tasks.append(task)

with open(OUT_FILE, 'w') as f:
    json.dump(tasks, f)
print(f"Created {OUT_FILE}. Upload this file to finish!")