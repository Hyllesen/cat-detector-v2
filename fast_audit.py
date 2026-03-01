import os
import cv2
import shutil

# Your exact local paths
BASE_DIR = "/Volumes/external-nvme256gb/cat-detector-v2/clips/dataset"
IMAGE_DIR = os.path.join(BASE_DIR, "positives")
LABEL_DIR = os.path.join(BASE_DIR, "labels")
TRASH_DIR = os.path.join(BASE_DIR, "trash")

os.makedirs(TRASH_DIR, exist_ok=True)

# 1. PRE-FILTERING
all_images = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
valid_images = []

for img_name in all_images:
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, label_name)
    img_path = os.path.join(IMAGE_DIR, img_name)
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            if f.read().strip():
                valid_images.append(img_name)
                continue
    
    # If no label or empty, move to trash
    shutil.move(img_path, os.path.join(TRASH_DIR, img_name))
    if os.path.exists(label_path):
        shutil.move(label_path, os.path.join(TRASH_DIR, label_name))

print(f"Starting audit for {len(valid_images)} images.")

# 2. THE AUDITOR
history = []
i = 0

while i < len(valid_images):
    img_name = valid_images[i]
    img_path = os.path.join(IMAGE_DIR, img_name)
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(LABEL_DIR, label_name)

    if not os.path.exists(img_path):
        i += 1
        continue

    img = cv2.imread(img_path)
    if img is None:
        i += 1
        continue

    h, w, _ = img.shape
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    data = lines[0].strip().split()
    _, x_c, y_c, bw, bh = map(float, data[:5])
    
    # Draw box
    x1, y1 = int((x_c - bw/2) * w), int((y_c - bh/2) * h)
    x2, y2 = int((x_c + bw/2) * w), int((y_c + bh/2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Overlay Info
    menu = "[1] Orange | [2] Squaky | [3] Kalaban | [4] Trash | [z] Undo"
    cv2.putText(img, menu, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, f"File: {img_name} ({i+1}/{len(valid_images)})", (10, h-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Fast Audit", img)
    key = cv2.waitKey(0) & 0xFF

    new_class = None
    if key == ord('1'): new_class = "0" # Orange
    elif key == ord('2'): new_class = "1" # Squaky
    elif key == ord('3'): new_class = "2" # Kalaban

    if new_class is not None:
        with open(label_path, 'w') as f:
            f.write(f"{new_class} {' '.join(data[1:])}\n")
        history.append((i, False))
        i += 1
    elif key == ord('4'): # Trash
        history.append((i, True))
        shutil.move(img_path, os.path.join(TRASH_DIR, img_name))
        shutil.move(label_path, os.path.join(TRASH_DIR, label_name))
        i += 1
    elif key in [ord('z'), ord('u')]:
        if history:
            prev_i, was_trashed = history.pop()
            if was_trashed:
                p_name = valid_images[prev_i]
                shutil.move(os.path.join(TRASH_DIR, p_name), os.path.join(IMAGE_DIR, p_name))
                shutil.move(os.path.join(TRASH_DIR, os.path.splitext(p_name)[0] + ".txt"), 
                            os.path.join(LABEL_DIR, os.path.splitext(p_name)[0] + ".txt"))
            i = prev_i
    elif key == ord('q'):
        break

cv2.destroyAllWindows()