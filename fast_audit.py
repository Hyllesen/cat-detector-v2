import os
import cv2
import shutil

# Paths
BASE_DIR = "/Volumes/external-nvme256gb/cat-detector-v2/clips/dataset"
IMAGE_DIR = os.path.join(BASE_DIR, "positives")
LABEL_DIR = os.path.join(BASE_DIR, "labels")
TRASH_DIR = os.path.join(BASE_DIR, "trash")

os.makedirs(TRASH_DIR, exist_ok=True)

all_images = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
valid_images = all_images # We don't pre-filter now, because we want to see potentially empty ones

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
    
    # Check if a box exists to draw it
    data = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            if lines:
                data = lines[0].strip().split()
                _, x_c, y_c, bw, bh = map(float, data[:5])
                x1, y1 = int((x_c - bw/2) * w), int((y_c - bh/2) * h)
                x2, y2 = int((x_c + bw/2) * w), int((y_c + bh/2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red box for checking

    # Menu
    menu = "[1] Orng [2] Squak [3] Kalab | [4] TRASH | [5] NEGATIVE (Nothing here)"
    cv2.putText(img, menu, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow("Fast Audit v2", img)
    
    key = cv2.waitKey(0) & 0xFF

    if key in [ord('1'), ord('2'), ord('3')]:
        cls = str(int(chr(key)) - 1)
        with open(label_path, 'w') as f:
            # We need coordinates to save a class. If no coords exist, we can't label it a cat.
            if data:
                f.write(f"{cls} {' '.join(data[1:])}\n")
                history.append((i, "labeled"))
                i += 1
            else:
                print("Cannot label as cat: No bounding box exists. Use 5 for Negative or 4 for Trash.")

    elif key == ord('4'): # TRASH: Remove from dataset entirely
        shutil.move(img_path, os.path.join(TRASH_DIR, img_name))
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(TRASH_DIR, label_name))
        history.append((i, "trashed"))
        i += 1

    elif key == ord('5'): # NEGATIVE: Keep image, but empty the label
        with open(label_path, 'w') as f:
            f.write("") # Create an empty file
        history.append((i, "negative"))
        print(f"Set {img_name} as a Background sample.")
        i += 1

    elif key == ord('z'): # Basic Undo
        if history:
            idx, act = history.pop()
            if act == "trashed":
                # Move back
                shutil.move(os.path.join(TRASH_DIR, valid_images[idx]), img_path)
            i = idx

    elif key == ord('q'):
        break

cv2.destroyAllWindows()