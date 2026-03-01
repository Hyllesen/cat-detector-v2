import os
import random
import shutil

BASE_DIR = "/Volumes/external-nvme256gb/cat-detector-v2/clips/dataset"
IMG_DIR = os.path.join(BASE_DIR, "positives")
LBL_DIR = os.path.join(BASE_DIR, "labels")

# Target folders
DATASET_ROOT = "/Volumes/external-nvme256gb/cat-detector-v2/yolo_ready"
for split in ['train', 'val']:
    os.makedirs(os.path.join(DATASET_ROOT, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(DATASET_ROOT, split, 'labels'), exist_ok=True)

# Get all current valid images
images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(images)

# Split 80% train, 20% val
split_idx = int(len(images) * 0.8)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

def move_files(file_list, target_split):
    for img_name in file_list:
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        # Copy Image
        shutil.copy(os.path.join(IMG_DIR, img_name), 
                    os.path.join(DATASET_ROOT, target_split, 'images', img_name))
        # Copy Label
        if os.path.exists(os.path.join(LBL_DIR, lbl_name)):
            shutil.copy(os.path.join(LBL_DIR, lbl_name), 
                        os.path.join(DATASET_ROOT, target_split, 'labels', lbl_name))

move_files(train_imgs, 'train')
move_files(val_imgs, 'val')

print(f"Split complete: {len(train_imgs)} train, {len(val_imgs)} val. Files are in: {DATASET_ROOT}")