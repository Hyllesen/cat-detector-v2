import os
import shutil
import cv2

# Paths
SRC_DIR = "dataset/positives"
NEG_DIR = "dataset/negatives"
TRASH_DIR = "dataset/.trash"

os.makedirs(NEG_DIR, exist_ok=True)
os.makedirs(TRASH_DIR, exist_ok=True)

# State Management
images = sorted([f for f in os.listdir(SRC_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
history = [] # Stores (filename, action_type, original_path)
index = 0

print("--- CAT DATA SORTER ---")
print(" [1] Keep (Next)")
print(" [2] Move to Negatives")
print(" [3] Delete (Move to Trash)")
print(" [Z] Undo Last Action")
print(" [ESC] Save & Quit")

while index < len(images):
    img_name = images[index]
    img_path = os.path.join(SRC_DIR, img_name)
    
    # Load and check image
    img = cv2.imread(img_path)
    if img is None:
        index += 1
        continue
    
    # Resize for Mac Mini Display
    h, w = img.shape[:2]
    display_img = cv2.resize(img, (1200, int(1200 * h / w)))
    
    cv2.imshow("Sorter", display_img)
    cv2.setWindowTitle("Sorter", f"[{index+1}/{len(images)}] - {img_name}")
    
    key = cv2.waitKey(0) & 0xFF

    if key == ord('1'): # KEEP
        history.append((img_name, 'keep', img_path))
        index += 1
        
    elif key == ord('2'): # MOVE TO NEGATIVE
        dest = os.path.join(NEG_DIR, img_name)
        shutil.move(img_path, dest)
        history.append((img_name, 'neg', dest))
        images.pop(index)
        print(f"Moved to Negatives: {img_name}")

    elif key == ord('3'): # DELETE (TO TRASH)
        dest = os.path.join(TRASH_DIR, img_name)
        shutil.move(img_path, dest)
        history.append((img_name, 'trash', dest))
        images.pop(index)
        print(f"Trashed: {img_name}")

    elif key == ord('z'): # UNDO
        if not history:
            print("Nothing to undo!")
            continue
            
        last_name, last_action, current_loc = history.pop()
        
        if last_action == 'keep':
            index -= 1
        else:
            # Move it back to the positives folder
            shutil.move(current_loc, os.path.join(SRC_DIR, last_name))
            # Put it back in our list at the current position
            images.insert(index, last_name)
        
        print(f"Undid: {last_action} for {last_name}")

    elif key == 27: # ESC
        break

cv2.destroyAllWindows()
print("Sorting session ended.")