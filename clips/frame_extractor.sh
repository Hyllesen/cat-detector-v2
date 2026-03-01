#!/bin/zsh
unsetopt NOMATCH

# Create the final image folders
mkdir -p dataset/positives
mkdir -p dataset/negatives

echo "Extracting frames at 1 FPS..."

# Process the Resident/Positive clips
for vid in cropped_videos/positives/*.mp4; do
  [ -f "$vid" ] || continue
  base=$(basename "${vid%.*}")
  ffmpeg -i "$vid" -vf "fps=1" "dataset/positives/pos_${base}_%04d.jpg"
done

# Process the Enemy/Negative clips
for vid in cropped_videos/negatives/*.mp4; do
  [ -f "$vid" ] || continue
  base=$(basename "${vid%.*}")
  ffmpeg -i "$vid" -vf "fps=1" "dataset/negatives/neg_${base}_%04d.jpg"
done

echo "Extraction complete. Check the 'dataset' folder!"
