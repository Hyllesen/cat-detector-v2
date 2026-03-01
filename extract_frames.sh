# Create output directories
mkdir -p dataset/images
mkdir -p dataset/labels

# Process Cat Videos (Positives)
for vid in clips/positives/*.{mp4,mkv}; do
  base=$(basename "${vid%.*}")
  ffmpeg -i "$vid" -vf "fps=1" "dataset/images/cat_${base}_%04d.jpg"
done

# Process Non-Cat Videos (Negatives)
for vid in clips/negatives/*.{mp4,mkv}; do
  base=$(basename "${vid%.*}")
  ffmpeg -i "$vid" -vf "fps=1" "dataset/images/neg_${base}_%04d.jpg"
done
