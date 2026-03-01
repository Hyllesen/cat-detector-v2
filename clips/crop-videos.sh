#!/bin/zsh

# 1. Enable recursive globbing
setopt globstar
setopt no_nomatch

# 2. Create the base output directory
mkdir -p cropped_videos

echo "Starting recursive crop..."

# 3. Loop through videos in all subdirectories (positives/ and negatives/)
for vid in **/*.{mp4,mkv,avi,mov}; do
  # Skip if it's already a cropped file to avoid infinite loops
  [[ "$vid" == cropped_videos/* ]] && continue
  
  if [ -f "$vid" ]; then
    # Determine the folder structure (e.g., "positives" or "negatives")
    subdir=$(dirname "$vid")
    mkdir -p "cropped_videos/$subdir"
    
    base=$(basename "${vid%.*}")
    echo "Processing: $vid -> cropped_videos/$subdir/cropped_${base}.mp4"
    
    # Run the crop
    ffmpeg -i "$vid" -vf "crop=in_w-200:in_h-220:0:220" -c:v libx264 -crf 18 -c:a copy "cropped_videos/$subdir/cropped_${base}.mp4"
  fi
done

echo "Done! Check the 'cropped_videos' folder."
