#!/bin/bash

# Loop through all .mp4 files in the current directory
for input_file in /scr/BEHAVIOR-1K/asset_pipeline/artifacts/pipeline/object_images/*.mp4; do
  # Define the output filename. This example adds "_h264" before the extension
  # and places it in the 'reencoded_h264' subdirectory.
  output_file="${input_file%.mp4}_h264.mp4"

  echo "Processing: $input_file"

  # Run ffmpeg to re-encode the video to H.264
  # -i: specifies the input file
  # -c:v libx264: sets the video codec to H.264 (using the libx264 encoder)
  # -crf 23: sets the Constant Rate Factor for quality (lower is better, 18-28 is a sane range)
  # -preset medium: sets the encoding speed/compression trade-off (slower presets offer better compression)
  # -c:a aac: sets the audio codec to AAC (a common choice for MP4)
  # -b:a 192k: sets the audio bitrate to 192 kbps (adjust as needed)
  # -movflags +faststart: optimizes the MP4 for web streaming (allows playback before full download)
  ffmpeg -i "$input_file" \
         -c:v libx264 \
         -crf 23 \
         -preset medium \
         -movflags +faststart \
         -y "$output_file"

done

echo "All .mp4 files have been processed."
