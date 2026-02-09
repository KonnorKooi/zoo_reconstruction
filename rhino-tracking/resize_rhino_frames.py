#!/usr/bin/env python3
"""
Resize rhino video frames to match reconstruction image resolution.
This helps COLMAP match features better.
"""

import cv2
from pathlib import Path

# Get reconstruction image size (from cameras.txt)
TARGET_WIDTH = 1009
TARGET_HEIGHT = 669

input_dir = Path("colmap_registration/new_images")
output_dir = Path("colmap_registration/new_images_resized")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Resizing rhino frames to {TARGET_WIDTH}x{TARGET_HEIGHT}")
print()

for img_path in input_dir.glob("*.jpg"):
    # Load image
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    print(f"Processing {img_path.name}")
    print(f"  Original: {w}x{h}")

    # Resize
    resized = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

    # Save
    output_path = output_dir / img_path.name
    cv2.imwrite(str(output_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"  Resized:  {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"  Saved to: {output_path.name}")
    print()

print("✓ Done! Now copy these resized images to outside/images/")
print(f"  cp {output_dir}/*.jpg outside/images/")
