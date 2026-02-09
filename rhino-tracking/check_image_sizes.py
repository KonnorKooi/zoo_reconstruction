#!/usr/bin/env python3
"""Check actual image dimensions in the images folder."""

import cv2
from pathlib import Path

image_dir = Path("outside/images")

print("Checking image dimensions...")
print()

# Check a few reconstruction images
recon_images = list(image_dir.glob("*.jpg"))[:5]
print("Sample reconstruction images:")
for img_path in recon_images:
    if "rhino" in img_path.name:
        continue
    img = cv2.imread(str(img_path))
    if img is not None:
        h, w = img.shape[:2]
        print(f"  {img_path.name}: {w}x{h}")

print()

# Check rhino images
rhino_images = list(image_dir.glob("*rhino*.jpg"))
print("Rhino images:")
for img_path in rhino_images:
    img = cv2.imread(str(img_path))
    if img is not None:
        h, w = img.shape[:2]
        print(f"  {img_path.name}: {w}x{h}")
