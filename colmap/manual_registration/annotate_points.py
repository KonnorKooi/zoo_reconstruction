#!/usr/bin/env python3
"""
Click on landmarks in stationary images to record 2D pixel coordinates.
After clicking all points, close the window. Then add the matching 3D
coordinates from COLMAP GUI into the output JSON file.

Usage:
    python annotate_points.py --stationary_dir ../stationary --output correspondences.json
"""

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path


def annotate_image(img_path):
    """Show image, let user click points. Returns list of [x, y] pixel coords."""
    img = mpimg.imread(str(img_path))
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(img)
    ax.set_title(f"{img_path.name}\nLeft-click to add points. Right-click to undo. Close window when done.")

    points = []
    markers = []
    labels = []

    def onclick(event):
        if event.inaxes != ax:
            return
        if event.button == 1:  # Left click - add point
            x, y = event.xdata, event.ydata
            points.append([round(x, 1), round(y, 1)])
            marker, = ax.plot(x, y, 'r+', markersize=15, markeredgewidth=2)
            label = ax.annotate(str(len(points)), (x, y),
                              textcoords="offset points", xytext=(8, 8),
                              fontsize=12, color='yellow', fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
            markers.append(marker)
            labels.append(label)
            print(f"  Point {len(points)}: [{x:.1f}, {y:.1f}]")
            fig.canvas.draw()
        elif event.button == 3:  # Right click - undo
            if points:
                points.pop()
                markers.pop().remove()
                labels.pop().remove()
                print(f"  Removed last point. {len(points)} points remaining.")
                fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()

    return points


def main():
    parser = argparse.ArgumentParser(description='Annotate landmarks in stationary images')
    parser.add_argument('--stationary_dir', required=True, help='Directory with stationary images')
    parser.add_argument('--output', default='correspondences.json', help='Output JSON file')
    args = parser.parse_args()

    stat_dir = Path(args.stationary_dir)
    stat_images = sorted([p for p in stat_dir.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])

    if not stat_images:
        print(f"No images found in {stat_dir}")
        return

    # Load existing file if it exists (to allow incremental editing)
    output_path = Path(args.output)
    data = {}
    if output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
        print(f"Loaded existing {args.output} with {len(data)} images")

    print(f"\nFound {len(stat_images)} stationary images")
    print("Instructions:")
    print("  1. Left-click on recognizable landmarks (log, rocks, corners, etc.)")
    print("  2. Right-click to undo the last point")
    print("  3. Close the window when done with each image")
    print("  4. Aim for 6-10 points per image")
    print()

    for img_path in stat_images:
        name = img_path.name
        print(f"\n--- {name} ---")

        if name in data and data[name].get("points_2d"):
            resp = input(f"  Already has {len(data[name]['points_2d'])} points. Re-annotate? [y/N] ")
            if resp.lower() != 'y':
                continue

        points_2d = annotate_image(img_path)

        if not points_2d:
            print("  No points clicked, skipping.")
            continue

        # Initialize with 2D points and placeholder 3D points
        data[name] = {
            "points_2d": points_2d,
            "points_3d": [["TODO_X", "TODO_Y", "TODO_Z"]] * len(points_2d)
        }
        print(f"  Saved {len(points_2d)} points for {name}")

    # Write output
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved to {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Open the COLMAP reconstruction in the GUI")
    print(f"  2. For each numbered point, find the same landmark in the 3D model")
    print(f"  3. Click the point in COLMAP to get its X, Y, Z coordinates")
    print(f"  4. Edit {args.output} and replace the TODO entries with real coordinates")
    print(f"  5. Run the pipeline with --manual_correspondences {args.output}")


if __name__ == '__main__':
    main()
