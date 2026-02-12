#!/usr/bin/env python3
"""
Click landmarks on stationary images and enter 3D coordinates in one step.

Usage:
    python annotate_points.py --stationary_dir ../stationary

Workflow per image:
  1. Left-click a landmark on the image
  2. Type X Y Z in the terminal (from COLMAP GUI)
  3. Repeat for 6-10 points
  4. Type 'done' when finished with the image
  5. Type 'undo' to remove the last point
"""

import argparse
import json
import threading
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path


CORRESPONDENCES_FILE = "correspondences.json"


def load_data():
    path = Path(CORRESPONDENCES_FILE)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_data(data):
    with open(CORRESPONDENCES_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def annotate_image(img_path, existing_2d=None, existing_3d=None):
    """
    Show image. User clicks a point, then types 3D coords in terminal.
    Returns (points_2d, points_3d) lists.
    """
    img = mpimg.imread(str(img_path))
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(img)
    ax.set_title(f"{img_path.name}\n"
                 f"Left-click a point, then enter X Y Z in terminal.\n"
                 f"Type 'done' to finish, 'undo' to remove last point.")

    points_2d = list(existing_2d) if existing_2d else []
    points_3d = list(existing_3d) if existing_3d else []
    markers = []
    labels = []
    pending_click = [None]  # mutable container for thread communication
    click_event = threading.Event()

    # Draw existing points
    for i, (pt2d, pt3d) in enumerate(zip(points_2d, points_3d)):
        marker, = ax.plot(pt2d[0], pt2d[1], '+', markersize=15, markeredgewidth=2, color='lime')
        label = ax.annotate(str(i + 1), (pt2d[0], pt2d[1]),
                           textcoords="offset points", xytext=(8, 8),
                           fontsize=12, color='yellow', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        markers.append(marker)
        labels.append(label)
    if points_2d:
        print(f"  Loaded {len(points_2d)} existing points")
    fig.canvas.draw()

    def onclick(event):
        if event.inaxes != ax:
            return
        if event.button == 1:
            pending_click[0] = (event.xdata, event.ydata)
            click_event.set()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.ion()
    plt.show()
    plt.pause(0.1)

    print(f"\n  Click a point on the image, then enter its X Y Z here.")
    print(f"  Commands: 'done' = finish, 'undo' = remove last, 'quit' = save & exit\n")

    while True:
        # Wait for a click or a command
        print(f"  [{len(points_2d)} points] Click a point or type a command: ", end='', flush=True)

        # Poll for either click or terminal input
        click_event.clear()

        import select
        import sys

        while True:
            plt.pause(0.1)
            if click_event.is_set():
                break
            # Check for terminal input (non-blocking)
            if select.select([sys.stdin], [], [], 0.0)[0]:
                break

        if click_event.is_set() and pending_click[0] is not None:
            x, y = pending_click[0]
            pending_click[0] = None
            print()  # newline after the prompt

            # Show the clicked point as red (pending)
            idx = len(points_2d) + 1
            marker, = ax.plot(x, y, '+', markersize=15, markeredgewidth=2, color='red')
            label = ax.annotate(str(idx), (x, y),
                               textcoords="offset points", xytext=(8, 8),
                               fontsize=12, color='yellow', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
            fig.canvas.draw()
            plt.pause(0.1)

            print(f"  Point {idx} clicked at pixel [{x:.1f}, {y:.1f}]")

            # Get 3D coordinates
            while True:
                raw = input(f"  Point {idx} X Y Z: ").strip()
                if raw.lower() == 'skip':
                    marker.remove()
                    label.remove()
                    fig.canvas.draw()
                    plt.pause(0.1)
                    print(f"  Skipped")
                    break
                try:
                    coords = [float(v.strip()) for v in raw.split(',')]
                    if len(coords) != 3:
                        print("  Need 3 values: X, Y, Z")
                        continue
                    points_2d.append([round(x, 1), round(y, 1)])
                    points_3d.append(coords)
                    # Turn marker green
                    marker.set_color('lime')
                    fig.canvas.draw()
                    plt.pause(0.1)
                    print(f"  Saved point {idx}: [{x:.1f}, {y:.1f}] -> [{coords[0]:.4f}, {coords[1]:.4f}, {coords[2]:.4f}]")
                    markers.append(marker)
                    labels.append(label)
                    break
                except ValueError:
                    print("  Invalid. Enter 3 numbers like: 1.23, -0.45, 2.67")
        else:
            # Terminal input
            raw = input().strip()
            if raw.lower() == 'done':
                break
            elif raw.lower() == 'quit':
                plt.close('all')
                return points_2d, points_3d, True  # signal quit
            elif raw.lower() == 'undo':
                if points_2d:
                    points_2d.pop()
                    points_3d.pop()
                    markers.pop().remove()
                    labels.pop().remove()
                    fig.canvas.draw()
                    plt.pause(0.1)
                    print(f"  Removed last point. {len(points_2d)} remaining.")
                else:
                    print(f"  No points to remove.")
            else:
                print(f"  Unknown command. Use 'done', 'undo', or 'quit'.")

    plt.close('all')
    return points_2d, points_3d, False


def main():
    parser = argparse.ArgumentParser(description='Annotate 2D-3D correspondences')
    parser.add_argument('--stationary_dir', required=True, help='Directory with stationary images')
    args = parser.parse_args()

    stat_dir = Path(args.stationary_dir)
    stat_images = sorted([p for p in stat_dir.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])

    if not stat_images:
        print(f"No images found in {stat_dir}")
        return

    data = load_data()
    if data:
        print(f"Loaded existing {CORRESPONDENCES_FILE} with {len(data)} images")

    print(f"Found {len(stat_images)} stationary images\n")

    for img_path in stat_images:
        name = img_path.name
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        existing_2d = data.get(name, {}).get("points_2d")
        existing_3d = data.get(name, {}).get("points_3d")

        # Filter out TODO entries from existing
        if existing_2d and existing_3d:
            valid = [(p2, p3) for p2, p3 in zip(existing_2d, existing_3d)
                     if not isinstance(p3[0], str)]
            if valid:
                existing_2d = [v[0] for v in valid]
                existing_3d = [v[1] for v in valid]
                resp = input(f"  Has {len(valid)} existing points. Add more? [Y/n/redo] ")
                if resp.lower() == 'n':
                    continue
                if resp.lower() == 'redo':
                    existing_2d = None
                    existing_3d = None
            else:
                existing_2d = None
                existing_3d = None

        pts_2d, pts_3d, quit_flag = annotate_image(
            img_path, existing_2d, existing_3d)

        if pts_2d:
            data[name] = {"points_2d": pts_2d, "points_3d": pts_3d}
            save_data(data)
            print(f"  Saved {len(pts_2d)} points for {name}")

        if quit_flag:
            break

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary — {CORRESPONDENCES_FILE}")
    print(f"{'='*60}")
    for stat_name, entry in data.items():
        n = len(entry["points_2d"])
        print(f"  {stat_name}: {n} correspondences")

    print(f"\nReady to run registration with this file.")


if __name__ == '__main__':
    main()
