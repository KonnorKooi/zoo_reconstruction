#!/usr/bin/env python3
"""
Extract frames from rhino videos for COLMAP registration.

This extracts keyframes at regular intervals to use for camera pose estimation.
"""

import cv2
import argparse
from pathlib import Path


def extract_frames(video_path, output_dir, interval=30, max_frames=50):
    """Extract frames from video at regular intervals.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        interval: Extract every Nth frame (default: 30 = 1 per second at 30fps)
        max_frames: Maximum number of frames to extract
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(video_path)
    video_name = video_path.stem

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_name}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Extracting every {interval} frames (max {max_frames})")
    print()

    frame_count = 0
    extracted = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract every Nth frame
        if frame_count % interval == 0:
            output_path = output_dir / f"{video_name}_frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted += 1
            print(f"  Extracted frame {frame_count} -> {output_path.name}")

            if extracted >= max_frames:
                break

        frame_count += 1

    cap.release()
    print(f"\n✓ Extracted {extracted} frames to {output_dir}")
    print()
    return extracted


def main():
    parser = argparse.ArgumentParser(description='Extract frames from rhino videos')
    parser.add_argument('--interval', type=int, default=30,
                        help='Extract every Nth frame (default: 30)')
    parser.add_argument('--max-frames', type=int, default=50,
                        help='Maximum frames to extract per video (default: 50)')
    parser.add_argument('--output', type=str, default='colmap_registration/new_images',
                        help='Output directory for frames')
    args = parser.parse_args()

    base_dir = Path(".")

    # Videos to process
    videos = [
        base_dir / "rhino" / "rhino_pool_1_trimmed.mp4",
        base_dir / "rhino" / "rhino_yard_1.mp4",
    ]

    output_dir = Path(args.output)

    print("=" * 70)
    print("EXTRACTING FRAMES FOR COLMAP REGISTRATION")
    print("=" * 70)
    print()

    total_extracted = 0
    for video in videos:
        if not video.exists():
            print(f"⚠ Video not found: {video}")
            continue

        extracted = extract_frames(
            video,
            output_dir,
            interval=args.interval,
            max_frames=args.max_frames
        )
        total_extracted += extracted

    print("=" * 70)
    print(f"DONE: Extracted {total_extracted} frames total")
    print(f"Output directory: {output_dir.absolute()}")
    print()
    print("Next steps:")
    print("  1. Open COLMAP GUI")
    print("  2. Load existing project")
    print("  3. Add these new images")
    print("  4. Run image registration")
    print("=" * 70)


if __name__ == "__main__":
    main()
