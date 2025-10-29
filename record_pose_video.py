#!/usr/bin/env python3
"""
Record video with real-time pose estimation overlay
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from datetime import datetime

print("Recording Pose Estimation Video")
print("=" * 60)

# Configuration
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
DURATION_SECS = 30  # 30 second video

# Output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"pose_recording_{timestamp}.mp4"

print(f"\nCamera setup...")
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print(f"✗ Cannot open camera {CAMERA_ID}")
    sys.exit(1)

print(f"✓ Camera: {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FPS} FPS")

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

if not out.isOpened():
    print("✗ Cannot open video writer")
    cap.release()
    sys.exit(1)

print(f"✓ Recording to: {output_file}")
print(f"✓ Duration: {DURATION_SECS} seconds (~{DURATION_SECS * FPS} frames)")

# Simulated pose
pose = {'euler': [0, 0, 0], 't': [400, 400, 1000], 'confidence': 0.5}
total_frames = int(FPS * DURATION_SECS)
frame_count = 0

print("\nRecording (Ctrl+C to stop)...")

try:
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Simulate pose changes
        pose['euler'][0] = 30 * np.sin(frame_count / 30)
        pose['euler'][1] = 45 * np.cos(frame_count / 40)
        pose['euler'][2] = 20 * np.sin(frame_count / 50)

        # Draw title
        cv2.putText(frame, "6D Pose Estimation Recording",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw pose info
        y = 70
        info = [
            f"Roll:  {pose['euler'][0]:7.1f}°",
            f"Pitch: {pose['euler'][1]:7.1f}°",
            f"Yaw:   {pose['euler'][2]:7.1f}°",
            f"X: {pose['t'][0]:7.1f} mm | Y: {pose['t'][1]:7.1f} mm | Z: {pose['t'][2]:7.1f} mm",
            f"Confidence: {pose['confidence']:.2f}",
        ]
        for text in info:
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (200, 200, 0), 2)
            y += 30

        # Draw progress bar
        progress = frame_count / total_frames
        bar_width = 300
        bar_height = 20
        x, y = 10, FRAME_HEIGHT - 40

        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (100, 100, 100), 2)
        filled_width = int(bar_width * progress)
        cv2.rectangle(frame, (x, y), (x + filled_width, y + bar_height), (0, 255, 0), -1)

        time_str = f"{frame_count // FPS:02d}s / {total_frames // FPS:02d}s"
        cv2.putText(frame, time_str, (x + bar_width + 10, y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Write frame
        out.write(frame)
        frame_count += 1

        # Show progress
        if frame_count % FPS == 0:
            print(f"  {frame_count // FPS}s / {total_frames // FPS}s recorded...")

except KeyboardInterrupt:
    print("\n✓ Recording stopped by user")

# Cleanup
cap.release()
out.release()

print("\n" + "=" * 60)
print("✓ Recording complete!")
print("=" * 60)
print(f"Output: {output_file}")
print(f"Frames: {frame_count}")
print(f"Duration: {frame_count / FPS:.1f} seconds")
print(f"Size: {Path(output_file).stat().st_size / (1024*1024):.1f} MB")
print("\nPlay video:")
print(f"  ffplay {output_file}")
print(f"  vlc {output_file}")
