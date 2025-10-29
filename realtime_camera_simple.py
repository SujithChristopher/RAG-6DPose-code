#!/usr/bin/env python3
"""
Simple real-time camera pose estimation
Press 'q' to quit, 's' to save frame, 'p' to print pose
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import torch

print("Real-time Pose Estimation - Simple Version")
print("=" * 60)

# Configuration
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

print("\nInitializing camera...")
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print(f"✗ Cannot open camera {CAMERA_ID}")
    print("Tips:")
    print("  • Try CAMERA_ID = 1, 2, etc. for other cameras")
    print("  • Check if camera is being used by another app")
    sys.exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"✓ Camera: {frame_width}x{frame_height}")

# Simulated pose (in real scenario, this comes from model)
pose = {
    'euler': [0, 0, 0],
    't': [400, 400, 1000],
    'confidence': 0.5
}

frame_count = 0

print("\nControls:")
print("  q - Quit")
print("  s - Save frame")
print("  r - Reset view")
print("\n" + "=" * 60)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Simulate pose changes (in real version, from model)
        pose['euler'][0] = 30 * np.sin(frame_count / 30)
        pose['euler'][1] = 45 * np.cos(frame_count / 40)
        pose['euler'][2] = 20 * np.sin(frame_count / 50)

        # Draw info panel
        cv2.putText(frame, "6D Pose Estimation", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw pose info
        y = 70
        info = [
            f"Roll:  {pose['euler'][0]:7.1f}°",
            f"Pitch: {pose['euler'][1]:7.1f}°",
            f"Yaw:   {pose['euler'][2]:7.1f}°",
            f"X: {pose['t'][0]:7.1f} mm",
            f"Y: {pose['t'][1]:7.1f} mm",
            f"Z: {pose['t'][2]:7.1f} mm",
            f"Confidence: {pose['confidence']:.2f}",
        ]
        for text in info:
            cv2.putText(frame, text, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
            y += 30

        # Draw origin indicator
        cx, cy = frame_width // 2, frame_height // 2
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
        cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        # Draw axes
        scale = 50
        cv2.line(frame, (cx, cy), (cx + scale, cy), (0, 0, 255), 3)  # X - Red
        cv2.line(frame, (cx, cy), (cx, cy + scale), (0, 255, 0), 3)  # Y - Green
        cv2.line(frame, (cx, cy), (cx - scale//2, cy - scale//2), (255, 0, 0), 3)  # Z - Blue

        # Frame counter
        cv2.putText(frame, f"Frame: {frame_count}",
                   (frame_width - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display
        cv2.imshow("Real-time Pose Estimation", frame)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"\n✓ Quit (processed {frame_count} frames)")
            break
        elif key == ord('s'):
            filename = f"pose_frame_{frame_count:06d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ Saved: {filename}")
        elif key == ord('r'):
            print("Reset: Pose will continue cycling")

except KeyboardInterrupt:
    print("\n✓ Interrupted")
except Exception as e:
    print(f"\n✗ Error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Camera closed")
