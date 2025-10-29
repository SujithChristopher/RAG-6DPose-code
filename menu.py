#!/usr/bin/env python3
"""
Interactive menu for RAG-6DPose pipeline
"""
import subprocess
import sys
from pathlib import Path

def print_menu():
    print("\n" + "="*70)
    print("RAG-6DPose - 6D Object Pose Estimation")
    print("="*70)
    print("\n📚 Available Commands:\n")

    commands = [
        ("1", "Train Model", "python simple_train.py", "Train the model on your dataset"),
        ("2", "Reduce Model Size", "python quantize_model.py", "Quantize model (50% smaller)"),
        ("3", "Batch Inference", "python infer_poses.py", "Estimate poses on test set"),
        ("4", "Visualize Results", "python simple_pose_viz.py --poses inference_output/estimated_poses.json", "Plot pose estimations"),
        ("5", "Live Camera (Simple)", "python realtime_camera_simple.py", "Live camera feed (easy)"),
        ("6", "Live Camera (Full)", "python realtime_pose_estimation.py", "Real-time inference"),
        ("7", "Record Video", "python record_pose_video.py", "Record 30s video with poses"),
        ("8", "Quick Test", "python test_single_batch.py", "Test single batch"),
        ("9", "Show Scripts", "ls -1 *.py | grep -E '^(simple_|infer_|quantize|realtime|record)'", "List all scripts"),
        ("0", "Exit", "", ""),
    ]

    for num, name, cmd, desc in commands:
        if num == "0":
            print(f"  {num} - {name}")
        else:
            print(f"  {num} - {name:25} | {desc}")

def run_command(cmd):
    """Run a shell command"""
    if not cmd:
        return

    print(f"\n▶ Running: {cmd}")
    print("-" * 70)

    try:
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"\n⚠ Command exited with code {result.returncode}")
    except KeyboardInterrupt:
        print("\n✓ Stopped by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")

def main():
    print("\n🎯 Welcome to RAG-6DPose!")

    commands_map = {
        "1": ("Train Model", "python simple_train.py"),
        "2": ("Reduce Model Size", "python quantize_model.py"),
        "3": ("Batch Inference", "python infer_poses.py"),
        "4": ("Visualize Results", "python simple_pose_viz.py --poses inference_output/estimated_poses.json"),
        "5": ("Live Camera (Simple)", "python realtime_camera_simple.py"),
        "6": ("Live Camera (Full)", "python realtime_pose_estimation.py"),
        "7": ("Record Video", "python record_pose_video.py"),
        "8": ("Quick Test", "python test_single_batch.py"),
        "9": ("Show Scripts", "ls -1 *.py | grep -E '^(simple_|infer_|quantize|realtime|record|test)'"),
    }

    while True:
        print_menu()
        print("\n" + "-"*70)
        choice = input("Choose option (0-9): ").strip()

        if choice == "0":
            print("\n👋 Goodbye!")
            sys.exit(0)
        elif choice in commands_map:
            name, cmd = commands_map[choice]
            print(f"\n📌 {name}")
            run_command(cmd)
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting...")
        sys.exit(0)
