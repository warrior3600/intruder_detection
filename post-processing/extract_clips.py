import cv2
import pandas as pd
import os

# Load the detections CSV
df = pd.read_csv("detections_with_timestamps.csv")

# Sort and get unique timestamps
timestamps = sorted(df["timestamp"].unique())

# Step 1: Create initial clips [start, end] with 4s before and 6s after
clips = []
for ts in timestamps:
    start = max(0, ts - 4)
    end = ts + 6
    clips.append([start, end])

# Step 2: Merge overlapping or close clips (within 5s)
merged_clips = []
if clips:
    current_start, current_end = clips[0]
    for start, end in clips[1:]:
        if start - current_end <= 5:
            current_end = max(current_end, end)
        else:
            merged_clips.append([current_start, current_end])
            current_start, current_end = start, end
    merged_clips.append([current_start, current_end])

# Step 3: Load video and extract clips
video_path = "/teamspace/studios/this_studio/robbery1.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

# Create output directory
output_dir = "output_clips"
os.makedirs(output_dir, exist_ok=True)

print("Creating clips...")

for idx, (start_sec, end_sec) in enumerate(merged_clips):
    # Convert seconds to frame indices
    start_frame = int(start_sec * fps)
    end_frame = int(min(end_sec * fps, frame_count))

    # Set video position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Define the codec and output path
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(output_dir, f"clip_{idx+1:03d}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()
    print(f"Saved: {out_path}")

cap.release()
print("All clips created.")