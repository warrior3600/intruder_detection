from tqdm import tqdm
import cv2
import torch
import torchvision.transforms as T
import numpy as np
import pandas as pd
from roboflow import download_dataset
from dotenv import load_dotenv
import supervision as sv
load_dotenv()
from rfdetr import RFDETRBase
# Load model and class labels
import moviepy as mp

# VIDEO_FILE = "/teamspace/studios/this_studio/robbery1.mp4"
# clip = mp.VideoFileClip(VIDEO_FILE)
# clip_resized = clip.resized(height=560) # make the height 360px ( According to moviePy documenation The width is then computed so that the width/height ratio is conserved.)
# clip_resized.write_videofile("robbery1_resized.mp4")
# print("Resized to required resolution...beginning processing")


MODEL_FILE = "/teamspace/studios/this_studio/output/checkpoint_best_regular.pth"

model = RFDETRBase(pretrain_weights = MODEL_FILE)
print(model)
# Create a bounding box annotator object.
box_annotator = sv.BoxAnnotator()
classes = ['intruder','no_mask','unknown']
dataset = download_dataset("https://universe.roboflow.com/warrior360/robbery-dataset/dataset/1", "coco")

box = sv.Detections.from_transformers

# Create a video_info object for use in the VideoSink.
video_info = sv.VideoInfo.from_video_path(video_path="robbery1_resized.mp4")
# Create a frame generator and video info object from supervision utilities.
frame_generator = sv.get_video_frames_generator("robbery1_resized.mp4")

# Yield a single frame from the generator.
frame = next(frame_generator)
detection_log = []
frame_idx = 0
# Create a VideoSink context manager to save our frames.
with sv.VideoSink(target_path="output.mp4", video_info=video_info) as sink:

    # Iterate through frames yielded from the frame_generator.
    for frame in tqdm(frame_generator, total=video_info.total_frames):
        # text_scale = sv.calculate_optimal_text_scale(resolution_wh=frame.size)
        # thickness = sv.calculate_optimal_line_thickness(resolution_wh=frame.size)
        height, width = frame.shape[:2]
        resolution_wh = (width, height)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)



        label_annotator = sv.LabelAnnotator(
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_thickness=thickness,
        smart_position=True)
        # Run inference on our frame.
        detections = model.predict(frame, imgsz = 560)
        detections = detections.with_nms(threshold=0.5)
        # # Parse the result into the detections data model.
        # detections = sv.Detections.from_inference(result)
        # print("Detections: ", detections)
        # Apply bounding box to detections on a copy of the frame.
        try:
            detections_labels = [
                f"{classes[class_id-1]} {confidence:.2f}"
                for class_id, confidence
                in zip(detections.class_id, detections.confidence)
            ]
        except IndexError as e:
            print(detections.class_id)
            raise e
        annotated_frame = box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )

        detections_image = label_annotator.annotate(annotated_frame, detections, detections_labels)
        fps = video_info.fps
        timestamp = frame_idx/ fps
        for detection in detections_labels:
            if "intruder" in detection:
                print("Entered condition")
                detection_log.append({'timestamp': round(timestamp, 2), 'class': "intruder"})

        # Write the annotated frame to the video sink.
        frame_idx+=1
        sink.write_frame(frame=annotated_frame)

# df_log = pd.DataFrame(detection_log)
# df_log.to_csv('detections_with_timestamps.csv', index=False)

# print("Detection log saved to 'detections_with_timestamps.csv'")