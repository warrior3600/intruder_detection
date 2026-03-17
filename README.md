# Intruder Detection Pipeline

This repository contains an end-to-end pipeline for intelligent intruder detection. The pipeline handles data preprocessing (video frame extraction), object detection model training (YOLO and DETR variants), video inference, and contextual analysis utilizing Large Language Models (LLMs) such as Gemini for generating structured reports (e.g., number of intruders, hostages, or observed weapons) from the footage.

The annotated video output highlighting class definitions and visual boxes over moving subjects successfully tracked by our tuned transformer object detection variants:

[Processed video](./assets/output.mp4)

## Project Structure

The project is structured into three main directories:

- `/helpers`: Contains utility scripts to process the raw video data and leverage LLM APIs.
  - `extract_clips.py`: Dynamically extracts smaller segmented video clips from continuous footage centered around detected timestamps from a CSV file.
  - `extract_frames.py`: Converts video files into sequential JPEG frames at specified intervals.
  - `get_response.py`: Generates structural JSON reports describing the observed scene using the Google Gemini generative AI model visually analyzing frames.
- `/scripts/inference`: Contains scripts to run object detection inferences on images or videos.
  - `yolo_inference.py`: Re-evaluates tracking loops using YOLOv11 weights.
  - `detr_inference_video.py` / `detr_inference_single_image.py`: Scripts leveraging Roboflow's DETR models for video/image bounding box generation indicating intruders/masks.
- `/scripts/training`: Scripts used to initiate fine-tuning cycles.
  - `yolo_training.py`: Model training entry-point focusing on YOLOv11 iterations over COCO8 configurations.
  - `detr_training.py`: Handles iterative RFDETRBase fine-tuning sweeps testing different optimizer learning-rates tracking loss.

## Setup Instructions

1.  **Clone the Repository** and make sure you're operating within the root directory `intruder_detection/`.
2.  **Environment Setup:** Create a `.env` file at the root to store required secrets (e.g., Roboflow API configuration if building new datasets dynamically).
    ```env
    # Example .env (Fill in your own tokens)
    ROBOFLOW_API_KEY="..."
    ```
3.  **Install Required Dependencies:** Run the following command from your active python virtual environment to install exactly what the scripts require:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure API Keys:** If you intend to utilize the LLM contextual analysis pipeline (`helpers/get_response.py`), you'll need to generate a Google Gemini API Key and insert it at the top `API_KEY = "KEY"` variable space (or update the script to use `os.environ` fetching logic).

## The Pipeline Workflow

### 1. Data Preprocessing
Raw surveillance output is processed through `helpers/extract_clips.py`. The system reads from `detections_with_timestamps.csv` allowing the python cv2 handler to isolate tight interval fragments capturing just the suspect behaviors without wasting cycle scanning empty security footage. If specific frame-wise data formats are needed, `helpers/extract_frames.py` down-samples the clip into physical `JPG` stacks.

### 2. Training New Models
Use scripts inside `/scripts/training/` to initiate tuning.
- YOLO variants: Run `python scripts/training/yolo_training.py` (ensure `yolo11l.pt` base weights exist locally).
- DETR versions: Run `python scripts/training/detr_training.py` which dynamically sweeps learning rates across RFDETRBase architectures. Check the generated `/output_[N]` directories for optimum checkpoints.

### 3. Model Inference Execution
The primary task executes trained model weights against visual feeds inside the `/scripts/inference/` catalog.

By executing `python scripts/inference/detr_inference_video.py`, the backend resamples the MP4 video, binds the optimum checkpoint path `checkpoint_best_regular.pth`, and dynamically calculates bounding dimensions annotating visual boxes using Supervision.

### 4. Semantic LLM Context Analysis
The structured output or target frames are pushed to the logic encoded within `helpers/get_response.py`. The Gemini flash model evaluates up to 5 sampled frames sequentially tracking distinct features:
- Intruders explicitly identified against hostages.
- Dress-code identification.
- Visible weapon tagging.

This step generates a clean telemetry payload inside `post_process_description.json`.

---

## Running Inference on the `data/` Folder

To run inference specifically over a directory of static images (such as the `/data/` folder), you can create and execute a small loop utilizing the core logic inside the `detr_inference_single_image.py` file.

Create a new file called `run_batch_inference.py` at the root directory:

```python
import os
import cv2
import supervision as sv
from PIL import Image
import numpy as np
from rfdetr import RFDETRBase

# Model Initialization
MODEL_FILE = "/teamspace/studios/this_studio/output/checkpoint_best_regular.pth"
model = RFDETRBase(pretrain_weights=MODEL_FILE)
classes = ['intruder', 'no_mask', 'unknown']

data_folder = "./data"
output_folder = "./output"
os.makedirs(output_folder, exist_ok=True)

# Iterate the Target Directory
for filename in os.listdir(data_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        file_path = os.path.join(data_folder, filename)
        image = Image.open(file_path).convert("RGB")
        process_image = image.copy()
        
        # Inference
        detections = model.predict(process_image)
        
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=process_image.size)
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=process_image.size)
        
        bbox_annotator = sv.BoxAnnotator(thickness=thickness)
        label_annotator = sv.LabelAnnotator(
            text_color=sv.Color.BLACK,
            text_scale=text_scale,
            text_thickness=thickness,
            smart_position=True
        )

        detections_labels = [
            f"{classes[class_id-1]} {confidence:.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]

        detections_image = image.copy()
        detections_image = bbox_annotator.annotate(detections_image, detections)
        detections_image = label_annotator.annotate(detections_image, detections, detections_labels)
        
        # Save Generated Result
        out_path = os.path.join(output_folder, f"det_{filename}")
        cv2.imwrite(out_path, cv2.cvtColor(np.array(detections_image), cv2.COLOR_BGR2RGB))
        print(f"Processed: {filename} -> Output Saved To: {out_path}")
```
To run it, open your terminal and simply trigger:
```bash
python run_batch_inference.py
```
