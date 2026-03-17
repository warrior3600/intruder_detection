import cv2
import torch
import torchvision.transforms as T
import numpy as np
from datetime import timedelta
from roboflow import download_dataset
from dotenv import load_dotenv
import supervision as sv
from PIL import Image
load_dotenv()
from rfdetr import RFDETRBase
# Load model and class labels
MODEL_FILE = "/teamspace/studios/this_studio/output/checkpoint_best_regular.pth"
model = RFDETRBase(pretrain_weights = MODEL_FILE)

dataset = download_dataset("https://universe.roboflow.com/warrior360/robbery-dataset/dataset/1", "coco")
import supervision as sv

# ds = sv.DetectionDataset.from_coco(
#     images_directory_path=f"{dataset.location}/test",
#     annotations_path=f"{dataset.location}/test/_annotations.coco.json",
# )
classes = ['intruder','no_mask','unknown']


# path, image, annotations = ds[10]
image = Image.open("/teamspace/studios/this_studio/3BB60A1300000578-0-image-a-20_1483088271705.jpg")

# # Define preprocessing

# process_image = cv2.resize(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB),(560,560))
process_image = image.copy()
detections = model.predict(process_image)

# height, width = process_image.shape[:2]
# resolution_wh = (width, height)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=process_image.size)
thickness = sv.calculate_optimal_line_thickness(resolution_wh=process_image.size)

bbox_annotator = sv.BoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(
    text_color=sv.Color.BLACK,
    text_scale=text_scale,
    text_thickness=thickness,
    smart_position=True)

print(detections.class_id)
print(detections.confidence)
detections_labels = [
    f"{classes[class_id-1]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]
# annotation_image = image.copy()
# annotation_image = bbox_annotator.annotate(annotation_image, annotations)
# annotation_image = label_annotator.annotate(annotation_image, annotations, annotations_labels)

detections_image = image.copy()
detections_image = bbox_annotator.annotate(detections_image, detections)
detections_image = label_annotator.annotate(detections_image, detections, detections_labels)
print(type(detections_image))
# sv.plot_images_grid(images=[annotation_image, detections_image], grid_size=(1, 2), titles=["Annotation", "Detection"])

cv2.imwrite("/teamspace/studios/this_studio/output/detection_BBox_raw.png",cv2.cvtColor(np.array(detections_image), cv2.COLOR_BGR2RGB))

# Replace with your actual class names
# class_names = ["intruder","no_mask"]



# # Inference and video writing
# def run_inference_on_video(video_path, output_path="/teamspace/studios/this_studio/robbery1.mp4", confidence_threshold=0.5):
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

#     out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
#     frame_idx = 0
#     all_detections = []

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         orig_frame = frame.copy()
#         input_tensor = transform(frame).unsqueeze(0).to(device)

#         with torch.no_grad():
#             outputs = model(input_tensor)

#         # Assuming DETR-like output
#         probs = outputs['pred_logits'].softmax(-1)[0]
#         boxes = outputs['pred_boxes'][0]  # in [cx, cy, w, h] normalized

#         for i, prob in enumerate(probs):
#             score, label = torch.max(prob[:-1], dim=0)  # skip the "no object" class
#             if score < confidence_threshold:
#                 continue

#             # Convert boxes to pixel coordinates
#             cx, cy, w, h = boxes[i]
#             cx, cy, w, h = cx.item(), cy.item(), w.item(), h.item()

#             x1 = int((cx - w/2) * frame_width)
#             y1 = int((cy - h/2) * frame_height)
#             x2 = int((cx + w/2) * frame_width)
#             y2 = int((cy + h/2) * frame_height)

#             # Draw box
#             cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             label_text = f"{class_names[label]}: {score:.2f}"
#             cv2.putText(orig_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#             # Append detection info
#             timestamp = str(timedelta(seconds=frame_idx / fps))
#             all_detections.append({
#                 "class": class_names[label],
#                 "timestamp": timestamp,
#                 "confidence": float(score),
#                 "bbox": [x1, y1, x2, y2]
#             })

#         out.write(orig_frame)
#         frame_idx += 1

#     cap.release()
#     out.release()
#     print(f"Video saved to {output_path}")
#     return all_detections


# # Example usage
# detections = run_inference_on_video("input_video.mp4")
# print(detections[:5])  # Preview a few detections
