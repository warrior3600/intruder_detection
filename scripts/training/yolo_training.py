from ultralytics import YOLO
from roboflow import download_dataset
from dotenv import load_dotenv

load_dotenv()

# dataset = download_dataset("https://universe.roboflow.com/beneadie/intruder-face-covering/dataset/4", "coco")
dataset = download_dataset("https://universe.roboflow.com/warrior360/robbery-dataset/dataset/1", "yolov8")
# Load a pretrained YOLO11n model
learning_rates = [1e-5, 3e-5, 5e-5, 1e-4, 3e-4]
for lr in learning_rates:
    model = YOLO("yolo11l.pt")

    # Train the model on the COCO8 dataset for 100 epochs
    train_results = model.train(
        data=f"{dataset.location}/data.yaml",  # Path to dataset configuration file
        epochs=50,  # Number of training epochs
        imgsz=640,  # Image size for training
        batch=6,
        device=0,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
        lr0 = lr
    )

    # Evaluate the model's performance on the validation set
    metrics = model.val()

# # Perform object detection on an image
# results = model("path/to/image.jpg")  # Predict on an image
# results[0].show()  # Display results