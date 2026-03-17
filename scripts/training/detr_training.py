from roboflow import download_dataset
from dotenv import load_dotenv

load_dotenv()

# dataset = download_dataset("https://universe.roboflow.com/beneadie/intruder-face-covering/dataset/4", "coco")
dataset = download_dataset("https://universe.roboflow.com/warrior360/robbery-dataset/dataset/1", "coco")

from rfdetr import RFDETRBase


learning_rates = [1e-5, 3e-5, 5e-5, 1e-4, 3e-4]

for i in range(len(learning_rates)):
    model = RFDETRBase()
    model.train(dataset_dir=dataset.location, epochs=50, batch_size=4, grad_accum_steps=4, use_ema=False, lr=learning_rates[i], output_dir=f"output_{i}")

