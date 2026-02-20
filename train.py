
from ultralytics import YOLO

# Load pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train
model.train(
    data="data.yaml",
    epochs=60,
    imgsz=640
)
