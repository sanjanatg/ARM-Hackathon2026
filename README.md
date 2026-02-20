Real-Time Road Anomaly Detection using YOLOv8
Project Overview

This project implements a 9-class road anomaly detection system using YOLOv8. The model is trained on urban infrastructure datasets and optimized for edge deployment on Raspberry Pi 4 using TensorFlow Lite with INT8 quantization.

The system detects the following classes:

Fallen Trees

Damaged Electrical Poles

Damaged Road Signs

Illegal Parking

Graffiti

Potholes and Road Cracks

Garbage

Damaged Concrete Structures

Dead Animals / Pollution

The trained model achieves high mAP@0.5 performance and supports near real-time inference.

Repository Structure
.
├── 9-class-test.ipynb
├── gpu-p100-road-anamoly.ipynb
├── best.pt
└── README.md
9-class-test.ipynb

Model testing notebook.
Includes inference on validation and dashcam datasets, bounding box visualization, and evaluation metrics.

gpu-p100-road-anamoly.ipynb

Training notebook executed on Kaggle using NVIDIA P100 GPU.
Includes dataset preprocessing, YOLOv8n training, and performance evaluation.

best.pt

Best-performing YOLOv8 model weights generated after training.
This file is used for inference and deployment. It can be exported to ONNX or TensorFlow Lite format for edge deployment.

Dataset Information
Primary Dataset

Urban Issues Dataset
Approximately 47,140 images across 9 classes.

Pothole Enhancement Dataset

PothRGBD Dataset
400 pothole images used to improve pothole detection recall.

Dashcam Dataset

Approximately 2,000 images used for testing under simulated real-world driving conditions.

Final balanced training dataset size: 3,521 images across 9 classes.

Training Configuration

Model: YOLOv8n
Epochs: 50
Image Size: 640 × 640
Batch Size: 16
Optimizer: SGD
Training Hardware: NVIDIA P100 GPU (Kaggle)

Performance (mAP@0.5)

FallenTrees: 0.961
DamagedElectricalPoles: 0.967
DamagedRoadSigns: 0.938
IllegalParking: 0.376
Graffiti: 0.943
Potholes and RoadCracks: 0.992
Garbage: 0.980
Damaged Concrete Structures: 0.995
DeadAnimalsPollution: 0.969

Deployment on Raspberry Pi 4

Deployment steps:

Export best.pt to TensorFlow Lite format.

Apply INT8 quantization.

Capture live camera feed using OpenCV.

Run inference using TensorFlow Lite runtime.

Log detections with timestamps.

Deployment results:

≥5 FPS inference speed

CPU-only execution

No external accelerators required

Requirements

Install training dependencies:

pip install ultralytics
pip install opencv-python
pip install numpy

For edge deployment:

pip install tflite-runtime
Future Improvements

Improve performance of IllegalParking class

Integrate GPS tagging for anomaly location

Develop cloud-based reporting dashboard

Add severity classification for potholes

Team

Sanjana T.G. (Team Leader)
K. Lakshmi Navyatha
Prema Malipatil

Cambridge Institute of Technology
Bharat AI-SoC Student Challenge – 2026
