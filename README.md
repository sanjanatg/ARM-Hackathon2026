# Real-Time Road Anomaly Detection

### YOLOv8 | Raspberry Pi 4 | Edge AI Deployment

---

## Overview

This project implements a 9-class road anomaly detection system using YOLOv8n.  
The trained model is optimized using INT8 quantization and deployed on Raspberry Pi 4 for CPU-only real-time inference.

The system processes dashcam footage and detects urban road anomalies with near real-time performance, without requiring external accelerators.

---

## Detected Classes

The model identifies the following anomaly categories:

- Fallen Trees  
- Damaged Electrical Poles  
- Damaged Road Signs  
- Illegal Parking  
- Graffiti  
- Potholes and Road Cracks  
- Garbage  
- Damaged Concrete Structures  
- Dead Animals / Pollution  

---

## Project Structure

```bash
.
├── gpu-p100-road-anamoly.ipynb
├── 9-class-test.ipynb
├── best.pt
└── README.md
```

## Dataset

- **Urban Issues Dataset** – 47,140 images (9 classes)  
- **PothRGBD Dataset** – 400 pothole images  
- **Dashcam Dataset** – ~2,000 images  

Final balanced training set: **3,521 images**

---

## Training Configuration

| Parameter   | Value        |
|------------|-------------|
| Model      | YOLOv8n     |
| Epochs     | 50          |
| Image Size | 640 × 640   |
| Batch Size | 16          |
| Optimizer  | SGD         |
| Hardware   | NVIDIA P100 |

---

## Performance (mAP@0.5)

| Class                        | mAP  |
|------------------------------|------|
| FallenTrees                  | 0.961 |
| DamagedElectricalPoles       | 0.967 |
| DamagedRoadSigns             | 0.938 |
| IllegalParking               | 0.376 |
| Graffiti                     | 0.943 |
| Potholes & RoadCracks        | 0.992 |
| Garbage                      | 0.980 |
| Damaged Concrete Structures  | 0.995 |
| DeadAnimalsPollution         | 0.969 |

---

## Raspberry Pi Deployment

### Steps

1. Export `best.pt` to TensorFlow Lite  
2. Apply INT8 quantization  
3. Capture camera feed using OpenCV  
4. Run inference using TFLite runtime  

### Deployment Result

- ≥5 FPS  
- CPU-only execution  

---

## Team

Sanjana T.G.  
K. Lakshmi Navyatha  
Prema Malipatil  

Cambridge Institute of Technology  
Bharat AI-SoC Student Challenge – 2026
