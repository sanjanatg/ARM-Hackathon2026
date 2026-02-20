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
