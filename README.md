# 🎭 Real-Time Emotion Recognition Using Computer Vision

This project detects and classifies human emotions in real time using computer vision and deep learning. By analyzing facial expressions captured from a webcam or video feed, the system identifies emotions such as **happy**, **sad**, **angry**, **surprised**, **neutral**, and more.  

Built with **Python**, **OpenCV**, and **YOLOv8**, this project demonstrates how modern deep learning models can be applied to emotion recognition tasks efficiently and accurately.

---

## 🚀 Features

- 🎥 **Real-time detection** from webcam or video feed  
- 🤖 **YOLOv8 model** for robust face detection and emotion classification  
- 🧠 **Multi-class emotion recognition** (e.g., happy, sad, angry, surprised, neutral)  
- 📊 **Live visualization** of detection results with bounding boxes and emotion labels  

---

## 🧩 Tech Stack

| Component | Description |
|------------|-------------|
| **Python** | Core programming language |
| **YOLOv8** | Object detection backbone for identifying faces |
| **PyTorch** | Deep learning framework for model inference |
| **OpenCV** | Real-time video capture and image processing |
| **NumPy / Pandas** | Data handling and processing |
| **Matplotlib** | Optional graphing of emotion trends |

---

## Install Dependancies

- pip install ultralytics
- pip install opencv-python


## How to Train Model

For detection:
yolo task=detect mode=train model=yolov8n-cls.pt data=dataset epochs=50 imgsz=64 batch=32 lr0=0.001 optimizer=Adam

For classification:
yolo task=classify mode=train model=yolov8n-cls.pt data=FER2013 epochs=50 imgsz=64 batch=32 lr0=0.001 optimizer=Adam


