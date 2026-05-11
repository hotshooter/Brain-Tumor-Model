# 🧠 Brain Tumor Detection System

AI-powered brain tumor detection and classification using **YOLOv8x** and **EfficientNetV2**.

---

## 🚀 Live Demo

Try the app online - no installation required!

👉 **[Launch Brain Tumor Detector](https://huggingface.co/spaces/hotshooter/brain-tumor-detector)**

---

## 📋 Overview

Upload an MRI scan to detect and classify brain tumors into four categories:

| Type | Description |
|------|-------------|
| **Glioma** | Tumor from glial cells |
| **Meningioma** | Tumor from meninges |
| **Pituitary** | Tumor in pituitary gland |
| **No Tumor** | Normal brain scan |

---

## 🗂️ Project Structure
├── app.ipynb # Main application notebook
├── YOLO8x.ipynb # YOLOv8x training notebook
├── EfficientNetV2S.ipynb # EfficientNetV2 model notebook
├── MOH_weights.weights.h5 # Model weights
├──── models/ # Pre-trained model files
│     ├─── MOH_full.keras # Classifier model (248 MB)
│     └── yolov8x.pt # Detection model (137 MB)
├─── Dataset_MRI_Images/ # MRI scan dataset
│     └── brain-tumor/ # Brain tumor dataset
│     ├── train/ # Training images & labels
│     ├── test/ # Test images & labels
│     ├── val/ # Validation images & labels    
├─── yoloruns/ # YOLO training results
│   └── detect/
│   └── brain_tumor_yolov8x/ # YOLOv8x detection run
│      ├── weights/
│      │  ├── best.pt # Best model weights
│      └── args.yaml # Training arguments
└── Brain-Tumor-Model/ # Cloned repository files

