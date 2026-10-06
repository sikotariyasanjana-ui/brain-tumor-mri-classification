# 🧠 Brain Tumor MRI Image Classification

## 📌 Project Overview

This project focuses on classifying brain MRI images into multiple categories using Deep Learning and Transfer Learning.

The project uses:

- Custom Convolutional Neural Network (CNN)
- MobileNetV2 Transfer Learning
- EfficientNetB0 Transfer Learning
- Image preprocessing and normalization
- Model evaluation
- Streamlit deployment

The application allows a user to upload a brain MRI image and receive a predicted tumor category with confidence.

> **Disclaimer:** This project is developed for educational and demonstration purposes. It is not intended to provide medical diagnosis or replace professional medical advice.

---

## 🎯 Project Objective

The main objective of this project is to develop a deep learning-based system that can classify brain MRI images into the following four categories:

1. Glioma
2. Meningioma
3. No Tumor
4. Pituitary

The project also compares different deep learning approaches and prepares the selected trained model for deployment using Streamlit.

---

## 📂 Dataset

The project uses a Brain Tumor MRI Multi-Class Dataset.

### Classes

| Class | Description |
|---|---|
| Glioma | MRI images associated with glioma |
| Meningioma | MRI images associated with meningioma |
| No Tumor | MRI images without a tumor |
| Pituitary | MRI images associated with pituitary tumor |

### Dataset Structure

```text
BRAIN TUMOR/
│
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
│
├── valid/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
│
└── test/
    ├── glioma/
    ├── meningioma/
    ├── no_tumor/
    └── pituitary/