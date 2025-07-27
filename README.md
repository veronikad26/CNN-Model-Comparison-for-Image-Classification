# 🧠 CNN Model Comparison for Image Classification

## 📌 Project Overview
This project focuses on training, evaluating, and comparing multiple Convolutional Neural Network (CNN) architectures for medical image classification. The primary objective is to identify the most efficient model for classifying **bone marrow cell images** accurately and efficiently.

## 🧪 Objective
To determine the best CNN model for bone marrow image classification based on:
- Accuracy & F1-Score
- Training Time
- Model Complexity (parameters & size)

## 🗂️ Dataset
- **Bone Marrow Cell Images Dataset**: A labeled medical dataset containing microscopic images of bone marrow cells across multiple classes.
- Dataset includes variations in staining, magnification, and cell morphology, making it ideal for deep learning-based evaluation.
- Format: RGB images, resized to 128×128 during preprocessing.
- Link: https://www.kaggle.com/datasets/andrewmvd/bone-marrow-cell-classification/data

## 🛠️ Models Compared

| Model           | Architecture Summary                          | Notable Features                        |
|------------------|-----------------------------------------------|------------------------------------------|
| **VGG16**        | Deep CNN with stacked 3×3 conv layers         | High accuracy, large number of parameters |
| **ResNet50**     | Residual connections for deep network training| Better convergence, lower overfitting     |
| **MobileNetV2**  | Lightweight with depthwise separable convolutions | Fast and efficient, mobile-ready       |

## ⚙️ Technical Details

### 🔹 Preprocessing
- Image resizing to 128×128
- Normalization to [0, 1]
- One-hot encoding of target classes
- Data augmentation: horizontal/vertical flip, random zoom, rotation

### 🔹 Training Pipeline
- Built using **TensorFlow/Keras**
- Batch size: 32 | Epochs: 50 (with early stopping)
- Optimizer: Adam | Loss Function: Categorical Crossentropy

### 🔹 Evaluation Metrics
- Accuracy, F1-Score, Precision, Recall
- Confusion matrix and classification report
- Recorded training time and model size for comparative study

## 🏁 Results Summary

| Model         | Accuracy | F1-Score | Training Time | Size (MB) |
|---------------|----------|----------|----------------|------------|
| VGG16         | 91.2%    | 90.9%    | 28 mins        | ~528 MB    |
| ResNet50      | **92.3%**| **92.0%**| 24 mins        | ~98 MB     |
| MobileNetV2   | 89.7%    | 89.2%    | **12 mins**    | ~14 MB     |

