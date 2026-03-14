# cnn-image-classification

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%2311557c.svg?style=for-the-badge&logo=Matplotlib&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%2311557c.svg?style=plastic&logo=Matplotlib&logoColor=white)

This project implements a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10** dataset using the PyTorch framework.

## 🚀 About the Project
The goal of this project is to build a deep learning model capable of recognizing 10 different categories of images with high accuracy.

### 🎯 Supported Categories
The model is trained to identify:
* ✈️ **Airplane** | 🚗 **Automobile** | 🐦 **Bird** | 🐱 **Cat** | 🦌 **Deer**
* 🐶 **Dog** | 🐸 **Frog** | 🐴 **Horse** | 🚢 **Ship** | 🚛 **Truck**

By the way! You can add as much data as you want.

---

## 🏗 Model Architecture
The network is designed with a standard CNN structure to balance performance and computational efficiency:

| Layer Type | Configuration |
| :--- | :--- |
| **Convolutional Layer 1** | 3 input channels, 32 filters, 3x3 kernel |
| **Max Pooling** | 2x2 window |
| **Convolutional Layer 2** | 32 input, 64 output channels, 3x3 kernel |
| **Fully Connected 1** | 64 * 6 * 6 units → 128 units |
| **Fully Connected 2** | 128 units → 10 classes |

---

## 🛠 Tech Stack & Tools
* **Language:** Python 3.x
* **Core Framework:** [PyTorch](https://pytorch.org/)
* **Data Handling:** Torchvision
* **Numerical Computing:** NumPy
* **Visualization:** Matplotlib
