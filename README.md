# MNIST-Classification-Model

A hand-built neural network for MNIST digit classification using **only JAX and NumPy**, with **manual forward & backward propagation**, a custom architecture with a **skip connection**, and zero external ML libraries.  
Finalist at **TensorCraft**, hosted by **NIT Trichy**.

---

## Project Overview

This model was developed as part of the **TensorCraft competition** organized by **NIT Trichy**, where participants were challenged to:

- Build a deep learning model using **only JAX and NumPy** (no libraries like Flax, Haiku, PyTorch, TensorFlow, etc.)
- Implement **manual forward and backward propagation**
- Include **at least one skip connection**
- Train on the **MNIST dataset** and optimize for performance

### Achievements

| Metric       | Score         |
|--------------|---------------|
| Accuracy     | **96.0%**     |
| F1 Score     | **95.3**      |
| Precision    | **95.32**     |

This performance qualified us as **finalists** in the competition.

---

## Model Architecture

- Fully connected layers (custom implementation)
- **ReLU**, Softmax, Cross-Entropy
- One **skip connection** between intermediate dense layers
- Batch-wise training
- Gradient computation via **manual backpropagation**
- No autograd, no external ML/DL libraries

---

## Dataset – MNIST

> **MNIST (Modified National Institute of Standards and Technology)** is a classic benchmark dataset in machine learning and deep learning.

- Contains **70,000 grayscale images** of handwritten digits (0–9)
  - **60,000** for training
  - **10,000** for testing
- Each image is **28x28 pixels**, flattened into a vector of **784 features**
- Labels range from **0 to 9**, one-hot encoded for training
- Dataset was preprocessed using:
  - Normalization (pixel values scaled between 0 and 1)
  - Shuffled and batched manually for training

We used the version available through `jax.example_libraries` and manually handled preprocessing to meet the low-level constraint rules of TensorCraft.

---


