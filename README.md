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


