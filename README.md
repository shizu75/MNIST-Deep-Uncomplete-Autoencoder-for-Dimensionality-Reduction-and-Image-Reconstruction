# MNIST Autoencoder for Dimensionality Reduction and Image Reconstruction

## Project Overview
This repository presents a **research-grade implementation of a fully connected Autoencoder** applied to the **MNIST handwritten digits dataset**. The project focuses on **unsupervised representation learning**, where high-dimensional image data is compressed into a low-dimensional latent space and reconstructed back with minimal information loss. This work demonstrates core concepts used in **feature learning, anomaly detection, and generative modeling**, making it suitable for **graduate-level AI/ML research portfolios**.

---

## Dataset Description
- **Dataset:** MNIST Handwritten Digits
- **Source:** TensorFlow / Keras built-in dataset
- **Image Resolution:** 28 × 28 pixels
- **Color Format:** Grayscale
- **Training Samples:** 60,000
- **Test Samples:** 10,000
- **Pixel Normalization:** Scaled to [0, 1]
- **Flattened Input Dimension:** 784 features per image

---

## Methodology

### Data Preprocessing
- Images are normalized to floating-point values in the range [0, 1]
- Training set is split into:
  - Training subset
  - Validation subset (first 10,000 samples)
- All images are reshaped from 2D (28×28) into 1D vectors (784)

---

## Autoencoder Architecture

### Encoder Network
- Input layer: 784-dimensional vector
- Dense layers:
  - 256 neurons (ReLU)
  - 128 neurons (ReLU)
- **Bottleneck layer:** 32 neurons (latent representation)

### Decoder Network
- Dense layers:
  - 128 neurons (ReLU)
  - 256 neurons (ReLU)
- Output layer:
  - 784 neurons (Sigmoid activation) for pixel-wise reconstruction

This symmetric architecture enforces **information compression**, encouraging the network to learn meaningful latent features.

---

## Model Configuration
- **Framework:** TensorFlow / Keras Functional API
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Batch Size:** 256
- **Epochs:** 50
- **Validation Strategy:** Reconstruction loss on test data

---

## Training and Evaluation

### Training
- The autoencoder is trained to reconstruct its input
- Validation loss is monitored to assess generalization
- Training history (loss curves) is visualized for convergence analysis

### Evaluation
- Reconstruction loss computed on the test dataset
- Original and reconstructed images are visually compared
- Pixel-wise reconstruction error is analyzed

---

## Results and Visualization
- Clear reconstruction of handwritten digits from compressed latent space
- Loss curves indicate stable convergence
- Visual comparison highlights preservation of digit structure despite compression
- Reconstruction error maps reveal pixel-level discrepancies

---

## Performance Analysis
- Latent space dimensionality reduced from 784 → 32
- Demonstrates effective dimensionality reduction with minimal perceptual loss
- Suitable for downstream tasks such as:
  - Anomaly detection
  - Feature extraction
  - Data compression
  - Pretraining for supervised learning

---

## Research Significance
This project reflects **core unsupervised learning principles** widely used in:
- Representation learning
- Medical imaging
- Signal compression
- Pretraining deep neural networks

The implementation follows **best practices in experimental deep learning**, including modular design, visualization, and reproducibility.

---

## Technologies Used
- Python 3
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn (metrics)

---

## Potential Extensions
- Convolutional Autoencoders for spatial feature learning
- Variational Autoencoders (VAE)
- Latent space visualization (t-SNE / PCA)
- Anomaly detection using reconstruction error thresholds

---

## Author Notes
This repository is designed to be **academically presentable, reproducible, and extensible**, suitable for inclusion in **research portfolios, MSc/PhD applications, and applied AI demonstrations**.
