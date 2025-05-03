# Generative Models on MNIST & Fashion‑MNIST

### *GAN vs VAE with Anomaly Detection*

## Project Overview

This repository contains a full, end‑to‑end workflow for training and evaluating **Generative Adversarial Networks (GANs)** and **Variational Autoencoders (VAEs)** on both **MNIST Digits** and **Fashion‑MNIST** datasets.  Beyond image generation, we also use the trained VAE as an **unsupervised anomaly‑detection engine**.

The work accompanies and demonstrates:

* How to conduct **Exploratory Data Analysis (EDA)** for image data.
* Building production‑ready TensorFlow / Keras implementations of GAN & VAE.
* Comparing model families on **image quality, training stability, and latent‑space structure**.
* Applying latent reconstructions for **out‑of‑distribution anomaly detection**.

---

## Key Features

* **Reproducible Experiments** – seeds, config files & tracked metrics.
* **GPU‑Ready** – verified on NVIDIA RTX 3050 (Laptop) with TensorFlow 2.10.
* **Live Visualisation** – Matplotlib loss curves & generated samples every epoch.
* **Anomaly Detection Demo** – flag unusual images via VAE reconstruction error.
* **Well‑Documented Code** – docstrings & type‑hints throughout.

---

## Dataset

| Dataset           | Train Images | Test Images | Classes | Image Shape |
| ----------------- | -----------: | ----------: | :-----: | :---------: |
| **MNIST**         |       60 000 |      10 000 |    10   | 28 × 28 × 1 |
| **Fashion‑MNIST** |       60 000 |      10 000 |    10   | 28 × 28 × 1 |

> All images are **grayscale** and normalised to **\[0, 1]** before modeling.  An extra channel dimension is added so that convolutional layers expect `(28, 28, 1)`.

---

## Methodology

### Exploratory Data Analysis

*Sample grids* of digits and fashion items are displayed, alongside basic dataset statistics and class mappings.

### Generative Adversarial Network (GAN)

| Component         | Layers (key)                                                                             | Output Shape |
| ----------------- | ---------------------------------------------------------------------------------------- | ------------ |
| **Generator**     | Dense (7×7×256) → BatchNorm → LeakyReLU → Conv2D<sup>T</sup> × 3 → Dropout → **Sigmoid** | 28 × 28 × 1  |
| **Discriminator** | Conv2D (64) → LeakyReLU → Dropout → Conv2D (128) → Conv2D (256) → Flatten → **Sigmoid**  | 1            |

* **Loss:** Binary cross‑entropy (real = 1 / fake = 0)
* **Optimiser:** Adam (learning‑rate = 1e‑4)
* \~2.38 M trainable params (generator), 1.03 M (discriminator)

### Variational Autoencoder (VAE)

* **Encoder:** Conv2D blocks → *µ* & *σ<sup>2</sup>* dense heads.
* **Decoder:** Transposed convolutions mirroring the encoder.
* **Loss:** Reconstruction (binary CE) + β‑KL divergence.

### Anomaly Detection Pipeline

1. Train the VAE on normal images.
2. **Reconstruct** a candidate image.
3. Compute pixel‑wise **reconstruction error**.
4. Flag as anomaly if error > threshold (μ + 3σ on validation set).

### Model Comparison

| Aspect           | **GAN**                              | **VAE**                                 |
| ---------------- | ------------------------------------ | --------------------------------------- |
| Image sharpness  | High (crisp digits & fashion items)  | Smooth, occasionally blurred            |
| Training time    | Long & delicate (oscillations)       | Faster, more stable                     |
| Latent structure | Unconstrained; harder to interpolate | Continuous & interpretable latent space |
| Resource usage   | Higher GPU footprint                 | Lighter                                 |

---

## Getting Started

### Prerequisites

* **Python 3.9+**
* GPU‑enabled **TensorFlow 2.10** (or higher)
* CUDA 11.x & cuDNN properly installed *(optional but recommended)*

```bash
# Core libraries
pip install -r requirements.txt
# Quick list:
# tensorflow==2.10  numpy  matplotlib  pandas  scikit-learn
```

### Installation

```bash
git clone https://github.com/abdurrehman-haroon/GenerativeModels-MNIST-FashionMNIST-GAN-VAE-AnomalyDetection
cd GenerativeModels-MNIST-FashionMNIST-GAN-VAE-AnomalyDetection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Results & Visualisations

<p align="center"><img src="outputs/images/gan_mnist_grid_epoch50.png" width="260">  <img src="outputs/images/vae_latent_walk.png" width="260">  <img src="outputs/images/anomaly_heatmap.png" width="260"></p>

|             Metric (MNIST) |  **GAN** |   **VAE**   |
| -------------------------: | :------: | :---------: |
| Fréchet Inception Distance | **12.3** |     29.8    |
| Peak Signal‑to‑Noise Ratio |  19.4 dB | **22.1 dB** |

> *Full quantitative tables are available in* `outputs/metrics/`.

---

## Troubleshooting & FAQ

| Issue                   | Possible Fix                                                                               |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| *Out‑of‑memory* errors  | Decrease `batch_size`, use `mixed_precision`, or restrict GPUs in `src/utils/tf_setup.py`. |
| GAN generator collapses | Lower `lr` for discriminator, add gradient penalty, or use label smoothing.                |
| Blurry VAE outputs      | Increase β in the VAE loss or enlarge the latent dimension.                                |

---

## Contributing

Pull requests are welcome!  Please open an issue first to discuss major changes.  Make sure your code is **PEP‑8 compliant** and includes unit tests (`pytest`).

---

## License

Distributed under the **MIT License**.  See [`LICENSE`](LICENSE) for details.

---

```
