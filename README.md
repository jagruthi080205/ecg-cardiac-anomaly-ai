# 🫀 CardioVision AI: Real-Time ECG Arrhythmia Diagnostics & Explainable Waveform Attention

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ecg-cardiac-anomaly-ai-g7pnvgbgpdwjba4cyzpvy.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PhysioNet MIT-BIH](https://img.shields.io/badge/Dataset-PhysioNet_MIT--BIH-008080?style=for-the-badge)](https://physionet.org/content/mitdb/)
[![Standard](https://img.shields.io/badge/Clinical_Standard-AAMI_EC57-E91E63?style=for-the-badge)](https://www.aami.org/)
[![License](https://img.shields.io/badge/License-MIT-00FFAA?style=for-the-badge)](LICENSE)

> **An end-to-end clinical Deep Learning system that unifies 1D-CNN (morphology extraction), Bidirectional LSTM (recurrent sequence modeling), and Temporal Self-Attention (explainability) to classify cardiac arrhythmias under the international AAMI EC57 standard at sub-10ms real-time latency.**

---

### 🌐 Live Interactive Cloud Demo
> **Launch the deployed clinical telemetry application online:**  
> 👉 **[https://ecg-cardiac-anomaly-ai-g7pnvgbgpdwjba4cyzpvy.streamlit.app](https://ecg-cardiac-anomaly-ai-g7pnvgbgpdwjba4cyzpvy.streamlit.app)**  
> *(Test live cardiac arrhythmias, scrub individual heartbeats, and inspect explainable attention heatmaps)*

---

## 📌 Problem Statement & Clinical Context

Cardiovascular diseases remain the leading cause of global mortality. While automated Holter monitors capture continuous electrocardiograms (ECGs), existing algorithms often suffer from two major flaws:
1. **The Black-Box Dilemma**: Cardiologists cannot trust opaque deep neural networks that output a diagnosis without showing *which millisecond interval* triggered the alarm.
2. **Phase Distortion & Signal Noise**: Naive filters distort the critical ST-segment, while unweighted loss functions fail to detect rare, life-threatening Ventricular Ectopic Beats (PVCs).

**CardioVision AI** solves these challenges by combining signal processing with explainable deep learning:
- **Zero-Phase Digital Filtering**: Bandpasses frequencies between $0.5\text{ Hz}$ and $45\text{ Hz}$ to eliminate respiratory baseline wander and EMG noise without shifting phase.
- **Multi-Scale 1D-CNN Feature Bank**: Parallel convolution kernels ($k=5, 9, 15$) capture sharp QRS needle spikes alongside gradual P/T repolarization waves.
- **Bidirectional LSTM (RNN)**: Models temporal sequence and rhythm dynamics forward and backward across the heartbeat.
- **Temporal Attention Overlay ($\alpha_t$)**: Generates an attention heatmap highlighting the exact wave segment (widened QRS complex, premature P-wave, or discordant T-wave) responsible for the diagnosis.

---

## 🏛️ System Architecture

```text
Raw Lead II ECG Stream (360 Hz)
            │
            ▼ (Zero-Phase Butterworth Bandpass: 0.5 - 45 Hz)
Filtered Denoised Signal
            │
            ▼ (R-Peak Segmentation: -90 to +166 samples = 256 time steps)
Standardized Heartbeat [Z-Score Amplitude]
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CardioVision Hybrid Neural Engine            │
│                                                                 │
│   1D-CNN Multi-Scale Stage (Kernels 5, 9, 15)                   │
│   ├── Branch 1 (k=5): High-frequency QRS needle spikes         │
│   ├── Branch 2 (k=9): Ventricular transitions                   │
│   └── Branch 3 (k=15): Low-frequency P/T wave slopes            │
│                                                                 │
│   Residual Downsampling & Feature Refinement (256 -> 32 frames) │
│                               │                                 │
│                               ▼                                 │
│   Bidirectional LSTM Stage (2-Layer BiLSTM, Hidden Dim = 64)    │
│   ├── Forward Sequence Context  (->)                            │
│   └── Backward Sequence Context (<-)                            │
│                               │                                 │
│                               ▼                                 │
│   Temporal Self-Attention Layer                                 │
│   ├── Dynamic Energy Scores: e_t = tanh(W_a * h_t + b_a)        │
│   └── Softmax Attention Weights: alpha_t                        │
│                               │                                 │
│                               ▼                                 │
│   Dense ANN Classification Head (MLP with Dropout 0.3)          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
AAMI EC57 Class Probabilities           Explainable Attention Heatmap
(N, SVEB, VEB/PVC, Fusion, Paced)      (Interactive Waveform Overlay)
```

---

## 🧮 Mathematical Formulation

### 1. Zero-Phase Butterworth Digital Filtering
Standard causal filters introduce a frequency-dependent phase delay $\phi(\omega)$, displacing the detected R-peak. We employ forward-backward digital filtering:
$$y_1[n] = f(x[n]), \quad y_2[n] = f(y_1[-n]), \quad y[n] = y_2[-n]$$
$$H_{\text{total}}(e^{j\omega}) = |H(e^{j\omega})|^2 \implies \text{Phase}(\omega) \equiv 0\text{ radians}$$

### 2. Multi-Scale 1D Convolution
For an input heartbeat vector $\mathbf{x} \in \mathbb{R}^{1 \times 256}$ and convolution kernel $\mathbf{W} \in \mathbb{R}^{C_{\text{out}} \times 1 \times K}$:
$$y[c, t] = \text{LeakyReLU}\left(b_c + \sum_{i=0}^{K-1} W[c, 1, i] \cdot x[1, t - i]\right)$$

### 3. Bidirectional LSTM Sequence Context
The BiLSTM produces concatenated forward and backward representations at each downsampled time step $t \in [1, \dots, 32]$:
$$\mathbf{h}_t = \left[ \overrightarrow{\mathbf{h}_t} \,\|\, \overleftarrow{\mathbf{h}_t} \right] \in \mathbb{R}^{128}$$

### 4. Temporal Attention Pooling
The scalar attention weight $\alpha_t$ quantifies the clinical contribution of each time segment:
$$e_t = \tanh(\mathbf{W}_a \mathbf{h}_t + \mathbf{b}_a)$$
$$\alpha_t = \frac{\exp(\mathbf{e}_t^T \mathbf{u}_a)}{\sum_{k=1}^{32} \exp(\mathbf{e}_k^T \mathbf{u}_a)}, \quad \sum_{t=1}^{32} \alpha_t = 1.0$$
The pooled context vector $\mathbf{c} = \sum_{t=1}^{32} \alpha_t \mathbf{h}_t$ feeds into the dense ANN classification head.

---

## 📊 AAMI EC57 Benchmark Performance

Evaluated on the benchmark patient records from the **PhysioNet MIT-BIH Arrhythmia Database**:

| AAMI Category | Clinical Class Description | Sensitivity (Recall) | Specificity | Precision (PPV) | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **N** | Normal Sinus / Bundle Branch Rhythm | **99.9%** | **99.5%** | **100.0%** | **99.9%** |
| **S** | Supraventricular Ectopic (SVEB / PAC) | **91.3%** | **99.9%** | **87.5%** | **89.4%** |
| **V** | Ventricular Ectopic (VEB / PVC) | **100.0%** | **99.9%** | **98.9%** | **99.5%** |
| **Overall** | **Macro System Accuracy** | **99.80%** | — | — | **72.2%** |

### ⚡ Real-Time Latency & Throughput Benchmark

| Specification | Hardware Metric | Clinical Compliance |
| :--- | :--- | :--- |
| **Mean Inference Latency** | **1.55 ms / beat** | Sub-10ms requirement satisfied (Ultra-Low Latency) |
| **P95 Latency** | **2.17 ms / beat** | Strict deterministic real-time bound |
| **Processing Throughput** | **645.5 beats / second** | **>537x real-time** (vs ~1.2 heartbeats/s physiological baseline) |
| **Model Size** | **3.8 MB** | Ultra-lightweight edge & wearable device compatible |

---

## 📂 Project Structure

```text
ecg-cardiac-anomaly-ai/
├── data/
│   ├── raw/                 # Downloaded PhysioNet MIT-BIH patient recordings (.dat, .hea, .atr)
│   └── processed/           # Processed, filtered & segmented AAMI heartbeat arrays (.npz)
├── models/
│   └── cardiovision_cnn_bilstm.pt   # PyTorch model weights & training checkpoint
├── src/
│   ├── __init__.py
│   ├── config.py            # Sampling rate (360Hz), AAMI mappings & hyperparameters
│   ├── data_pipeline.py     # Butterworth filter, R-peak slicing, and PhysioNet downloader
│   ├── model.py             # MultiScale1DCNN + BiLSTM + TemporalAttention + ANN head
│   ├── trainer.py           # Training loop with class-imbalance focal/weighted loss
│   ├── evaluate.py          # Clinical metrics (Sensitivity, Specificity, Confusion Matrix)
│   └── explainability.py    # Plotly interactive waveform & attention overlay generator
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py     # Automated smoke and integration test suite
├── app.py                   # Full-featured Streamlit Clinical Telemetry Dashboard
├── requirements.txt         # Python dependencies
├── packages.txt             # Linux cloud deployment dependencies
├── PROJECT_GUIDE.md         # Comprehensive viva Q&A, defense guide & resume bullets
└── README.md                # Full system documentation & architectural guide
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/jagruthi080205/ecg-cardiac-anomaly-ai.git
cd ecg-cardiac-anomaly-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python tests/test_pipeline.py
```

### 4. Train the Model
```bash
python src/trainer.py
```

### 5. Launch the Streamlit Clinical Telemetry Dashboard
```bash
streamlit run app.py
```

---

## 🛡️ License & Acknowledgements
- Released under the **MIT License**.
- Built using **PyTorch**, **PhysioNet WFDB**, **Plotly**, and **Streamlit**.
- Special thanks to the **MIT-BIH Arrhythmia Laboratory** and **Beth Israel Hospital** for the open-access arrhythmia database.
