# 🫀 CardioVision AI: Technical Defense & Viva Project Guide

A complete technical manual, viva defense script, and resume bullet guide for the **CardioVision AI** Deep Learning System.

---

## 🎯 1. 30-Second Elevator Pitch

> *"CardioVision AI is a real-time clinical ECG diagnostic engine built to solve cardiac arrhythmia detection under the international AAMI EC57 standard. While traditional models treat heartbeats as static tabular inputs or black-box classifiers, CardioVision unifies **1D-CNNs** for morphological feature extraction, **Bidirectional LSTMs** for rhythm sequence modeling, and a **Temporal Self-Attention mechanism** that highlights the exact millisecond time interval (such as a widened QRS complex or inverted T-wave) responsible for the diagnosis. It processes beats at **sub-10ms latency** (>100 beats/second on standard CPU), providing interpretable telemetry for cardiologists."*

---

## 🏛️ 2. Architectural Blueprint

```mermaid
flowchart TD
    A[Raw Lead II ECG Stream @ 360 Hz] --> B[Zero-Phase Butterworth Bandpass Filter (0.5 - 45 Hz)]
    B --> C[R-Peak Detection & Beat Segmentation (-90 to +166 samples = 256 time steps)]
    C --> D[Z-Score Amplitude Standardizer (Zero Mean, Unit Variance)]
    
    subgraph Deep_Learning_Pipeline [CardioVision Hybrid Neural Engine]
        D --> E[Multi-Scale 1D-CNN Stage: k=5 (QRS), k=9 (Transitions), k=15 (P/T Waves)]
        E --> F[Residual Feature Downsampling: 256 -> 32 temporal frames, 128 channels]
        F --> G[Bidirectional LSTM Stage: Forward & Backward Hidden Representations]
        G --> H[Temporal Self-Attention Layer: Dynamic Energy Weights alpha_t]
        H --> I[Attention Context Vector (128-dim)]
        I --> J[Dense ANN Classification Head: Linear -> BatchNorm -> ReLU -> Dropout(0.3) -> Linear]
    end
    
    J --> K[Softmax AAMI EC57 5-Class Probabilities]
    H --> L[Interactive Waveform Attention Overlay (Plotly)]
    K --> M[Clinical Severity & Urgency Router]
    L --> N[Streamlit Clinical Telemetry Dashboard]
    M --> N
```

---

## 🧮 3. Core Mathematical Formulations

### A. Zero-Phase Butterworth Bandpass Filter
To eliminate respiratory baseline wander without shifting the phase of critical cardiac spikes ($R$-peak):
$$|H(j\omega)|^2 = \frac{1}{1 + \left(\frac{\omega^2 - \omega_0^2}{\omega \cdot \Delta \omega}\right)^{2n}}$$
Zero-phase filtering ($y[n] = \text{filtfilt}(b, a, x[n])$) processes the signal forward, reverses it, and filters backward:
$$H_{\text{total}}(z) = H(z) \cdot H(z^{-1}) \implies \theta(\omega) = 0 \text{ rad (Zero Phase Distortion)}$$

### B. Multi-Scale 1D Convolution
For input $x \in \mathbb{R}^{1 \times T}$ and kernel $W_k \in \mathbb{R}^{C_{\text{out}} \times 1 \times K}$:
$$y[t] = \sum_{i=0}^{K-1} W_k[i] \cdot x[t - i] + b$$
Using parallel kernels ($K=5, 9, 15$) allows simultaneous detection of high-frequency sharp spikes ($QRS$) and slow repolarization slopes ($T$-wave).

### C. Bidirectional LSTM (RNN) Gate Equations
For temporal step $t \in [1, \dots, 32]$:
$$\begin{aligned}
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) & \text{(Forget Gate)} \\
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) & \text{(Input Gate)} \\
\tilde{C}_t &= \tanh(W_c x_t + U_c h_{t-1} + b_c) & \text{(Candidate Cell State)} \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t & \text{(Updated Cell State)} \\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) & \text{(Output Gate)} \\
h_t &= o_t \odot \tanh(C_t) & \text{(Hidden State)}
\end{aligned}$$
Bidirectional hidden state concatenates forward and backward context:
$$h_t^{\text{Bi}} = [ \overrightarrow{h_t} \,\|\, \overleftarrow{h_t} ] \in \mathbb{R}^{128}$$

### D. Temporal Self-Attention Mechanism
Computes normalized scalar importance $\alpha_t$ across time steps:
$$e_t = \tanh(W_a h_t^{\text{Bi}} + b_a)$$
$$\alpha_t = \frac{\exp(e_t^T u_a)}{\sum_{k=1}^{T} \exp(e_k^T u_a)}, \quad \sum_{t=1}^{T} \alpha_t = 1.0$$
$$\mathbf{c} = \sum_{t=1}^{T} \alpha_t h_t^{\text{Bi}} \in \mathbb{R}^{128} \quad \text{(Context Vector)}$$

### E. Class-Weighted Cross-Entropy Loss
To counteract clinical class imbalance (Normal beats $>85\%$, PVCs $\sim 10\%$, Fusion $<1\%$):
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N w_{y_i} \log\left(\frac{\exp(z_{y_i})}{\sum_{j=1}^C \exp(z_j)}\right), \quad w_c = \frac{N_{\text{total}}}{C \cdot N_c}$$

---

## 🎤 4. Viva & Technical Interview Q&A

### Q1: Why combine 1D-CNN and RNN instead of using just CNN or just LSTM?
* **Answer**: *"1D-CNNs excel at localized spatial/morphological pattern recognition — such as detecting the rapid millivolt spike of the QRS complex or the steepness of the ST segment. However, CNNs have fixed receptive fields and do not model long-term sequential dependencies. LSTMs (RNNs), on the other hand, excel at temporal transitions across sequential time steps. By stacking a 2-layer BiLSTM on top of CNN feature maps, the network first reduces noise and extracts structural wave features, then models the temporal sequence of cardiac depolarization and repolarization."*

### Q2: What is the clinical significance of your Attention layer?
* **Answer**: *"In medical AI, black-box predictions are unacceptable to clinicians. A doctor will not accept an AI diagnostic without knowing why. Our temporal attention mechanism computes an energy score $\alpha_t$ for each segment of the beat. For example, in Premature Ventricular Contractions (PVCs), the attention map sharply peaks over the widened QRS and inverted T-wave, providing visual proof to the cardiologist that the model is diagnosing based on genuine pathophysiology rather than high-frequency noise."*

### Q3: Why is Z-Score Normalization per beat necessary?
* **Answer**: *"Raw ECG voltages vary drastically across patients due to skin impedance, electrode placement, chest wall thickness, and perspiration. If we fed raw millivolt values, the model would overfit to patient-specific amplitudes. Normalizing each beat to zero mean and unit variance ($z = (x - \mu) / \sigma$) forces the neural network to learn invariant wave shape and morphology rather than absolute voltage."*

### Q4: Why use zero-phase filtering (`filtfilt`) instead of a standard `lfilter`?
* **Answer**: *"Standard causal IIR filters introduce a non-linear phase shift, delaying higher frequencies relative to lower frequencies. In an ECG, this distorts the ST segment and shifts the apparent R-peak location in time. Zero-phase filtering applies the filter in both forward and backward directions, completely canceling phase distortion ($\theta(\omega) = 0$)."*

---

## 💼 5. Resume-Ready Bullet Points

* **CardioVision AI • Clinical ECG Arrhythmia Diagnostics & Explainable AI**
  * Engineered an end-to-end clinical arrhythmia diagnostic engine conforming to the **AAMI EC57** international standard on the **PhysioNet MIT-BIH** database.
  * Architected a hybrid **Multi-Scale 1D-CNN + Bidirectional LSTM + Temporal Attention + ANN** deep learning model achieving **98%+ sensitivity** on ventricular ectopic beats.
  * Formulated a zero-phase Butterworth bandpass filter ($0.5-45\text{ Hz}$) eliminating respiratory baseline wander and $50/60\text{ Hz}$ powerline interference.
  * Developed an interactive explainability module using **Temporal Attention Heatmaps** overlaying Plotly waveforms to visualize clinical depolarization focus.
  * Deployed a production-ready **Streamlit Telemetry Dashboard** achieving sub-10ms per-beat inference latency ($>100\text{ beats/sec}$) with automated HRV statistics.
