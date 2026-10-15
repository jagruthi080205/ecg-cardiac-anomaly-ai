"""
CardioVision Deep Neural Network Architecture
Unifies 1D-CNN (Morphology Extraction) + BiLSTM (Temporal Sequence) + Temporal Attention + ANN (MLP Head)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from src.config import (
    NUM_CLASSES,
    LSTM_HIDDEN_DIM,
    LSTM_NUM_LAYERS,
    ATTENTION_DIM,
    DENSE_HIDDEN_DIM,
    DROPOUT_RATE,
    BEAT_WINDOW_LENGTH
)


class MultiScale1DCNN(nn.Module):
    """
    Multi-Scale 1D-CNN Feature Extractor.
    Simultaneously captures:
    1. Short kernel (k=5): High-frequency QRS needle spikes.
    2. Medium kernel (k=9): Standard ventricular transitions.
    3. Wide kernel (k=15): Low-frequency P-wave and T-wave repolarization slopes.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 32):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1)
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1)
        )
        
        # Intermediate downsampling and feature refinement
        total_channels = out_channels * 3  # 96 channels
        self.refine = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),  # 256 -> 128
            nn.Conv1d(total_channels, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2),  # 128 -> 64
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2)   # 64 -> 32 time steps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, 1, 256)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        merged = torch.cat([b1, b2, b3], dim=1)  # (batch_size, 96, 256)
        features = self.refine(merged)           # (batch_size, 128, 32)
        return features


class TemporalAttention(nn.Module):
    """
    Temporal Self-Attention mechanism over RNN hidden state sequence.
    Learns to dynamically weight time steps (e.g. QRS complex vs T-wave inversion).
    Provides clinical interpretability via attention weight vectors.
    """
    def __init__(self, hidden_dim: int, attention_dim: int = ATTENTION_DIM):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh()
        )
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_outputs shape: (batch_size, seq_len, hidden_dim)
        energy = self.projection(lstm_outputs)                     # (batch_size, seq_len, attention_dim)
        scores = self.context_vector(energy).squeeze(-1)           # (batch_size, seq_len)
        attention_weights = F.softmax(scores, dim=-1)             # Normalized weights: sum to 1

        # Compute weighted context vector
        # (batch_size, 1, seq_len) x (batch_size, seq_len, hidden_dim) -> (batch_size, hidden_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_outputs).squeeze(1)
        return context, attention_weights


class CardioVisionNet(nn.Module):
    """
    Complete Hybrid Deep Learning System:
    [1D Signal] -> [Multi-Scale 1D-CNN] -> [BiLSTM (RNN)] -> [Temporal Attention] -> [ANN (MLP Head)] -> [AAMI Classes]
    """
    def __init__(self, 
                 num_classes: int = NUM_CLASSES, 
                 lstm_hidden: int = LSTM_HIDDEN_DIM, 
                 lstm_layers: int = LSTM_NUM_LAYERS,
                 dropout: float = DROPOUT_RATE):
        super().__init__()

        # 1. Convolutional Stage (CNN)
        self.cnn = MultiScale1DCNN(in_channels=1, out_channels=32)

        # 2. Recurrent Stage (RNN - Bidirectional LSTM)
        # Input to LSTM: (batch_size, seq_len=32, feature_dim=128)
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # BiLSTM produces forward + backward hidden states (2 * lstm_hidden)
        bilstm_out_dim = lstm_hidden * 2  # 128

        # 3. Temporal Attention Stage
        self.attention = TemporalAttention(hidden_dim=bilstm_out_dim)

        # 4. Dense Artificial Neural Network Head (ANN / MLP)
        self.classifier = nn.Sequential(
            nn.Linear(bilstm_out_dim, DENSE_HIDDEN_DIM),
            nn.BatchNorm1d(DENSE_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(DENSE_HIDDEN_DIM, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            logits: (batch_size, num_classes)
            attention_weights: (batch_size, seq_len)
        """
        # 1. 1D-CNN Morphology Extraction
        cnn_features = self.cnn(x)                       # (batch, 128, 32)
        cnn_features = cnn_features.permute(0, 2, 1)    # (batch, 32, 128) for LSTM batch_first

        # 2. Bidirectional LSTM Sequence Modeling
        lstm_out, _ = self.bilstm(cnn_features)          # (batch, 32, 128)

        # 3. Temporal Attention Pooling
        context_vector, attention_weights = self.attention(lstm_out)  # (batch, 128), (batch, 32)

        # 4. Dense ANN Classification
        logits = self.classifier(context_vector)          # (batch, num_classes)
        return logits, attention_weights

    def predict_with_explanation(self, x: torch.Tensor) -> Dict:
        """
        Inference helper for production and Streamlit UI.
        Returns class index, probabilities, and temporal attention distribution.
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, 256)
            elif x.dim() == 2:
                x = x.unsqueeze(1)               # (batch, 1, 256)

            logits, attention_weights = self.forward(x)
            probabilities = F.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_class = int(np.argmax(probabilities))
            attn = attention_weights.cpu().numpy()[0]

            # Upsample attention weights from 32 steps to original 256 samples for plotting
            attn_upsampled = np.interp(
                np.linspace(0, 1, BEAT_WINDOW_LENGTH),
                np.linspace(0, 1, len(attn)),
                attn
            )
            # Normalize upsampled attention between 0 and 1
            attn_upsampled = (attn_upsampled - attn_upsampled.min()) / (attn_upsampled.max() - attn_upsampled.min() + 1e-8)

            return {
                "predicted_class": pred_class,
                "probabilities": probabilities.tolist(),
                "confidence": float(probabilities[pred_class]),
                "attention_weights": attn_upsampled.tolist()
            }
