"""
Automated Smoke & Integration Test Suite for CardioVision AI
Validates 1D-CNN + BiLSTM + Attention forward pass, tensor dimensions, and signal filtering.
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import sys

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BEAT_WINDOW_LENGTH,
    NUM_CLASSES,
    SAMPLING_RATE
)
from src.data_pipeline import (
    butter_bandpass_filter,
    z_score_normalize,
    generate_physiologic_synthetic_ecg,
    ECGDataset
)
from src.model import CardioVisionNet, MultiScale1DCNN, TemporalAttention
from src.evaluate import benchmark_inference_latency
from src.explainability import compute_clinical_vitals


def test_bandpass_filter_and_normalization():
    """Validates that digital filtering preserves length and normalization yields zero-mean."""
    raw_signal = np.sin(np.linspace(0, 10, 1000)) + np.random.normal(0, 0.1, 1000)
    filtered = butter_bandpass_filter(raw_signal, fs=SAMPLING_RATE)
    assert len(filtered) == len(raw_signal)
    assert not np.isnan(filtered).any()

    norm_beat = z_score_normalize(filtered[:256])
    assert len(norm_beat) == 256
    assert abs(np.mean(norm_beat)) < 1e-4
    assert abs(np.std(norm_beat) - 1.0) < 1e-2


def test_synthetic_ecg_generator():
    """Validates synthetic ECG generator produces valid AAMI classes and dimensions."""
    beats, labels = generate_physiologic_synthetic_ecg(num_beats=50)
    assert beats.shape == (50, BEAT_WINDOW_LENGTH)
    assert len(labels) == 50
    assert set(np.unique(labels)).issubset({0, 1, 2, 3, 4})


def test_multiscale_1dcnn_dimensions():
    """Validates 1D-CNN multi-kernel branches and temporal downsampling."""
    cnn = MultiScale1DCNN(in_channels=1, out_channels=32)
    dummy_input = torch.randn(4, 1, BEAT_WINDOW_LENGTH)  # Batch of 4 beats
    out = cnn(dummy_input)
    # Output should be downsampled by 8 (256 -> 32) with 128 channels
    assert out.shape == (4, 128, 32)


def test_temporal_attention_constraint():
    """Validates that attention weights strictly sum to 1.0 across time steps."""
    attention = TemporalAttention(hidden_dim=128, attention_dim=64)
    dummy_lstm_out = torch.randn(4, 32, 128)
    context, attn_weights = attention(dummy_lstm_out)

    assert context.shape == (4, 128)
    assert attn_weights.shape == (4, 32)
    # Softmax condition: sum across time steps must equal 1.0
    sums = torch.sum(attn_weights, dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_full_cardiovision_net_forward():
    """Validates complete hybrid network forward pass and prediction explanation."""
    model = CardioVisionNet(num_classes=NUM_CLASSES)
    model.eval()
    dummy_beat = torch.randn(8, 1, BEAT_WINDOW_LENGTH)

    logits, attn = model(dummy_beat)
    assert logits.shape == (8, NUM_CLASSES)
    assert attn.shape == (8, 32)

    # Test single-beat explanation pipeline
    explanation = model.predict_with_explanation(dummy_beat[0, 0])
    assert "predicted_class" in explanation
    assert "probabilities" in explanation
    assert len(explanation["probabilities"]) == NUM_CLASSES
    assert len(explanation["attention_weights"]) == BEAT_WINDOW_LENGTH
    assert abs(sum(explanation["probabilities"]) - 1.0) < 1e-4


def test_clinical_vitals_computation():
    """Validates Heart Rate and HRV statistics."""
    # Simulated R-peaks 360 samples apart at 360 Hz (exactly 1 beat per second = 60 BPM)
    r_peaks = np.array([360, 720, 1080, 1440, 1800])
    vitals = compute_clinical_vitals(r_peaks, fs=360)
    assert vitals["bpm"] == 60.0
    assert vitals["mean_rr_ms"] == 1000.0


if __name__ == "__main__":
    test_bandpass_filter_and_normalization()
    test_synthetic_ecg_generator()
    test_multiscale_1dcnn_dimensions()
    test_temporal_attention_constraint()
    test_full_cardiovision_net_forward()
    test_clinical_vitals_computation()
    print("SUCCESS: ALL 6 TESTS PASSED!")
