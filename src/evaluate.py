"""
CardioVision Clinical Evaluation & Latency Benchmarking Engine
Evaluates AAMI EC57 metrics: Sensitivity (Recall), Specificity, Precision, and Real-Time Latency.
"""

import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from typing import Dict, Tuple

from src.config import (
    CHECKPOINT_PATH,
    NUM_CLASSES,
    CLASS_NAMES,
    BEAT_WINDOW_LENGTH
)
from src.model import CardioVisionNet
from src.data_pipeline import build_or_load_dataset


def load_trained_model(checkpoint_path: Path = CHECKPOINT_PATH, device: str = "cpu") -> CardioVisionNet:
    """Loads model weights from checkpoint or initializes fresh."""
    model = CardioVisionNet(num_classes=NUM_CLASSES)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[Loaded] Checkpoint from {checkpoint_path}")
    else:
        print("[Notice] No checkpoint found. Initializing untrained baseline model.")
    model.to(device)
    model.eval()
    return model


def evaluate_clinical_metrics(model: CardioVisionNet, X: np.ndarray, y: np.ndarray) -> Dict:
    """
    Computes rigorous clinical metrics:
    - Sensitivity (Recall)
    - Specificity
    - Positive Predictive Value (Precision)
    - Macro F1-Score
    - Confusion Matrix
    """
    device = next(model.parameters()).device
    model.eval()

    tensor_X = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
    
    with torch.no_grad():
        logits, attention_weights = model(tensor_X)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

    cm = confusion_matrix(y, preds, labels=list(range(NUM_CLASSES)))
    acc = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average='macro', zero_division=0)

    # Calculate per-class Sensitivity and Specificity
    per_class_metrics = {}
    total_samples = len(y)

    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = total_samples - (tp + fn + fp)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

        per_class_metrics[CLASS_NAMES[i]] = {
            "sensitivity_recall": round(float(sensitivity), 4),
            "specificity": round(float(specificity), 4),
            "precision_ppv": round(float(precision), 4),
            "f1_score": round(float(f1), 4),
            "support": int(np.sum(cm[i, :]))
        }

    return {
        "overall_accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_metrics
    }


def benchmark_inference_latency(model: CardioVisionNet, num_iterations: int = 500) -> Dict:
    """
    Measures per-beat inference latency in milliseconds and real-time processing throughput.
    """
    device = next(model.parameters()).device
    dummy_beat = torch.randn(1, 1, BEAT_WINDOW_LENGTH).to(device)

    # Warmup
    for _ in range(50):
        _ = model(dummy_beat)

    latencies = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        _ = model(dummy_beat)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    throughput_fps = float(1000.0 / avg_latency)

    return {
        "mean_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "throughput_beats_per_sec": round(throughput_fps, 1),
        "real_time_factor": f"{throughput_fps / 1.2:.1f}x real-time (Assuming ~1.2 heartbeats/sec)"
    }


def run_full_benchmark():
    """Runs complete clinical and throughput benchmark and prints table."""
    print("[Benchmark] Loading data and model...")
    X, y = build_or_load_dataset()
    model = load_trained_model()

    print("\n--- CLINICAL ACCURACY & SENSITIVITY METRICS ---")
    metrics = evaluate_clinical_metrics(model, X, y)
    print(f"Overall Accuracy: {metrics['overall_accuracy']*100:.2f}%")
    print(f"Macro F1-Score:   {metrics['macro_f1']*100:.2f}%\n")

    print(f"{'Class Name':<35} | {'Sensitivity':<12} | {'Specificity':<12} | {'Precision':<10} | {'F1':<8} | {'Support'}")
    print("-" * 95)
    for c_name, scores in metrics["per_class"].items():
        print(f"{c_name:<35} | {scores['sensitivity_recall']*100:>10.1f}% | {scores['specificity']*100:>10.1f}% | {scores['precision_ppv']*100:>8.1f}% | {scores['f1_score']*100:>6.1f}% | {scores['support']:>6}")

    print("\n--- REAL-TIME LATENCY BENCHMARK ---")
    latency = benchmark_inference_latency(model)
    print(f"Mean Inference Latency: {latency['mean_latency_ms']} ms / beat")
    print(f"P95 Inference Latency:  {latency['p95_latency_ms']} ms / beat")
    print(f"Throughput:             {latency['throughput_beats_per_sec']} beats/second ({latency['real_time_factor']})")

    return metrics, latency


if __name__ == "__main__":
    run_full_benchmark()
