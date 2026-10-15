"""
System Configuration & Hyperparameters for CardioVision AI
ECG Arrhythmia Classification & Temporal Attention Explainability Engine
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
CHECKPOINT_PATH = MODEL_DIR / "cardiovision_cnn_bilstm.pt"

# Signal Processing Hyperparameters
SAMPLING_RATE = 360  # Hz (MIT-BIH Arrhythmia standard)
LOWCUT_FREQ = 0.5    # Hz (Filters baseline wander caused by respiration)
HIGHCUT_FREQ = 45.0  # Hz (Filters powerline 50/60Hz interference & EMG noise)
FILTER_ORDER = 3     # Butterworth bandpass filter order

# Beat Segmentation Window (in samples at 360 Hz)
# 90 samples pre-R-peak (0.25s) and 166 samples post-R-peak (0.46s) = 256 samples total
PRE_R_SAMPLES = 90
POST_R_SAMPLES = 166
BEAT_WINDOW_LENGTH = PRE_R_SAMPLES + POST_R_SAMPLES  # 256 time steps

# AAMI EC57 Standard Categorization
AAMI_CLASSES = {
    "N": 0,  # Non-ectopic (Normal, LBBB, RBBB, Atrial/Nodal escape)
    "S": 1,  # Supraventricular ectopic (Atrial premature, Aberrant, Nodal premature)
    "V": 2,  # Ventricular ectopic (Premature Ventricular Contractions - PVC, Ventricular escape)
    "F": 3,  # Fusion of ventricular and normal beat
    "Q": 4   # Paced, unclassifiable, or unknown
}

# Raw MIT-BIH annotation symbols mapped to AAMI classes
MIT_TO_AAMI = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q'
}

CLASS_NAMES = [
    "Normal (N)",
    "Supraventricular Ectopic (SVEB)",
    "Ventricular Ectopic (VEB / PVC)",
    "Fusion Beat (F)",
    "Paced / Unknown (Q)"
]

CLASS_COLORS = {
    0: "#00E5FF",  # Cyan (Normal)
    1: "#FFD600",  # Amber/Yellow (SVEB)
    2: "#FF1744",  # Red/Alert (VEB / PVC)
    3: "#D500F9",  # Purple (Fusion)
    4: "#76FF03"   # Lime (Paced/Unknown)
}

CLASS_SEVERITY = {
    0: {"level": "Normal", "urgency": "Routine", "desc": "Regular sinus or bundle branch rhythm."},
    1: {"level": "Moderate Risk", "urgency": "Elevated", "desc": "Premature atrial activation; monitor for SVT/Atrial Fibrillation."},
    2: {"level": "High Clinical Risk", "urgency": "Urgent", "desc": "Premature Ventricular Contraction (PVC); can trigger Ventricular Tachycardia."},
    3: {"level": "Moderate Risk", "urgency": "Clinical Review", "desc": "Hybrid ventricular and supraventricular depolarization wave."},
    4: {"level": "Observational", "urgency": "Device Review", "desc": "Artificial cardiac pacemaker stimulus or artifact."}
}

# Selected Representative Patient Records from MIT-BIH Arrhythmia Database
BENCHMARK_RECORDS = [
    "100",  # Normal sinus rhythm with rare PACs
    "101",  # Normal with pacemaker rhythm
    "106",  # Severe Ventricular Ectopic Beats (PVCs)
    "119",  # Frequent PVCs in couplets & bigeminy
    "200",  # Mixed Ventricular Ectopic and Supraventricular Ectopic
    "208"   # High-frequency PVCs and fusion beats
]

# Deep Learning Architecture Hyperparameters
NUM_CLASSES = len(CLASS_NAMES)
CNN_FILTERS = [32, 64, 128]
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 2
ATTENTION_DIM = 64
DENSE_HIDDEN_DIM = 64
DROPOUT_RATE = 0.3

# Training Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 15
DEVICE = "cuda" if False else "cpu"
