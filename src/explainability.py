"""
CardioVision Explainability & Clinical Waveform Visualization Engine
Provides interactive Plotly visualizations with Temporal Attention overlays and Rhythm Strips.
"""

import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Optional

from src.config import (
    CLASS_NAMES,
    CLASS_COLORS,
    SAMPLING_RATE,
    BEAT_WINDOW_LENGTH,
    PRE_R_SAMPLES
)


def create_attention_overlay_figure(
    beat: np.ndarray,
    attention_weights: List[float],
    predicted_class: int,
    confidence: float,
    true_label: Optional[int] = None,
    fs: int = SAMPLING_RATE
) -> go.Figure:
    """
    Renders an interactive Plotly waveform figure with a temporal attention heatmap.
    Allows cardiologists to visually confirm whether the model triggered on the QRS spike,
    missing P-wave, or inverted T-wave.
    """
    time_ms = np.linspace(-PRE_R_SAMPLES / fs * 1000, 
                          (len(beat) - PRE_R_SAMPLES) / fs * 1000, 
                          len(beat))
    
    attn_array = np.array(attention_weights)
    pred_name = CLASS_NAMES[predicted_class]
    pred_color = CLASS_COLORS[predicted_class]

    fig = go.Figure()

    # 1. Attention Heatmap Area (Shaded region indicating model focus)
    fig.add_trace(go.Scatter(
        x=time_ms,
        y=attn_array * (np.max(beat) * 0.8),
        fill='tozeroy',
        mode='none',
        fillcolor='rgba(255, 23, 68, 0.22)',
        name='Temporal Attention Focus (α)',
        hoverinfo='skip'
    ))

    # 2. Main Lead II ECG Voltage Signal
    fig.add_trace(go.Scatter(
        x=time_ms,
        y=beat,
        mode='lines',
        name='ECG Lead II (Normalized)',
        line=dict(color='#00E5FF', width=2.5),
        hovertemplate='Time: %{x:.1f} ms<br>Amplitude: %{y:.2f} mV<extra></extra>'
    ))

    # 3. Mark the R-Peak reference at t = 0 ms
    r_peak_idx = PRE_R_SAMPLES
    fig.add_trace(go.Scatter(
        x=[0],
        y=[beat[r_peak_idx]],
        mode='markers+text',
        name='R-Peak Reference',
        text=['R-Peak (0 ms)'],
        textposition='top center',
        textfont=dict(color='#FFD600', size=11),
        marker=dict(color='#FFD600', size=9, symbol='diamond')
    ))

    # Layout styling: Dark medical telemetry monitor aesthetic
    title_text = f"<b>Predicted:</b> {pred_name} (Confidence: {confidence*100:.1f}%)"
    if true_label is not None:
        title_text += f" | <b>Ground Truth:</b> {CLASS_NAMES[true_label]}"

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=15, color='#F0F6FC')
        ),
        paper_bgcolor='#0D1117',
        plot_bgcolor='#161B22',
        xaxis=dict(
            title="Time relative to R-peak (ms)",
            gridcolor='#21262D',
            zerolinecolor='#30363D',
            color='#8B949E'
        ),
        yaxis=dict(
            title="Standardized Amplitude",
            gridcolor='#21262D',
            zerolinecolor='#30363D',
            color='#8B949E'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#C9D1D9')
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=380
    )

    return fig


def create_rhythm_strip_figure(
    continuous_signal: np.ndarray,
    r_peaks: np.ndarray,
    beat_labels: List[int],
    fs: int = SAMPLING_RATE
) -> go.Figure:
    """
    Renders a continuous 10-second rhythm strip with annotated, color-coded beat markers.
    """
    time_sec = np.arange(len(continuous_signal)) / fs

    fig = go.Figure()

    # Continuous ECG Trace
    fig.add_trace(go.Scatter(
        x=time_sec,
        y=continuous_signal,
        mode='lines',
        name='Continuous Rhythm (Lead II)',
        line=dict(color='#58A6FF', width=1.5),
        hoverinfo='skip'
    ))

    # Add markers for each detected heartbeat
    for peak_idx, label in zip(r_peaks, beat_labels):
        if peak_idx < len(time_sec):
            t_val = time_sec[peak_idx]
            y_val = continuous_signal[peak_idx]
            color = CLASS_COLORS[label]
            name = CLASS_NAMES[label].split()[0]

            fig.add_trace(go.Scatter(
                x=[t_val],
                y=[y_val],
                mode='markers',
                marker=dict(color=color, size=8, symbol='circle'),
                name=name,
                showlegend=False,
                hovertemplate=f"Beat: {name}<br>Time: {t_val:.2f} s<extra></extra>"
            ))

    fig.update_layout(
        paper_bgcolor='#0D1117',
        plot_bgcolor='#161B22',
        xaxis=dict(
            title="Time (seconds)",
            gridcolor='#21262D',
            zerolinecolor='#30363D',
            color='#8B949E'
        ),
        yaxis=dict(
            title="Voltage (mV)",
            gridcolor='#21262D',
            zerolinecolor='#30363D',
            color='#8B949E'
        ),
        margin=dict(l=40, r=40, t=30, b=40),
        height=260
    )

    return fig


def compute_clinical_vitals(r_peaks: np.ndarray, fs: int = SAMPLING_RATE) -> Dict:
    """
    Calculates Heart Rate (BPM), Heart Rate Variability (HRV - RMSSD), and RR interval statistics.
    """
    if len(r_peaks) < 2:
        return {"bpm": 72.0, "hrv_rmssd_ms": 35.0, "mean_rr_ms": 833.0, "rhythm_regularity": "Insufficient data"}

    rr_intervals_sec = np.diff(r_peaks) / fs
    rr_intervals_ms = rr_intervals_sec * 1000.0

    bpm = float(60.0 / np.mean(rr_intervals_sec))
    mean_rr = float(np.mean(rr_intervals_ms))
    
    # Root Mean Square of Successive Differences (RMSSD)
    if len(rr_intervals_ms) >= 2:
        rmssd = float(np.sqrt(np.mean(np.diff(rr_intervals_ms) ** 2)))
    else:
        rmssd = 0.0

    std_rr = float(np.std(rr_intervals_ms))
    regularity = "Regular Sinus" if std_rr < 60.0 else "Irregular (Arrhythmic)"

    return {
        "bpm": round(bpm, 1),
        "mean_rr_ms": round(mean_rr, 1),
        "hrv_rmssd_ms": round(rmssd, 1),
        "rhythm_regularity": regularity
    }
