"""
Project: Low-Light Image Enhancement Using Deep Learning Techniques
Streamlit Web Application & Digital Image Processing Suite
"""

import os
import sys
import time
from datetime import datetime
import streamlit as st
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# Import existing project modules (No model duplication in app.py)
from model import load_zero_dce_model
from enhancement import (
    load_image,
    enhance_low_light,
    save_enhanced_image,
    get_image_download_bytes,
    generate_grayscale_comparison,
    generate_histogram_comparison_figure,
    visualize_curve_parameters_figure,
)
from metrics import compute_all_metrics, compare_metrics


# ============================================================================
# Streamlit Page Setup & Dark Navy Theme Styling
# ============================================================================
st.set_page_config(
    page_title="Low-Light Image Enhancement | Deep Learning",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark Navy & Blue Academic CSS Styling
st.markdown(
    """
    <style>
    /* Global background and font styling */
    .stApp {
        background-color: #0b132b;
        color: #e2e8f0;
    }
    
    /* Header Card */
    .header-box {
        background: linear-gradient(135deg, #0a192f 0%, #1e3a8a 100%);
        border: 1px solid #1e40af;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #60a5fa;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #cbd5e1;
        line-height: 1.5;
    }
    
    /* Status Badge styling */
    .status-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 18px;
    }
    .badge-dl {
        background-color: rgba(59, 130, 246, 0.15);
        color: #38bdf8;
        border: 1px solid #0284c7;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-ok {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid #059669;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-warn {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid #d97706;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    /* Metrics Card styling */
    .metric-container {
        background-color: #111e38;
        border: 1px solid #1e3a8a;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    .metric-delta-pos {
        color: #34d399;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Image Section Card */
    .image-card {
        background-color: #0d1b38;
        border: 1px solid #1e3a8a;
        border-radius: 10px;
        padding: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "data", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")
WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "Epoch99.pth")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# Cached Model Initialization (Evaluated Once)
# ============================================================================
@st.cache_resource(show_spinner="Initializing Deep Learning Enhancement Engine...")
def get_cached_zero_dce_model():
    return load_zero_dce_model(weights_path=WEIGHTS_PATH)


try:
    model, model_status = get_cached_zero_dce_model()
except Exception as e:
    st.error(f"Critical error loading model: {str(e)}")
    st.stop()


# ============================================================================
# Header Section (Title & Short Academic Project Description)
# ============================================================================
st.markdown(
    """
    <div class="header-box">
        <div class="main-title">🌌 Low-Light Image Enhancement Using Deep Learning Techniques</div>
        <div class="sub-title">
            Digital Image Processing & Zero-Reference Deep Curve Estimation (Zero-DCE) Project.<br>
            Restores luminance, dynamic contrast, and structural texture from underexposed photographs 
            through iterative deep curve estimation.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Sidebar: Configuration, Model Status, and Tuning
# ============================================================================
with st.sidebar:
    st.markdown("### ⚙️ System & Controls")

    # Enhancement Method & Model Status
    st.markdown("#### Model Information")
    st.markdown('<div class="badge-dl">Method: Zero-DCE (Deep Learning)</div>', unsafe_allow_html=True)
    st.write("")

    if model_status["is_pretrained"]:
        st.markdown(
            '<div class="badge-ok">✓ Pretrained Checkpoint: Epoch99.pth</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="badge-warn">⚠ Fallback Demonstration Mode</div>',
            unsafe_allow_html=True,
        )
        st.caption(model_status.get("message", "Pretrained weights not available."))

    st.caption(f"**Compute Device:** `{model_status['device'].upper()}`")
    st.divider()

    # Enhancement Parameters
    st.markdown("#### Interactive Tuning")
    brightness_factor = st.slider(
        "Brightness Multiplier (α):",
        min_value=0.2,
        max_value=2.0,
        value=1.0,
        step=0.05,
        help="Scales the deep curve parameter maps A_n. 1.0 represents standard Zero-DCE output.",
    )

    iterations = st.slider(
        "LE-Curve Iterations (n):",
        min_value=1,
        max_value=8,
        value=8,
        step=1,
        help="Number of recursive Light-Enhancement curve iterations applied (1 to 8).",
    )

    st.divider()
    st.caption("College Project Implementation | PyTorch, OpenCV, NumPy, Streamlit")


# ============================================================================
# Image Input Selection (Upload Widget + Benchmark Sample Selector)
# ============================================================================
col_mode1, col_mode2 = st.columns([1, 1])
with col_mode1:
    input_mode = st.radio(
        "Select Image Input Source:",
        ["📤 Upload an Image", "🖼️ Use Benchmark Sample Image"],
        horizontal=True,
    )

image_source = None
image_name = "image.jpg"

if input_mode == "📤 Upload an Image":
    uploaded_file = st.file_uploader(
        "Upload a low-light image (JPG, JPEG, PNG):",
        type=["jpg", "jpeg", "png"],
        help="Drag and drop or select an underexposed photo from your computer.",
    )
    if uploaded_file is not None:
        image_source = uploaded_file
        image_name = uploaded_file.name
else:
    sample_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if sample_files:
        selected_sample = st.selectbox(
            "Choose a preloaded low-light benchmark sample:",
            sample_files,
            index=0,
        )
        image_source = os.path.join(INPUT_DIR, selected_sample)
        image_name = selected_sample
    else:
        st.warning("No sample images found in `data/input/`. Please upload an image.")


# If no image is provided yet, show clean waiting message
if image_source is None:
    st.info("👆 Please upload a low-light photograph above or choose a benchmark sample to begin enhancement.")
    st.stop()


# ============================================================================
# Processing & Inference (With Progress Indicator & Error Handling)
# ============================================================================
progress_placeholder = st.empty()

try:
    with progress_placeholder.container():
        with st.spinner("⏳ Processing low-light image through Zero-DCE neural network..."):
            # 1. Read image with OpenCV (handles BGR -> RGB)
            original_img = load_image(image_source)
            h, w, _ = original_img.shape

            # 2. Run Zero-DCE inference (evaluation mode without gradients)
            enhanced_img, curve_params, inference_time = enhance_low_light(
                model=model,
                image_rgb=original_img,
                brightness_factor=brightness_factor,
                iterations=iterations,
                device=model_status["device"],
            )

            # 3. Calculate quantitative evaluation metrics
            orig_metrics = compute_all_metrics(original_img)
            enh_metrics = compute_all_metrics(enhanced_img)
            comparison_df = compare_metrics(original_img, enhanced_img)

            # 4. Save enhanced image to data/output/
            saved_path = save_enhanced_image(
                image_rgb=enhanced_img,
                output_dir=OUTPUT_DIR,
                base_filename=image_name,
                file_format="PNG",
            )

    progress_placeholder.empty()

except FileNotFoundError as fnf_err:
    progress_placeholder.empty()
    st.error(f"📁 File Error: {str(fnf_err)}")
    st.stop()
except ValueError as val_err:
    progress_placeholder.empty()
    st.error(f"⚠️ Image Format Error: {str(val_err)}")
    st.stop()
except Exception as e:
    progress_placeholder.empty()
    st.error(f"❌ Processing Error: Unexpected exception occurred: {str(e)}")
    st.stop()


# ============================================================================
# Status & Model Execution Summary
# ============================================================================
st.markdown("---")

col_summary1, col_summary2 = st.columns([2, 1])
with col_summary1:
    method_label = "Zero-Reference Deep Curve Estimation (Zero-DCE)" if model_status["is_pretrained"] else "Adaptive Fallback"
    st.markdown(
        f"**Method:** `{method_label}` | **Resolution:** `{w} × {h} px` | "
        f"**Execution Time:** `{inference_time * 1000:.1f} ms`"
    )
with col_summary2:
    st.caption(f"💾 Preserved locally at: `data/output/{os.path.basename(saved_path)}`")


# ============================================================================
# Side-by-Side Image Display (Original vs Enhanced)
# ============================================================================
col_img_orig, col_img_enh = st.columns(2)

with col_img_orig:
    st.markdown("### 📷 Original Low-Light Image")
    st.image(
        original_img,
        caption=f"Input: {image_name} ({w}×{h} px)",
        use_container_width=True,
    )

with col_img_enh:
    st.markdown(f"### 🌟 Enhanced Image (Zero-DCE)")
    st.image(
        enhanced_img,
        caption=f"Enhanced Output (α={brightness_factor}, n={iterations})",
        use_container_width=True,
    )

# Download Button Section directly below enhanced image
col_dl1, col_dl2, _ = st.columns([1, 1, 2])

with col_dl1:
    png_buffer = get_image_download_bytes(enhanced_img, file_format="PNG")
    st.download_button(
        label="⬇️ Download Enhanced (PNG)",
        data=png_buffer,
        file_name=f"enhanced_{os.path.splitext(image_name)[0]}.png",
        mime="image/png",
        use_container_width=True,
    )

with col_dl2:
    jpg_buffer = get_image_download_bytes(enhanced_img, file_format="JPEG")
    st.download_button(
        label="⬇️ Download Enhanced (JPG)",
        data=jpg_buffer,
        file_name=f"enhanced_{os.path.splitext(image_name)[0]}.jpg",
        mime="image/jpeg",
        use_container_width=True,
    )


# ============================================================================
# Quantitative Evaluation Metrics Section
# ============================================================================
st.markdown("---")
st.markdown("### 📊 Quantitative Evaluation Metrics")

# Metric KPI Cards: Mean Intensity, Standard Deviation, Shannon Entropy
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    diff_bright = enh_metrics["mean_brightness"] - orig_metrics["mean_brightness"]
    pct_bright = (diff_bright / max(orig_metrics["mean_brightness"], 1e-5)) * 100
    st.metric(
        label="Mean Intensity (μ)",
        value=f"{enh_metrics['mean_brightness']:.1f} / 255",
        delta=f"{diff_bright:+.1f} ({pct_bright:+.0f}%)",
    )

with kpi2:
    diff_std = enh_metrics["std_intensity"] - orig_metrics["std_intensity"]
    pct_std = (diff_std / max(orig_metrics["std_intensity"], 1e-5)) * 100
    st.metric(
        label="Standard Deviation / Contrast (σ)",
        value=f"{enh_metrics['std_intensity']:.1f}",
        delta=f"{diff_std:+.1f} ({pct_std:+.0f}%)",
    )

with kpi3:
    diff_entropy = enh_metrics["entropy"] - orig_metrics["entropy"]
    pct_entropy = (diff_entropy / max(orig_metrics["entropy"], 1e-5)) * 100
    st.metric(
        label="Shannon Entropy (H)",
        value=f"{enh_metrics['entropy']:.3f} bits",
        delta=f"{diff_entropy:+.3f} bits ({pct_entropy:+.0f}%)",
    )

with kpi4:
    diff_under = enh_metrics["underexposed_percentage"] - orig_metrics["underexposed_percentage"]
    st.metric(
        label="Crushed Shadows (<25)",
        value=f"{enh_metrics['underexposed_percentage']:.1f}%",
        delta=f"{diff_under:+.1f}%",
        delta_color="inverse",
    )

# Metrics comparison table
with st.expander("📋 View Detailed Metrics Comparison Table", expanded=True):
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)


# ============================================================================
# Grayscale & Brightness Histogram Comparisons
# ============================================================================
st.markdown("---")
st.markdown("### 🔬 Grayscale & Histogram Diagnostics")

# 1. Grayscale Comparison Side by Side
col_gray1, col_gray2 = st.columns(2)
orig_gray, enh_gray = generate_grayscale_comparison(original_img, enhanced_img)

with col_gray1:
    st.markdown("#### Original Grayscale Luminance")
    st.image(
        orig_gray,
        caption="ITU-R BT.601 Grayscale (Y = 0.299R + 0.587G + 0.114B)",
        use_container_width=True,
    )

with col_gray2:
    st.markdown("#### Enhanced Grayscale Luminance")
    st.image(
        enh_gray,
        caption="Restored Grayscale Luminance Distribution",
        use_container_width=True,
    )

# 2. Brightness & Color Histogram Figure (Matplotlib)
st.markdown("#### Brightness & RGB Channel Histogram Analysis")
fig_hist = generate_histogram_comparison_figure(original_img, enhanced_img)
st.pyplot(fig_hist)


# ============================================================================
# Deep Curve Parameter Maps (Collapsible Diagnostic)
# ============================================================================
if curve_params is not None:
    with st.expander("🔬 View Learned Zero-DCE Curve Parameter Maps (A_1 to A_8)"):
        st.write(
            "Below are the 8 pixel-wise curve parameter maps estimated by DCE-Net. "
            "Cool blue tones indicate negative parameter values (active dynamic brightening)."
        )
        fig_curves = visualize_curve_parameters_figure(curve_params)
        st.pyplot(fig_curves)
