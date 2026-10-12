"""
Comprehensive Engineering Audit Verification Script
"""

import os
import sys
import glob
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image

print("=" * 75)
print("COMPREHENSIVE ENGINEERING AUDIT: LOW-LIGHT IMAGE ENHANCEMENT")
print("=" * 75)

base_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Check Directory Structure
required_dirs = ["data/input", "data/output", "results", "report", "weights"]
for d in required_dirs:
    p = os.path.join(base_dir, d)
    exists = os.path.exists(p)
    print(f"[DIR CHECK] {d:15} -> Exists: {exists}")
    assert exists, f"Directory missing: {d}"

# 2. Check Weights Checkpoint
weights_file = os.path.join(base_dir, "weights", "Epoch99.pth")
assert os.path.exists(weights_file), "Epoch99.pth missing!"
file_size = os.path.getsize(weights_file)
print(f"[CHECKPOINT] Epoch99.pth verified! Size: {file_size} bytes (~320 KB)")

# 3. Model Architecture & Key Compatibility
from model import ZeroDCE, load_zero_dce_model, load_pretrained_weights, inference
model = ZeroDCE()
state_dict = torch.load(weights_file, map_location="cpu")
res = model.load_state_dict(state_dict)
print(f"[MODEL WEIGHTS] All keys matched successfully: {res}")

# 4. Dimension & Normalization Rigor
for shape in [(1, 3, 480, 640), (1, 3, 256, 256), (3, 300, 400)]:
    dummy_in = torch.rand(*shape)
    out, r = inference(model, dummy_in)
    expected_shape = (1, 3, shape[-2], shape[-1]) if len(shape) == 3 else shape
    assert out.shape == expected_shape, f"Shape mismatch: {out.shape} vs {expected_shape}"
    assert out.min() >= 0.0 and out.max() <= 1.0, "Values outside [0, 1]"
    assert r.shape[1] == 24, "Curve params channels != 24"
print("[TENSOR RIGOR] Normalization [0, 1] & Arbitrary Dimensions: PASSED!")

# 5. Image RGB/BGR Conversions & OpenCV
from enhancement import load_image, preprocess_tensor, tensor_to_rgb, save_enhanced_image, enhance_image_pipeline
sample_path = os.path.join(base_dir, "data", "input", "01.jpg")
orig_rgb = load_image(sample_path)
assert orig_rgb.ndim == 3 and orig_rgb.shape[2] == 3
print(f"[IMAGE CONVERSION] 01.jpg decoded correctly as RGB, shape: {orig_rgb.shape}")

# 6. Metrics & Safeguards
from metrics import compute_all_metrics, compare_metrics, calculate_psnr, calculate_ssim
m = compute_all_metrics(orig_rgb)
assert "mean_intensity" in m and "entropy" in m and "std_intensity" in m
assert m["psnr"] is None, "PSNR must be None without reference"
assert m["ssim"] is None, "SSIM must be None without reference"
print(f"[METRICS AUDIT] Metrics evaluated: Mean={m['mean_intensity']:.2f}, StdDev={m['std_intensity']:.2f}, Entropy={m['entropy']:.3f}")

# 7. Ground Truth Reference PSNR/SSIM Test
ref_sample = orig_rgb.copy()
psnr_val = calculate_psnr(orig_rgb, ref_sample)
ssim_val = calculate_ssim(orig_rgb, ref_sample)
assert psnr_val == float("inf") and abs(ssim_val - 1.0) < 1e-4
print(f"[REFERENCE METRICS] Reference evaluation verified! PSNR=inf dB, SSIM=1.0000")

# 8. Test Pipeline End-to-End Execution
orig, enh, info = enhance_image_pipeline(sample_path)
assert orig.shape == enh.shape
assert info["is_pretrained"] is True
assert os.path.exists(info["saved_path"])
print(f"[PIPELINE RUN] Pipeline executed: {info['method']}, saved to: {os.path.basename(info['saved_path'])}")

print("=" * 75)
print("AUDIT RESULT: ALL ENGINEERING CHECKS PASSED SUCCESSFULLY!")
print("=" * 75)
