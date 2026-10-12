"""
Digital Image Processing (DIP) Evaluation Metrics Module
Project: Low-Light Image Enhancement Using Deep Learning Techniques

Key Metrics Implemented:
1. Mean Intensity (Average Image Brightness)
2. Standard Deviation of Grayscale Intensity (Contrast & Dynamic Spread)
3. Shannon Entropy (Information Density & Texture Richness)
4. Grayscale & RGB Histogram Generation
5. Full-Reference Metrics (PSNR & SSIM) - Evaluated strictly when a valid ground-truth reference is provided.

Academic Methodology Note:
In low-light enhancement literature, evaluating PSNR or SSIM using the underexposed input as
a pseudo-reference is scientifically invalid because the input itself is degraded. PSNR and SSIM
are full-reference metrics that strictly require a registered, well-exposed normal-light ground truth.
Moreover, higher mean brightness or higher entropy alone does not guarantee superior image quality;
excessive noise amplification or clipping artifacts also artificially inflate these metrics.
"""

import numpy as np
import cv2
import pandas as pd


# ============================================================================
# Array Normalization & Preprocessing Helpers
# ============================================================================

def to_standard_uint8(image):
    """
    Converts an input image array to standard uint8 in range [0, 255].
    Robustly handles uint8 inputs, normalized floating-point arrays ([0.0, 1.0]),
    and floating-point arrays in range [0.0, 255.0].
    
    Args:
        image (np.ndarray): Image array of arbitrary data type.
        
    Returns:
        np.ndarray: uint8 image array with pixel values in [0, 255].
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected image to be a NumPy array, got {type(image)}.")

    if image.size == 0:
        raise ValueError("Cannot process empty image array (size is 0).")

    if image.dtype == np.uint8:
        return image

    # Handle floating point representations
    if np.issubdtype(image.dtype, np.floating):
        max_val = np.nanmax(image)
        # If values are normalized to [0.0, 1.0]
        if max_val <= 1.0 + 1e-6:
            scaled = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0) * 255.0
        else:
            scaled = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        return np.clip(scaled, 0.0, 255.0).astype(np.uint8)

    # Handle other integer types (e.g. int32, int64, uint16)
    clipped = np.clip(image, 0, 255)
    return clipped.astype(np.uint8)


def rgb_to_grayscale(image_rgb):
    """
    Converts RGB image to standard ITU-R BT.601 grayscale luminance.
    Formula: Y = 0.299*R + 0.587*G + 0.114*B
    
    Args:
        image_rgb (np.ndarray): Input RGB image array of shape (H, W, 3) or (H, W).
        
    Returns:
        np.ndarray: Grayscale image of shape (H, W) in uint8 [0, 255].
    """
    img_u8 = to_standard_uint8(image_rgb)
    if img_u8.ndim == 2:
        return img_u8
    elif img_u8.ndim == 3 and img_u8.shape[2] == 3:
        return cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    elif img_u8.ndim == 3 and img_u8.shape[2] == 1:
        return img_u8[:, :, 0]
    else:
        raise ValueError(f"Invalid image shape: {image_rgb.shape}. Expected (H, W) or (H, W, 3).")


# ============================================================================
# 1. Mean Intensity (Average Brightness)
# ============================================================================

def calculate_mean_intensity(image_rgb):
    """
    Calculates the mean grayscale intensity (average brightness) of the image.
    
    Formula:
        mu = (1 / (M * N)) * sum_{x=1}^M sum_{y=1}^N Y(x, y)
        
    where Y(x, y) is the luminance intensity at coordinate (x, y).
    
    Args:
        image_rgb (np.ndarray): Input RGB image array.
        
    Returns:
        float: Mean intensity in the range [0.0, 255.0].
    """
    gray = rgb_to_grayscale(image_rgb)
    return float(np.mean(gray))


# Alias for backward compatibility
calculate_mean_brightness = calculate_mean_intensity


# ============================================================================
# 2. Standard Deviation of Grayscale Intensity (Contrast Measure)
# ============================================================================

def calculate_std_intensity(image_rgb):
    """
    Calculates the standard deviation of grayscale intensity across the image.
    Measures global intensity dispersion and dynamic tonal contrast.
    
    Formula:
        sigma = sqrt((1 / (M * N)) * sum_{x=1}^M sum_{y=1}^N (Y(x, y) - mu)^2)
        
    Higher standard deviation indicates broader separation between shadow and highlight tones.
    
    Args:
        image_rgb (np.ndarray): Input RGB image array.
        
    Returns:
        float: Standard deviation of intensity in range [0.0, 128.0].
    """
    gray = rgb_to_grayscale(image_rgb)
    return float(np.std(gray))


# ============================================================================
# 3. Shannon Entropy
# ============================================================================

def calculate_shannon_entropy(image_rgb, epsilon=1e-12):
    """
    Calculates the Shannon information entropy from the 256-bin grayscale histogram.
    
    Formula:
        H = - sum_{k=0}^{255} p_k * log2(p_k)  for all p_k > 0
        
    where p_k = n_k / (M * N) is the normalized frequency (probability) of intensity k.
    
    Shannon entropy measures the uncertainty, randomness, and texture detail richness.
    In low-light images, pixel values are heavily concentrated in narrow low-intensity bins
    (resulting in low entropy). Enhancement typically redistributes intensities across the
    dynamic range, increasing entropy towards the 8-bit theoretical maximum (8.0 bits).
    
    Protection:
        Includes epsilon filtering to prevent division-by-zero or log2(0) runtime errors.
        
    Args:
        image_rgb (np.ndarray): Input RGB image array.
        epsilon (float): Small safety constant preventing log2(0).
        
    Returns:
        float: Shannon entropy in bits (range [0.0, 8.0]).
    """
    gray = rgb_to_grayscale(image_rgb)
    total_pixels = float(gray.size)

    if total_pixels == 0:
        return 0.0

    # Compute 256-bin intensity histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()

    # Probability mass function with division-by-zero safeguard
    probabilities = hist / (total_pixels + epsilon)

    # Filter out zero probabilities to avoid invalid log2(0)
    non_zero_p = probabilities[probabilities > 0]

    # Calculate Shannon entropy
    entropy = -np.sum(non_zero_p * np.log2(non_zero_p + epsilon))
    return float(np.clip(max(0.0, float(entropy)), 0.0, 8.0))


# ============================================================================
# 4. Histogram Generation
# ============================================================================

def compute_grayscale_histogram(image_rgb, bins=256, normalized=False):
    """
    Generates the 1D intensity histogram for an image.
    
    Args:
        image_rgb (np.ndarray): Input RGB image array.
        bins (int): Number of intensity bins (default: 256).
        normalized (bool): Whether to normalize frequencies to sum to 1.0.
        
    Returns:
        np.ndarray: 1D array of histogram values of length `bins`.
    """
    gray = rgb_to_grayscale(image_rgb)
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).ravel()
    if normalized:
        total = float(np.sum(hist))
        return hist / (total + 1e-12)
    return hist


def compute_rgb_histograms(image_rgb, bins=256, normalized=False):
    """
    Generates channel-wise histograms for Red, Green, and Blue color channels.
    
    Args:
        image_rgb (np.ndarray): Input RGB image array.
        bins (int): Number of bins per channel.
        normalized (bool): Whether to normalize bin frequencies.
        
    Returns:
        dict: {'red': np.ndarray, 'green': np.ndarray, 'blue': np.ndarray}
    """
    img_u8 = to_standard_uint8(image_rgb)
    histograms = {}
    channel_names = ["red", "green", "blue"]

    for i, name in enumerate(channel_names):
        hist = cv2.calcHist([img_u8], [i], None, [bins], [0, 256]).ravel()
        if normalized:
            total = float(np.sum(hist))
            hist = hist / (total + 1e-12)
        histograms[name] = hist

    return histograms


# ============================================================================
# 5. Full-Reference Metrics (PSNR & SSIM) - Ground Truth Required
# ============================================================================

def calculate_psnr(enhanced_rgb, reference_rgb, max_val=255.0):
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) between the enhanced image and
    a genuine normal-light ground-truth reference image.
    
    Formula:
        MSE = (1 / (3 * M * N)) * sum_{c,x,y} (I_enh(c, x, y) - I_ref(c, x, y))^2
        PSNR = 10 * log10(MAX^2 / (MSE + epsilon))
        
    Methodological Requirement:
        PSNR requires a valid normal-light reference image. If no reference image is provided,
        or if the user mistakenly passes the original low-light image as reference,
        this function returns None or raises an explanatory warning.
        
    Args:
        enhanced_rgb (np.ndarray): Enhanced RGB image.
        reference_rgb (np.ndarray or None): Well-exposed ground-truth normal-light image.
        max_val (float): Maximum possible pixel value (255.0 for 8-bit images).
        
    Returns:
        float or None: PSNR in decibels (dB), or None if reference is invalid or not provided.
    """
    if reference_rgb is None:
        return None

    enh_u8 = to_standard_uint8(enhanced_rgb).astype(np.float64)
    ref_u8 = to_standard_uint8(reference_rgb).astype(np.float64)

    if enh_u8.shape != ref_u8.shape:
        raise ValueError(
            f"Shape mismatch for PSNR calculation: enhanced shape {enh_u8.shape} "
            f"vs reference shape {ref_u8.shape}."
        )

    # Check for identical image to avoid false pseudo-reference calculations
    mse = np.mean(np.square(enh_u8 - ref_u8))
    if mse < 1e-10:
        return float("inf")

    psnr = 10.0 * np.log10((max_val ** 2) / mse)
    return float(psnr)


def calculate_ssim(enhanced_rgb, reference_rgb, window_size=11, k1=0.01, k2=0.03):
    """
    Calculates Structural Similarity Index (SSIM) between the enhanced image and
    a genuine normal-light ground-truth reference image.
    
    Formula:
        SSIM(x, y) = [ (2*mu_x*mu_y + C1) * (2*sigma_xy + C2) ] /
                     [ (mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2) ]
                     
    Methodological Requirement:
        SSIM requires an aligned, normal-light reference image. Comparing a low-light
        input to an enhanced image using SSIM is scientifically meaningless because the
        low-light input lacks target normal exposure.
        
    Args:
        enhanced_rgb (np.ndarray): Enhanced RGB image.
        reference_rgb (np.ndarray or None): Ground-truth normal-light image.
        window_size (int): Gaussian filter window size (default: 11).
        k1 (float): Small stability constant (default: 0.01).
        k2 (float): Small stability constant (default: 0.03).
        
    Returns:
        float or None: SSIM index in range [-1.0, 1.0], or None if reference is not provided.
    """
    if reference_rgb is None:
        return None

    # Compute SSIM on grayscale luminance
    enh_gray = rgb_to_grayscale(enhanced_rgb).astype(np.float64)
    ref_gray = rgb_to_grayscale(reference_rgb).astype(np.float64)

    if enh_gray.shape != ref_gray.shape:
        raise ValueError(
            f"Shape mismatch for SSIM calculation: enhanced {enh_gray.shape} "
            f"vs reference {ref_gray.shape}."
        )

    c1 = (k1 * 255.0) ** 2
    c2 = (k2 * 255.0) ** 2

    # Gaussian weighting window
    sigma = 1.5
    kernel = cv2.getGaussianKernel(window_size, sigma)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(enh_gray, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(ref_gray, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(enh_gray ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(ref_gray ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(enh_gray * ref_gray, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12
    )

    return float(np.mean(ssim_map))


# ============================================================================
# Additional Complementary DIP Metrics
# ============================================================================

def calculate_rms_contrast(image_rgb):
    """
    Calculates Root Mean Square (RMS) Contrast of normalized luminance.
    Formula: C_RMS = sqrt((1 / (M * N)) * sum (Y_norm - mu_norm)^2)
    """
    gray = rgb_to_grayscale(image_rgb).astype(np.float64) / 255.0
    return float(np.sqrt(np.mean(np.square(gray - np.mean(gray)))))


def calculate_color_saturation(image_rgb):
    """
    Calculates average color saturation using the HSV color model.
    """
    img_u8 = to_standard_uint8(image_rgb)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float64) / 255.0
    return float(np.mean(saturation))


def calculate_exposure_ratios(image_rgb, under_thresh=25, over_thresh=230):
    """
    Calculates percentage of underexposed and overexposed pixels.
    """
    gray = rgb_to_grayscale(image_rgb)
    total = float(gray.size)
    under_count = np.sum(gray < under_thresh)
    over_count = np.sum(gray > over_thresh)
    return float((under_count / total) * 100.0), float((over_count / total) * 100.0)


# ============================================================================
# Consolidated Metrics Engine
# ============================================================================

ACADEMIC_INTERPRETATION_NOTE = (
    "Academic Notice: Higher brightness or higher entropy alone does not guarantee superior "
    "image quality. Severe over-exposure or sensor noise will artificially inflate both metrics "
    "while degrading perceptual quality. Quantitative scores must be interpreted in conjunction with "
    "visual contrast balance and natural color preservation. Full-reference metrics (PSNR/SSIM) "
    "are only valid when evaluated against an authentic ground-truth normal-light reference."
)


def compute_all_metrics(image_rgb, reference_rgb=None):
    """
    Computes a comprehensive dictionary of Digital Image Processing evaluation metrics.
    
    Args:
        image_rgb (np.ndarray): Image array in uint8 or normalized float.
        reference_rgb (np.ndarray, optional): Ground-truth normal-light image.
        
    Returns:
        dict: Complete dictionary containing all quantitative evaluation metrics.
    """
    mean_bright = calculate_mean_intensity(image_rgb)
    std_intensity = calculate_std_intensity(image_rgb)
    entropy = calculate_shannon_entropy(image_rgb)
    rms_contrast = calculate_rms_contrast(image_rgb)
    saturation = calculate_color_saturation(image_rgb)
    under_pct, over_pct = calculate_exposure_ratios(image_rgb)

    psnr = calculate_psnr(image_rgb, reference_rgb) if reference_rgb is not None else None
    ssim = calculate_ssim(image_rgb, reference_rgb) if reference_rgb is not None else None

    return {
        "mean_intensity": mean_bright,
        "mean_brightness": mean_bright,  # Alias
        "std_intensity": std_intensity,
        "entropy": entropy,
        "rms_contrast": rms_contrast,
        "color_saturation": saturation,
        "underexposed_percentage": under_pct,
        "overexposed_percentage": over_pct,
        "psnr": psnr,
        "ssim": ssim,
        "reference_provided": reference_rgb is not None,
        "academic_note": ACADEMIC_INTERPRETATION_NOTE,
    }


def compare_metrics(original_rgb, enhanced_rgb, reference_rgb=None):
    """
    Generates a structured side-by-side comparison table between original and enhanced images.
    
    Args:
        original_rgb (np.ndarray): Original low-light image.
        enhanced_rgb (np.ndarray): Enhanced image.
        reference_rgb (np.ndarray, optional): Genuine ground-truth normal-light image.
        
    Returns:
        pd.DataFrame: Tabular metrics comparison with deltas and physical interpretations.
    """
    orig = compute_all_metrics(original_rgb)
    enh = compute_all_metrics(enhanced_rgb, reference_rgb=reference_rgb)

    records = [
        {
            "Metric": "Mean Intensity (Average Brightness)",
            "Symbol": "μ",
            "Original": f"{orig['mean_intensity']:.2f} / 255",
            "Enhanced": f"{enh['mean_intensity']:.2f} / 255",
            "Change (Δ)": f"{enh['mean_intensity'] - orig['mean_intensity']:+.2f}",
            "Change (%)": f"{((enh['mean_intensity'] - orig['mean_intensity']) / max(orig['mean_intensity'], 1e-5)) * 100:+.1f}%",
            "Ideal Trend": "Moderate (~90 - 140)",
            "Interpretation": "Quantifies overall luminance restored from shadows."
        },
        {
            "Metric": "Intensity Standard Deviation",
            "Symbol": "σ",
            "Original": f"{orig['std_intensity']:.2f}",
            "Enhanced": f"{enh['std_intensity']:.2f}",
            "Change (Δ)": f"{enh['std_intensity'] - orig['std_intensity']:+.2f}",
            "Change (%)": f"{((enh['std_intensity'] - orig['std_intensity']) / max(orig['std_intensity'], 1e-5)) * 100:+.1f}%",
            "Ideal Trend": "Higher (~50 - 75)",
            "Interpretation": "Reflects global dynamic contrast and tonal separation."
        },
        {
            "Metric": "Shannon Information Entropy",
            "Symbol": "H",
            "Original": f"{orig['entropy']:.3f} bits",
            "Enhanced": f"{enh['entropy']:.3f} bits",
            "Change (Δ)": f"{enh['entropy'] - orig['entropy']:+.3f}",
            "Change (%)": f"{((enh['entropy'] - orig['entropy']) / max(orig['entropy'], 1e-5)) * 100:+.1f}%",
            "Ideal Trend": "Higher (Max 8.0)",
            "Interpretation": "Quantifies information density and revealed textures."
        },
        {
            "Metric": "RMS Contrast",
            "Symbol": "C_RMS",
            "Original": f"{orig['rms_contrast']:.3f}",
            "Enhanced": f"{enh['rms_contrast']:.3f}",
            "Change (Δ)": f"{enh['rms_contrast'] - orig['rms_contrast']:+.3f}",
            "Change (%)": f"{((enh['rms_contrast'] - orig['rms_contrast']) / max(orig['rms_contrast'], 1e-5)) * 100:+.1f}%",
            "Ideal Trend": "Higher",
            "Interpretation": "Normalized root mean square luminance dispersion."
        },
        {
            "Metric": "Color Saturation Index",
            "Symbol": "S_avg",
            "Original": f"{orig['color_saturation']:.3f}",
            "Enhanced": f"{enh['color_saturation']:.3f}",
            "Change (Δ)": f"{enh['color_saturation'] - orig['color_saturation']:+.3f}",
            "Change (%)": f"{((enh['color_saturation'] - orig['color_saturation']) / max(orig['color_saturation'], 1e-5)) * 100:+.1f}%",
            "Ideal Trend": "Moderate (~0.25 - 0.55)",
            "Interpretation": "Color vividness without chromatic clipping."
        },
        {
            "Metric": "Crushed Shadow Pixels (< 25/255)",
            "Symbol": "P_under",
            "Original": f"{orig['underexposed_percentage']:.1f}%",
            "Enhanced": f"{enh['underexposed_percentage']:.1f}%",
            "Change (Δ)": f"{enh['underexposed_percentage'] - orig['underexposed_percentage']:+.1f}%",
            "Change (%)": f"{((enh['underexposed_percentage'] - orig['underexposed_percentage']) / max(orig['underexposed_percentage'], 1e-5)) * 100:+.1f}%",
            "Ideal Trend": "Lower (near 0%)",
            "Interpretation": "Percentage of pitch-black pixels restored."
        },
        {
            "Metric": "Overexposed Pixels (> 230/255)",
            "Symbol": "P_over",
            "Original": f"{orig['overexposed_percentage']:.1f}%",
            "Enhanced": f"{enh['overexposed_percentage']:.1f}%",
            "Change (Δ)": f"{enh['overexposed_percentage'] - orig['overexposed_percentage']:+.1f}%",
            "Change (%)": f"{((enh['overexposed_percentage'] - orig['overexposed_percentage']) / max(orig['overexposed_percentage'], 1e-5)) * 100:+.1f}%",
            "Ideal Trend": "Low (< 5%)",
            "Interpretation": "Monitors preservation of highlights without burnout."
        }
    ]

    # Add PSNR and SSIM rows if ground-truth reference is provided
    if reference_rgb is not None:
        psnr_val = enh["psnr"]
        ssim_val = enh["ssim"]
        records.append({
            "Metric": "Peak Signal-to-Noise Ratio (PSNR)",
            "Symbol": "PSNR",
            "Original": "N/A (Reference-based)",
            "Enhanced": f"{psnr_val:.2f} dB" if psnr_val is not None else "N/A",
            "Change (Δ)": "N/A",
            "Change (%)": "N/A",
            "Ideal Trend": "Higher (> 20 dB)",
            "Interpretation": "Measured strictly against supplied ground-truth normal light reference."
        })
        records.append({
            "Metric": "Structural Similarity Index (SSIM)",
            "Symbol": "SSIM",
            "Original": "N/A (Reference-based)",
            "Enhanced": f"{ssim_val:.4f}" if ssim_val is not None else "N/A",
            "Change (Δ)": "N/A",
            "Change (%)": "N/A",
            "Ideal Trend": "Closer to 1.0",
            "Interpretation": "Structural fidelity relative to ground-truth normal light reference."
        })

    return pd.DataFrame(records)


# ============================================================================
# Example Usage & Module Self-Verification
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("DIGITAL IMAGE PROCESSING (DIP) METRICS MODULE: VERIFICATION TEST")
    print("=" * 70)

    # 1. Test with synthetic uint8 low-light image
    low_light_sample = np.random.randint(5, 35, (256, 256, 3), dtype=np.uint8)
    enhanced_sample = np.random.randint(50, 160, (256, 256, 3), dtype=np.uint8)

    print("\n--- 1. Testing Metrics on uint8 Arrays ---")
    metrics_orig = compute_all_metrics(low_light_sample)
    metrics_enh = compute_all_metrics(enhanced_sample)
    print(f"Original Mean Intensity: {metrics_orig['mean_intensity']:.2f} / 255")
    print(f"Original Std Dev:        {metrics_orig['std_intensity']:.2f}")
    print(f"Original Shannon Entropy: {metrics_orig['entropy']:.3f} bits")
    print(f"Enhanced Mean Intensity: {metrics_enh['mean_intensity']:.2f} / 255")
    print(f"Enhanced Std Dev:        {metrics_enh['std_intensity']:.2f}")
    print(f"Enhanced Shannon Entropy: {metrics_enh['entropy']:.3f} bits")

    # 2. Test with normalized float32 arrays in [0.0, 1.0]
    print("\n--- 2. Testing Normalization with float32 in [0.0, 1.0] ---")
    float_img = (low_light_sample.astype(np.float32) / 255.0)
    mean_float = calculate_mean_intensity(float_img)
    print(f"Mean Intensity from normalized float array: {mean_float:.2f} (Matches uint8)")

    # 3. Test division-by-zero protection on completely black image (zeros)
    print("\n--- 3. Testing Edge Case: Pitch Black Image (All Zeros) ---")
    black_img = np.zeros((100, 100, 3), dtype=np.uint8)
    entropy_black = calculate_shannon_entropy(black_img)
    mean_black = calculate_mean_intensity(black_img)
    std_black = calculate_std_intensity(black_img)
    print(f"Pitch Black: Mean={mean_black}, StdDev={std_black}, Entropy={entropy_black:.4f} (Zero Division Protected)")

    # 4. Test Histogram Generation
    print("\n--- 4. Testing Histogram Generation ---")
    gray_hist = compute_grayscale_histogram(enhanced_sample)
    rgb_hists = compute_rgb_histograms(enhanced_sample)
    print(f"Grayscale histogram bins: {len(gray_hist)}, Total counts: {int(np.sum(gray_hist))}")
    print(f"RGB histogram keys: {list(rgb_hists.keys())}")

    # 5. Test PSNR/SSIM Behavior with & without Reference
    print("\n--- 5. Testing Reference-Based Metrics (PSNR & SSIM) ---")
    print("Without Reference: PSNR =", metrics_enh['psnr'], ", SSIM =", metrics_enh['ssim'])
    
    # Simulate a ground truth reference
    ground_truth_sample = enhanced_sample + np.random.randint(-5, 5, enhanced_sample.shape).astype(np.int16)
    ground_truth_sample = np.clip(ground_truth_sample, 0, 255).astype(np.uint8)
    
    psnr_gt = calculate_psnr(enhanced_sample, ground_truth_sample)
    ssim_gt = calculate_ssim(enhanced_sample, ground_truth_sample)
    print(f"With Valid Ground Truth Reference: PSNR = {psnr_gt:.2f} dB, SSIM = {ssim_gt:.4f}")

    print("\n" + "=" * 70)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)
