"""
Digital Image Processing and Deep Learning Enhancement Engine
Project: Low-Light Image Enhancement Using Deep Learning Techniques
Architecture: Zero-Reference Deep Curve Estimation (Zero-DCE)
"""

import os
import io
import time
from datetime import datetime
import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model import ZeroDCE, load_zero_dce_model, load_pretrained_weights, inference


# ============================================================================
# Image Loading & Preprocessing with OpenCV
# ============================================================================

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def load_image(source):
    """
    Reads an input image using OpenCV, validates format, and converts BGR to RGB.
    Preserves original image dimensions.
    
    Args:
        source: File path (str/Path), file-like buffer (Streamlit UploadedFile / BytesIO),
                or existing NumPy array.
                
    Returns:
        np.ndarray: Original image in uint8 [0, 255] RGB format of shape (H, W, 3).
        
    Raises:
        FileNotFoundError: If input path does not exist.
        ValueError: If file is corrupted, empty, or an unsupported image format.
        TypeError: If input source type is invalid.
    """
    if source is None:
        raise ValueError("Image source cannot be None.")

    img_bgr = None

    try:
        # Case 1: File path on disk
        if isinstance(source, (str, os.PathLike)):
            file_path = str(source)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Image file does not exist: '{file_path}'")

            ext = os.path.splitext(file_path)[1].lower()
            if ext and ext not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported image extension '{ext}'. Supported formats: {sorted(list(SUPPORTED_EXTENSIONS))}"
                )

            # Read image using OpenCV (loads as BGR)
            img_bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError(f"OpenCV could not decode image from '{file_path}'. File may be corrupted.")

        # Case 2: Buffer / Streamlit UploadedFile / BytesIO
        elif hasattr(source, "read"):
            data = source.read()
            if hasattr(source, "seek"):
                source.seek(0)  # Reset buffer pointer for future reads

            if not data or len(data) == 0:
                raise ValueError("Uploaded file buffer is empty (0 bytes).")

            nparr = np.frombuffer(data, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("OpenCV failed to decode image buffer. Format may be corrupted or unsupported.")

        # Case 3: Already a NumPy array
        elif isinstance(source, np.ndarray):
            if source.ndim == 2:
                # Grayscale to RGB
                return cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
            elif source.ndim == 3 and source.shape[2] == 3:
                # Assume already RGB if passed directly as 3-channel array
                return source.astype(np.uint8)
            else:
                raise ValueError(f"Unsupported NumPy array shape: {source.shape}. Expected (H, W, 3).")

        # Case 4: PIL Image
        elif isinstance(source, Image.Image):
            rgb_pil = source.convert("RGB")
            return np.array(rgb_pil, dtype=np.uint8)

        else:
            raise TypeError(f"Unsupported image source type: {type(source)}")

    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError, TypeError)):
            raise
        raise ValueError(f"Failed to load image: {str(e)}")

    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Decoded image is empty or invalid.")

    # Convert BGR (OpenCV default) to RGB correctly
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def preprocess_tensor(image_rgb, device=None):
    """
    Converts RGB NumPy array into a normalized PyTorch tensor.
    - Normalizes pixel values to [0.0, 1.0].
    - Converts shape from (H, W, 3) to (1, 3, H, W).
    - Preserves exact original spatial dimensions (H, W).
    
    Args:
        image_rgb (np.ndarray): Input RGB array of shape (H, W, 3) in uint8 [0, 255].
        device (torch.device, optional): Target compute device.
        
    Returns:
        torch.Tensor: Normalized tensor of shape (1, 3, H, W) in [0.0, 1.0].
    """
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image_rgb)}")

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected image with shape (H, W, 3), got {image_rgb.shape}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Normalize pixel values to [0.0, 1.0] and permute to (1, 3, H, W)
    tensor = torch.from_numpy(image_rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def tensor_to_rgb(tensor):
    """
    Converts an enhanced PyTorch tensor back to an RGB NumPy array.
    Strictly clamps values to [0, 255] in uint8.
    
    Args:
        tensor (torch.Tensor): Shape (1, 3, H, W) or (3, H, W) in [0.0, 1.0].
        
    Returns:
        np.ndarray: RGB image array of shape (H, W, 3) in uint8 [0, 255].
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # Permute from (3, H, W) back to (H, W, 3)
    np_img = tensor.permute(1, 2, 0).detach().cpu().numpy()
    np_img = np.clip(np_img * 255.0, 0.0, 255.0).astype(np.uint8)
    return np_img


# ============================================================================
# Labeled Fallback Enhancement (When Pretrained Weights Are Unavailable)
# ============================================================================

def enhance_labeled_fallback(image_rgb, clip_limit=2.5, tile_grid_size=(8, 8)):
    """
    Explicitly labeled classical fallback using Contrast Limited Adaptive
    Histogram Equalization (CLAHE) on the L-channel in LAB space.
    
    Used when deep learning weights are unavailable to prevent misleading claims.
    
    Args:
        image_rgb (np.ndarray): Input RGB image.
        clip_limit (float): CLAHE contrast threshold.
        tile_grid_size (tuple): Equalization grid size.
        
    Returns:
        np.ndarray: Enhanced RGB image preserving exact original dimensions.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)

    enhanced_lab = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def enhance_clahe_baseline(image_rgb, clip_limit=2.5, tile_grid_size=(8, 8)):
    """
    Alias for classical CLAHE baseline comparison.
    """
    return enhance_labeled_fallback(image_rgb, clip_limit, tile_grid_size)


# ============================================================================
# Zero-DCE Inference & Enhancement Engine
# ============================================================================

def enhance_low_light(model, image_rgb, brightness_factor=1.0, iterations=8, device=None):
    """
    Enhances a low-light RGB image using Zero-DCE deep learning architecture.
    Runs inference without gradients and preserves original image dimensions.
    
    If model weights are untrained (not pretrained), automatically branches to an
    explicitly labeled fallback to avoid falsely presenting unweighted outputs as deep learning.
    
    Args:
        model (ZeroDCE): Instantiated Zero-DCE model.
        image_rgb (np.ndarray): Input RGB image array of shape (H, W, 3).
        brightness_factor (float): Multiplier for curve parameters (default: 1.0).
        iterations (int): Number of LE-curve iterations (1 to 8, default: 8).
        device (torch.device, optional): Device for tensor computation.
        
    Returns:
        tuple: (enhanced_rgb, curve_params_np, inference_time_sec)
            - enhanced_rgb (np.ndarray): Enhanced RGB image (H, W, 3) in uint8.
            - curve_params_np (np.ndarray or None): Estimated parameter maps (24, H, W).
            - inference_time_sec (float): Execution duration in seconds.
    """
    orig_h, orig_w = image_rgb.shape[:2]

    # Check whether the model has genuine pretrained weights
    is_model_pretrained = getattr(model, "is_pretrained", False)

    if not is_model_pretrained:
        # Explicit fallback execution
        start_time = time.time()
        enhanced_rgb = enhance_labeled_fallback(image_rgb)
        inf_time = time.time() - start_time
        return enhanced_rgb, None, inf_time

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    # 1. Normalize image pixels to [0, 1] and convert into PyTorch tensor
    input_tensor = preprocess_tensor(image_rgb, device=device)

    # 2. Run inference in evaluation mode without gradients
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        # Forward pass predicting 24 curve parameter maps
        _, curve_params = model(input_tensor)

        # Apply iterative nonlinear enhancement curves
        enhanced_tensor = model.apply_curves(
            input_tensor,
            curve_params,
            brightness_factor=brightness_factor,
            iterations=iterations,
        )

    inf_time = time.time() - start_time

    # 3. Convert enhanced tensor back to RGB NumPy array
    enhanced_rgb = tensor_to_rgb(enhanced_tensor)

    # 4. Verify original dimensions are strictly preserved
    assert enhanced_rgb.shape[:2] == (orig_h, orig_w), (
        f"Dimension mismatch: expected ({orig_h}, {orig_w}), got {enhanced_rgb.shape[:2]}"
    )

    curve_params_np = curve_params.squeeze(0).cpu().numpy()
    return enhanced_rgb, curve_params_np, inf_time


def enhance_image_pipeline(
    image_source,
    model=None,
    checkpoint_path=None,
    brightness_factor=1.0,
    iterations=8,
    save_output=True,
    output_dir=None,
    device=None,
):
    """
    High-level end-to-end pipeline:
    - Reads input image with OpenCV.
    - Converts BGR to RGB.
    - Loads Zero-DCE model and weights if not provided.
    - Runs inference without gradients.
    - Preserves dimensions.
    - Converts to BGR only when saving to data/output/ with OpenCV.
    - Returns original image and enhanced image for display in Streamlit.
    
    Args:
        image_source: File path, buffer, or NumPy array.
        model (ZeroDCE, optional): Pre-loaded model.
        checkpoint_path (str, optional): Path to .pth checkpoint.
        brightness_factor (float): Brightness scaling factor (default: 1.0).
        iterations (int): LE-curve iterations (1 to 8, default: 8).
        save_output (bool): Whether to save to data/output/.
        output_dir (str, optional): Destination folder.
        device (torch.device, optional): Compute device.
        
    Returns:
        tuple: (original_rgb, enhanced_rgb, info_dict)
            - original_rgb (np.ndarray): Original RGB image.
            - enhanced_rgb (np.ndarray): Enhanced RGB image.
            - info_dict (dict): Status metadata (method, saved_path, inference_time_ms, is_pretrained).
    """
    # 1. Read input image using OpenCV and convert BGR to RGB
    original_rgb = load_image(image_source)

    # 2. Initialize or verify model
    if model is None:
        model, model_status = load_zero_dce_model(weights_path=checkpoint_path, device=device)
    else:
        model_status = {
            "is_pretrained": getattr(model, "is_pretrained", False),
            "message": "User-supplied model instance."
        }

    # 3. Enhance image
    enhanced_rgb, curve_params, inf_time = enhance_low_light(
        model=model,
        image_rgb=original_rgb,
        brightness_factor=brightness_factor,
        iterations=iterations,
        device=device,
    )

    saved_path = None
    if save_output:
        if output_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(base_dir, "data", "output")
        saved_path = save_enhanced_image(enhanced_rgb, output_dir=output_dir)

    is_pretrained = getattr(model, "is_pretrained", False)
    method_name = "Zero-DCE (Deep Learning)" if is_pretrained else "Adaptive CLAHE (Traditional DIP Fallback)"

    info_dict = {
        "method": method_name,
        "is_pretrained": is_pretrained,
        "inference_time_ms": inf_time * 1000.0,
        "saved_path": saved_path,
        "curve_params": curve_params,
        "original_shape": original_rgb.shape,
        "enhanced_shape": enhanced_rgb.shape,
    }

    return original_rgb, enhanced_rgb, info_dict


# ============================================================================
# Saving & Exporting with OpenCV (Converts RGB to BGR)
# ============================================================================

def save_enhanced_image(image_rgb, output_dir=None, base_filename="enhanced_image", file_format="PNG"):
    """
    Saves the enhanced RGB image using OpenCV.
    Converts RGB to BGR strictly before saving with cv2.imwrite.
    
    Args:
        image_rgb (np.ndarray): Enhanced RGB image in uint8.
        output_dir (str, optional): Target folder (defaults to data/output/).
        base_filename (str): Base filename identifier.
        file_format (str): 'PNG', 'JPG', or 'JPEG'.
        
    Returns:
        str: Absolute path of the saved image file.
    """
    if output_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "data", "output")

    os.makedirs(output_dir, exist_ok=True)

    clean_base = os.path.splitext(os.path.basename(base_filename))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = file_format.lower()
    if ext == "jpeg":
        ext = "jpg"

    filename = f"{clean_base}_{timestamp}.{ext}"
    out_path = os.path.join(output_dir, filename)

    # Convert RGB to BGR only when saving with OpenCV
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    success = cv2.imwrite(out_path, img_bgr)
    if not success:
        raise IOError(f"OpenCV failed to save image to '{out_path}'.")

    return out_path


def get_image_download_bytes(image_rgb, file_format="PNG"):
    """
    Encodes RGB image to in-memory bytes for Streamlit st.download_button.
    Uses OpenCV imencode for speed and fidelity.
    """
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ext = ".jpg" if file_format.upper() in ["JPG", "JPEG"] else ".png"
    success, buffer = cv2.imencode(ext, img_bgr)
    if not success:
        raise ValueError("Failed to encode image to bytes.")
    return io.BytesIO(buffer.tobytes())


# ============================================================================
# Grayscale, Histograms & Diagnostics for Streamlit
# ============================================================================

def generate_grayscale_comparison(original_rgb, enhanced_rgb):
    """
    Converts both original and enhanced images to ITU-R BT.601 grayscale for DIP visual inspection.
    """
    orig_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    enh_gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)
    return orig_gray, enh_gray


def generate_histogram_comparison_figure(original_rgb, enhanced_rgb):
    """
    Creates a publication-quality Matplotlib figure comparing histograms before and after enhancement.
    
    Plots:
        1. RGB Color Histograms (Original vs Enhanced)
        2. Grayscale Luminance Intensity Distribution
        3. Cumulative Distribution Function (CDF) Analysis
    """
    plt.style.use("default")
    fig = plt.figure(figsize=(14, 8), dpi=150)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.25)

    colors = ("#e74c3c", "#2ecc71", "#3498db")
    labels = ("Red Channel", "Green Channel", "Blue Channel")

    # 1. Original RGB Channel Histograms
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (col, lbl) in enumerate(zip(colors, labels)):
        hist = cv2.calcHist([original_rgb], [i], None, [256], [0, 256])
        ax1.plot(hist, color=col, alpha=0.85, linewidth=1.5, label=lbl)
    ax1.set_title("Original Image: RGB Channel Histogram", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Pixel Intensity [0 - 255]", fontsize=9)
    ax1.set_ylabel("Pixel Frequency", fontsize=9)
    ax1.set_xlim([0, 255])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8)

    # 2. Enhanced RGB Channel Histograms
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (col, lbl) in enumerate(zip(colors, labels)):
        hist = cv2.calcHist([enhanced_rgb], [i], None, [256], [0, 256])
        ax2.plot(hist, color=col, alpha=0.85, linewidth=1.5, label=lbl)
    ax2.set_title("Zero-DCE Enhanced: RGB Channel Histogram", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Pixel Intensity [0 - 255]", fontsize=9)
    ax2.set_ylabel("Pixel Frequency", fontsize=9)
    ax2.set_xlim([0, 255])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)

    # 3. Grayscale Luminance Comparison
    orig_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    enh_gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)

    ax3 = fig.add_subplot(gs[1, 0])
    hist_orig_g = cv2.calcHist([orig_gray], [0], None, [256], [0, 256])
    hist_enh_g = cv2.calcHist([enh_gray], [0], None, [256], [0, 256])

    ax3.fill_between(range(256), hist_orig_g.ravel(), color="#34495e", alpha=0.5, label="Original (Low-Light)")
    ax3.fill_between(range(256), hist_enh_g.ravel(), color="#f39c12", alpha=0.5, label="Enhanced (Zero-DCE)")
    ax3.set_title("Grayscale Luminance Distribution (Overlaid)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Luminance Intensity [0 - 255]", fontsize=9)
    ax3.set_ylabel("Pixel Count", fontsize=9)
    ax3.set_xlim([0, 255])
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=8)

    # 4. Cumulative Distribution Function (CDF)
    ax4 = fig.add_subplot(gs[1, 1])
    cdf_orig = hist_orig_g.cumsum() / hist_orig_g.sum()
    cdf_enh = hist_enh_g.cumsum() / hist_enh_g.sum()

    ax4.plot(cdf_orig, color="#34495e", linewidth=2.0, label="Original CDF (Steep in shadows)")
    ax4.plot(cdf_enh, color="#27ae60", linewidth=2.0, label="Enhanced CDF (Balanced spread)")
    ax4.set_title("Cumulative Distribution Function (CDF)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Luminance Level [0 - 255]", fontsize=9)
    ax4.set_ylabel("Cumulative Probability", fontsize=9)
    ax4.set_xlim([0, 255])
    ax4.set_ylim([0, 1.05])
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="lower right", fontsize=8)

    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95, hspace=0.35, wspace=0.25)
    return fig


def visualize_curve_parameters_figure(curve_params_np):
    """
    Renders the learned curve parameter maps across the 8 Zero-DCE iterations.
    Each map illustrates where the network assigned positive/negative curve weights.
    
    Args:
        curve_params_np (np.ndarray): Array of shape (24, H, W).
        
    Returns:
        matplotlib.figure.Figure: Rendered 2x4 grid figure of parameter maps.
    """
    if curve_params_np is None:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Parameter maps not available (Traditional Fallback Mode)",
                horizontalalignment="center", verticalalignment="center")
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), dpi=130)
    fig.suptitle("Zero-DCE Estimated Curve Parameter Maps (Iterations 1 through 8)", fontsize=12, fontweight="bold")

    for i in range(8):
        row = i // 4
        col = i % 4
        ax = axes[row, col]

        # Average across the 3 color channels for iteration i
        iter_map = np.mean(curve_params_np[i * 3 : (i + 1) * 3, :, :], axis=0)
        im = ax.imshow(iter_map, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_title(f"Iteration {i + 1} ($A_{i + 1}$)", fontsize=10)
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.046, pad=0.1)
    cbar.set_label("Curve Parameter Weight $A_n(x)$  [-1.0: Maximum Brightening  <--->  +1.0: Darkening]", fontsize=9)

    return fig
