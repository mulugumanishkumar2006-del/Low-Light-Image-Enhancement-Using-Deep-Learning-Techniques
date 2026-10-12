"""
Zero-Reference Deep Curve Estimation (Zero-DCE) Neural Network Architecture
Project: Low-Light Image Enhancement Using Deep Learning Techniques
Reference: Li et al., "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement", CVPR 2020.
"""

import os
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Zero-DCE Neural Network (DCE-Net)
# ============================================================================

class ZeroDCE(nn.Module):
    """
    Zero-Reference Deep Curve Estimation Network (DCE-Net).
    
    Architecture Overview:
        DCE-Net is a lightweight, fully convolutional network designed to estimate
        pixel-wise, higher-order Light-Enhancement (LE) curves without downsampling.
        
        The network consists of 7 convolutional layers with symmetrical skip connections:
        - Layer 1 (e_conv1): 3 -> 32 channels. Extracts initial low-level spatial features from RGB input.
        - Layer 2 (e_conv2): 32 -> 32 channels. Intermediate spatial feature extraction.
        - Layer 3 (e_conv3): 32 -> 32 channels. Deeper contextual representation.
        - Layer 4 (e_conv4): 32 -> 32 channels. Bottleneck feature representation.
        - Layer 5 (e_conv5): (32 + 32) -> 32 channels. Skip connection concatenating Layer 3 and Layer 4.
        - Layer 6 (e_conv6): (32 + 32) -> 32 channels. Skip connection concatenating Layer 2 and Layer 5.
        - Layer 7 (e_conv7): (32 + 32) -> 24 channels. Output layer concatenating Layer 1 and Layer 6.
          Activated by Tanh to bound curve parameters to [-1, 1].
          The 24 output channels correspond to 8 curve iterations x 3 color channels (R, G, B).
          
    Enhancement Equation:
        The iterative nonlinear Light-Enhancement (LE) curve is defined as:
            LE_n(x) = LE_{n-1}(x) + A_n(x) * [LE_{n-1}(x)^2 - LE_{n-1}(x)]
            
        where:
            - x is the input pixel normalized to [0, 1].
            - LE_0(x) = x is the initial low-light image.
            - A_n(x) in [-1, 1] is the learned curve parameter map for iteration n (n in 1..8).
            - Since (x^2 - x) <= 0 for all x in [0, 1], a negative parameter A_n < 0 guarantees
              monotonic brightening, while avoiding value clipping and preserving zero-slope
              stability at the boundaries (x=0 and x=1).
    """

    def __init__(self, number_f=32):
        super(ZeroDCE, self).__init__()
        
        # Tracks whether compatible pretrained weights have been loaded
        self.is_pretrained = False

        self.relu = nn.ReLU(inplace=True)

        # --------------------------------------------------------------------
        # Layer 1: Initial feature extraction
        # Input: 3 channels (RGB) -> Output: 32 feature maps
        # Kernel: 3x3, Stride: 1, Padding: 1 (preserves spatial resolution H x W)
        # --------------------------------------------------------------------
        self.e_conv1 = nn.Conv2d(3, number_f, kernel_size=3, stride=1, padding=1, bias=True)

        # --------------------------------------------------------------------
        # Layer 2: Shallow feature transformation
        # Input: 32 channels -> Output: 32 feature maps
        # --------------------------------------------------------------------
        self.e_conv2 = nn.Conv2d(number_f, number_f, kernel_size=3, stride=1, padding=1, bias=True)

        # --------------------------------------------------------------------
        # Layer 3: Mid-level contextual feature representation
        # Input: 32 channels -> Output: 32 feature maps
        # --------------------------------------------------------------------
        self.e_conv3 = nn.Conv2d(number_f, number_f, kernel_size=3, stride=1, padding=1, bias=True)

        # --------------------------------------------------------------------
        # Layer 4: Deep bottleneck feature representation
        # Input: 32 channels -> Output: 32 feature maps
        # --------------------------------------------------------------------
        self.e_conv4 = nn.Conv2d(number_f, number_f, kernel_size=3, stride=1, padding=1, bias=True)

        # --------------------------------------------------------------------
        # Layer 5: First symmetrical skip-connection
        # Concatenates output of Layer 3 and Layer 4 (32 + 32 = 64 channels)
        # Output: 32 feature maps
        # --------------------------------------------------------------------
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, kernel_size=3, stride=1, padding=1, bias=True)

        # --------------------------------------------------------------------
        # Layer 6: Second symmetrical skip-connection
        # Concatenates output of Layer 2 and Layer 5 (32 + 32 = 64 channels)
        # Output: 32 feature maps
        # --------------------------------------------------------------------
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, kernel_size=3, stride=1, padding=1, bias=True)

        # --------------------------------------------------------------------
        # Layer 7: Final parameter estimation layer with skip-connection
        # Concatenates output of Layer 1 and Layer 6 (32 + 32 = 64 channels)
        # Output: 24 channels (8 iterations x 3 RGB color channels)
        # Tanh activation bounds all parameters to the range [-1.0, 1.0]
        # --------------------------------------------------------------------
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        """
        Forward pass predicting 8 sets of curve parameters and applying the enhancement curves.
        
        Args:
            x (torch.Tensor): Input RGB tensor of shape (B, 3, H, W) normalized to [0, 1].
                             Supports arbitrary valid H and W dimensions.
                             
        Returns:
            enhanced_image (torch.Tensor): Enhanced image tensor (B, 3, H, W) clamped to [0, 1].
            curve_params (torch.Tensor): Predicted parameter maps (B, 24, H, W) in [-1, 1].
        """
        if x.dim() == 3:
            # Add batch dimension if single image (3, H, W)
            x = x.unsqueeze(0)

        # 1. Forward feature extraction through conv layers with skip connections
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))
        
        # 24 parameter maps bounded to [-1, 1] via tanh activation
        curve_params = torch.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))

        # 2. Split 24 channels into 8 sets of 3-channel parameter maps (r1 through r8)
        r_list = torch.split(curve_params, 3, dim=1)

        # 3. Apply iterative nonlinear enhancement curve repeatedly for all 8 parameter maps:
        #    enhanced = image + curve * (image * image - image)
        curr = x
        for r in r_list:
            curr = curr + r * (torch.pow(curr, 2) - curr)

        # 4. Strictly maintain pixel values within the range [0.0, 1.0]
        enhanced_image = torch.clamp(curr, 0.0, 1.0)

        return enhanced_image, curve_params

    def apply_curves(self, image, curve_params, brightness_factor=1.0, iterations=8):
        """
        Applies predicted curve parameter maps with custom brightness scaling and iteration depth.
        
        Enhancement Equation:
            curr = curr + (brightness_factor * r) * (curr^2 - curr)
            
        Args:
            image (torch.Tensor): Input tensor (B, 3, H, W) in [0, 1].
            curve_params (torch.Tensor): Estimated parameter maps (B, 24, H, W).
            brightness_factor (float): Multiplier for parameter maps (default: 1.0).
            iterations (int): Number of iterations to apply (1 to 8, default: 8).
            
        Returns:
            torch.Tensor: Enhanced image tensor clamped strictly to [0.0, 1.0].
        """
        r_list = torch.split(curve_params, 3, dim=1)
        curr = image
        num_iters = min(max(1, int(iterations)), len(r_list))

        for i in range(num_iters):
            r = r_list[i] * brightness_factor
            # Nonlinear enhancement curve equation
            curr = curr + r * (torch.pow(curr, 2) - curr)

        return torch.clamp(curr, 0.0, 1.0)

    def inference(self, image_tensor, brightness_factor=1.0, iterations=8):
        """
        Convenience method to run inference using evaluation mode and torch.no_grad().
        
        Args:
            image_tensor (torch.Tensor): Input tensor (B, 3, H, W) or (3, H, W) in [0, 1].
            brightness_factor (float): Optional brightness multiplier (default: 1.0).
            iterations (int): Number of curve iterations (1 to 8, default: 8).
            
        Returns:
            tuple: (enhanced_tensor, curve_params_tensor)
        """
        return inference(self, image_tensor, brightness_factor=brightness_factor, iterations=iterations)


# ============================================================================
# Inference Function
# ============================================================================

def inference(model, image_tensor, brightness_factor=1.0, iterations=8, device=None):
    """
    Performs feed-forward low-light enhancement using evaluation mode and torch.no_grad().
    
    Args:
        model (ZeroDCE): ZeroDCE model instance.
        image_tensor (torch.Tensor): Input tensor of shape (B, 3, H, W) or (3, H, W) in [0, 1].
        brightness_factor (float): Optional multiplier for curve parameters (default: 1.0).
        iterations (int): Number of LE-curve iterations (1 to 8, default: 8).
        device (torch.device, optional): Target compute device.
        
    Returns:
        tuple: (enhanced_tensor, curve_params_tensor)
            - enhanced_tensor: Clamped tensor in range [0.0, 1.0].
            - curve_params_tensor: Predicted 24-channel parameter maps in [-1.0, 1.0].
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected model to be an instance of torch.nn.Module, got {type(model)}.")

    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError(f"Expected image_tensor to be a torch.Tensor, got {type(image_tensor)}.")

    # Validate tensor shape
    if image_tensor.dim() == 3:
        if image_tensor.size(0) != 3:
            raise ValueError(f"Expected 3 channels (RGB) for 3D tensor (3, H, W), got shape {image_tensor.shape}.")
        image_tensor = image_tensor.unsqueeze(0)
    elif image_tensor.dim() == 4:
        if image_tensor.size(1) != 3:
            raise ValueError(f"Expected 3 channels (RGB) for 4D tensor (B, 3, H, W), got shape {image_tensor.shape}.")
    else:
        raise ValueError(f"Invalid tensor dimensions: expected 3D or 4D tensor, got {image_tensor.dim()}D tensor.")

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
        model = model.to(device)

    image_tensor = image_tensor.to(device)

    # Set evaluation mode
    model.eval()

    # Inference without gradient calculation
    with torch.no_grad():
        _, curve_params = model(image_tensor)
        enhanced_tensor = model.apply_curves(
            image_tensor,
            curve_params,
            brightness_factor=brightness_factor,
            iterations=iterations,
        )

    return enhanced_tensor, curve_params


# ============================================================================
# Checkpoint Compatibility Verification & Weight Loader
# ============================================================================

EXPECTED_LAYER_KEYS = [
    "e_conv1.weight", "e_conv1.bias",
    "e_conv2.weight", "e_conv2.bias",
    "e_conv3.weight", "e_conv3.bias",
    "e_conv4.weight", "e_conv4.bias",
    "e_conv5.weight", "e_conv5.bias",
    "e_conv6.weight", "e_conv6.bias",
    "e_conv7.weight", "e_conv7.bias",
]


def load_pretrained_weights(model, checkpoint_path, device="cpu", strict=True):
    """
    Loads pretrained weights from a checkpoint file into a ZeroDCE model.
    Checks checkpoint existence, integrity, and architectural compatibility.
    
    Args:
        model (ZeroDCE): ZeroDCE model instance to populate.
        checkpoint_path (str): File path to .pth or .pt checkpoint.
        device (str or torch.device): Device to map tensors to ('cpu' or 'cuda').
        strict (bool): Whether to enforce exact key match (default: True).
        
    Returns:
        ZeroDCE: The model loaded with pretrained weights.
        
    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        ValueError: If checkpoint format is invalid or missing required layer weights.
        RuntimeError: If layer tensor dimensions do not match the model architecture.
    """
    if not isinstance(model, ZeroDCE):
        raise TypeError(f"Expected model to be an instance of ZeroDCE, got {type(model)}.")

    if not checkpoint_path:
        raise ValueError("checkpoint_path cannot be empty or None.")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: '{checkpoint_path}'.")

    device = torch.device(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint file '{checkpoint_path}': {str(e)}")

    # Handle checkpoints saved as raw state_dict or nested dictionary (e.g., {'state_dict': ...} or {'model': ...})
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint structure. Expected state dict, got {type(checkpoint)}.")

    # Strip potential 'module.' prefixes if saved from DataParallel
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[clean_key] = v

    # Check key compatibility
    model_state = model.state_dict()
    missing_keys = [k for k in EXPECTED_LAYER_KEYS if k not in cleaned_state_dict]
    if missing_keys:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' is incompatible with Zero-DCE architecture.\n"
            f"Missing required layer keys: {missing_keys}"
        )

    # Check shape compatibility for each layer
    mismatched_shapes = []
    for k in EXPECTED_LAYER_KEYS:
        expected_shape = tuple(model_state[k].shape)
        checkpoint_shape = tuple(cleaned_state_dict[k].shape)
        if expected_shape != checkpoint_shape:
            mismatched_shapes.append(f"Layer '{k}': expected {expected_shape}, got {checkpoint_shape}")

    if mismatched_shapes:
        raise RuntimeError(
            f"Weight shape mismatch in checkpoint '{checkpoint_path}':\n"
            + "\n".join(mismatched_shapes)
        )

    # Load weights into model
    model.load_state_dict(cleaned_state_dict, strict=strict)
    model.to(device)
    model.eval()
    model.is_pretrained = True

    return model


# ============================================================================
# Factory Model Loader (Maintained for backward compatibility)
# ============================================================================

OFFICIAL_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/Zero-DCE_code/snapshots/Epoch99.pth"
)


def load_zero_dce_model(weights_path=None, device=None):
    """
    Initializes the ZeroDCE model and loads pretrained weights.
    If compatible pretrained weights are absent, explicitly flags that the network
    is operating in untrained demonstration mode.
    
    Args:
        weights_path (str, optional): Path to .pth checkpoint file.
        device (torch.device or str, optional): Device to run model on.
        
    Returns:
        tuple: (model, status_dict)
            status_dict contains 'is_pretrained': bool, 'weights_path': str, 'device': str, 'message': str
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = ZeroDCE().to(device)
    model.is_pretrained = False

    if weights_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(base_dir, "weights", "Epoch99.pth")

    status = {
        "is_pretrained": False,
        "weights_path": weights_path,
        "device": str(device),
        "message": ""
    }

    # If file does not exist, attempt automatic download of the official CVPR 2020 weights
    if not os.path.isfile(weights_path):
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        try:
            print(f"Downloading official Zero-DCE pretrained weights from {OFFICIAL_WEIGHTS_URL}...")
            urllib.request.urlretrieve(OFFICIAL_WEIGHTS_URL, weights_path)
        except Exception as e:
            status["is_pretrained"] = False
            status["message"] = (
                f"Pretrained weights not found at '{weights_path}' and automated download failed: {str(e)}. "
                "NOTICE: Model is running with randomly initialized weights. "
                "Untrained weights will NOT produce valid low-light enhancement results."
            )
            model.eval()
            return model, status

    # Load and verify checkpoint
    try:
        load_pretrained_weights(model, weights_path, device=device, strict=True)
        status["is_pretrained"] = True
        status["message"] = f"Successfully loaded official Zero-DCE pretrained weights from '{os.path.basename(weights_path)}'."
    except Exception as e:
        model.is_pretrained = False
        status["is_pretrained"] = False
        status["message"] = (
            f"Checkpoint compatibility failure: {str(e)}. "
            "NOTICE: Model is running with randomly initialized weights (demonstration only)."
        )
        model.eval()

    return model, status


# ============================================================================
# Non-Reference Loss Functions (Academic Reference)
# ============================================================================

class SpatialConsistencyLoss(nn.Module):
    """
    Spatial Consistency Loss (L_spa):
    Preserves structural spatial coherence between input and enhanced images
    across 4-directional neighboring regions.
    """

    def __init__(self):
        super(SpatialConsistencyLoss, self).__init__()
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)

        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        d_org_left = F.conv2d(org_pool, self.weight_left, padding=1)
        d_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        d_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        d_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        d_enh_left = F.conv2d(enhance_pool, self.weight_left, padding=1)
        d_enh_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        d_enh_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        d_enh_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        loss = (
            torch.pow(d_org_left - d_enh_left, 2)
            + torch.pow(d_org_right - d_enh_right, 2)
            + torch.pow(d_org_up - d_enh_up, 2)
            + torch.pow(d_org_down - d_enh_down, 2)
        )
        return torch.mean(loss)


class ExposureControlLoss(nn.Module):
    """
    Exposure Control Loss (L_exp):
    Penalizes regions where the average intensity deviates from target constant E = 0.6.
    """

    def __init__(self, patch_size=16, mean_val=0.6):
        super(ExposureControlLoss, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        mean_intensity = torch.mean(x, 1, keepdim=True)
        patch_mean = self.pool(mean_intensity)
        return torch.mean(torch.pow(patch_mean - self.mean_val, 2))


class ColorConstancyLoss(nn.Module):
    """
    Color Constancy Loss (L_col):
    Enforces Gray-World hypothesis to prevent color distortion across R, G, B channels.
    """

    def __init__(self):
        super(ColorConstancyLoss, self).__init__()

    def forward(self, x):
        mean_rgb = torch.mean(x, [2, 3])
        mr, mg, mb = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
        d_rg = torch.pow(mr - mg, 2)
        d_rb = torch.pow(mr - mb, 2)
        d_gb = torch.pow(mg - mb, 2)
        return torch.mean(torch.sqrt(d_rg + d_rb + d_gb + 1e-6))


class IlluminationSmoothnessLoss(nn.Module):
    """
    Illumination Smoothness Loss (L_tv):
    Enforces total variation smoothness on estimated curve parameter maps.
    """

    def __init__(self):
        super(IlluminationSmoothnessLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (h_x - 1) * w_x
        count_w = h_x * (w_x - 1)
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, : h_x - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, : w_x - 1], 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size
