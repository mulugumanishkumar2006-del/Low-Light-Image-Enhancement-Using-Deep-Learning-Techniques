# Low-Light Image Enhancement Using Deep Learning Techniques

An academic college project implementing **Zero-Reference Deep Curve Estimation (Zero-DCE)** for low-light digital image enhancement, complete with an interactive **Streamlit** user interface and a **Digital Image Processing (DIP)** quantitative evaluation suite.

---

## 📌 Project Overview

Digital photographs captured in low-light environments often suffer from poor contrast, diminished visibility, noise, and color distortion. Traditional digital image processing methods (such as Histogram Equalization and Gamma Correction) apply hand-crafted transformations that either overamplify sensor noise or wash out highlights. Meanwhile, standard supervised deep learning models require paired low-light and normal-light images, which are difficult to capture in the real world.

This project implements the **Zero-DCE** (CVPR 2020) deep learning architecture. Instead of predicting an enhanced image or an illumination map directly, the network estimates **pixel-wise dynamic Light-Enhancement curves (LE-curves)** iteratively. The network is trained with four specialized **zero-reference loss functions**, requiring **no ground-truth normal-light images**.

---

## 🛠️ Technology Stack

- **Language:** Python 3.9+
- **Deep Learning Framework:** PyTorch (`torch`, `torch.nn`)
- **Computer Vision:** OpenCV (`opencv-python`)
- **Scientific Computing:** NumPy, Pandas
- **Visualization:** Matplotlib
- **Web UI & Dashboard:** Streamlit
- **Image Handling:** Pillow (PIL)

---

## 📂 Project Directory Structure

```
low_light_enhancement_zero_dce/
│
├── app.py                      # Main Streamlit web application
├── model.py                    # Genuine Zero-DCE neural network architecture & loss functions
├── enhancement.py              # DIP processing engine, inference, brightness control, and plots
├── metrics.py                  # Quantitative DIP metrics (Mean, Std Dev, Shannon Entropy, etc.)
├── requirements.txt            # Pinned dependency requirements
├── README.md                   # Complete documentation and setup guide
│
├── weights/                    # Pretrained PyTorch model weights
│   └── Epoch99.pth             # Official CVPR 2020 Zero-DCE checkpoint (~320 KB)
│
├── data/
│   ├── input/                  # Input test images (JPG, JPEG, PNG, benchmark samples)
│   └── output/                 # Automatically saved enhanced images with timestamps
│
├── results/                    # Generated comparison plots, metrics CSVs, and benchmarks
│
└── report/
    └── project_report.md       # Comprehensive academic project report for college submission
```

---

## 🔬 Mathematical Formulation of Zero-DCE

### 1. Light-Enhancement (LE) Curve
The network maps an input image $x \in [0, 1]$ through an 8-iteration recursive curve formulation:

$$LE_n(x) = LE_{n-1}(x) + A_n(x) \cdot \left( LE_{n-1}(x)^2 - LE_{n-1}(x) \right)$$

where $A_n(x) \in [-1, 1]$ represents the pixel-wise curve parameter map learned by the network.

- Since $LE^2 - LE \le 0$ for all values in $[0, 1]$, a negative parameter $A_n < 0$ guarantees monotonic brightening without value clipping.
- By iteratively applying 8 curves, the model achieves high-order dynamic range mapping.

### 2. Four Non-Reference Loss Functions
Zero-DCE requires **no ground-truth reference images** during training:
1. **Spatial Consistency Loss ($L_{spa}$):** Preserves spatial gradients and structural coherence across 4 local neighbor regions.
2. **Exposure Control Loss ($L_{exp}$):** Keeps the average intensity of local $16 \times 16$ patches centered around target $E = 0.6$.
3. **Color Constancy Loss ($L_{col}$):** Enforces the Gray-World hypothesis to eliminate color tint shifts.
4. **Illumination Smoothness Loss ($L_{tv}$):** Total variation regularizer ensuring smooth curve transitions across adjacent pixels.

---

## 📊 Digital Image Processing Evaluation Metrics

The system calculates comprehensive quantitative metrics to validate enhancement:

1. **Mean Brightness (Luminance) $\mu$:**
   $$\mu = \frac{1}{M \cdot N} \sum_{x=1}^{M} \sum_{y=1}^{N} Y(x, y)$$
   *(Target: restoration from $< 25$ up to $100 - 140$ on a 255-scale).*

2. **Standard Deviation of Intensity $\sigma$ (Global Contrast):**
   $$\sigma = \sqrt{\frac{1}{M \cdot N} \sum_{x=1}^{M} \sum_{y=1}^{N} (Y(x, y) - \mu)^2}$$
   *(Measures the dynamic spread across tonal levels).*

3. **Shannon Information Entropy $H$:**
   $$H = - \sum_{k=0}^{255} p_k \log_2(p_k)$$
   *(Quantifies the texture detail and information richness revealed from the shadows).*

4. **RMS Contrast & Exposure Percentages:**
   Measures root-mean-square contrast and monitors under-exposed ($< 25/255$) vs over-exposed ($> 230/255$) pixel ratios.

---

## 💻 Installation & Execution Guide (Windows)

### Prerequisites
- Python 3.9, 3.10, or 3.11 installed. Verify with:
  ```powershell
  python --version
  ```

### Step 1: Open PowerShell or Command Prompt
Navigate to the project root directory:
```powershell
cd C:\Users\mulug\.gemini\antigravity\scratch\low_light_enhancement_zero_dce
```

### Step 2: Create and Activate a Virtual Environment (Recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If script execution is disabled on PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### Step 3: Install Required Packages
```powershell
pip install -r requirements.txt
```

### Step 4: Verify Pretrained Weights
The official Zero-DCE checkpoint `Epoch99.pth` is pre-located in `weights/Epoch99.pth`.  
If missing, the system will automatically download it from the official CVPR 2020 repository upon startup.

### Step 5: Run the Streamlit Application
```powershell
streamlit run app.py
```
Streamlit will launch and display the local URL (typically `http://localhost:8501`). Open this URL in any web browser.

---

## 🖥️ Application Features Walkthrough

1. **Image Upload:** Upload JPG, JPEG, or PNG images via drag-and-drop, or select pre-loaded benchmark images from the DICM dataset.
2. **Side-by-Side Comparison:** Compare original low-light and enhanced images side-by-side with resolution and processing duration indicators.
3. **Interactive Brightness Adjustment:** Continuous slider controlling the curve parameter scaling factor $\alpha$ ($0.2$ to $2.0$).
4. **Curve Iteration Depth Control:** Control recursive iteration count from $1$ to $8$ to observe multi-level light estimation.
5. **Grayscale & Histogram Analysis:**
   - Grayscale luminance comparison (ITU-R BT.601 standard).
   - 4-panel diagnostic plot: RGB channel histograms, overlaid luminance, and Cumulative Distribution Function (CDF).
6. **Quantitative Metric Table:** Live evaluation of Mean Brightness, Standard Deviation, Shannon Entropy, and shadow restoration.
7. **Curve Parameter Map Visualizer:** 8-panel inspection of internal $A_1 \dots A_8$ parameter maps showing spatial brightening regions.
8. **Export & Download:** Download enhanced images in PNG or JPG format with timestamped filenames and automated archival in `data/output/`.

---

## 🎓 College Viva-Voce Defense (Q&A)

**Q1: What is the main innovation of Zero-DCE over RetinexNet?**  
*Answer:* RetinexNet decomposes images into reflectance and illumination maps, which is an ill-posed inverse problem that often produces color artifacts and over-smoothing. Zero-DCE formulates the problem as non-linear curve estimation where pixels are transformed directly through monotonic curves, preserving realistic color and texture.

**Q2: How does Zero-DCE avoid overexposing bright regions?**  
*Answer:* The LE-curve formulation $x + A \cdot (x^2 - x)$ has zero slope at the boundaries ($x = 0$ and $x = 1$). Highlights close to $1$ experience minimal adjustment, while mid-tones and shadows experience higher dynamic gain. In addition, the Exposure Control Loss ($L_{exp}$) penalizes local patches exceeding the optimal exposure value.

**Q3: Why is high Shannon Entropy desirable in an enhanced image?**  
*Answer:* In a low-light photograph, almost all pixel intensities are clustered into a few low-value bins, leading to a low entropy value (little distinguishable information). When enhanced, details in previously black shadows become visible across diverse intensity levels, distributing the probability mass and raising Shannon entropy.

---

## 📜 References

1. C. Li, C. Guo, and C. C. Loy, *"Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement"*, CVPR 2020.
2. R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th Edition, Pearson.
3. Official Zero-DCE Repository: `https://github.com/Li-Chongyi/Zero-DCE`
