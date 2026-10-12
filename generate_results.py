"""
Batch Processing and Benchmark Result Generation Script
Project: Low-Light Image Enhancement Using Deep Learning Techniques
"""

import os
import glob
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import load_zero_dce_model
from enhancement import (
    load_image,
    enhance_low_light,
    save_enhanced_image,
    generate_histogram_comparison_figure,
    visualize_curve_parameters_figure,
)
from metrics import compute_all_metrics, compare_metrics


def run_benchmark_evaluation():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "data", "input")
    output_dir = os.path.join(base_dir, "data", "output")
    results_dir = os.path.join(base_dir, "results")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("LOW-LIGHT IMAGE ENHANCEMENT: BENCHMARK EVALUATION")
    print("=" * 70)

    # 1. Load Model
    model, status = load_zero_dce_model()
    print(f"Model status: {status['message']}")
    print(f"Device: {status['device'].upper()}")

    # 2. Gather input images
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.[jJ][pP][gG]")) + 
                         glob.glob(os.path.join(input_dir, "*.[pP][nN][gG]")))

    if not image_paths:
        print(f"No input images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} benchmark image(s) for evaluation.")

    summary_rows = []

    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.basename(img_path)
        print(f"\n[{idx}/{len(image_paths)}] Processing: {filename}...")

        # Load image
        img_rgb = load_image(img_path)

        # Enhance with Zero-DCE
        enhanced_rgb, curve_params, inf_time = enhance_low_light(model, img_rgb)
        print(f"   Enhanced in {inf_time*1000:.1f} ms")

        # Save enhanced image
        out_path = save_enhanced_image(enhanced_rgb, output_dir, base_filename=f"eval_{filename}")

        # Metrics
        m_orig = compute_all_metrics(img_rgb)
        m_enh = compute_all_metrics(enhanced_rgb)

        summary_rows.append({
            "Filename": filename,
            "Orig_Mean_Bright": round(m_orig["mean_brightness"], 2),
            "Enh_Mean_Bright": round(m_enh["mean_brightness"], 2),
            "Mean_Bright_Gain(%)": round(((m_enh["mean_brightness"] - m_orig["mean_brightness"]) / max(m_orig["mean_brightness"], 1e-5)) * 100, 1),
            "Orig_StdDev": round(m_orig["std_intensity"], 2),
            "Enh_StdDev": round(m_enh["std_intensity"], 2),
            "Orig_Entropy": round(m_orig["entropy"], 3),
            "Enh_Entropy": round(m_enh["entropy"], 3),
            "Entropy_Gain(bits)": round(m_enh["entropy"] - m_orig["entropy"], 3),
            "Orig_Underexposed(%)": round(m_orig["underexposed_percentage"], 1),
            "Enh_Underexposed(%)": round(m_enh["underexposed_percentage"], 1),
            "Inference_Time(ms)": round(inf_time * 1000, 1)
        })

        # Generate and save comparison plot for the first image
        if idx == 1:
            plot_path = os.path.join(results_dir, "benchmark_comparison_plot.png")
            fig = generate_histogram_comparison_figure(img_rgb, enhanced_rgb)
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"   Saved diagnostic histogram plot to: {plot_path}")

            curves_path = os.path.join(results_dir, "curve_parameter_maps.png")
            fig_curves = visualize_curve_parameters_figure(curve_params)
            fig_curves.savefig(curves_path, dpi=130, bbox_inches="tight")
            plt.close(fig_curves)
            print(f"   Saved curve parameter visualization to: {curves_path}")

    # 3. Save Summary DataFrame
    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(results_dir, "benchmark_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nBenchmark summary saved to: {csv_path}")
    print("\nBenchmark Summary Table:")
    print(df.to_string(index=False))
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark_evaluation()
