"""
metrics_utils.py
-----------------
Classification metrics (ridge-vs-valley pixel classification against the
pseudo ground truth) and matplotlib chart builders. Every algorithm is
tested standalone, directly on the uploaded image (no synthetic clean
reference is generated), so there is no PSNR/SSIM path here — the only
image-quality signal is the no-reference Ridge Clarity Score computed in
evaluation.py.

The interactive, on-screen versions of these charts are rendered with
native Streamlit components directly in app.py (st.dataframe / st.bar_chart);
the matplotlib figures below exist only so the PDF export has something
embeddable, since ReportLab needs static images.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def compute_classification_metrics(gt_mask, pred_mask):
    gt = gt_mask.flatten()
    pred = pred_mask.flatten()
    cm = confusion_matrix(gt, pred, labels=[0, 1])
    return {
        "confusion_matrix": cm,
        "accuracy": accuracy_score(gt, pred),
        "precision": precision_score(gt, pred, zero_division=0),
        "recall": recall_score(gt, pred, zero_division=0),
        "f1": f1_score(gt, pred, zero_division=0),
    }


# --------------------------------------------------------------------------- #
# Plot builders (PDF export only — see module docstring)
# --------------------------------------------------------------------------- #

def plot_confusion_matrices(results):
    names = list(results.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6))
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        cm = results[name]["confusion_matrix"]
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=8, wrap=True)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Valley(0)", "Ridge(1)"], fontsize=7)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Valley(0)", "Ridge(1)"], fontsize=7)
        vmax = cm.max() if cm.max() > 0 else 1
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > vmax * 0.5 else "black"
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color=color, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=7)
        ax.set_ylabel("Actual", fontsize=7)
    fig.tight_layout()
    return fig


def plot_metric_bars(results):
    names = list(results.keys())
    short_names = [n.split(":")[0] for n in names]
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(metrics):
        vals = [results[n][m] for n in names]
        ax.bar(x + i * width, vals, width, label=m.capitalize())
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(short_names, rotation=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Classification Metric Comparison (Ridge vs Valley)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_clarity_bars(results):
    names = list(results.keys())
    short_names = [n.split(":")[0] for n in names]
    vals = [results[n]["clarity"] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(short_names, vals, color="mediumpurple")
    ax.set_title("Ridge Orientation Clarity Score\n(no-reference, 0-1, higher = better)")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_histograms(image_dict):
    names = list(image_dict.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 2.8))
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        ax.hist(image_dict[name].flatten(), bins=32, color="steelblue")
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("Intensity", fontsize=7)
        ax.set_ylabel("Pixel count", fontsize=7)
    fig.tight_layout()
    return fig


def plot_sample_gallery(image, enhanced_dict, image_label="Original (uploaded)"):
    names = [image_label] + list(enhanced_dict.keys())
    images = [image] + list(enhanced_dict.values())
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.8))
    for ax, name, img in zip(axes, names, images):
        ax.imshow(img, cmap="gray")
        ax.set_title(name, fontsize=8, wrap=True)
        ax.axis("off")
    fig.tight_layout()
    return fig