# -*- coding: utf-8 -*-
"""
5-fold stability (Dice) — boxplot + fold scatter

Panels:
  a) Lacrimal sac:                     [Fine, Cascade]
  b) Maxilla of target site:           [Coarse, Fine, Cascade]
  c) Nasal cavity of target approach:  [Coarse, Fine, Cascade]

Usage:
  python plot_5fold_stability_dice.py \
      --csv /path/to/table_cv.csv \
      --out-dir /path/to/output_figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# =========================
# Matplotlib style
# =========================
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "savefig.dpi": 600,
})

# Colors
C_COARSE  = "#F28E2B"
C_FINE    = "#2CB1BC"
C_CASCADE = "#A7C7E7"

MODEL_COLOR = {
    "Coarse segmentation model": C_COARSE,
    "Fine segmentation model": C_FINE,
    "Two-Stage cascade model": C_CASCADE,
}

PANELS = [
    ("a", "Lacrimal sac", "C1_Dice",
     ["Fine segmentation model", "Two-Stage cascade model"]),
    ("b", "Maxilla of target site", "C2_Dice",
     ["Coarse segmentation model", "Fine segmentation model", "Two-Stage cascade model"]),
    ("c", "Nasal cavity of target approach", "C3_Dice",
     ["Coarse segmentation model", "Fine segmentation model", "Two-Stage cascade model"]),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot 5-fold Dice stability (boxplot + fold scatter) from a CV summary CSV."
    )
    p.add_argument("--csv", required=True, help="Path to CV summary CSV (e.g., table_cv.csv)")
    p.add_argument("--out-dir", required=True, help="Output directory for figures")
    p.add_argument("--out-stem", default="fivefold_stability_dice_3panels",
                   help="Output filename stem (default: fivefold_stability_dice_3panels)")
    return p.parse_args()


def load_and_validate(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = ["Fold", "Model", "C1_Dice", "C2_Dice", "C3_Dice"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df["Fold"] = pd.to_numeric(df["Fold"], errors="coerce")
    for c in ["C1_Dice", "C2_Dice", "C3_Dice"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Use standard 5 folds only
    df = df[df["Fold"].isin([0, 1, 2, 3, 4])].copy()
    return df


def _short_label(model_name: str) -> str:
    if model_name.startswith("Coarse"):
        return "Coarse"
    if model_name.startswith("Fine"):
        return "Fine"
    return "Cascade"


def draw_box_with_fold_scatter(ax, sub, dice_col, models, letter, title):
    data = [sub.loc[sub["Model"] == m, dice_col].dropna().values for m in models]
    x = np.arange(1, len(models) + 1)

    bp = ax.boxplot(
        data,
        positions=x,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        medianprops=dict(color="0.15", linewidth=1.2),
        boxprops=dict(linewidth=0.9, color="0.25"),
        whiskerprops=dict(linewidth=0.9, color="0.25"),
        capprops=dict(linewidth=0.9, color="0.25"),
    )

    for i, m in enumerate(models):
        bp["boxes"][i].set_facecolor(MODEL_COLOR[m])
        bp["boxes"][i].set_alpha(0.25)

        pts = sub.loc[sub["Model"] == m, dice_col].dropna().values

        # deterministic jitter for reproducibility
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(pts))
        ax.scatter(
            x[i] + jitter,
            pts,
            s=18,
            color=MODEL_COLOR[m],
            alpha=0.65,
            zorder=3,
            edgecolors="none",
        )

    # Axis range/ticks (kept consistent with your original design)
    ax.set_ylim(0.80, 0.90)
    ax.set_yticks(np.arange(0.80, 0.91, 0.025))
    ax.set_ylabel("Dice (DSC)")

    ax.set_title(title, fontsize=9, pad=8)

    ax.set_xticks(x)
    ax.set_xticklabels([_short_label(m) for m in models])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.18)

    ax.text(
        -0.14, 1.02, letter,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=10,
        va="bottom"
    )


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_png = out_dir / f"{args.out_stem}.png"
    out_pdf = out_dir / f"{args.out_stem}.pdf"

    df = load_and_validate(csv_path)

    fig = plt.figure(figsize=(8.2, 3.3))
    gs = fig.add_gridspec(1, 3, wspace=0.38)

    for i, (letter, title, col, models) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, i])
        sub = df[df["Model"].isin(models)].copy()
        draw_box_with_fold_scatter(ax, sub, col, models, letter, title)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(out_png)
    print(out_pdf)


if __name__ == "__main__":
    main()