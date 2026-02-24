# -*- coding: utf-8 -*-
"""
maxilla_thickness_axial_half_surgery.py
------------------------------------------------------------
Axial half-side maxilla local thickness feature extraction + visualization
for DCR-related analysis.

Labels:
  1 = lacrimal
  2 = maxilla (bone)
  3 = nasal space

Design summary:
- Determine surgical side from lacrimal centroid (left/right by mid-sagittal plane)
- Use only ipsilateral half maxilla on axial slice where lacrimal area is maximal
- Build local ROI by dilating (lacrimal ∪ nasal) on that slice
- Compute local thickness map (mm) via 2 * distance_transform inside maxilla ROI
- Export per-case features (max/mean/p95 thickness, etc.)
- Optional visualization overlays for selected cases

Notes:
- This script can run on predicted segmentations and/or GT segmentations.
- Visualization title intentionally keeps only "patient + z*" (no dataset/side text).
------------------------------------------------------------
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import SimpleITK as sitk

from scipy.ndimage import distance_transform_edt
from skimage.measure import find_contours

import matplotlib as mpl
import matplotlib.pyplot as plt


# =========================
# 0) figure style
# =========================
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "savefig.dpi": 600,
})


# =========================
# 1) Defaults / labels / params
# =========================
LAC = 1
MAX = 2
NAS = 3

DEFAULT_LOCAL_RADIUS_MM = 15.0
DEFAULT_HEAT_ALPHA = 0.55


# =========================
# 2) CLI
# =========================
def build_parser():
    p = argparse.ArgumentParser(
        description="Extract axial half-side maxilla local thickness features and generate overlay figures."
    )

    # output
    p.add_argument("--out-dir", required=True, type=str,
                   help="Output root directory for CSVs and figures")

    # thickness / visualization params
    p.add_argument("--local-radius-mm", type=float, default=DEFAULT_LOCAL_RADIUS_MM)
    p.add_argument("--heat-alpha", type=float, default=DEFAULT_HEAT_ALPHA)

    # dataset triplets (optional; provide any subset)
    # Internal
    p.add_argument("--int-images", type=str, default=None)
    p.add_argument("--int-gt", type=str, default=None)
    p.add_argument("--int-pred", type=str, default=None)

    # External 1
    p.add_argument("--ext1-images", type=str, default=None)
    p.add_argument("--ext1-gt", type=str, default=None)
    p.add_argument("--ext1-pred", type=str, default=None)

    # External 2
    p.add_argument("--ext2-images", type=str, default=None)
    p.add_argument("--ext2-gt", type=str, default=None)
    p.add_argument("--ext2-pred", type=str, default=None)

    # visualization selected cases
    p.add_argument("--select-cases", nargs="*", default=[],
                   help="Case IDs to visualize (will search ext1 then ext2 by default)")
    p.add_argument("--patient-title-map-json", type=str, default=None,
                   help='Optional JSON file mapping case_id -> display title, e.g. {"81":"patient 81"}')

    return p


# =========================
# 3) IO helpers
# =========================
def find_case_file(folder: Path, case_id: str, is_image: bool) -> Optional[Path]:
    folder = Path(folder)
    cid = str(case_id)
    cand: List[Path] = []
    if is_image:
        cand += [folder / f"{cid}_0000.nii.gz", folder / f"{cid}_0000.nii"]
    cand += [folder / f"{cid}.nii.gz", folder / f"{cid}.nii"]
    for p in cand:
        if p.exists():
            return p
    for fp in folder.glob("*.nii*"):
        if fp.name.lower().startswith(cid.lower()):
            return fp
    return None


def read_nii_zyx(path: Path) -> Tuple[np.ndarray, sitk.Image]:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # (z,y,x)
    return arr, img


def normalize_ct(ct_zyx: np.ndarray) -> np.ndarray:
    x = ct_zyx.astype(np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((x - lo) / (hi - lo), 0, 1)


# =========================
# 4) Geometry helpers
# =========================
def axial_max_area_slice(mask_zyx: np.ndarray) -> Optional[int]:
    if mask_zyx.sum() == 0:
        return None
    areas = mask_zyx.reshape(mask_zyx.shape[0], -1).sum(axis=1)
    if areas.max() == 0:
        return None
    return int(np.argmax(areas))


def mid_sagittal_x_index(size_xyz: Tuple[int, int, int]) -> float:
    sizeX = int(size_xyz[0])  # (X,Y,Z)
    return (sizeX - 1) / 2.0


def centroid_zyx(mask_zyx: np.ndarray) -> Optional[np.ndarray]:
    pts = np.argwhere(mask_zyx > 0)
    if pts.size == 0:
        return None
    return pts.mean(axis=0).astype(np.float64)


def surgical_side_from_lacrimal(lac_mask_zyx: np.ndarray, size_xyz: Tuple[int, int, int]) -> Optional[str]:
    c = centroid_zyx(lac_mask_zyx)
    if c is None:
        return None
    midx = mid_sagittal_x_index(size_xyz)
    return "L" if float(c[2]) < midx else "R"


def half_mask_by_side_2d(mask_yx: np.ndarray, side: str) -> np.ndarray:
    h, w = mask_yx.shape
    midx = (w - 1) / 2.0
    xs = np.arange(w)
    keep = xs < midx if side == "L" else xs >= midx
    out = np.zeros_like(mask_yx, dtype=np.uint8)
    out[:, keep] = mask_yx[:, keep]
    return out


def dilate_2d_mm(bin_yx: np.ndarray, sy: float, sx: float, radius_mm: float) -> np.ndarray:
    if bin_yx.max() == 0:
        return np.zeros_like(bin_yx, dtype=np.uint8)
    dt = distance_transform_edt((bin_yx == 0).astype(np.uint8), sampling=(sy, sx))
    return (dt <= radius_mm).astype(np.uint8)


def local_thickness_map_2d_mm(bin_yx: np.ndarray, sy: float, sx: float) -> np.ndarray:
    """
    Local thickness proxy in ROI:
      thickness = 2 * distance-to-background (mm)
    Only defined inside bin_yx.
    """
    if bin_yx.max() == 0:
        return np.zeros_like(bin_yx, dtype=np.float32)
    dt_in = distance_transform_edt(bin_yx.astype(np.uint8), sampling=(sy, sx))
    th = (2.0 * dt_in).astype(np.float32)
    th[bin_yx == 0] = 0.0
    return th


def stats_on_mask(th_map: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    vals = th_map[mask > 0]
    if vals.size == 0:
        return {"max": 0.0, "mean": 0.0, "p95": 0.0}
    return {
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "p95": float(np.percentile(vals, 95)),
    }


# =========================
# 5) Core extraction from one seg
# =========================
def extract_maxilla_thickness_from_seg(
    seg_zyx: np.ndarray,
    seg_img: sitk.Image,
    local_radius_mm: float
) -> Dict[str, float]:
    size_xyz = seg_img.GetSize()  # (X,Y,Z)
    sx, sy, sz = map(float, seg_img.GetSpacing())

    lac = (seg_zyx == LAC).astype(np.uint8)
    maxa = (seg_zyx == MAX).astype(np.uint8)
    nas = (seg_zyx == NAS).astype(np.uint8)

    out: Dict[str, float] = {
        "z_star_lacrimal_max_area": -1.0,
        "surgical_side": -1.0,  # 0=Left,1=Right
        "maxilla_thickness_max_mm": 0.0,
        "maxilla_thickness_mean_mm": 0.0,
        "maxilla_thickness_p95_mm": 0.0,
        "maxilla_pixels_in_roi": 0.0,
    }

    side = surgical_side_from_lacrimal(lac, size_xyz)
    if side is None:
        return out
    out["surgical_side"] = 0.0 if side == "L" else 1.0

    z_star = axial_max_area_slice(lac)
    if z_star is None:
        return out
    out["z_star_lacrimal_max_area"] = float(z_star)

    lac_yx = lac[z_star].astype(np.uint8)
    nas_yx = nas[z_star].astype(np.uint8)
    max_yx = maxa[z_star].astype(np.uint8)

    # ipsilateral maxilla only
    max_half = half_mask_by_side_2d(max_yx, side)

    # local ROI seeded by lacrimal + nasal
    roi_seed = ((lac_yx > 0) | (nas_yx > 0)).astype(np.uint8)
    if roi_seed.max() == 0:
        roi_seed = lac_yx.copy()

    roi_dil = dilate_2d_mm(roi_seed, sy=sy, sx=sx, radius_mm=local_radius_mm)
    max_roi = (max_half.astype(bool) & roi_dil.astype(bool)).astype(np.uint8)

    th_map = local_thickness_map_2d_mm(max_roi, sy=sy, sx=sx)
    st = stats_on_mask(th_map, max_roi)

    out["maxilla_thickness_max_mm"] = st["max"]
    out["maxilla_thickness_mean_mm"] = st["mean"]
    out["maxilla_thickness_p95_mm"] = st["p95"]
    out["maxilla_pixels_in_roi"] = float(max_roi.sum())
    return out


# =========================
# 6) Batch extraction
# =========================
def list_case_ids_from_seg_folder(seg_dir: Path) -> List[str]:
    cids = []
    for fn in sorted(os.listdir(seg_dir)):
        if fn.endswith(".nii.gz") or fn.endswith(".nii"):
            cids.append(fn.replace(".nii.gz", "").replace(".nii", ""))
    return cids


def run_features_for_dataset(name: str, images_dir: Path, seg_dir: Path, out_csv: Path, local_radius_mm: float):
    rows = []
    case_ids = list_case_ids_from_seg_folder(seg_dir)
    for i, cid in enumerate(case_ids, 1):
        img_fp = find_case_file(images_dir, cid, is_image=True)
        seg_fp = find_case_file(seg_dir, cid, is_image=False)
        if img_fp is None or seg_fp is None:
            print(f"[WARN] missing: {cid} | img={img_fp} seg={seg_fp}")
            continue
        try:
            _, ct_img = read_nii_zyx(img_fp)
            seg_zyx, seg_img = read_nii_zyx(seg_fp)

            if ct_img.GetSize() != seg_img.GetSize():
                print(f"[WARN] size mismatch: {cid} | ct={ct_img.GetSize()} seg={seg_img.GetSize()} -> skip")
                continue

            feats = extract_maxilla_thickness_from_seg(
                seg_zyx.astype(np.int16),
                seg_img,
                local_radius_mm=local_radius_mm
            )
            feats["case_id"] = str(cid)
            feats["dataset"] = name
            rows.append(feats)
        except Exception as e:
            print(f"[FAIL] {cid}: {e}")
            rows.append({"case_id": str(cid), "dataset": name, "failed": 1.0})

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f">> Saved: {out_csv}")
    return df


# =========================
# 7) Visualization
# =========================
def draw_contours(ax, bin2d: np.ndarray, color, lw=1.0, linestyle="solid"):
    if bin2d is None:
        return
    m = (bin2d > 0).astype(np.uint8)
    if m.max() == 0:
        return
    cs = find_contours(m, 0.5)
    for c in cs:
        ax.plot(c[:, 1], c[:, 0], color=color, lw=lw, linestyle=linestyle)


def format_patient_title(case_id: str, patient_title_map: Optional[Dict[str, str]] = None) -> str:
    if patient_title_map is None:
        return str(case_id)
    return patient_title_map.get(case_id, case_id)


def visualize_case(
    case_id: str,
    dataset_name: str,
    images_dir: Path,
    gt_dir: Path,
    pred_dir: Path,
    fig_dir: Path,
    local_radius_mm: float,
    heat_alpha: float,
    patient_title_map: Optional[Dict[str, str]] = None
):
    img_fp  = find_case_file(images_dir, case_id, is_image=True)
    gt_fp   = find_case_file(gt_dir, case_id, is_image=False)
    pr_fp   = find_case_file(pred_dir, case_id, is_image=False)
    if img_fp is None or gt_fp is None or pr_fp is None:
        raise FileNotFoundError(f"Missing file for {case_id}\nimg={img_fp}\ngt={gt_fp}\npred={pr_fp}")

    ct_zyx, ct_img = read_nii_zyx(img_fp)
    gt_zyx, gt_img = read_nii_zyx(gt_fp)
    pr_zyx, pr_img = read_nii_zyx(pr_fp)

    if ct_img.GetSize() != gt_img.GetSize() or ct_img.GetSize() != pr_img.GetSize():
        raise RuntimeError(f"Size mismatch: {case_id}")

    ct_n = normalize_ct(ct_zyx)
    sx, sy, sz = map(float, ct_img.GetSpacing())

    # choose z* based on GT lacrimal for stable figure plane
    gt_lac = (gt_zyx == LAC).astype(np.uint8)
    z_star = axial_max_area_slice(gt_lac)
    if z_star is None:
        z_star = ct_zyx.shape[0] // 2

    # determine side (kept for computation, removed from title text)
    size_xyz = pr_img.GetSize()
    side = surgical_side_from_lacrimal((pr_zyx == LAC).astype(np.uint8), size_xyz)
    if side is None:
        side = surgical_side_from_lacrimal((gt_zyx == LAC).astype(np.uint8), size_xyz)
    if side is None:
        side = "L"

    # build thickness map from PRED on z_star
    pr_seg = pr_zyx.astype(np.int16)
    max_yx = (pr_seg[z_star] == MAX).astype(np.uint8)
    lac_yx = (pr_seg[z_star] == LAC).astype(np.uint8)
    nas_yx = (pr_seg[z_star] == NAS).astype(np.uint8)

    max_half = half_mask_by_side_2d(max_yx, side)
    roi_seed = ((lac_yx > 0) | (nas_yx > 0)).astype(np.uint8)
    if roi_seed.max() == 0:
        roi_seed = (gt_zyx[z_star] == LAC).astype(np.uint8)

    roi_dil = dilate_2d_mm(roi_seed, sy=sy, sx=sx, radius_mm=local_radius_mm)
    max_roi = (max_half.astype(bool) & roi_dil.astype(bool)).astype(np.uint8)

    th_map = local_thickness_map_2d_mm(max_roi, sy=sy, sx=sx)
    st = stats_on_mask(th_map, max_roi)

    # contours (full maxilla) for GT and Pred
    gt2 = (gt_zyx[z_star] == MAX).astype(np.uint8)
    pr2 = (pr_zyx[z_star] == MAX).astype(np.uint8)

    fig = plt.figure(figsize=(6.2, 5.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(ct_n[z_star], cmap="gray", vmin=0, vmax=1)

    masked_th = np.ma.masked_where(th_map <= 0, th_map)
    im = ax.imshow(masked_th, alpha=heat_alpha)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Maxilla local thickness (mm)")

    draw_contours(ax, gt2, color="lime", lw=1.0, linestyle="solid")
    draw_contours(ax, pr2, color="yellow", lw=1.2, linestyle=(0, (1.8, 1.2)))

    # Title intentionally simplified (patient + z* only)
    title = f"{format_patient_title(case_id, patient_title_map)} | z*={z_star}"
    ax.set_title(title, fontsize=9.5, fontweight="bold")

    txt = (
        f"Pred bone-window ROI thickness\n"
        f"max={st['max']:.2f} mm\n"
        f"mean={st['mean']:.2f} mm\n"
        f"p95={st['p95']:.2f} mm"
    )
    ax.text(
        0.02, 0.98, txt, transform=ax.transAxes, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", alpha=0.85),
        fontsize=8
    )

    ax.set_axis_off()
    fig.tight_layout()

    fig_dir.mkdir(parents=True, exist_ok=True)
    out_png = fig_dir / f"{dataset_name}_{case_id}_maxilla_thickness_overlay.png"
    out_pdf = fig_dir / f"{dataset_name}_{case_id}_maxilla_thickness_overlay.pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f">> Saved figure: {out_png}")
    print(f"   stats: max={st['max']:.3f} mean={st['mean']:.3f} p95={st['p95']:.3f}")


# =========================
# 8) Dataset config helpers
# =========================
def load_patient_title_map(json_path: Optional[str]) -> Optional[Dict[str, str]]:
    if json_path is None:
        return None
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"patient title map json not found: {p}")
    import json
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items()}


def has_triplet(images_dir: Optional[str], gt_dir: Optional[str], pred_dir: Optional[str]) -> bool:
    return images_dir is not None and gt_dir is not None and pred_dir is not None


# =========================
# 9) MAIN
# =========================
def main():
    args = build_parser().parse_args()

    out_dir = Path(args.out_dir)
    feature_dir = out_dir / "csv"
    fig_dir = out_dir / "figures"
    feature_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    patient_title_map = load_patient_title_map(args.patient_title_map_json)

    # ---------- Batch feature extraction ----------
    # internal_pred / internal_gt
    if args.int_images and args.int_pred:
        run_features_for_dataset(
            name="internal_oof_pred",
            images_dir=Path(args.int_images),
            seg_dir=Path(args.int_pred),
            out_csv=feature_dir / "internal_oof_pred_maxilla_thickness.csv",
            local_radius_mm=args.local_radius_mm,
        )
    if args.int_images and args.int_gt:
        run_features_for_dataset(
            name="internal_gt",
            images_dir=Path(args.int_images),
            seg_dir=Path(args.int_gt),
            out_csv=feature_dir / "internal_gt_maxilla_thickness.csv",
            local_radius_mm=args.local_radius_mm,
        )

    # external1_pred / external1_gt
    if args.ext1_images and args.ext1_pred:
        run_features_for_dataset(
            name="external1_pred",
            images_dir=Path(args.ext1_images),
            seg_dir=Path(args.ext1_pred),
            out_csv=feature_dir / "external1_pred_maxilla_thickness.csv",
            local_radius_mm=args.local_radius_mm,
        )
    if args.ext1_images and args.ext1_gt:
        run_features_for_dataset(
            name="external1_gt",
            images_dir=Path(args.ext1_images),
            seg_dir=Path(args.ext1_gt),
            out_csv=feature_dir / "external1_gt_maxilla_thickness.csv",
            local_radius_mm=args.local_radius_mm,
        )

    # external2_pred / external2_gt
    if args.ext2_images and args.ext2_pred:
        run_features_for_dataset(
            name="external2_pred",
            images_dir=Path(args.ext2_images),
            seg_dir=Path(args.ext2_pred),
            out_csv=feature_dir / "external2_pred_maxilla_thickness.csv",
            local_radius_mm=args.local_radius_mm,
        )
    if args.ext2_images and args.ext2_gt:
        run_features_for_dataset(
            name="external2_gt",
            images_dir=Path(args.ext2_images),
            seg_dir=Path(args.ext2_gt),
            out_csv=feature_dir / "external2_gt_maxilla_thickness.csv",
            local_radius_mm=args.local_radius_mm,
        )

    # ---------- Visualization for selected cases ----------
    # Search external1 first, then external2 (same as your original logic)
    for cid in args.select_cases:
        done = False

        if has_triplet(args.ext1_images, args.ext1_gt, args.ext1_pred):
            e1_ok = (
                find_case_file(Path(args.ext1_images), cid, True) and
                find_case_file(Path(args.ext1_gt), cid, False) and
                find_case_file(Path(args.ext1_pred), cid, False)
            )
            if e1_ok:
                visualize_case(
                    case_id=cid,
                    dataset_name="external1",
                    images_dir=Path(args.ext1_images),
                    gt_dir=Path(args.ext1_gt),
                    pred_dir=Path(args.ext1_pred),
                    fig_dir=fig_dir,
                    local_radius_mm=args.local_radius_mm,
                    heat_alpha=args.heat_alpha,
                    patient_title_map=patient_title_map
                )
                done = True

        if (not done) and has_triplet(args.ext2_images, args.ext2_gt, args.ext2_pred):
            e2_ok = (
                find_case_file(Path(args.ext2_images), cid, True) and
                find_case_file(Path(args.ext2_gt), cid, False) and
                find_case_file(Path(args.ext2_pred), cid, False)
            )
            if e2_ok:
                visualize_case(
                    case_id=cid,
                    dataset_name="external2",
                    images_dir=Path(args.ext2_images),
                    gt_dir=Path(args.ext2_gt),
                    pred_dir=Path(args.ext2_pred),
                    fig_dir=fig_dir,
                    local_radius_mm=args.local_radius_mm,
                    heat_alpha=args.heat_alpha,
                    patient_title_map=patient_title_map
                )
                done = True

        if not done:
            print(f"[WARN] Cannot locate case {cid} in external1/external2 folders. Check filenames.")

    print(">> ALL DONE")


if __name__ == "__main__":
    main()