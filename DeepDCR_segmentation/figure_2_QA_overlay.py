# -*- coding: utf-8 -*-
"""
Prepare an external dataset for nnU-Net inference/testing.

What it does:
1) Match image/mask pairs from source directories
2) (Optional) resample mask to image grid using nearest-neighbor
3) Enforce allowed label set (unexpected labels -> 0)
4) Export nnU-Net-style files:
      imagesTs/case_0000.nii.gz
      labelsTs/case.nii.gz
5) Save QA overlay PNGs (axial slice with largest foreground area)
6) Save per-case report and global foreground statistics

Usage example:
  python prepare_external_to_nnunet.py \
      --src-img-dir /path/to/images_nrrd \
      --src-mask-dir /path/to/masks_nii_gz \
      --dst-root /path/to/nnUNet_raw/DatasetXXX_Name \
      --split-tag external_test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import matplotlib.pyplot as plt


# =======================
# Defaults / constants
# =======================
CHANNEL_SUFFIX = "_0000"

LABEL_MAP = {
    0: "background",
    1: "lacrimal_sac",
    2: "maxilla",
    3: "nasal_cavity",
}
ALLOWED_LABELS = set(LABEL_MAP.keys())


# =======================
# CLI
# =======================
def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare external dataset into nnU-Net imagesTs/labelsTs format with QA overlays and foreground stats."
    )
    p.add_argument("--src-img-dir", required=True, help="Source image directory (e.g., *.nrrd)")
    p.add_argument("--src-mask-dir", required=True, help="Source mask directory (e.g., *.nii.gz)")
    p.add_argument("--dst-root", required=True, help="Destination nnU-Net dataset root (contains imagesTs/labelsTs)")
    p.add_argument("--split-tag", default="external_test", help="Tag used in summary outputs (default: external_test)")
    p.add_argument("--qa-subdir", default="test", help="QA overlay subdirectory name under _QA_overlays (default: test)")
    p.add_argument("--image-glob", default="*.nrrd", help="Image glob pattern (default: *.nrrd)")
    p.add_argument("--mask-glob", default="*.nii.gz", help="Mask glob pattern (default: *.nii.gz)")
    p.add_argument("--resample-mask-to-image", action="store_true", default=True,
                   help="Resample mask to image grid if mismatch (nearest-neighbor). Enabled by default.")
    p.add_argument("--no-resample-mask-to-image", dest="resample_mask_to_image", action="store_false",
                   help="Disable mask resampling when image/mask grids mismatch.")
    p.add_argument("--enforce-label-set", action="store_true", default=True,
                   help="Set unexpected labels to 0. Enabled by default.")
    p.add_argument("--no-enforce-label-set", dest="enforce_label_set", action="store_false",
                   help="Disable label-set enforcement.")
    p.add_argument("--qa-dpi", type=int, default=220, help="DPI for QA overlay PNGs (default: 220)")
    return p.parse_args()


# =======================
# Grid / IO helpers
# =======================
def same_grid(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing())
        and np.allclose(a.GetOrigin(), b.GetOrigin())
        and np.allclose(a.GetDirection(), b.GetDirection())
    )


def resample_mask_to_ref(mask_img: sitk.Image, ref_img: sitk.Image) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    out = resampler.Execute(mask_img)
    return sitk.Cast(out, sitk.sitkUInt8)


def read_image_as_float32(path: Path) -> sitk.Image:
    img = sitk.ReadImage(str(path))
    return sitk.Cast(img, sitk.sitkFloat32)


def read_mask_as_uint8(path: Path) -> sitk.Image:
    img = sitk.ReadImage(str(path))
    return sitk.Cast(img, sitk.sitkUInt8)


def write_nii_gz(img: sitk.Image, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path))


def enforce_labels(mask_np: np.ndarray):
    uniq = set(np.unique(mask_np).tolist())
    bad = sorted(list(uniq - ALLOWED_LABELS))
    if bad:
        for v in bad:
            mask_np[mask_np == v] = 0
    return mask_np, bad


# =======================
# Foreground stats
# =======================
def compute_fg_stats(mask_np: np.ndarray):
    total = int(mask_np.size)
    vox1 = int(np.sum(mask_np == 1))
    vox2 = int(np.sum(mask_np == 2))
    vox3 = int(np.sum(mask_np == 3))
    fg = vox1 + vox2 + vox3
    fg_ratio = fg / total if total > 0 else np.nan
    return {
        "total_vox": total,
        "fg_vox": fg,
        "fg_ratio": fg_ratio,
        "vox_l1": vox1,
        "vox_l2": vox2,
        "vox_l3": vox3,
        "ratio_l1": vox1 / total if total > 0 else np.nan,
        "ratio_l2": vox2 / total if total > 0 else np.nan,
        "ratio_l3": vox3 / total if total > 0 else np.nan,
    }


# =======================
# QA overlay helpers
# =======================
def choose_best_slice(mask_np_123: np.ndarray) -> int:
    fg = (mask_np_123 > 0)
    if not np.any(fg):
        return mask_np_123.shape[0] // 2
    areas = fg.reshape(fg.shape[0], -1).sum(axis=1)
    return int(np.argmax(areas))


def robust_window(ct_slice: np.ndarray, p_low=1, p_high=99):
    v = ct_slice[np.isfinite(ct_slice)]
    if v.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(v, p_low))
    hi = float(np.percentile(v, p_high))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def save_overlay_png(ct_np: np.ndarray, mask_np: np.ndarray, out_png: Path,
                     title: str, note: str, dpi: int = 220):
    z = choose_best_slice(mask_np)
    ct = ct_np[z].astype(np.float32)
    mk = mask_np[z].astype(np.uint8)

    vmin, vmax = robust_window(ct, 1, 99)
    ct_show = np.clip(ct, vmin, vmax)

    # RGB overlay: label1->R, label2->G, label3->B
    rgb = np.zeros((mk.shape[0], mk.shape[1], 3), dtype=np.float32)
    rgb[mk == 1, 0] = 1.0
    rgb[mk == 2, 1] = 1.0
    rgb[mk == 3, 2] = 1.0

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    ax = plt.gca()
    ax.imshow(ct_show, cmap="gray", vmin=vmin, vmax=vmax)
    ax.imshow(rgb, alpha=0.35)

    # optional contour outlines
    try:
        for lbl, color in [(1, "r"), (2, "g"), (3, "b")]:
            m = (mk == lbl).astype(np.uint8)
            if m.sum() > 0:
                ax.contour(m, levels=[0.5], colors=color, linewidths=0.9)
    except Exception:
        pass

    ax.set_axis_off()
    ax.set_title(title, fontsize=10)

    ax.text(
        0.01, 0.01, note,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7)
    )

    fig.tight_layout(pad=0.1)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# =======================
# Name helpers
# =======================
def image_case_id_from_path(p: Path) -> str:
    """
    Default behavior:
    - for *.nrrd: Path.stem usually returns base name without .nrrd
    - for other names, caller can customize if needed
    """
    return p.stem


def mask_case_id_from_path(p: Path) -> str:
    """
    Handles *.nii.gz and common single-suffix cases.
    """
    if p.name.endswith(".nii.gz"):
        return p.name[:-7]
    return p.stem


# =======================
# Main
# =======================
def main():
    args = parse_args()

    src_img_dir = Path(args.src_img_dir)
    src_mask_dir = Path(args.src_mask_dir)
    dst_root = Path(args.dst_root)

    images_ts = dst_root / "imagesTs"
    labels_ts = dst_root / "labelsTs"
    qa_dir = dst_root / "_QA_overlays" / args.qa_subdir

    images_ts.mkdir(parents=True, exist_ok=True)
    labels_ts.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(src_img_dir.glob(args.image_glob))
    mask_files = sorted(src_mask_dir.glob(args.mask_glob))

    img_id_to_path = {image_case_id_from_path(p): p for p in img_files}
    mask_id_to_path = {mask_case_id_from_path(p): p for p in mask_files}

    img_ids = set(img_id_to_path.keys())
    mask_ids = set(mask_id_to_path.keys())

    common = sorted(list(img_ids & mask_ids))
    missing_img = sorted(list(mask_ids - img_ids))
    missing_mask = sorted(list(img_ids - mask_ids))

    print(f"[INFO] images found: {len(img_files)}")
    print(f"[INFO] masks found:  {len(mask_files)}")
    print(f"[INFO] matched pairs: {len(common)}")
    if missing_img:
        print(f"[WARN] masks without image: {len(missing_img)} (example: {missing_img[:10]})")
    if missing_mask:
        print(f"[WARN] images without mask: {len(missing_mask)} (example: {missing_mask[:10]})")

    rows = []
    global_total = 0
    global_fg = 0
    global_l1 = 0
    global_l2 = 0
    global_l3 = 0

    for case_id in tqdm(common, desc="Preparing dataset -> imagesTs/labelsTs"):
        src_img = img_id_to_path[case_id]
        src_msk = mask_id_to_path[case_id]

        try:
            ct_img = read_image_as_float32(src_img)
            msk_img = read_mask_as_uint8(src_msk)
        except Exception as e:
            rows.append({"case_id": case_id, "status": "failed_read", "error": str(e)})
            continue

        was_resampled = False
        if args.resample_mask_to_image and (not same_grid(msk_img, ct_img)):
            msk_img = resample_mask_to_ref(msk_img, ct_img)
            was_resampled = True

        ct_np = sitk.GetArrayFromImage(ct_img).astype(np.float32)
        m_np = sitk.GetArrayFromImage(msk_img).astype(np.uint8)

        bad_labels = []
        if args.enforce_label_set:
            m_np, bad_labels = enforce_labels(m_np)
            # rebuild with image geometry
            msk_img = sitk.GetImageFromArray(m_np.astype(np.uint8))
            msk_img.CopyInformation(ct_img)

        if ct_np.shape != m_np.shape:
            rows.append({
                "case_id": case_id,
                "status": "failed_shape_mismatch",
                "ct_shape": str(ct_np.shape),
                "mask_shape": str(m_np.shape),
                "was_resampled": was_resampled,
            })
            continue

        dst_img = images_ts / f"{case_id}{CHANNEL_SUFFIX}.nii.gz"
        dst_msk = labels_ts / f"{case_id}.nii.gz"

        write_nii_gz(ct_img, dst_img)
        write_nii_gz(msk_img, dst_msk)

        fgst = compute_fg_stats(m_np)
        global_total += fgst["total_vox"]
        global_fg += fgst["fg_vox"]
        global_l1 += fgst["vox_l1"]
        global_l2 += fgst["vox_l2"]
        global_l3 += fgst["vox_l3"]

        qa_png = qa_dir / f"{case_id}.png"
        z = choose_best_slice(m_np)
        title = f"{case_id} | axial z={z} ({args.split_tag})"
        note = (
            f"FG ratio={fgst['fg_ratio']:.6f}\n"
            f"label1={fgst['vox_l1']}  label2={fgst['vox_l2']}  label3={fgst['vox_l3']}\n"
            f"resampled={was_resampled}  bad_labels->0={','.join(map(str, bad_labels)) if bad_labels else 'None'}\n"
            f"Overlay colors: label1=red, label2=green, label3=blue"
        )
        save_overlay_png(ct_np, m_np, qa_png, title=title, note=note, dpi=args.qa_dpi)

        rows.append({
            "case_id": case_id,
            "status": "ok",
            # Keep only filenames (not absolute paths) to avoid leaking local filesystem structure
            "src_img_name": src_img.name,
            "src_mask_name": src_msk.name,
            "dst_img_name": dst_img.name,
            "dst_mask_name": dst_msk.name,
            "qa_png_name": qa_png.name,
            "was_resampled": was_resampled,
            "bad_labels_set_to_0": ",".join(map(str, bad_labels)) if bad_labels else "",
            **fgst,
            "image_size_xyz": str(ct_img.GetSize()),
            "image_spacing_xyz": str(ct_img.GetSpacing()),
        })

    df = pd.DataFrame(rows)

    report_csv = dst_root / f"prepare_report_with_fgstats_{args.qa_subdir}.csv"
    df.to_csv(report_csv, index=False, encoding="utf-8-sig")

    ok_df = df[df["status"] == "ok"].copy()
    summary = {
        "split": args.split_tag,
        "n_cases_ok": int(len(ok_df)),
        "global_total_vox": int(global_total),
        "global_fg_vox": int(global_fg),
        "global_fg_ratio_weighted": float(global_fg / global_total) if global_total > 0 else np.nan,
        "global_l1_vox": int(global_l1),
        "global_l2_vox": int(global_l2),
        "global_l3_vox": int(global_l3),
        "global_l1_ratio_weighted": float(global_l1 / global_total) if global_total > 0 else np.nan,
        "global_l2_ratio_weighted": float(global_l2 / global_total) if global_total > 0 else np.nan,
        "global_l3_ratio_weighted": float(global_l3 / global_total) if global_total > 0 else np.nan,
        "fg_ratio_mean_per_case": float(ok_df["fg_ratio"].mean()) if len(ok_df) else np.nan,
        "fg_ratio_median_per_case": float(ok_df["fg_ratio"].median()) if len(ok_df) else np.nan,
        "l1_ratio_mean_per_case": float(ok_df["ratio_l1"].mean()) if len(ok_df) else np.nan,
        "l2_ratio_mean_per_case": float(ok_df["ratio_l2"].mean()) if len(ok_df) else np.nan,
        "l3_ratio_mean_per_case": float(ok_df["ratio_l3"].mean()) if len(ok_df) else np.nan,
    }

    summary_json = dst_root / f"foreground_summary_{args.qa_subdir}.json"
    summary_csv = dst_root / f"foreground_summary_{args.qa_subdir}.csv"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    pd.DataFrame([summary]).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(f"imagesTs: {images_ts}")
    print(f"labelsTs: {labels_ts}")
    print(f"QA overlays: {qa_dir}")
    print(f"Per-case report: {report_csv}")
    print(f"Global FG summary: {summary_json} / {summary_csv}")


if __name__ == "__main__":
    main()