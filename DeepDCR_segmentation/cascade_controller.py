# cascade_controller.py
# ---------------------------------------------------------
# Generic controller for a 2-stage segmentation cascade:
#   (1) crop ROI from full-FOV images using coarse masks
#   (2) (optional) run nnUNetv2_predict on ROI images
#   (3) paste ROI predictions back to full-FOV space
#
# Features:
#   - split tag: tr/ts (for logging only)
#   - fold id selection
#   - arbitrary input/output paths via CLI
#   - optional passthrough args to nnUNetv2_predict
# ---------------------------------------------------------

from __future__ import annotations

import argparse
import ast
import csv
import math
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import SimpleITK as sitk


# =========================
# Crop utilities
# =========================
def get_case_id_from_0000(name: str) -> str:
    # Typical nnU-Net image name: case_0000.nii.gz
    return name.replace("_0000.nii.gz", "")


def get_case_id_plain(name: str) -> str:
    # Typical prediction/label name: case.nii.gz
    return name.replace(".nii.gz", "")


def _allclose(a, b, rtol=1e-5, atol=1e-4) -> bool:
    return np.allclose(np.array(a, dtype=np.float64), np.array(b, dtype=np.float64), rtol=rtol, atol=atol)


def ensure_same_grid_or_align(img: sitk.Image, lbl: sitk.Image, case: str) -> Optional[sitk.Image]:
    # size mismatch -> cannot crop reliably
    if img.GetSize() != lbl.GetSize():
        print(f"[ERROR] {case}: image/mask size mismatch, skip.")
        print("  image size:", img.GetSize(), "mask size:", lbl.GetSize())
        return None

    same_spacing = _allclose(img.GetSpacing(), lbl.GetSpacing(), rtol=1e-6, atol=1e-6)
    same_origin = _allclose(img.GetOrigin(), lbl.GetOrigin(), rtol=1e-5, atol=1e-3)
    same_dir = _allclose(img.GetDirection(), lbl.GetDirection(), rtol=1e-5, atol=1e-3)

    if not (same_spacing and same_origin and same_dir):
        print(
            f"[WARN] {case}: header mismatch -> aligning mask header to image "
            f"(spacing:{same_spacing}, origin:{same_origin}, dir:{same_dir})"
        )
        lbl2 = sitk.Image(lbl)
        lbl2.CopyInformation(img)
        return lbl2

    return lbl


def make_anchor_mask(lbl_img: sitk.Image, labels=(2, 3)) -> sitk.Image:
    m = sitk.Image(lbl_img.GetSize(), sitk.sitkUInt8)
    m.CopyInformation(lbl_img)
    for v in labels:
        m = m | sitk.Equal(lbl_img, int(v))
    return sitk.Cast(m, sitk.sitkUInt8)


def bbox_from_mask_sitk(mask: sitk.Image):
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    labs = stats.GetLabels()
    if len(labs) == 0:
        return None
    return stats.GetBoundingBox(labs[0])  # (x,y,z,sx,sy,sz)


def expand_bbox_mm(bb, img: sitk.Image, margin_mm: float):
    x0, y0, z0, sx, sy, sz = bb
    sizeX, sizeY, sizeZ = img.GetSize()
    spX, spY, spZ = img.GetSpacing()

    mx = int(math.ceil(margin_mm / abs(spX)))
    my = int(math.ceil(margin_mm / abs(spY)))
    mz = int(math.ceil(margin_mm / abs(spZ)))

    x1 = x0 + sx - 1
    y1 = y0 + sy - 1
    z1 = z0 + sz - 1

    x0 = max(0, x0 - mx)
    x1 = min(sizeX - 1, x1 + mx)
    y0 = max(0, y0 - my)
    y1 = min(sizeY - 1, y1 + my)
    z0 = max(0, z0 - mz)
    z1 = min(sizeZ - 1, z1 + mz)
    return (x0, x1, y0, y1, z0, z1)


def apply_anterior_keep(img: sitk.Image, y0: int, y1: int, keep_ratio: float):
    if keep_ratio >= 0.999:
        return y0, y1

    _, sizeY, _ = img.GetSize()
    keep_len = max(1, int(math.ceil(sizeY * keep_ratio)))

    p0 = img.TransformIndexToPhysicalPoint((0, 0, 0))
    p1 = img.TransformIndexToPhysicalPoint((0, sizeY - 1, 0))
    y_phys_0 = float(p0[1])
    y_phys_1 = float(p1[1])

    # "anterior" here is defined as the smaller physical y end
    if y_phys_0 <= y_phys_1:
        ant_min, ant_max = 0, keep_len - 1
    else:
        ant_min, ant_max = sizeY - keep_len, sizeY - 1

    ny0 = max(y0, ant_min)
    ny1 = min(y1, ant_max)
    if ny0 > ny1:
        ny0, ny1 = ant_min, ant_max
    return int(ny0), int(ny1)


def crop_roi(img: sitk.Image, bbox6):
    x0, x1, y0, y1, z0, z1 = bbox6
    return sitk.RegionOfInterest(
        img,
        size=[x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1],
        index=[x0, y0, z0],
    )


# =========================
# Paste utilities
# =========================
def load_bbox_map(crop_log: Path) -> Dict[str, Tuple[int, int, int, int, int, int]]:
    mp = {}
    with open(crop_log, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            case = row["case"]
            bbox6 = ast.literal_eval(row["final_bbox6(x0,x1,y0,y1,z0,z1)"])
            mp[case] = tuple(int(x) for x in bbox6)
    return mp


def paste_back_fullfov(full_img: sitk.Image, roi_pred: sitk.Image, bbox6):
    x0, x1, y0, y1, z0, z1 = bbox6

    full_arr = np.zeros(sitk.GetArrayFromImage(full_img).shape, dtype=np.uint8)  # (Z,Y,X)
    roi_arr = sitk.GetArrayFromImage(roi_pred).astype(np.uint8)

    # bbox uses (x,y,z), numpy uses (z,y,x)
    full_arr[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] = roi_arr

    out_img = sitk.GetImageFromArray(full_arr)
    out_img.CopyInformation(full_img)
    return out_img


# =========================
# Commands
# =========================
def cmd_crop(args):
    out_root = Path(args.out_root)
    out_images = out_root / "imagesTs"
    out_labels = out_root / "labelsTs"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    log_csv = out_root / "crop_log.csv"

    images_dir = Path(args.images_dir)
    coarse_dir = Path(args.coarse_dir)

    split_name = args.split.lower()
    anchor_labels = tuple(int(x) for x in args.anchor_labels)

    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "split",
                "case",
                "orig_size",
                "orig_spacing",
                "anchor_bbox(x,y,z,sx,sy,sz)",
                "final_bbox6(x0,x1,y0,y1,z0,z1)",
                "cropped_size",
            ]
        )

        img_paths = sorted(images_dir.glob("*.nii.gz"))
        if not img_paths:
            raise RuntimeError(f"No .nii.gz found in images_dir: {images_dir}")

        for img_path in img_paths:
            case = get_case_id_from_0000(img_path.name)
            lbl_path = coarse_dir / f"{case}.nii.gz"
            if not lbl_path.exists():
                print(f"[SKIP] missing coarse mask: {case}")
                continue

            img = sitk.ReadImage(str(img_path))
            lbl = sitk.ReadImage(str(lbl_path))

            lbl = ensure_same_grid_or_align(img, lbl, case)
            if lbl is None:
                continue

            anchor = make_anchor_mask(lbl, anchor_labels)
            bb = bbox_from_mask_sitk(anchor)
            if bb is None:
                print(f"[WARN] no anchor labels {anchor_labels} found in coarse mask: {case} (skip)")
                continue

            bbox6 = expand_bbox_mm(bb, img, float(args.margin_mm))
            y0, y1 = apply_anterior_keep(img, bbox6[2], bbox6[3], float(args.ant_keep))
            bbox6 = (bbox6[0], bbox6[1], y0, y1, bbox6[4], bbox6[5])

            img_c = crop_roi(img, bbox6)
            lbl_c = crop_roi(lbl, bbox6)

            # Keep original image naming convention (case_0000.nii.gz) for nnU-Net input
            sitk.WriteImage(img_c, str(out_images / img_path.name))
            # Save cropped coarse label for QA/debug (case.nii.gz)
            sitk.WriteImage(lbl_c, str(out_labels / f"{case}.nii.gz"))

            w.writerow([split_name, case, img.GetSize(), img.GetSpacing(), bb, bbox6, img_c.GetSize()])
            print(f"[OK] {case} -> cropped size {img_c.GetSize()}")

    print("[DONE] ROI images:", out_images)
    print("[DONE] crop log:", log_csv)


def cmd_predict(args):
    """
    Wrapper for nnUNetv2_predict on ROI images.
    """
    roi_images = Path(args.roi_images)
    out_roi_pred = Path(args.out_roi_pred)
    out_roi_pred.mkdir(parents=True, exist_ok=True)

    cmd = [
        "nnUNetv2_predict",
        "-i",
        str(roi_images),
        "-o",
        str(out_roi_pred),
        "-d",
        str(args.dataset_id),
        "-c",
        str(args.config),
        "-f",
        str(args.fold),
    ]

    if args.chk:
        cmd += ["-chk", str(args.chk)]

    # Pass-through extra args, e.g. --disable_tta
    if args.extra:
        cmd += args.extra

    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[DONE] ROI predictions:", out_roi_pred)


def cmd_paste(args):
    full_img_dir = Path(args.full_img_dir)
    crop_log = Path(args.crop_log)
    roi_pred_dir = Path(args.roi_pred_dir)
    out_full_pred = Path(args.out_full_pred)
    out_full_pred.mkdir(parents=True, exist_ok=True)

    bbox_map = load_bbox_map(crop_log)

    full_paths = sorted(full_img_dir.glob("*.nii.gz"))
    if not full_paths:
        raise RuntimeError(f"No .nii.gz found in full_img_dir: {full_img_dir}")

    for img_path in full_paths:
        # Full images are usually named case_0000.nii.gz
        case = get_case_id_from_0000(img_path.name)
        if case not in bbox_map:
            print(f"[SKIP] {case} not found in crop_log")
            continue

        roi_pred_path = roi_pred_dir / f"{case}.nii.gz"
        if not roi_pred_path.exists():
            print(f"[SKIP] missing ROI prediction: {case}")
            continue

        full_img = sitk.ReadImage(str(img_path))
        roi_pred = sitk.ReadImage(str(roi_pred_path))
        out_img = paste_back_fullfov(full_img, roi_pred, bbox_map[case])

        sitk.WriteImage(out_img, str(out_full_pred / f"{case}.nii.gz"))
        print(f"[OK] pasted: {case}")

    print("[DONE] full-FOV predictions:", out_full_pred)


def cmd_cascade(args):
    """
    crop -> (optional predict) -> paste
    """
    # 1) crop
    cmd_crop(args)

    # 2) optional predict
    if args.run_predict:
        out_root = Path(args.out_root)
        roi_images = out_root / "imagesTs"
        out_roi_pred = Path(args.out_roi_pred)

        predict_ns = argparse.Namespace(
            roi_images=str(roi_images),
            out_roi_pred=str(out_roi_pred),
            dataset_id=args.dataset_id,
            config=args.config,
            fold=args.fold,
            chk=args.chk,
            extra=args.extra,
        )
        cmd_predict(predict_ns)

    # 3) paste
    out_root = Path(args.out_root)
    crop_log = out_root / "crop_log.csv"
    paste_ns = argparse.Namespace(
        full_img_dir=args.full_img_dir,
        crop_log=str(crop_log),
        roi_pred_dir=args.out_roi_pred,
        out_full_pred=args.out_full_pred,
    )
    cmd_paste(paste_ns)


def build_parser():
    p = argparse.ArgumentParser(
        description="Controller for 2-stage cascade segmentation workflow: crop / predict / paste"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # Common crop args
    def add_crop_args(sp):
        sp.add_argument("--split", choices=["tr", "ts"], default="ts", help="Tag written into crop_log.csv")
        sp.add_argument("--images-dir", required=True, help="Full-FOV images directory (e.g., imagesTr/imagesTs)")
        sp.add_argument("--coarse-dir", required=True, help="Coarse mask directory (files like case.nii.gz)")
        sp.add_argument(
            "--out-root",
            required=True,
            help="Output root for ROI dataset (creates imagesTs, labelsTs, crop_log.csv)",
        )
        sp.add_argument("--margin-mm", type=float, default=30.0)
        sp.add_argument("--ant-keep", type=float, default=1.0)
        sp.add_argument(
            "--anchor-labels",
            nargs="+",
            default=[2, 3],
            help="Labels used to form anchor mask, default: 2 3",
        )

    # crop
    sp = sub.add_parser("crop", help="Crop ROI images + crop_log.csv from full-FOV images using coarse masks")
    add_crop_args(sp)
    sp.set_defaults(func=cmd_crop)

    # predict
    sp = sub.add_parser("predict", help="Run nnUNetv2_predict on ROI images")
    sp.add_argument("--roi-images", required=True, help="ROI image directory created by crop (imagesTs)")
    sp.add_argument("--out-roi-pred", required=True, help="Output directory for ROI predictions (case.nii.gz)")
    sp.add_argument("--dataset-id", type=int, default=103, help="nnU-Net dataset id for stage-2 model")
    sp.add_argument("--config", default="3d_fullres")
    sp.add_argument("--fold", type=int, required=True)
    sp.add_argument("--chk", default="checkpoint_best.pth")
    sp.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        help="Extra args passed to nnUNetv2_predict (e.g., --disable_tta)",
        default=None,
    )
    sp.set_defaults(func=cmd_predict)

    # paste
    sp = sub.add_parser("paste", help="Paste ROI predictions back to full-FOV using crop_log.csv")
    sp.add_argument("--full-img-dir", required=True, help="Full-FOV image directory (same space as target grid)")
    sp.add_argument("--crop-log", required=True, help="crop_log.csv produced by crop")
    sp.add_argument("--roi-pred-dir", required=True, help="ROI prediction directory (case.nii.gz)")
    sp.add_argument("--out-full-pred", required=True, help="Output directory for full-FOV predictions (case.nii.gz)")
    sp.set_defaults(func=cmd_paste)

    # cascade
    sp = sub.add_parser("cascade", help="crop -> (optional predict) -> paste")
    add_crop_args(sp)
    sp.add_argument("--full-img-dir", required=True, help="Full-FOV image directory (same space as reference)")
    sp.add_argument("--out-roi-pred", required=True, help="Directory to store ROI predictions")
    sp.add_argument("--out-full-pred", required=True, help="Directory to store full-FOV predictions")

    sp.add_argument("--fold", type=int, required=True)
    sp.add_argument("--dataset-id", type=int, default=103)
    sp.add_argument("--config", default="3d_fullres")
    sp.add_argument("--chk", default="checkpoint_best.pth")
    sp.add_argument("--run-predict", action="store_true", help="If set, call nnUNetv2_predict automatically")
    sp.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra args passed to nnUNetv2_predict", default=None)
    sp.set_defaults(func=cmd_cascade)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()