# -*- coding: utf-8 -*-
"""
extract_nnunet_embeddings.py
--------------------------------------------------
Extract DL embeddings using nnU-Net encoder (e.g., Dataset103 fine model)

ROI cropping STRICTLY follows the specified crop logic:
- anchor labels
- mm margin expansion
- anterior keep
- header/grid alignment

Crop is guided by segmentation masks (typically full-FOV predictions):
- internal OOF predictions
- external predictions/ensembles

Key features:
1) Robust network loading via nnUNetPredictor (version-friendly)
2) Force network + encoder to target DEVICE (avoid CUDA/CPU mismatch)
3) Resume/flush checkpointed CSV writing for safe interrupt-resume
4) PCA fit on internal set only (no leakage), then transform external sets
--------------------------------------------------
"""

from __future__ import annotations

import os
import math
import csv
import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional, List

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from sklearn.decomposition import PCA

# nnU-Net v2 predictor (version-robust loader)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


# =========================
# CLI
# =========================
def build_parser():
    p = argparse.ArgumentParser(description="Extract nnU-Net encoder embeddings with strict ROI crop + optional PCA.")

    # ---- checkpoint / model ----
    p.add_argument("--nnunet-ckpt", required=True, type=str,
                   help="Path to nnU-Net checkpoint file (e.g., fold_0/checkpoint_final.pth)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                   help="Device for inference. 'auto' uses cuda if available.")

    # ---- crop config ----
    p.add_argument("--margin-mm", type=float, default=30.0)
    p.add_argument("--ant-keep", type=float, default=1.0)
    p.add_argument("--anchor-labels", nargs="+", type=int, default=[2, 3],
                   help="Labels used to build anchor mask (default: 2 3)")
    p.add_argument("--flush-every", type=int, default=10, help="Flush partial embeddings to CSV every N cases")

    # ---- PCA config ----
    p.add_argument("--target-pca-dim", type=int, default=30)
    p.add_argument("--skip-pca", action="store_true", help="If set, only export raw embeddings (no PCA)")

    # ---- Mode A: single dataset extraction ----
    p.add_argument("--ct-dir", type=str, default=None, help="CT directory (expects case_0000.nii.gz)")
    p.add_argument("--seg-dir", type=str, default=None, help="Segmentation directory (case.nii.gz)")
    p.add_argument("--tag", type=str, default=None, help="Tag for outputs in single-dataset mode")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory for embeddings/crop logs")

    # ---- Mode B: 3-dataset batch extraction + PCA ----
    p.add_argument("--internal-ct-dir", type=str, default=None)
    p.add_argument("--internal-seg-dir", type=str, default=None)
    p.add_argument("--external1-ct-dir", type=str, default=None)
    p.add_argument("--external1-seg-dir", type=str, default=None)
    p.add_argument("--external2-ct-dir", type=str, default=None)
    p.add_argument("--external2-seg-dir", type=str, default=None)

    # ---- Optional PCA output filenames prefix ----
    p.add_argument("--pca-prefix", type=str, default="DL_embedding",
                   help="Prefix for PCA CSVs: {prefix}_internal_oof.csv, etc.")

    return p


# =========================
# Crop / IO utilities
# =========================
def get_case_id_from_seg_name(seg_name: str) -> str:
    return seg_name.replace(".nii.gz", "")


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

    x0 = max(0, x0 - mx); x1 = min(sizeX - 1, x1 + mx)
    y0 = max(0, y0 - my); y1 = min(sizeY - 1, y1 + my)
    z0 = max(0, z0 - mz); z1 = min(sizeZ - 1, z1 + mz)
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

    # anterior = smaller physical y end
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
        index=[x0, y0, z0]
    )


def _allclose(a, b, rtol=1e-5, atol=1e-4):
    return np.allclose(np.array(a, dtype=np.float64), np.array(b, dtype=np.float64), rtol=rtol, atol=atol)


def ensure_same_grid_or_align(img: sitk.Image, lbl: sitk.Image, case: str):
    # size mismatch -> cannot crop reliably
    if img.GetSize() != lbl.GetSize():
        print(f"[ERROR] {case} size mismatch! skip")
        print("  img size:", img.GetSize(), "lbl size:", lbl.GetSize())
        return None

    same_spacing = _allclose(img.GetSpacing(), lbl.GetSpacing(), rtol=1e-6, atol=1e-6)
    same_origin  = _allclose(img.GetOrigin(),  lbl.GetOrigin(),  rtol=1e-5, atol=1e-3)
    same_dir     = _allclose(img.GetDirection(), lbl.GetDirection(), rtol=1e-5, atol=1e-3)

    if not (same_spacing and same_origin and same_dir):
        print(f"[WARN] {case} header mismatch -> align seg header to CT "
              f"(spacing:{same_spacing}, origin:{same_origin}, dir:{same_dir})")
        lbl2 = sitk.Image(lbl)
        lbl2.CopyInformation(img)
        return lbl2
    return lbl


def zscore_norm(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-6:
        sd = 1e-6
    return (x - mu) / sd


def find_ct_path(ct_dir: Path, case_id: str) -> Path:
    return ct_dir / f"{case_id}_0000.nii.gz"


# =========================
# nnU-Net encoder loader
# =========================
def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_nnunet_encoder(nnunet_ckpt: Path, device: torch.device):
    print(">> Loading nnU-Net encoder (via nnUNetPredictor)...")

    if not nnunet_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {nnunet_ckpt}")

    fold_dir = nnunet_ckpt.parent
    model_folder = fold_dir.parent
    checkpoint_name = nnunet_ckpt.name

    print(f"  model_folder: {model_folder}")
    print(f"  checkpoint:   {checkpoint_name}")
    print(f"  device:       {device}")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )

    # load a single fold (embedding extraction is usually consistent enough with one encoder)
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(0,),
        checkpoint_name=checkpoint_name
    )

    net = predictor.network
    net.eval()
    for p in net.parameters():
        p.requires_grad = False

    # force device placement (avoid CPU weight vs CUDA input mismatch)
    net = net.to(device)
    encoder = net.encoder.to(device)

    print(">> nnU-Net encoder loaded and moved to device.")
    return predictor, net, encoder


# =========================
# ROI tensor loading
# =========================
def load_roi_ct_tensor(
    ct_path: Path,
    seg_path: Path,
    case: str,
    device: torch.device,
    margin_mm: float,
    ant_keep: float,
    anchor_labels: Tuple[int, ...]
) -> Tuple[torch.Tensor, Dict]:
    """
    Return:
      roi_tensor: (1,1,Z,Y,X) float32 on DEVICE
      meta: dict with bbox, cropped_size, spacing
    """
    ct_img = sitk.ReadImage(str(ct_path))
    seg_img = sitk.ReadImage(str(seg_path))

    seg_img = ensure_same_grid_or_align(ct_img, seg_img, case)
    if seg_img is None:
        raise RuntimeError("CT/SEG grid mismatch")

    anchor = make_anchor_mask(seg_img, anchor_labels)
    bb = bbox_from_mask_sitk(anchor)
    if bb is None:
        raise RuntimeError(f"No anchor labels {anchor_labels} found in seg")

    bbox6 = expand_bbox_mm(bb, ct_img, margin_mm)
    y0, y1 = apply_anterior_keep(ct_img, bbox6[2], bbox6[3], ant_keep)
    bbox6 = (bbox6[0], bbox6[1], y0, y1, bbox6[4], bbox6[5])

    ct_crop = crop_roi(ct_img, bbox6)

    # sitk array is (Z,Y,X)
    ct_arr = sitk.GetArrayFromImage(ct_crop).astype(np.float32)
    ct_arr = zscore_norm(ct_arr)

    roi_tensor = torch.from_numpy(ct_arr)[None, None].float().to(device)

    meta = {
        "anchor_bbox_xyzw": bb,
        "final_bbox6": bbox6,
        "cropped_size_xyz": ct_crop.GetSize(),   # (X,Y,Z)
        "spacing_xyz": ct_crop.GetSpacing()
    }
    return roi_tensor, meta


@torch.no_grad()
def extract_embedding(roi_tensor: torch.Tensor, encoder: torch.nn.Module, device: torch.device) -> np.ndarray:
    roi_tensor = roi_tensor.to(device, non_blocking=True)
    feats = encoder(roi_tensor)
    bottleneck = feats[-1]  # (1, C, d, h, w)
    emb = F.adaptive_avg_pool3d(bottleneck, 1).view(1, -1)
    return emb.detach().cpu().numpy()[0]


# =========================
# Dataset processing (resume + flush)
# =========================
def process_dataset(
    ct_dir: Path,
    seg_dir: Path,
    tag: str,
    out_dir: Path,
    encoder: torch.nn.Module,
    device: torch.device,
    margin_mm: float,
    ant_keep: float,
    anchor_labels: Tuple[int, ...],
    flush_every: int = 10
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"embeddings_raw_{tag}.csv"
    log_path = out_dir / f"crop_log_{tag}.csv"

    # resume: existing embeddings
    done = set()
    rows_existing = []
    if out_csv.exists():
        try:
            old = pd.read_csv(out_csv)
            if "case_id" in old.columns:
                rows_existing = old.to_dict("records")
                done = set(old["case_id"].astype(str).tolist())
                print(f">> Resume: {len(done)} cases already done for {tag} ({out_csv})")
        except Exception as e:
            print(f"[WARN] Failed to read existing {out_csv}: {e}")
            done = set()
            rows_existing = []

    # crop log append
    log_exists = log_path.exists()
    f_log = open(log_path, "a", newline="", encoding="utf-8")
    w = csv.writer(f_log)
    if not log_exists:
        w.writerow([
            "tag", "case",
            "ct_path", "seg_path",
            "anchor_bbox(x,y,z,sx,sy,sz)",
            "final_bbox6(x0,x1,y0,y1,z0,z1)",
            "cropped_size_xyz",
            "spacing_xyz"
        ])

    rows_new = []
    n_since_flush = 0

    try:
        for fn in sorted(os.listdir(seg_dir)):
            if not fn.endswith(".nii.gz"):
                continue

            case_id = get_case_id_from_seg_name(fn)
            if case_id in done:
                continue

            ct_path = find_ct_path(ct_dir, case_id)
            seg_path = seg_dir / fn

            if not ct_path.exists():
                print(f"[WARN] CT missing: {case_id} | expected: {ct_path}")
                continue

            try:
                roi_tensor, meta = load_roi_ct_tensor(
                    ct_path=ct_path,
                    seg_path=seg_path,
                    case=case_id,
                    device=device,
                    margin_mm=margin_mm,
                    ant_keep=ant_keep,
                    anchor_labels=anchor_labels
                )
                emb = extract_embedding(roi_tensor, encoder=encoder, device=device)

                row = {"case_id": case_id}
                for i, v in enumerate(emb):
                    row[f"emb_{i:03d}"] = float(v)
                rows_new.append(row)
                done.add(case_id)

                w.writerow([
                    tag, case_id,
                    str(ct_path), str(seg_path),
                    meta["anchor_bbox_xyzw"],
                    meta["final_bbox6"],
                    meta["cropped_size_xyz"],
                    meta["spacing_xyz"]
                ])

                n_since_flush += 1
                if n_since_flush >= flush_every:
                    df_tmp = pd.DataFrame(rows_existing + rows_new)
                    df_tmp.to_csv(out_csv, index=False)
                    print(f">> Flushed {len(rows_new)} new cases to {out_csv}")
                    rows_existing = df_tmp.to_dict("records")
                    rows_new = []
                    n_since_flush = 0

            except Exception as e:
                print(f"[FAIL] {case_id}: {e}")

    finally:
        # final flush
        if rows_new:
            df_tmp = pd.DataFrame(rows_existing + rows_new)
            df_tmp.to_csv(out_csv, index=False)
            print(f">> Final flush {len(rows_new)} new cases to {out_csv}")
            rows_existing = df_tmp.to_dict("records")
            rows_new = []
        f_log.close()

    df_final = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(rows_existing)
    print(f">> Saved {out_csv}")
    print(f">> Saved crop log {log_path}")
    return df_final


# =========================
# PCA helpers
# =========================
def save_pca_df(df: pd.DataFrame, X_pca: np.ndarray, tag: str, out_dir: Path, pca_prefix: str, pca_dim: int):
    out = pd.DataFrame(X_pca, columns=[f"dl_{i:02d}" for i in range(pca_dim)])
    out["case_id"] = df["case_id"].values
    out_path = out_dir / f"{pca_prefix}_{tag}.csv"
    out.to_csv(out_path, index=False)
    print(f">> Saved {out_path}")


def run_pca_no_leakage(
    df_internal: pd.DataFrame,
    df_ext1: Optional[pd.DataFrame],
    df_ext2: Optional[pd.DataFrame],
    target_pca_dim: int,
    out_dir: Path,
    pca_prefix: str
):
    print(">> PCA fitting on internal set only (no leakage)")

    emb_cols = [c for c in df_internal.columns if c.startswith("emb_")]
    if len(emb_cols) == 0:
        raise RuntimeError("No embedding columns found in internal set. Check extraction output.")

    n_comp = min(target_pca_dim, len(emb_cols), max(1, len(df_internal)))
    if n_comp < target_pca_dim:
        print(f"[WARN] target_pca_dim={target_pca_dim} reduced to {n_comp} due to data dimensionality/sample size.")

    pca = PCA(n_components=n_comp, random_state=0)
    X_int_pca = pca.fit_transform(df_internal[emb_cols].values)

    save_pca_df(df_internal, X_int_pca, "internal_oof", out_dir, pca_prefix, n_comp)

    if df_ext1 is not None and len(df_ext1):
        X_ext1_pca = pca.transform(df_ext1[emb_cols].values)
        save_pca_df(df_ext1, X_ext1_pca, "external1", out_dir, pca_prefix, n_comp)

    if df_ext2 is not None and len(df_ext2):
        X_ext2_pca = pca.transform(df_ext2[emb_cols].values)
        save_pca_df(df_ext2, X_ext2_pca, "external2", out_dir, pca_prefix, n_comp)

    # Save PCA metadata for reproducibility
    meta = pd.DataFrame({
        "component": np.arange(n_comp),
        "explained_variance_ratio": pca.explained_variance_ratio_
    })
    meta.to_csv(out_dir / f"{pca_prefix}_pca_explained_variance.csv", index=False)
    print(f">> Saved {out_dir / f'{pca_prefix}_pca_explained_variance.csv'}")


# =========================
# Main
# =========================
def main():
    args = build_parser().parse_args()

    device = resolve_device(args.device)
    anchor_labels = tuple(int(x) for x in args.anchor_labels)

    nnunet_ckpt = Path(args.nnunet_ckpt)
    _, _, encoder = load_nnunet_encoder(nnunet_ckpt=nnunet_ckpt, device=device)

    # -------- Mode A: single dataset --------
    if args.ct_dir and args.seg_dir and args.tag and args.out_dir:
        out_dir = Path(args.out_dir)
        print(f">> Extract embeddings for single dataset: {args.tag}")

        df_single = process_dataset(
            ct_dir=Path(args.ct_dir),
            seg_dir=Path(args.seg_dir),
            tag=args.tag,
            out_dir=out_dir,
            encoder=encoder,
            device=device,
            margin_mm=args.margin_mm,
            ant_keep=args.ant_keep,
            anchor_labels=anchor_labels,
            flush_every=args.flush_every
        )

        if not args.skip_pca:
            print("[INFO] Single-dataset mode detected. PCA is skipped by default design unless you separately provide internal/external splits.")
        print(">> DONE (single dataset mode)")
        return

    # -------- Mode B: 3-dataset batch (internal + external1 + external2) --------
    batch_keys = [
        "internal_ct_dir", "internal_seg_dir",
        "external1_ct_dir", "external1_seg_dir",
        "external2_ct_dir", "external2_seg_dir",
        "out_dir"
    ]
    batch_present = any(getattr(args, k) is not None for k in batch_keys)

    if batch_present:
        missing = [k for k in batch_keys if getattr(args, k) is None]
        if missing:
            raise ValueError(f"Batch mode requires all arguments. Missing: {missing}")

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(">> Extract internal OOF embeddings")
        df_internal = process_dataset(
            ct_dir=Path(args.internal_ct_dir),
            seg_dir=Path(args.internal_seg_dir),
            tag="internal_oof",
            out_dir=out_dir,
            encoder=encoder,
            device=device,
            margin_mm=args.margin_mm,
            ant_keep=args.ant_keep,
            anchor_labels=anchor_labels,
            flush_every=args.flush_every
        )

        print(">> Extract external1 embeddings")
        df_ext1 = process_dataset(
            ct_dir=Path(args.external1_ct_dir),
            seg_dir=Path(args.external1_seg_dir),
            tag="external1",
            out_dir=out_dir,
            encoder=encoder,
            device=device,
            margin_mm=args.margin_mm,
            ant_keep=args.ant_keep,
            anchor_labels=anchor_labels,
            flush_every=args.flush_every
        )

        print(">> Extract external2 embeddings")
        df_ext2 = process_dataset(
            ct_dir=Path(args.external2_ct_dir),
            seg_dir=Path(args.external2_seg_dir),
            tag="external2",
            out_dir=out_dir,
            encoder=encoder,
            device=device,
            margin_mm=args.margin_mm,
            ant_keep=args.ant_keep,
            anchor_labels=anchor_labels,
            flush_every=args.flush_every
        )

        if not args.skip_pca:
            run_pca_no_leakage(
                df_internal=df_internal,
                df_ext1=df_ext1,
                df_ext2=df_ext2,
                target_pca_dim=args.target_pca_dim,
                out_dir=out_dir,
                pca_prefix=args.pca_prefix
            )

        print(">> DONE (batch mode)")
        return

    raise ValueError(
        "Provide either:\n"
        "  (A) single dataset mode: --ct-dir --seg-dir --tag --out-dir\n"
        "or\n"
        "  (B) batch mode: --internal-*/--external1-*/--external2-* plus --out-dir"
    )


if __name__ == "__main__":
    main()