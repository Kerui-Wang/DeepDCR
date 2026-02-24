# -*- coding: utf-8 -*-
"""
dcr_surgery_morph_features_extraction.py
------------------------------------------------------------
Surgery-related morphological feature extraction for DCR
Labels: 1 = lacrimal, 2 = maxilla (bone), 3 = nasal space

Key design goals:
1) Scientific + interpretable + surgery-relevant thickness/distance:
   - Bone-window path thickness (mm): along the shortest line segment
     from lacrimal surface (near maxilla) to nasal surface (near maxilla),
     measure the length passing through label=2 (maxilla/bone).
   - Remove unrelated bone by restricting to ipsilateral (surgical-side) half
     determined by mid-sagittal plane.

2) Feature groups:
   2.1 Lacrimal features (volume + morphology)
   2.2 Maxilla thickness features (bone-window path thickness + local bone burden)
   2.3 Other maxilla frontal-process related features for DCR difficulty
   2.4 Nasal space features (ipsilateral nasal volume + spatial)
   2.5 Relative position features:
       (1) distance between lacrimal center and nasal center (mm)
       (2) angle on axial slice where lacrimal area is maximal:
           major-axis of ipsilateral maxilla mask vs mid-sagittal line

Input segmentation examples:
- Internal OOF predictions (leakage-free internal features)
- External full-FOV predictions (ensemble or single model)

Output examples:
- internal_oof_features.csv
- external1_features.csv
- external2_features.csv

Notes:
- This script computes features from predicted segmentation masks (not GT labels).
- CT images are used for geometry/spacing and case matching.
------------------------------------------------------------
"""

from __future__ import annotations

import os
import math
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk

from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.spatial import cKDTree

from skimage import measure


# =========================
# Feature / ROI Config (CLI-overridable)
# =========================
LABEL_LACRIMAL = 1
LABEL_MAXILLA = 2
LABEL_NASAL = 3

# "near maxilla" surface selection: keep surface voxels within this distance to maxilla (mm)
DEFAULT_NEAR_MAXILLA_MM = 3.0

# line sampling step for thickness computation (mm)
DEFAULT_LINE_STEP_MM = 0.2

# bone burden ROI around lacrimal: dilate lacrimal by radius_mm then intersect maxilla(ipsi)
DEFAULT_BONE_BURDEN_RADIUS_MM = 10.0

# for surface extraction erosion iterations (voxel)
DEFAULT_ERODE_ITERS = 1


# -------------------------
# CLI
# -------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Extract surgery-related DCR morphological features from predicted segmentation masks."
    )

    # Single-run mode (one seg dir + one ct dir + one output csv)
    p.add_argument("--seg-dir", type=str, help="Segmentation directory (*.nii.gz)")
    p.add_argument("--ct-dir", type=str, help="CT image directory (expects case_0000.nii.gz)")
    p.add_argument("--out-csv", type=str, help="Output CSV path for extracted features")

    # Batch 3-dataset mode (internal/external1/external2), optional
    p.add_argument("--internal-seg-dir", type=str, default=None)
    p.add_argument("--internal-ct-dir", type=str, default=None)
    p.add_argument("--external1-seg-dir", type=str, default=None)
    p.add_argument("--external1-ct-dir", type=str, default=None)
    p.add_argument("--external2-seg-dir", type=str, default=None)
    p.add_argument("--external2-ct-dir", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory for 3-dataset mode (writes internal_oof_features.csv/external1_features.csv/external2_features.csv)")

    # Feature params
    p.add_argument("--near-maxilla-mm", type=float, default=DEFAULT_NEAR_MAXILLA_MM)
    p.add_argument("--line-step-mm", type=float, default=DEFAULT_LINE_STEP_MM)
    p.add_argument("--bone-burden-radius-mm", type=float, default=DEFAULT_BONE_BURDEN_RADIUS_MM)
    p.add_argument("--erode-iters", type=int, default=DEFAULT_ERODE_ITERS)

    return p


# -------------------------
# utilities
# -------------------------
def case_id_from_seg(seg_file: str) -> str:
    return seg_file.replace(".nii.gz", "")


def find_ct_path(ct_dir: Path, case_id: str) -> Optional[Path]:
    p = ct_dir / f"{case_id}_0000.nii.gz"
    return p if p.exists() else None


def sitk_read(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def sitk_arr(img: sitk.Image) -> np.ndarray:
    # (Z,Y,X)
    return sitk.GetArrayFromImage(img)


def spacing_xyz(img: sitk.Image) -> Tuple[float, float, float]:
    # (sx,sy,sz) in mm
    return tuple(map(float, img.GetSpacing()))


def voxel_volume_mm3(sp: Tuple[float, float, float]) -> float:
    return float(sp[0] * sp[1] * sp[2])


def to_sampling_zyx(sp_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    sx, sy, sz = sp_xyz
    return (sz, sy, sx)


def safe_centroid_zyx(mask_zyx: np.ndarray) -> Optional[np.ndarray]:
    coords = np.argwhere(mask_zyx > 0)
    if coords.size == 0:
        return None
    return coords.mean(axis=0).astype(np.float64)  # (z,y,x)


def centroid_mm_from_zyx(centroid_zyx: np.ndarray, sp_xyz: Tuple[float, float, float]) -> np.ndarray:
    sx, sy, sz = sp_xyz
    sp_zyx = np.array([sz, sy, sx], dtype=np.float64)
    return centroid_zyx * sp_zyx


def surface_voxels(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    # boundary = mask - eroded(mask)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    er = binary_erosion(mask, iterations=iters, border_value=0)
    surf = (mask.astype(bool) & (~er.astype(bool))).astype(np.uint8)
    return surf


def marching_cubes_surface_area(mask_zyx: np.ndarray, sp_xyz: Tuple[float, float, float]) -> float:
    # skimage expects spacing in (z,y,x)
    if mask_zyx.sum() == 0:
        return 0.0
    sp_zyx = to_sampling_zyx(sp_xyz)
    try:
        verts, faces, _, _ = measure.marching_cubes(mask_zyx.astype(np.float32), level=0.5, spacing=sp_zyx)
        return float(measure.mesh_surface_area(verts, faces))
    except Exception:
        return 0.0


def sphericity(volume_mm3: float, surface_area_mm2: float) -> float:
    # sphericity = pi^(1/3) * (6V)^(2/3) / A
    if volume_mm3 <= 0 or surface_area_mm2 <= 0:
        return 0.0
    return float((math.pi ** (1.0 / 3.0)) * ((6.0 * volume_mm3) ** (2.0 / 3.0)) / surface_area_mm2)


def pca_elongation_3d(mask_zyx: np.ndarray, sp_xyz: Tuple[float, float, float]) -> float:
    # elongation = sqrt(lambda2/lambda1) with lambda1>=lambda2>=lambda3; use coords in mm
    coords = np.argwhere(mask_zyx > 0)
    if coords.shape[0] < 10:
        return 0.0
    sx, sy, sz = sp_xyz
    sp_zyx = np.array([sz, sy, sx], dtype=np.float64)
    pts = coords.astype(np.float64) * sp_zyx
    pts -= pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts.T)
    w = np.linalg.eigvalsh(cov)
    w = np.sort(w)[::-1]
    if w[0] <= 1e-12:
        return 0.0
    return float(math.sqrt(max(w[1], 1e-12) / w[0]))


def equiv_diameter_mm(mask_zyx: np.ndarray, sp_xyz: Tuple[float, float, float]) -> float:
    vox = int(mask_zyx.sum())
    if vox == 0:
        return 0.0
    V = vox * voxel_volume_mm3(sp_xyz)
    return float(2.0 * ((3.0 * V) / (4.0 * math.pi)) ** (1.0 / 3.0))


def get_mid_sagittal_x_index(size_xyz: Tuple[int, int, int]) -> float:
    # x dimension is sizeX
    sizeX = size_xyz[0]
    return (sizeX - 1) / 2.0


def ipsilateral_side_from_lacrimal(lacrimal_centroid_zyx: np.ndarray, size_xyz: Tuple[int, int, int]) -> str:
    # centroid_zyx[2] is x index
    midx = get_mid_sagittal_x_index(size_xyz)
    return "L" if float(lacrimal_centroid_zyx[2]) < midx else "R"


def half_mask_by_side(mask_zyx: np.ndarray, side: str, size_xyz: Tuple[int, int, int]) -> np.ndarray:
    # keep half in x: Left = x < mid, Right = x >= mid
    sizeX = size_xyz[0]
    midx = get_mid_sagittal_x_index(size_xyz)
    out = np.zeros_like(mask_zyx, dtype=np.uint8)
    if mask_zyx.sum() == 0:
        return out
    # mask_zyx dims: (Z,Y,X)
    xs = np.arange(sizeX)
    if side == "L":
        keep = xs < midx
    else:
        keep = xs >= midx
    out[:, :, keep] = mask_zyx[:, :, keep]
    return (out > 0).astype(np.uint8)


def min_surface_distance_mm(a_mask_zyx: np.ndarray, b_mask_zyx: np.ndarray,
                            sp_xyz: Tuple[float, float, float], erode_iters: int) -> float:
    # compute min distance between surfaces using EDT: distance from every voxel to nearest b
    if a_mask_zyx.sum() == 0 or b_mask_zyx.sum() == 0:
        return 0.0
    sp_zyx = to_sampling_zyx(sp_xyz)
    dt = distance_transform_edt((b_mask_zyx == 0).astype(np.uint8), sampling=sp_zyx)
    a_surf = surface_voxels(a_mask_zyx, iters=erode_iters).astype(bool)
    vals = dt[a_surf]
    return float(vals.min()) if vals.size else 0.0


def select_surface_near_maxilla(
    target_surf_zyx: np.ndarray,
    maxilla_mask_zyx: np.ndarray,
    sp_xyz: Tuple[float, float, float],
    near_mm: float
) -> np.ndarray:
    if target_surf_zyx.sum() == 0 or maxilla_mask_zyx.sum() == 0:
        return np.zeros_like(target_surf_zyx, dtype=np.uint8)
    sp_zyx = to_sampling_zyx(sp_xyz)
    dt_to_maxilla = distance_transform_edt((maxilla_mask_zyx == 0).astype(np.uint8), sampling=sp_zyx)  # mm
    sel = (target_surf_zyx.astype(bool) & (dt_to_maxilla <= near_mm))
    return sel.astype(np.uint8)


def phys_points_from_mask(mask_zyx: np.ndarray, sp_xyz: Tuple[float, float, float]) -> np.ndarray:
    coords = np.argwhere(mask_zyx > 0)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    sx, sy, sz = sp_xyz
    sp_zyx = np.array([sz, sy, sx], dtype=np.float64)
    return coords.astype(np.float64) * sp_zyx


def sample_line_points_mm(p0: np.ndarray, p1: np.ndarray, step_mm: float) -> np.ndarray:
    # p0,p1 in mm (z,y,x) physical in mm-aligned grid
    v = p1 - p0
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        return p0[None, :].copy()
    n = max(2, int(math.ceil(L / step_mm)) + 1)
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    return p0[None, :] + t[:, None] * v[None, :]


def label_at_physical_zyx(seg_img: sitk.Image, p_zyx_mm: np.ndarray) -> int:
    """
    Query label by converting (z,y,x)-mm coordinates back to index space via spacing.
    Uses array indexing (not origin/direction transforms), assuming segmentation and CT
    are aligned in index space (typical for exported full-FOV nnU-Net predictions).
    """
    sp_xyz = spacing_xyz(seg_img)
    sx, sy, sz = sp_xyz
    z = int(round(p_zyx_mm[0] / sz))
    y = int(round(p_zyx_mm[1] / sy))
    x = int(round(p_zyx_mm[2] / sx))
    arr = sitk_arr(seg_img)
    z = max(0, min(arr.shape[0] - 1, z))
    y = max(0, min(arr.shape[1] - 1, y))
    x = max(0, min(arr.shape[2] - 1, x))
    return int(arr[z, y, x])


def bone_window_path_thickness_mm(
    seg_img: sitk.Image,
    lacrimal_mask_zyx: np.ndarray,
    nasal_mask_zyx: np.ndarray,
    maxilla_mask_zyx: np.ndarray,
    sp_xyz: Tuple[float, float, float],
    near_mm: float,
    step_mm: float,
    erode_iters: int,
) -> Dict[str, float]:
    """
    Shortest line between:
      lacrimal surface voxels NEAR maxilla
      nasal surface voxels NEAR maxilla
    Then measure length along this segment that passes through maxilla label=2.
    """
    out = {
        "bone_path_thickness_mm": 0.0,
        "bone_path_total_len_mm": 0.0,
        "bone_path_frac_in_bone": 0.0,
        "bone_path_min_lacrimal_to_nasal_mm": 0.0
    }

    if lacrimal_mask_zyx.sum() == 0 or nasal_mask_zyx.sum() == 0 or maxilla_mask_zyx.sum() == 0:
        return out

    lac_surf = surface_voxels(lacrimal_mask_zyx, iters=erode_iters)
    nas_surf = surface_voxels(nasal_mask_zyx, iters=erode_iters)

    lac_surf_near = select_surface_near_maxilla(lac_surf, maxilla_mask_zyx, sp_xyz, near_mm)
    nas_surf_near = select_surface_near_maxilla(nas_surf, maxilla_mask_zyx, sp_xyz, near_mm)

    # fallback if too strict: use full surfaces
    if lac_surf_near.sum() < 10:
        lac_surf_near = lac_surf
    if nas_surf_near.sum() < 10:
        nas_surf_near = nas_surf

    A = phys_points_from_mask(lac_surf_near, sp_xyz)  # (N,3) in (z,y,x) mm
    B = phys_points_from_mask(nas_surf_near, sp_xyz)

    if A.shape[0] == 0 or B.shape[0] == 0:
        return out

    # nearest pair via KDTree
    tree = cKDTree(B)
    dists, idx = tree.query(A, k=1, workers=-1)
    j = int(np.argmin(dists))
    p0 = A[j]
    p1 = B[int(idx[j])]
    min_dist = float(dists[j])
    out["bone_path_min_lacrimal_to_nasal_mm"] = min_dist

    # sample along segment
    pts = sample_line_points_mm(p0, p1, step_mm=step_mm)
    total_len = float(np.linalg.norm(p1 - p0))
    out["bone_path_total_len_mm"] = total_len

    # count points inside bone
    in_bone = []
    for p in pts:
        lab = label_at_physical_zyx(seg_img, p)
        in_bone.append(lab == LABEL_MAXILLA)
    in_bone = np.array(in_bone, dtype=bool)

    # thickness = length of segment portions in bone (approx by counting steps)
    if pts.shape[0] >= 2:
        seglens = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        mids = (in_bone[1:] & in_bone[:-1])
        thick = float(seglens[mids].sum()) if mids.any() else 0.0
    else:
        thick = 0.0

    out["bone_path_thickness_mm"] = thick
    out["bone_path_frac_in_bone"] = float(thick / total_len) if total_len > 1e-6 else 0.0
    return out


def dilate_mask_mm(mask_zyx: np.ndarray, sp_xyz: Tuple[float, float, float], radius_mm: float) -> np.ndarray:
    if mask_zyx.sum() == 0:
        return np.zeros_like(mask_zyx, dtype=np.uint8)
    sp_zyx = to_sampling_zyx(sp_xyz)
    dt = distance_transform_edt((mask_zyx == 0).astype(np.uint8), sampling=sp_zyx)
    return (dt <= radius_mm).astype(np.uint8)


def axial_slice_max_area(mask_zyx: np.ndarray) -> Optional[int]:
    if mask_zyx.sum() == 0:
        return None
    areas = mask_zyx.sum(axis=(1, 2))
    return int(np.argmax(areas))


def major_axis_angle_to_sagittal_deg(maxilla_half_slice_yx: np.ndarray) -> float:
    """
    In axial slice (Y,X):
      - compute 2D PCA major axis of maxilla_half
      - angle to mid-sagittal line direction in axial plane: along +Y (vector [1,0] in (y,x))
    Return in degrees in [0, 90].
    """
    pts = np.argwhere(maxilla_half_slice_yx > 0)
    if pts.shape[0] < 50:
        return 0.0
    pts = pts.astype(np.float64)
    pts -= pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts.T)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    v = v[:, order]
    major = v[:, 0]  # in (y,x)
    ref = np.array([1.0, 0.0], dtype=np.float64)  # +Y direction
    cosang = float(abs(np.dot(major, ref)) / (np.linalg.norm(major) * np.linalg.norm(ref) + 1e-12))
    cosang = max(0.0, min(1.0, cosang))
    ang = math.degrees(math.acos(cosang))
    if ang > 90.0:
        ang = 180.0 - ang
    return float(ang)


# =========================
# Feature extraction config container
# =========================
class FeatureConfig:
    def __init__(self, near_maxilla_mm: float, line_step_mm: float,
                 bone_burden_radius_mm: float, erode_iters: int):
        self.near_maxilla_mm = float(near_maxilla_mm)
        self.line_step_mm = float(line_step_mm)
        self.bone_burden_radius_mm = float(bone_burden_radius_mm)
        self.erode_iters = int(erode_iters)


# =========================
# Extract features per case
# =========================
def extract_case_features(seg_path: Path, ct_path: Path, cfg: FeatureConfig) -> Dict[str, Union[str, float]]:
    seg_img = sitk_read(seg_path)
    ct_img = sitk_read(ct_path)  # for spacing/size consistency

    seg = sitk_arr(seg_img).astype(np.int16)  # (Z,Y,X)
    sp = spacing_xyz(ct_img)                  # (sx,sy,sz)
    vv = voxel_volume_mm3(sp)
    size_xyz = ct_img.GetSize()               # (X,Y,Z)

    # masks (fullFOV)
    lac = (seg == LABEL_LACRIMAL).astype(np.uint8)
    maxilla = (seg == LABEL_MAXILLA).astype(np.uint8)
    nasal = (seg == LABEL_NASAL).astype(np.uint8)

    feats: Dict[str, Union[str, float]] = {}
    case_id = case_id_from_seg(seg_path.name)
    feats["case_id"] = case_id

    # -------------------------
    # surgical side + ipsilateral halves
    # -------------------------
    lac_cent = safe_centroid_zyx(lac)
    if lac_cent is None:
        feats["has_lacrimal"] = 0.0
        return feats

    feats["has_lacrimal"] = 1.0
    side = ipsilateral_side_from_lacrimal(lac_cent, size_xyz)
    feats["surgical_side"] = 0.0 if side == "L" else 1.0  # 0=Left, 1=Right

    nasal_ipsi = half_mask_by_side(nasal, side, size_xyz)
    maxilla_ipsi = half_mask_by_side(maxilla, side, size_xyz)

    # -------------------------
    # 2.1 Lacrimal Features
    # -------------------------
    lac_vox = int(lac.sum())
    lac_vol = lac_vox * vv
    lac_area = marching_cubes_surface_area(lac, sp)
    feats["lac_volume_mm3"] = float(lac_vol)
    feats["lac_surface_area_mm2"] = float(lac_area)
    feats["lac_sphericity"] = sphericity(lac_vol, lac_area)
    feats["lac_equiv_diameter_mm"] = equiv_diameter_mm(lac, sp)
    feats["lac_elongation_pca"] = pca_elongation_3d(lac, sp)

    # centroid (mm)
    lac_cent_mm = centroid_mm_from_zyx(lac_cent, sp)
    feats["lac_centroid_z_mm"] = float(lac_cent_mm[0])
    feats["lac_centroid_y_mm"] = float(lac_cent_mm[1])
    feats["lac_centroid_x_mm"] = float(lac_cent_mm[2])

    # -------------------------
    # 2.4 Nasal Space Features (ipsilateral only)
    # -------------------------
    nas_vox = int(nasal_ipsi.sum())
    nas_vol = nas_vox * vv
    nas_area = marching_cubes_surface_area(nasal_ipsi, sp)
    feats["nas_ipsi_volume_mm3"] = float(nas_vol)
    feats["nas_ipsi_surface_area_mm2"] = float(nas_area)

    nas_cent = safe_centroid_zyx(nasal_ipsi)
    if nas_cent is not None:
        nas_cent_mm = centroid_mm_from_zyx(nas_cent, sp)
        feats["nas_ipsi_centroid_z_mm"] = float(nas_cent_mm[0])
        feats["nas_ipsi_centroid_y_mm"] = float(nas_cent_mm[1])
        feats["nas_ipsi_centroid_x_mm"] = float(nas_cent_mm[2])
    else:
        feats["nas_ipsi_centroid_z_mm"] = 0.0
        feats["nas_ipsi_centroid_y_mm"] = 0.0
        feats["nas_ipsi_centroid_x_mm"] = 0.0

    # -------------------------
    # 2.5 Relative Position Features
    # -------------------------
    if nas_cent is not None:
        d_cent = float(np.linalg.norm((lac_cent - nas_cent) * np.array(to_sampling_zyx(sp), dtype=np.float64)))
        feats["dist_lac_cent_to_nas_cent_mm"] = d_cent
    else:
        feats["dist_lac_cent_to_nas_cent_mm"] = 0.0

    feats["min_dist_lac_surf_to_nas_surf_mm"] = min_surface_distance_mm(
        lac, nasal_ipsi, sp, erode_iters=cfg.erode_iters
    )

    # -------------------------
    # 2.2 Maxilla Thickness Features (surgery-relevant)
    # -------------------------
    path_stats = bone_window_path_thickness_mm(
        seg_img=seg_img,
        lacrimal_mask_zyx=lac,
        nasal_mask_zyx=nasal_ipsi,
        maxilla_mask_zyx=maxilla_ipsi,
        sp_xyz=sp,
        near_mm=cfg.near_maxilla_mm,
        step_mm=cfg.line_step_mm,
        erode_iters=cfg.erode_iters
    )
    feats.update(path_stats)

    # -------------------------
    # 2.3 Other Maxilla Frontal-process related features
    # -------------------------
    max_vox = int(maxilla_ipsi.sum())
    feats["max_ipsi_volume_mm3"] = float(max_vox * vv)
    max_area = marching_cubes_surface_area(maxilla_ipsi, sp)
    feats["max_ipsi_surface_area_mm2"] = float(max_area)

    lac_dil = dilate_mask_mm(lac, sp, radius_mm=cfg.bone_burden_radius_mm)
    bone_burden = (lac_dil.astype(bool) & maxilla_ipsi.astype(bool)).astype(np.uint8)
    feats["max_bone_burden_within_lac_dilate_mm3"] = float(int(bone_burden.sum()) * vv)

    lac_surf = surface_voxels(lac, iters=cfg.erode_iters).astype(bool)
    if maxilla_ipsi.sum() > 0 and lac_surf.any():
        sp_zyx = to_sampling_zyx(sp)
        dt_to_max = distance_transform_edt((maxilla_ipsi == 0).astype(np.uint8), sampling=sp_zyx)
        feats["lac_surf_area_near_maxilla_vox"] = float(np.sum(lac_surf & (dt_to_max <= max(sp_zyx))))  # ~within 1 voxel
    else:
        feats["lac_surf_area_near_maxilla_vox"] = 0.0

    # -------------------------
    # 2.5 Relative Position Feature (angle)
    # -------------------------
    z_star = axial_slice_max_area(lac)
    if z_star is None:
        feats["angle_maxilla_majoraxis_to_sagittal_deg"] = 0.0
        feats["lac_max_area_slice_z_index"] = -1.0
    else:
        feats["lac_max_area_slice_z_index"] = float(z_star)
        max_slice = maxilla_ipsi[z_star, :, :]
        feats["angle_maxilla_majoraxis_to_sagittal_deg"] = major_axis_angle_to_sagittal_deg(max_slice)

    return feats


# =========================
# Batch run
# =========================
def run_dataset(seg_dir: Path, ct_dir: Path, out_csv: Path, cfg: FeatureConfig) -> pd.DataFrame:
    seg_files = sorted([f for f in os.listdir(seg_dir) if f.endswith(".nii.gz")])
    rows: List[Dict[str, Union[str, float]]] = []

    for i, fn in enumerate(seg_files, 1):
        case_id = case_id_from_seg(fn)
        seg_path = seg_dir / fn
        ct_path = find_ct_path(ct_dir, case_id)

        if ct_path is None:
            print(f"[WARN] CT missing for {case_id} in {ct_dir}")
            continue

        print(f"[{i:04d}/{len(seg_files):04d}] {case_id}")
        try:
            feats = extract_case_features(seg_path, ct_path, cfg)
            rows.append(feats)
        except Exception as e:
            print(f"[FAIL] {case_id}: {e}")
            rows.append({"case_id": case_id, "failed": 1.0})

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f">> Saved: {out_csv}")
    return df


def main():
    args = build_parser().parse_args()

    cfg = FeatureConfig(
        near_maxilla_mm=args.near_maxilla_mm,
        line_step_mm=args.line_step_mm,
        bone_burden_radius_mm=args.bone_burden_radius_mm,
        erode_iters=args.erode_iters,
    )

    # Mode 1: single dataset
    if args.seg_dir and args.ct_dir and args.out_csv:
        run_dataset(
            seg_dir=Path(args.seg_dir),
            ct_dir=Path(args.ct_dir),
            out_csv=Path(args.out_csv),
            cfg=cfg
        )
        print(">> DONE (single dataset mode)")
        return

    # Mode 2: 3-dataset batch mode
    batch_args_present = any([
        args.internal_seg_dir, args.internal_ct_dir,
        args.external1_seg_dir, args.external1_ct_dir,
        args.external2_seg_dir, args.external2_ct_dir,
        args.out_dir
    ])

    if batch_args_present:
        required = [
            "internal_seg_dir", "internal_ct_dir",
            "external1_seg_dir", "external1_ct_dir",
            "external2_seg_dir", "external2_ct_dir",
            "out_dir"
        ]
        missing = [k for k in required if getattr(args, k) is None]
        if missing:
            raise ValueError(f"Batch mode requires all arguments. Missing: {missing}")

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        run_dataset(
            seg_dir=Path(args.internal_seg_dir),
            ct_dir=Path(args.internal_ct_dir),
            out_csv=out_dir / "internal_oof_features.csv",
            cfg=cfg
        )
        run_dataset(
            seg_dir=Path(args.external1_seg_dir),
            ct_dir=Path(args.external1_ct_dir),
            out_csv=out_dir / "external1_features.csv",
            cfg=cfg
        )
        run_dataset(
            seg_dir=Path(args.external2_seg_dir),
            ct_dir=Path(args.external2_ct_dir),
            out_csv=out_dir / "external2_features.csv",
            cfg=cfg
        )

        print(">> ALL DONE (batch mode)")
        return

    raise ValueError(
        "Provide either:\n"
        "  (A) --seg-dir --ct-dir --out-csv\n"
        "or\n"
        "  (B) --internal-*/--external1-*/--external2-* plus --out-dir"
    )


if __name__ == "__main__":
    main()