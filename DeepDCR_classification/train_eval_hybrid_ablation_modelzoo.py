# -*- coding: utf-8 -*-
"""
train_eval_hybrid_ablation_modelzoo.py

Ablation (3) x ModelZoo, Nested CV (outer eval / inner tuning), paper-style outputs.

Ablations:
  A1: clinical only
  A2: clinical + morph + thickness
  A3: clinical + morph + thickness + DL (full hybrid)

ModelZoo:
  - ElasticNet Logistic Regression
  - Linear SVM + probability calibration (CalibratedClassifierCV)
  - Random Forest
  - XGBoost (optional; auto-skip if not installed)

Windows/joblib safety:
  - Force matplotlib backend to "Agg" BEFORE importing pyplot.

Also supports:
  - Force integer-coded categorical columns to pandas 'category' dtype
  - External-set predicted probabilities CSV
  - Internal OOF probabilities CSV
  - ROC/PR/Calibration/DCA panel figure
  - Threshold sensitivity table
  - Missing-rate report
  - Summary CSV
"""

import os

# ------------------------------------------------------------------
# MUST BE BEFORE importing matplotlib.pyplot (Windows + joblib fix)
# ------------------------------------------------------------------
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    brier_score_loss, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# optional xgboost
try:
    import xgboost as xgb  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# -----------------------------
# TASKS
# -----------------------------
TASKS = {
    "difficulty": {
        "label_col": "target",
        "pos_label": 1,
        "name": "difficulty",
    },
    "outcome_failure": {
        "label_col": "result",
        "pos_label": 2,
        "name": "outcome_failure",
    }
}

# -----------------------------
# FORCE-CATEGORICAL COLUMNS
# -----------------------------
FORCE_CATEGORICAL_COLS = [
    "surgical_eye",
    "sex",
    "symptoms_eye",
    "severity_of_symptoms(MUNK)",
    "previous_treatment_history",
    "systemic_medical_history",
]


# =============================================================================
# CLI / Config
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Nested-CV ablation + model-zoo training/evaluation for hybrid tabular classification."
    )
    p.add_argument("--internal-csv", required=True, help="Internal cohort CSV")
    p.add_argument("--external1-csv", required=True, help="External cohort 1 CSV")
    p.add_argument("--external2-csv", required=True, help="External cohort 2 CSV")
    p.add_argument(
        "--feature-group-json",
        default=str(Path(__file__).with_name("feature_groups.json")),
        help="Path to feature_groups.json (default: same directory as script)"
    )
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["difficulty", "outcome_failure"],
        choices=["difficulty", "outcome_failure"],
        help="Tasks to run (default: both)"
    )
    p.add_argument("--outer-splits", type=int, default=5, help="Outer CV folds (default: 5)")
    p.add_argument("--inner-splits", type=int, default=3, help="Inner CV folds for nested CV (default: 3)")
    p.add_argument("--final-inner-splits", type=int, default=5, help="Inner CV folds for final-model tuning (default: 5)")
    p.add_argument("--n-jobs", type=int, default=-1, help="GridSearchCV n_jobs (default: -1)")
    p.add_argument("--random-seed", type=int, default=42, help="Random seed (default: 42)")
    return p.parse_args()


# =============================================================================
# Utilities
# =============================================================================
def _read_csv_auto(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path, encoding="utf-8", errors="ignore")


def _to_binary_y(df: pd.DataFrame, task: str) -> np.ndarray:
    spec = TASKS[task]
    col = spec["label_col"]
    if col not in df.columns:
        raise KeyError(f"Label column '{col}' not found in {task} CSV.")

    y_raw = df[col].values
    if task == "difficulty":
        y = pd.to_numeric(pd.Series(y_raw), errors="coerce").values
        if np.nanmax(y) > 1:
            raise ValueError("difficulty/target should be 0/1.")
        return y.astype(int)

    y_num = pd.to_numeric(pd.Series(y_raw), errors="coerce").values
    return (y_num == 2).astype(int)


def _clean_feature_groups(df_all_cols: List[str], json_path: Path) -> Dict[str, List[str]]:
    if not json_path.exists():
        raise FileNotFoundError(
            f"Missing {json_path}. Create feature_groups.json with keys: clinical/morph/thickness/dl."
        )

    groups = json.loads(json_path.read_text(encoding="utf-8"))

    cleaned: Dict[str, List[str]] = {}
    for g, cols in groups.items():
        cols = cols or []
        out, seen = [], set()
        for c in cols:
            if c is None:
                continue
            c = str(c).strip()
            if c == "":
                continue
            if c in seen:
                continue
            out.append(c)
            seen.add(c)
        cleaned[g] = out

    all_cols = set(df_all_cols)
    for g, cols in cleaned.items():
        missing = [c for c in cols if c not in all_cols]
        if missing:
            raise ValueError(f"[feature_groups.json] group '{g}' has missing columns in CSV: {missing[:30]}")
    return cleaned


def _bootstrap_ci_metric(y_true: np.ndarray, y_prob: np.ndarray, metric: str,
                         n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)

    def _score(yt, yp):
        if metric == "auroc":
            return roc_auc_score(yt, yp)
        if metric == "auprc":
            return average_precision_score(yt, yp)
        raise ValueError(metric)

    base = _score(y_true, y_prob)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        boots.append(_score(yt, y_prob[idx]))

    boots = np.asarray(boots, dtype=float)
    if boots.size < 50:
        return float(base), np.nan, np.nan

    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(base), lo, hi


def _calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    eps = 1e-6
    p = np.clip(y_prob, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    X = np.vstack([np.ones_like(logit), logit]).T
    y = y_true.astype(float)

    beta = np.zeros(2)
    for _ in range(50):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu)
        z = eta + (y - mu) / np.maximum(W, 1e-9)

        XtW = X.T * W
        H = XtW @ X
        g = XtW @ z
        try:
            beta_new = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new
            break
        beta = beta_new

    intercept = float(beta[0])
    slope = float(beta[1])
    return slope, intercept


def _youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def _dca_net_benefit(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    y_true = y_true.astype(int)
    n = len(y_true)
    out = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        w = t / (1 - t + 1e-12)
        out.append((tp / n) - (fp / n) * w)
    return np.array(out, dtype=float)


def _nature_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.grid(False)


def _save_fig(fig, path_pdf: Path, path_png: Path, dpi=600):
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def _force_categoricals(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Force selected columns to pandas Categorical if they exist.
    Keeps NaN as NaN; works for int/float/str encoded categories.
    """
    df2 = df.copy()
    for c in cols:
        if c in df2.columns:
            df2[c] = df2[c].astype("category")
    return df2


# =============================================================================
# Preprocess + Models
# =============================================================================
@dataclass
class ModelSpec:
    name: str
    estimator: object
    param_grid: Dict[str, List]


def _make_onehot_dense():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _make_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    # detect categorical by dtype AFTER we forced categoricals in load_dataset()
    cat_cols = [c for c in X.columns if str(X[c].dtype) in ("object", "category", "bool")]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
        ("onehot", _make_onehot_dense()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _calibrated_linear_svm(random_seed: int) -> object:
    base = LinearSVC(class_weight="balanced", dual=True, random_state=random_seed)
    try:
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)


def build_modelzoo(task: str, random_seed: int) -> List[ModelSpec]:
    # task is currently not used but kept for future task-specific grids
    _ = task
    specs: List[ModelSpec] = []

    lr = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        class_weight="balanced",
        max_iter=8000,
        tol=1e-4,
        random_state=random_seed,
    )
    specs.append(ModelSpec(
        name="ElasticNetLR",
        estimator=lr,
        param_grid={
            "clf__C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
            "clf__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
        }
    ))

    specs.append(ModelSpec(
        name="LinearSVM-Cal",
        estimator=_calibrated_linear_svm(random_seed),
        param_grid={}  # built dynamically
    ))

    rf = RandomForestClassifier(
        n_estimators=800,
        random_state=random_seed,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    specs.append(ModelSpec(
        name="RandomForest",
        estimator=rf,
        param_grid={
            "clf__max_depth": [None, 3, 5, 8],
            "clf__min_samples_leaf": [1, 2, 5],
            "clf__max_features": ["sqrt", 0.3, 0.6],
        }
    ))

    if HAS_XGB:
        xgbc = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=1200,
            random_state=random_seed,
            n_jobs=-1,
            tree_method="hist",
        )
        specs.append(ModelSpec(
            name="XGBoost",
            estimator=xgbc,
            param_grid={
                "clf__max_depth": [2, 3, 4],
                "clf__learning_rate": [0.01, 0.05, 0.1],
                "clf__subsample": [0.7, 0.9, 1.0],
                "clf__colsample_bytree": [0.6, 0.8, 1.0],
            }
        ))

    return specs


def build_pipeline(preprocess: ColumnTransformer, spec: ModelSpec) -> Tuple[Pipeline, Dict[str, List]]:
    pipe = Pipeline(steps=[("pre", preprocess), ("clf", spec.estimator)])

    if spec.name != "LinearSVM-Cal":
        return pipe, spec.param_grid

    # ONLY ONE valid key (auto-detect), to avoid GridSearchCV crashing in joblib workers
    C_list = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    candidates = ["clf__estimator__C", "clf__base_estimator__C", "clf__classifier__C"]

    valid_key = None
    for k in candidates:
        try:
            pipe.set_params(**{k: C_list[0]})
            valid_key = k
            break
        except Exception:
            continue

    if valid_key is None:
        keys = list(pipe.get_params().keys())
        raise RuntimeError(
            "Cannot find valid parameter path for LinearSVC(C) inside CalibratedClassifierCV.\n"
            f"Tried: {candidates}\n"
            "Pipeline param keys (first 120):\n" + "\n".join(keys[:120])
        )

    return pipe, {valid_key: C_list}


# =============================================================================
# Feature sets (ablation)
# =============================================================================
def build_ablation_sets(groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    clinical = groups.get("clinical", [])
    morph = groups.get("morph", [])
    thickness = groups.get("thickness", [])
    dl = groups.get("dl", [])

    return {
        "A1_clinical": clinical,
        "A2_clin_morph_thick": clinical + morph + thickness,
        "A3_full_hybrid": clinical + morph + thickness + dl,
    }


# =============================================================================
# Plotting
# =============================================================================
def fig2_panel(y_true: np.ndarray, y_prob: np.ndarray, title: str,
               out_pdf: Path, out_png: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    frac_pos = float(np.mean(y_true))
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")

    thr = np.linspace(0.01, 0.99, 99)
    nb_model = _dca_net_benefit(y_true, y_prob, thr)
    prev = frac_pos
    nb_all = prev - (1 - prev) * (thr / (1 - thr + 1e-12))
    nb_none = np.zeros_like(thr)

    fig = plt.figure(figsize=(7.0, 2.2))
    gs = fig.add_gridspec(1, 4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])

    ax1.plot(fpr, tpr, lw=1.8)
    ax1.plot([0, 1], [0, 1], lw=1.0, linestyle="--")
    ax1.set_xlabel("False positive rate", fontsize=9)
    ax1.set_ylabel("True positive rate", fontsize=9)
    ax1.set_title("ROC", fontsize=10)
    _nature_axes(ax1)

    ax2.plot(rec, prec, lw=1.8)
    ax2.hlines(frac_pos, 0, 1, linestyles="--", lw=1.0)
    ax2.set_xlabel("Recall", fontsize=9)
    ax2.set_ylabel("Precision", fontsize=9)
    ax2.set_title("PR", fontsize=10)
    _nature_axes(ax2)

    ax3.plot(prob_pred, prob_true, marker="o", lw=1.6)
    ax3.plot([0, 1], [0, 1], lw=1.0, linestyle="--")
    ax3.set_xlabel("Predicted", fontsize=9)
    ax3.set_ylabel("Observed", fontsize=9)
    ax3.set_title("Calibration", fontsize=10)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    _nature_axes(ax3)

    ax4.plot(thr, nb_model, lw=1.8, label="Model")
    ax4.plot(thr, nb_all, lw=1.2, linestyle="--", label="Treat all")
    ax4.plot(thr, nb_none, lw=1.2, linestyle=":", label="Treat none")
    ax4.set_xlabel("Threshold probability", fontsize=9)
    ax4.set_ylabel("Net benefit", fontsize=9)
    ax4.set_title("DCA", fontsize=10)
    _nature_axes(ax4)
    ax4.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle(title, fontsize=11, y=1.05)
    _save_fig(fig, out_pdf, out_png, dpi=600)


def threshold_sensitivity_table(y_true: np.ndarray, y_prob: np.ndarray,
                                thresholds: List[float]) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-12)
        spec = tn / (tn + fp + 1e-12)
        ppv = tp / (tp + fp + 1e-12)
        npv = tn / (tn + fn + 1e-12)
        rows.append({
            "threshold": t,
            "sensitivity": sens,
            "specificity": spec,
            "PPV": ppv,
            "NPV": npv,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn
        })
    return pd.DataFrame(rows)


# =============================================================================
# Core nested CV runner
# =============================================================================
def run_nested_cv_one(
    X: pd.DataFrame, y: np.ndarray,
    model_spec: ModelSpec,
    inner_splits: int = 3,
    outer_splits: int = 5,
    random_seed: int = 42,
    n_jobs: int = -1
) -> Tuple[np.ndarray, List[dict]]:
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_seed)
    oof_prob = np.full(len(y), np.nan, dtype=float)
    fold_logs = []

    for i, (tr, te) in enumerate(outer.split(X, y), start=1):
        Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
        ytr, yte = y[tr], y[te]

        preprocess = _make_preprocess(Xtr)
        pipe, grid = build_pipeline(preprocess, model_spec)

        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_seed + i)

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring="roc_auc",
            cv=inner,
            n_jobs=n_jobs,
            refit=True,
        )
        gs.fit(Xtr, ytr)

        if hasattr(gs.best_estimator_, "predict_proba"):
            prob = gs.best_estimator_.predict_proba(Xte)[:, 1]
        else:
            score = gs.best_estimator_.decision_function(Xte)
            prob = 1 / (1 + np.exp(-score))

        oof_prob[te] = prob

        auc = roc_auc_score(yte, prob) if len(np.unique(yte)) == 2 else np.nan
        ap = average_precision_score(yte, prob) if len(np.unique(yte)) == 2 else np.nan
        fold_logs.append({
            "outer_fold": i,
            "auc": auc,
            "ap": ap,
            "best_params": gs.best_params_
        })
        print(f"[{model_spec.name}] Outer fold {i}/{outer_splits} | AUC={auc:.4f} AP={ap:.4f}")

    return oof_prob, fold_logs


def fit_final_model(X: pd.DataFrame, y: np.ndarray, model_spec: ModelSpec,
                    inner_splits: int = 5, random_seed: int = 42, n_jobs: int = -1):
    preprocess = _make_preprocess(X)
    pipe, grid = build_pipeline(preprocess, model_spec)
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_seed)

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        scoring="roc_auc",
        cv=inner,
        n_jobs=n_jobs,
        refit=True,
    )
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_, float(gs.best_score_)


def eval_dataset(tag: str, model, X: pd.DataFrame, y: np.ndarray, random_seed: int = 42) -> dict:
    """
    Returns metrics + probabilities (for saving *_probs.csv)
    """
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        score = model.decision_function(X)
        prob = 1 / (1 + np.exp(-score))

    auroc, lo, hi = _bootstrap_ci_metric(y, prob, "auroc", seed=random_seed)
    auprc, plo, phi = _bootstrap_ci_metric(y, prob, "auprc", seed=random_seed)
    brier = brier_score_loss(y, prob)
    slope, intercept = _calibration_slope_intercept(y, prob)
    return {
        "tag": tag, "n": int(len(y)),
        "auroc": float(auroc), "auroc_lo": float(lo), "auroc_hi": float(hi),
        "auprc": float(auprc), "auprc_lo": float(plo), "auprc_hi": float(phi),
        "brier": float(brier),
        "cal_slope": float(slope),
        "cal_intercept": float(intercept),
        "prob": np.asarray(prob, dtype=float),
    }


def load_dataset(path: Path, task: str) -> Tuple[pd.DataFrame, np.ndarray]:
    df = _read_csv_auto(path)
    y = _to_binary_y(df, task)

    # drop label cols
    drop_cols = ["target", "result"]
    keep = [c for c in df.columns if c not in drop_cols]
    df = df[keep].copy()

    # force integer-coded categoricals into 'category'
    df = _force_categoricals(df, FORCE_CATEGORICAL_COLS)

    return df, y


def missing_report(X: pd.DataFrame) -> pd.DataFrame:
    miss = X.isna().mean().sort_values(ascending=False)
    return miss.reset_index().rename(columns={"index": "feature", 0: "missing_rate"})


# =============================================================================
# Main runner
# =============================================================================
def run_task(task: str, cfg):
    print("\n" + "=" * 70)
    print(f">>> TASK = {task} | Ablation(3) x ModelZoo")
    print("=" * 70)

    df_int, y_int = load_dataset(cfg["internal_csv"], task)
    df_e1, y_e1 = load_dataset(cfg["external1_csv"], task)
    df_e2, y_e2 = load_dataset(cfg["external2_csv"], task)

    groups = _clean_feature_groups(df_int.columns.tolist(), cfg["feature_group_json"])
    ablations = build_ablation_sets(groups)
    model_specs = build_modelzoo(task, cfg["random_seed"])

    task_dir = cfg["outdir"] / task
    task_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for abl_name, feat_cols in ablations.items():
        X_int = df_int[feat_cols].copy()
        X_e1 = df_e1[feat_cols].copy()
        X_e2 = df_e2[feat_cols].copy()

        mr = missing_report(X_int)
        mr.to_csv(task_dir / f"{task}_{abl_name}_missing_rate.csv", index=False)

        for spec in model_specs:
            print(f"\n--- {task} | {abl_name} | {spec.name} ---")

            oof_prob, fold_logs = run_nested_cv_one(
                X_int, y_int, spec,
                inner_splits=cfg["inner_splits"],
                outer_splits=cfg["outer_splits"],
                random_seed=cfg["random_seed"],
                n_jobs=cfg["n_jobs"],
            )

            auroc, lo, hi = _bootstrap_ci_metric(y_int, oof_prob, "auroc", seed=cfg["random_seed"])
            auprc, plo, phi = _bootstrap_ci_metric(y_int, oof_prob, "auprc", seed=cfg["random_seed"])
            brier = brier_score_loss(y_int, oof_prob)
            slope, intercept = _calibration_slope_intercept(y_int, oof_prob)
            thr = _youden_threshold(y_int, oof_prob)

            final_model, best_params, best_cv_auc = fit_final_model(
                X_int, y_int, spec,
                inner_splits=cfg["final_inner_splits"],
                random_seed=cfg["random_seed"],
                n_jobs=cfg["n_jobs"],
            )

            ext1 = eval_dataset("external1", final_model, X_e1, y_e1, random_seed=cfg["random_seed"])
            ext2 = eval_dataset("external2", final_model, X_e2, y_e2, random_seed=cfg["random_seed"])

            # External probs
            pd.DataFrame({"y_true": y_e1.astype(int), "y_prob": ext1["prob"].astype(float)}).to_csv(
                task_dir / f"{task}_{abl_name}_{spec.name}_External1_probs.csv", index=False
            )
            pd.DataFrame({"y_true": y_e2.astype(int), "y_prob": ext2["prob"].astype(float)}).to_csv(
                task_dir / f"{task}_{abl_name}_{spec.name}_External2_probs.csv", index=False
            )

            # Internal OOF probs
            pd.DataFrame({"y_true": y_int.astype(int), "y_prob": oof_prob.astype(float)}).to_csv(
                task_dir / f"{task}_{abl_name}_{spec.name}_internal_oof_probs.csv", index=False
            )

            # Threshold sensitivity analysis (on OOF)
            ts = threshold_sensitivity_table(
                y_int, oof_prob,
                thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, float(thr), 0.6, 0.7, 0.8, 0.9]
            )
            ts.to_csv(task_dir / f"{task}_{abl_name}_{spec.name}_threshold_sensitivity.csv", index=False)

            # Panel figure (OOF)
            fig_pdf = task_dir / f"Fig_{task}_{abl_name}_{spec.name}_OOF_ROC_PR_Cal_DCA.pdf"
            fig_png = task_dir / f"Fig_{task}_{abl_name}_{spec.name}_OOF_ROC_PR_Cal_DCA.png"
            fig2_panel(
                y_int, oof_prob,
                title=f"{task} | {abl_name} | {spec.name} (Internal OOF)",
                out_pdf=fig_pdf, out_png=fig_png
            )

            # Save fold logs
            pd.DataFrame(fold_logs).to_csv(task_dir / f"{task}_{abl_name}_{spec.name}_outer_fold_logs.csv", index=False)

            summary_rows.append({
                "task": task,
                "ablation": abl_name,
                "model": spec.name,
                "internal_AUROC": auroc, "internal_AUROC_lo": lo, "internal_AUROC_hi": hi,
                "internal_AUPRC": auprc, "internal_AUPRC_lo": plo, "internal_AUPRC_hi": phi,
                "internal_Brier": brier,
                "internal_cal_slope": slope,
                "internal_cal_intercept": intercept,
                "youden_thr_oof": thr,
                "final_best_params": json.dumps(best_params, ensure_ascii=False),
                "final_innerCV_best_auc": best_cv_auc,
                "external1_AUROC": ext1["auroc"], "external1_AUROC_lo": ext1["auroc_lo"], "external1_AUROC_hi": ext1["auroc_hi"],
                "external1_AUPRC": ext1["auprc"], "external1_AUPRC_lo": ext1["auprc_lo"], "external1_AUPRC_hi": ext1["auprc_hi"],
                "external1_Brier": ext1["brier"], "external1_cal_slope": ext1["cal_slope"], "external1_cal_intercept": ext1["cal_intercept"],
                "external2_AUROC": ext2["auroc"], "external2_AUROC_lo": ext2["auroc_lo"], "external2_AUROC_hi": ext2["auroc_hi"],
                "external2_AUPRC": ext2["auprc"], "external2_AUPRC_lo": ext2["auprc_lo"], "external2_AUPRC_hi": ext2["auprc_hi"],
                "external2_Brier": ext2["brier"], "external2_cal_slope": ext2["cal_slope"], "external2_cal_intercept": ext2["cal_intercept"],
            })

            print(f"[INTERNAL OOF] AUROC={auroc:.3f} ({lo:.3f}-{hi:.3f}) | AUPRC={auprc:.3f} ({plo:.3f}-{phi:.3f})")
            print(f"[INTERNAL OOF] Brier={brier:.4f} | Cal slope={slope:.3f} intercept={intercept:.3f} | Youden thr={thr:.3f}")
            print(f"[FINAL] best_params={best_params} | innerCV(AUC)={best_cv_auc:.4f}")
            print(f"[EXTERNAL1] AUROC={ext1['auroc']:.3f} ({ext1['auroc_lo']:.3f}-{ext1['auroc_hi']:.3f}) | AUPRC={ext1['auprc']:.3f}")
            print(f"[EXTERNAL2] AUROC={ext2['auroc']:.3f} ({ext2['auroc_lo']:.3f}-{ext2['auroc_hi']:.3f}) | AUPRC={ext2['auprc']:.3f}")

    summary = pd.DataFrame(summary_rows)
    summary_path = task_dir / f"{task}_ablation_modelzoo_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n[OK] Saved summary: {summary_path}")
    if not HAS_XGB:
        print("[NOTE] xgboost not installed -> XGBoost model skipped (expected behavior).")


def main():
    args = parse_args()

    # propagate global seed to module-level behavior if desired
    global RANDOM_SEED
    RANDOM_SEED = int(args.random_seed)
    np.random.seed(RANDOM_SEED)

    cfg = {
        "internal_csv": Path(args.internal_csv),
        "external1_csv": Path(args.external1_csv),
        "external2_csv": Path(args.external2_csv),
        "feature_group_json": Path(args.feature_group_json),
        "outdir": Path(args.outdir),
        "outer_splits": int(args.outer_splits),
        "inner_splits": int(args.inner_splits),
        "final_inner_splits": int(args.final_inner_splits),
        "n_jobs": int(args.n_jobs),
        "random_seed": int(args.random_seed),
    }

    cfg["outdir"].mkdir(parents=True, exist_ok=True)

    for p in [cfg["internal_csv"], cfg["external1_csv"], cfg["external2_csv"]]:
        if not p.exists():
            raise FileNotFoundError(f"Missing CSV: {p}")

    if not cfg["feature_group_json"].exists():
        raise FileNotFoundError(f"Missing feature group JSON: {cfg['feature_group_json']}")

    for task in args.tasks:
        run_task(task, cfg)

    print("\nDONE. Outputs in:", cfg["outdir"])


if __name__ == "__main__":
    main()