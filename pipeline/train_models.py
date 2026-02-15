# pipeline/07_train_models.py
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# sklearn imports (fail fast with a clear message if not installed)
try:
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
        roc_auc_score,
        log_loss,
        average_precision_score,
    )
except Exception as e:
    raise RuntimeError(
        "scikit-learn is required. Install with: pip install scikit-learn joblib\n"
        f"Original import error: {repr(e)}"
    )

try:
    import joblib
except Exception as e:
    raise RuntimeError(
        "joblib is required. Install with: pip install joblib\n"
        f"Original import error: {repr(e)}"
    )

from pipeline.config import Config, ensure_project_dirs


def _sanitize_target_name(target: str) -> str:
    s = str(target).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def _load_labels(labels_path: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_path)
    if "patient_id" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Labels file must contain columns ['patient_id','label']: {labels_path}")
    df = df[["patient_id", "label"]].copy()
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    return df


def _auto_find_labels_file(datasets_dir: Path, target: str, prefer_binary: bool) -> Path:
    key = _sanitize_target_name(target)
    candidates = sorted(datasets_dir.glob("labels_*.csv"))

    prim: List[Path] = []
    bina: List[Path] = []
    for p in candidates:
        stem = p.stem  # labels_<something> or labels_<something>_binary
        if not stem.startswith("labels_"):
            continue
        rest = stem[len("labels_"):]
        is_bin = False
        if rest.endswith("_binary"):
            is_bin = True
            rest = rest[: -len("_binary")]
        cand_key = _sanitize_target_name(rest)
        if cand_key == key:
            (bina if is_bin else prim).append(p)

    if prefer_binary and bina:
        return max(bina, key=lambda p: p.stat().st_mtime)
    if prim:
        return max(prim, key=lambda p: p.stat().st_mtime)
    if bina:
        return max(bina, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(
        f"Could not find labels file for target='{target}' under {datasets_dir}.\n"
        f"Searched labels_*.csv and matched by normalized key='{key}'."
    )


def _load_features_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Features parquet not found: {path}")
    df = pd.read_parquet(path)
    df.index = df.index.astype(str)
    df.index.name = "patient_id"
    return df


def _load_clinical_features(path: Path) -> pd.DataFrame:
    """
    Load clinical features created by pipeline.prepare_clinical_datasets.py
    Expected: patient_id column + numeric columns.
    Keeps only numeric cols.
    """
    if not path.exists():
        raise FileNotFoundError(f"Clinical features not found: {path}")

    if path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if "patient_id" not in df.columns:
        raise ValueError(f"Clinical features must have a 'patient_id' column: {path}")

    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["patient_id"], keep="first").set_index("patient_id")

    # keep numeric only
    num = df.select_dtypes(include=["number"]).copy()
    num.index = num.index.astype(str)
    num.index.name = "patient_id"
    return num


def _concat_features(*dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Inner join on patient_id index and concat columns.
    """
    if len(dfs) < 2:
        return dfs[0]
    common = dfs[0].index
    for d in dfs[1:]:
        common = common.intersection(d.index)
    out = pd.concat([d.loc[common] for d in dfs], axis=1)
    out.index = out.index.astype(str)
    out.index.name = "patient_id"
    return out


def _infer_pet_subset(df_fused: pd.DataFrame, df_pet: Optional[pd.DataFrame] = None) -> List[str]:
    if df_pet is not None and len(df_pet) > 0:
        return list(df_pet.index.astype(str))

    pet_cols = [c for c in df_fused.columns if c.startswith("pet__")]
    if not pet_cols:
        return []
    has_any_pet = df_fused[pet_cols].notna().any(axis=1)
    return list(df_fused.index[has_any_pet].astype(str))


@dataclass
class CVResult:
    metrics_summary: Dict[str, object]
    fold_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame


def _make_model(model_name: str, random_state: int, C: float, l1_ratio: float) -> Pipeline:
    model_name = model_name.lower().strip()

    if model_name in ("logreg", "logreg_l2", "logistic", "logistic_regression"):
        clf = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            penalty="l2",
            C=C,
            random_state=random_state,
        )
    elif model_name in ("logreg_l1", "l1"):
        clf = LogisticRegression(
            max_iter=8000,
            class_weight="balanced",
            solver="saga",
            penalty="l1",
            C=C,
            random_state=random_state,
        )
    elif model_name in ("elasticnet", "enet"):
        clf = LogisticRegression(
            max_iter=12000,
            class_weight="balanced",
            solver="saga",
            penalty="elasticnet",
            l1_ratio=l1_ratio,
            C=C,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model '{model_name}'. Supported: logreg_l2, logreg_l1, elasticnet")

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", clf),
        ]
    )
    return pipe


def _best_threshold_for_f1(y_true: np.ndarray, p1: np.ndarray) -> float:
    if len(y_true) < 10 or np.allclose(p1, p1[0]):
        return 0.5
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        y_pred = (p1 >= t).astype(int)
        f = f1_score(y_true, y_pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = float(t)
    return best_t


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    is_multiclass: bool,
    labels_sorted: List[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    if is_multiclass:
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
        metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        if y_proba is not None:
            try:
                metrics["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
            except Exception:
                metrics["roc_auc_ovr_macro"] = float("nan")
            try:
                metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=labels_sorted))
            except Exception:
                metrics["log_loss"] = float("nan")
    else:
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == 2:
            p1 = y_proba[:, 1]
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, p1))
            except Exception:
                metrics["roc_auc"] = float("nan")
            try:
                metrics["pr_auc"] = float(average_precision_score(y_true, p1))
            except Exception:
                metrics["pr_auc"] = float("nan")
            try:
                metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=labels_sorted))
            except Exception:
                metrics["log_loss"] = float("nan")

    return metrics


def _train_cv(
    X: pd.DataFrame,
    y_raw: pd.Series,
    model_name: str,
    n_splits: int,
    random_state: int,
    C: float,
    l1_ratio: float,
    threshold_mode: str,
) -> Tuple[CVResult, Pipeline, LabelEncoder]:
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))
    classes = list(le.classes_)
    is_multiclass = len(classes) > 2
    labels_sorted = list(range(len(classes)))

    # Guard against impossible CV splits when some classes are rare.
    if len(classes) < 2:
        raise ValueError(f"Need at least 2 classes to train. Got classes={classes}")
    counts = pd.Series(y).value_counts()
    min_class = int(counts.min()) if len(counts) else 0
    effective_splits = min(int(n_splits), int(min_class))
    if effective_splits < 2:
        raise ValueError(
            f"Not enough samples per class for StratifiedKFold: min_class_count={min_class}, requested_splits={n_splits}"
        )
    if effective_splits != n_splits:
        print(f"[cv] Adjusting n_splits from {n_splits} to {effective_splits} (min_class_count={min_class}).")

    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)

    oof_pred = np.full(shape=(len(y),), fill_value=-1, dtype=int)
    oof_proba = None
    fold_rows = []
    thresholds_used: List[float] = []

    X_values = X.values

    for fold, (tr, te) in enumerate(skf.split(X_values, y), start=1):
        X_tr, X_te = X_values[tr], X_values[te]
        y_tr, y_te = y[tr], y[te]

        m = _make_model(model_name, random_state=random_state, C=C, l1_ratio=l1_ratio)
        m.fit(X_tr, y_tr)

        y_proba_te = None
        if hasattr(m, "predict_proba"):
            try:
                y_proba_te = m.predict_proba(X_te)
            except Exception:
                y_proba_te = None

        if is_multiclass or y_proba_te is None or y_proba_te.shape[1] != 2:
            y_hat = m.predict(X_te)
            thr = float("nan")
        else:
            p1_te = y_proba_te[:, 1]
            if threshold_mode == "train_f1":
                y_proba_tr = m.predict_proba(X_tr)
                thr = _best_threshold_for_f1(y_tr, y_proba_tr[:, 1])
            else:
                thr = 0.5
            y_hat = (p1_te >= thr).astype(int)

        oof_pred[te] = y_hat

        if y_proba_te is not None:
            if oof_proba is None:
                oof_proba = np.full((len(y), y_proba_te.shape[1]), np.nan, dtype=float)
            oof_proba[te, :] = y_proba_te

        fold_metrics = _compute_metrics(
            y_true=y_te,
            y_pred=y_hat,
            y_proba=y_proba_te,
            is_multiclass=is_multiclass,
            labels_sorted=labels_sorted,
        )
        fold_metrics["fold"] = fold
        fold_metrics["n_train"] = int(len(tr))
        fold_metrics["n_test"] = int(len(te))
        if not is_multiclass:
            fold_metrics["threshold"] = float(thr) if thr == thr else float("nan")
            thresholds_used.append(float(thr) if thr == thr else 0.5)
        fold_rows.append(fold_metrics)

    fold_df = pd.DataFrame(fold_rows).sort_values("fold")

    summary: Dict[str, object] = {}
    for c in fold_df.columns:
        if c in ("fold", "n_train", "n_test"):
            continue
        if pd.api.types.is_numeric_dtype(fold_df[c]):
            summary[c] = float(np.nanmean(fold_df[c].values))

    cm = confusion_matrix(y, oof_pred, labels=labels_sorted)
    summary["confusion_matrix"] = cm.tolist()

    if not is_multiclass:
        summary["threshold_mode"] = threshold_mode
        if thresholds_used:
            summary["mean_threshold"] = float(np.mean(thresholds_used))

    oof_df = pd.DataFrame({"patient_id": X.index.astype(str), "y_true": y, "y_pred": oof_pred})
    if oof_proba is not None:
        for i, cls in enumerate(classes):
            oof_df[f"proba__{cls}"] = oof_proba[:, i]

    final_model = _make_model(model_name, random_state=random_state, C=C, l1_ratio=l1_ratio)
    final_model.fit(X_values, y)

    return CVResult(metrics_summary=summary, fold_metrics=fold_df, oof_predictions=oof_df), final_model, le


def _align_X_y(features: pd.DataFrame, labels: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Drop rows with missing/empty labels so we never train a bogus 'nan' class.
    lab = labels.dropna(subset=["patient_id", "label"]).copy()
    lab["patient_id"] = lab["patient_id"].astype(str).str.strip()
    lab["label"] = lab["label"].astype(str).str.strip()
    # Some CSVs may contain literal strings like 'nan'/'NA' — treat as missing.
    bad = {"", "nan", "na", "n/a", "none", "unknown"}
    lab = lab[~lab["label"].str.lower().isin(bad)]
    lab = lab.drop_duplicates(subset=["patient_id"], keep="last").set_index("patient_id")

    common = features.index.intersection(lab.index)
    X = features.loc[common].copy()
    y = lab.loc[common, "label"].copy()
    return X, y

def _drop_target_leaky_cols(df_clin: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Extra safety: if someone accidentally included target-derived cols in clinical features,
    drop them based on the training target.
    """
    t = (target or "").strip().lower()
    drop = set()

    for c in df_clin.columns:
        cl = c.lower()
        if cl == "label" or cl.endswith("_label") or cl.startswith("label_"):
            drop.add(c)

    # stage targets
    if "m-stage" in t or t.replace("_", "-") == "m-stage":
        drop |= {c for c in df_clin.columns if c.lower() in ("m_num", "m_bin")}
    if "t-stage" in t or t.replace("_", "-") == "t-stage":
        drop |= {c for c in df_clin.columns if c.lower() in ("t_num", "t_bin")}
    if "n-stage" in t or t.replace("_", "-") == "n-stage":
        drop |= {c for c in df_clin.columns if c.lower() in ("n_num", "n_bin")}

    # grading target
    if "grading" in t:
        drop |= {c for c in df_clin.columns if c.lower().startswith("grade_raw_") or c.lower() == "grading_num"}

    if drop:
        print(f"[clinical] Dropping leaky cols for target='{target}': {sorted(drop)}")
        df_clin = df_clin.drop(columns=list(drop), errors="ignore")

    return df_clin


def _save_bundle(out_dir: Path, name: str, res: CVResult, model: Pipeline, le: LabelEncoder) -> None:
    joblib.dump(model, out_dir / f"model_{name}.joblib")
    joblib.dump(le, out_dir / f"label_encoder_{name}.joblib")
    res.fold_metrics.to_csv(out_dir / f"cv_folds_{name}.csv", index=False)
    res.oof_predictions.to_csv(out_dir / f"oof_predictions_{name}.csv", index=False)


def _pick_default_clinical_features(datasets_dir: Path) -> Optional[Path]:
    """
    Prefer parquet, fallback to csv.
    """
    p1 = datasets_dir / "clinical_features.parquet"
    if p1.exists():
        return p1
    p2 = datasets_dir / "clinical_features.csv"
    if p2.exists():
        return p2
    return None


def _ensure_clinical_artifacts(
    cfg: Config,
    datasets_dir: Path,
    target: str,
    prefer_binary: bool,
    clinical_xlsx: str,
    clinical_sheet: str,
    update_manifest: bool,
    exclude_from_features_csv: str,
) -> None:
    """
    If clinical XLSX provided, build missing artifacts (clinical_features + labels) robustly.
    """
    if not clinical_xlsx.strip():
        return

    # Determine whether we need to build features and/or labels
    need_features = not (_pick_default_clinical_features(datasets_dir) is not None)

    need_labels = False
    if target.strip():
        try:
            _auto_find_labels_file(datasets_dir, target, prefer_binary=prefer_binary)
            need_labels = False
        except FileNotFoundError:
            need_labels = True

    if not (need_features or need_labels):
        return

    # Import lazily so this script still works if you don't use clinical prep
    try:
        from pipeline.prepare_clinical_datasets import prepare_clinical_datasets
    except Exception as e:
        raise RuntimeError(
            "Clinical auto-build requested but pipeline.prepare_clinical_datasets is not importable.\n"
            "Make sure you added pipeline/prepare_clinical_datasets.py with prepare_clinical_datasets().\n"
            f"Import error: {repr(e)}"
        )

    exclude_list = [x.strip() for x in exclude_from_features_csv.split(",") if x.strip()]

    print("[clinical] Auto-building missing clinical artifacts...")
    prepare_clinical_datasets(
        cfg=cfg,
        xlsx_path=Path(clinical_xlsx),
        sheet=clinical_sheet,
        out_dir=datasets_dir,
        targets=[target] if target.strip() else [],
        binary_stage_labels=bool(prefer_binary),
        update_manifest=bool(update_manifest),
        exclude_from_features=exclude_list,
    )
    print("[clinical] Done.")


def main() -> None:
    ap = argparse.ArgumentParser()

    # Labels
    ap.add_argument("--labels", type=str, default="", help="Path to labels CSV with columns: patient_id,label")
    ap.add_argument("--target", type=str, default="", help="Target name to auto-find labels file (e.g., 'M-Stage').")
    ap.add_argument("--prefer-binary", action="store_true", help="Prefer *_binary labels if available.")

    # Features
    ap.add_argument("--datasets-dir", type=str, default="", help="Override data/datasets directory.")
    ap.add_argument("--ct-features", type=str, default="tumor_bbox_features_ct.parquet")
    ap.add_argument("--fused-features", type=str, default="tumor_bbox_features_fused.parquet")
    ap.add_argument("--pet-features", type=str, default="tumor_bbox_features_pet.parquet")

    # Clinical
    ap.add_argument("--clinical-features", type=str, default="",
                    help="Optional clinical features file (csv/parquet) with patient_id column. "
                         "If omitted with --add-clinical, will try datasets_dir/clinical_features.parquet/csv.")
    ap.add_argument("--add-clinical", action="store_true",
                    help="If set, train Clinical-only and CT+Clinical (and fair PET-subset versions).")

    # Clinical auto-build (from XLSX)
    ap.add_argument("--clinical-xlsx", type=str, default="",
                    help="If set, can auto-build labels + clinical features into datasets_dir if missing.")
    ap.add_argument("--clinical-sheet", type=str, default="Clinical Data")
    ap.add_argument("--no-auto-clinical-build", dest="auto_clinical_build", action="store_false", default=True,
                    help="Disable auto-building clinical artifacts even if --clinical-xlsx is provided.")
    ap.add_argument("--no-update-manifest", dest="update_manifest", action="store_false", default=True,
                    help="Disable per-case manifest updates during clinical prep (default: updates enabled).")
    ap.add_argument("--exclude-from-features", type=str, default="",
                    help="Comma-separated canonical columns to exclude from clinical features (extra leakage guard).")

    # Training
    ap.add_argument("--model", type=str, default="logreg_l2", help="logreg_l2, logreg_l1, elasticnet")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--l1-ratio", type=float, default=0.5)
    ap.add_argument("--threshold", type=str, default="fixed_0_5",
                    choices=["fixed_0_5", "train_f1"])

    # Output
    ap.add_argument("--run-name", type=str, default="", help="Optional name for output folder.")
    ap.add_argument("--out-dir", type=str, default="", help="Override output directory.")

    args = ap.parse_args()

    cfg = Config()
    ensure_project_dirs(cfg)

    datasets_dir = Path(args.datasets_dir) if args.datasets_dir.strip() else (cfg.DATA_DIR / "datasets")
    if not datasets_dir.exists():
        raise FileNotFoundError(f"Datasets dir not found: {datasets_dir}")

    # Clinical auto-build (if requested)
    if args.auto_clinical_build and args.clinical_xlsx.strip():
        _ensure_clinical_artifacts(
            cfg=cfg,
            datasets_dir=datasets_dir,
            target=args.target.strip(),
            prefer_binary=bool(args.prefer_binary),
            clinical_xlsx=args.clinical_xlsx,
            clinical_sheet=args.clinical_sheet,
            update_manifest=bool(args.update_manifest),
            exclude_from_features_csv=args.exclude_from_features,
        )

    # Labels path
    if args.labels.strip():
        labels_path = Path(args.labels)
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
    else:
        if not args.target.strip():
            raise ValueError("Provide either --labels <file.csv> or --target <target-name>.")
        # after auto-build attempt, try again
        labels_path = _auto_find_labels_file(datasets_dir, args.target, prefer_binary=args.prefer_binary)

    labels_df = _load_labels(labels_path).dropna(subset=["label"])

    # Load imaging features
    ct_path = datasets_dir / args.ct_features
    fused_path = datasets_dir / args.fused_features
    pet_path = datasets_dir / args.pet_features

    df_ct = _load_features_parquet(ct_path)
    df_fused = _load_features_parquet(fused_path)
    df_pet = _load_features_parquet(pet_path) if pet_path.exists() else None

    # PET subset ids
    pet_case_ids = set(_infer_pet_subset(df_fused, df_pet))

    # Joins for imaging experiments
    X_ct_full, y_ct_full = _align_X_y(df_ct, labels_df)
    X_fused_all, y_fused_all = _align_X_y(df_fused, labels_df)

    pet_common = [pid for pid in X_fused_all.index.astype(str) if pid in pet_case_ids]
    X_ctpet = X_fused_all.loc[pet_common].copy()
    y_ctpet = y_fused_all.loc[pet_common].copy()

    # FAIR baseline: CT-only on the same PET subset
    pet_common_set = set(pet_common)
    pet_common_ct = [pid for pid in X_ct_full.index.astype(str) if pid in pet_common_set]
    X_ct_on_pet = X_ct_full.loc[pet_common_ct].copy()
    y_ct_on_pet = y_ct_full.loc[pet_common_ct].copy()

    # Clinical features (optional)
    df_clin = None
    clinical_features_path: Optional[Path] = None

    if args.add_clinical:
        if args.clinical_features.strip():
            p = Path(args.clinical_features)
            clinical_features_path = p if p.is_absolute() else (datasets_dir / p)
        else:
            # auto-pick defaults created by prepare_clinical_datasets.py
            clinical_features_path = _pick_default_clinical_features(datasets_dir)

        # If still missing and user provided XLSX, try another build attempt (features-only)
        if clinical_features_path is None and args.auto_clinical_build and args.clinical_xlsx.strip():
            _ensure_clinical_artifacts(
                cfg=cfg,
                datasets_dir=datasets_dir,
                target=args.target.strip(),  # harmless if labels already exist
                prefer_binary=bool(args.prefer_binary),
                clinical_xlsx=args.clinical_xlsx,
                clinical_sheet=args.clinical_sheet,
                update_manifest=bool(args.update_manifest),
                exclude_from_features_csv=args.exclude_from_features,
            )
            clinical_features_path = _pick_default_clinical_features(datasets_dir)

        if clinical_features_path is None:
            raise ValueError(
                "--add-clinical requested but no clinical features were found.\n"
                "Either pass --clinical-features <file> or pass --clinical-xlsx so it can auto-build."
            )

        df_clin = _load_clinical_features(clinical_features_path)
        df_clin = _drop_target_leaky_cols(df_clin, args.target)

    # FULL clinical joins
    X_clin_full = y_clin_full = None
    X_ctclin_full = y_ctclin_full = None

    # PET-subset clinical joins (fair)
    X_clin_on_pet = y_clin_on_pet = None
    X_ctclin_on_pet = y_ctclin_on_pet = None

    if args.add_clinical and df_clin is not None:
        X_clin_full, y_clin_full = _align_X_y(df_clin, labels_df)
        X_ctclin = _concat_features(df_ct, df_clin)
        X_ctclin_full, y_ctclin_full = _align_X_y(X_ctclin, labels_df)

        # Fair PET-subset versions
        pet_idx = pd.Index(pet_common)
        X_clin_on_pet_tmp = df_clin.loc[df_clin.index.intersection(pet_idx)].copy()
        X_clin_on_pet, y_clin_on_pet = _align_X_y(X_clin_on_pet_tmp, labels_df)

        X_ctclin_on_pet_tmp = X_ctclin_full.loc[X_ctclin_full.index.intersection(pet_idx)].copy()
        X_ctclin_on_pet, y_ctclin_on_pet = _align_X_y(X_ctclin_on_pet_tmp, labels_df)

    # Sanity
    if len(X_ct_full) < 20:
        raise RuntimeError(f"Too few labeled CT samples after join: {len(X_ct_full)}")
    if len(X_ctpet) < 20:
        raise RuntimeError(f"Too few labeled CT+PET samples after join: {len(X_ctpet)}")
    if len(X_ct_on_pet) != len(X_ctpet):
        print(f"WARNING: CT-on-PET subset size {len(X_ct_on_pet)} != CT+PET subset size {len(X_ctpet)}")

    # Output dirs
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
    base_name = args.run_name.strip() or f"train_{_sanitize_target_name(args.target) if args.target else labels_path.stem}_{stamp}"
    out_dir = Path(args.out_dir) if args.out_dir.strip() else (cfg.DATA_DIR / "models" / base_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Join report
    join_report = {
        "labels_file": str(labels_path),
        "n_labels_rows": int(len(labels_df)),
        "n_labels_unique": int(labels_df["patient_id"].nunique()),
        "n_ct_features": int(len(df_ct)),
        "n_fused_features": int(len(df_fused)),
        "n_ct_full_labeled": int(len(X_ct_full)),
        "n_pet_subset_labeled": int(len(X_ctpet)),
        "add_clinical": bool(args.add_clinical),
        "clinical_features_path": str(clinical_features_path) if clinical_features_path else "",
        "n_clin_full_labeled": int(len(X_clin_full)) if X_clin_full is not None else 0,
        "n_ct_clin_full_labeled": int(len(X_ctclin_full)) if X_ctclin_full is not None else 0,
        "n_clin_pet_subset_labeled": int(len(X_clin_on_pet)) if X_clin_on_pet is not None else 0,
        "n_ct_clin_pet_subset_labeled": int(len(X_ctclin_on_pet)) if X_ctclin_on_pet is not None else 0,
        "clinical_auto_build": bool(args.auto_clinical_build),
        "clinical_xlsx": str(args.clinical_xlsx) if args.clinical_xlsx else "",
        "clinical_sheet": str(args.clinical_sheet),
        "clinical_update_manifest": bool(args.update_manifest),
        "exclude_from_features": str(args.exclude_from_features),
    }
    (out_dir / "join_report.json").write_text(json.dumps(join_report, indent=2))

    # ---- Train experiments ----
    results = {}

    # 1) CT-only FULL
    res_ct_full, m_ct_full, le_ct_full = _train_cv(
        X_ct_full, y_ct_full, args.model, args.splits, args.seed, args.C, args.l1_ratio, args.threshold
    )
    _save_bundle(out_dir, "ct_only_full", res_ct_full, m_ct_full, le_ct_full)
    results["ct_only_full"] = {"n": int(len(X_ct_full)), "metrics": res_ct_full.metrics_summary}

    # 2) CT-only PET-subset (FAIR baseline)
    res_ct_pet, m_ct_pet, le_ct_pet = _train_cv(
        X_ct_on_pet, y_ct_on_pet, args.model, args.splits, args.seed, args.C, args.l1_ratio, args.threshold
    )
    _save_bundle(out_dir, "ct_only_pet_subset", res_ct_pet, m_ct_pet, le_ct_pet)
    results["ct_only_pet_subset"] = {"n": int(len(X_ct_on_pet)), "metrics": res_ct_pet.metrics_summary}

    # 3) CT+PET PET-subset
    res_ctpet, m_ctpet, le_ctpet = _train_cv(
        X_ctpet, y_ctpet, args.model, args.splits, args.seed, args.C, args.l1_ratio, args.threshold
    )
    _save_bundle(out_dir, "ct_pet_pet_subset", res_ctpet, m_ctpet, le_ctpet)
    results["ct_pet_pet_subset"] = {"n": int(len(X_ctpet)), "metrics": res_ctpet.metrics_summary}

    # 4+) Clinical models
    if args.add_clinical and df_clin is not None:
        # Clinical-only FULL
        res_clin, m_clin, le_clin = _train_cv(
            X_clin_full, y_clin_full, args.model, args.splits, args.seed, args.C, args.l1_ratio, args.threshold
        )
        _save_bundle(out_dir, "clinical_only_full", res_clin, m_clin, le_clin)
        results["clinical_only_full"] = {"n": int(len(X_clin_full)), "metrics": res_clin.metrics_summary}

        # CT + Clinical FULL
        res_ctclin_full, m_ctclin_full, le_ctclin_full = _train_cv(
            X_ctclin_full, y_ctclin_full, args.model, args.splits, args.seed, args.C, args.l1_ratio, args.threshold
        )
        _save_bundle(out_dir, "ct_clinical_full", res_ctclin_full, m_ctclin_full, le_ctclin_full)
        results["ct_clinical_full"] = {"n": int(len(X_ctclin_full)), "metrics": res_ctclin_full.metrics_summary}

        # Clinical-only PET-subset (fair)
        res_clin_pet, m_clin_pet, le_clin_pet = _train_cv(
            X_clin_on_pet, y_clin_on_pet, args.model, args.splits, args.seed, args.C, args.l1_ratio, args.threshold
        )
        _save_bundle(out_dir, "clinical_only_pet_subset", res_clin_pet, m_clin_pet, le_clin_pet)
        results["clinical_only_pet_subset"] = {"n": int(len(X_clin_on_pet)), "metrics": res_clin_pet.metrics_summary}

        # CT + Clinical PET-subset (fair)
        res_ctclin_pet, m_ctclin_pet, le_ctclin_pet = _train_cv(
            X_ctclin_on_pet, y_ctclin_on_pet, args.model, args.splits, args.seed, args.C, args.l1_ratio, args.threshold
        )
        _save_bundle(out_dir, "ct_clinical_pet_subset", res_ctclin_pet, m_ctclin_pet, le_ctclin_pet)
        results["ct_clinical_pet_subset"] = {"n": int(len(X_ctclin_on_pet)), "metrics": res_ctclin_pet.metrics_summary}

    # Save summary.json (all)
    summary = {
        "labels_file": str(labels_path),
        "target": args.target if args.target else "",
        "model": args.model,
        "splits": args.splits,
        "seed": args.seed,
        "C": args.C,
        "l1_ratio": args.l1_ratio,
        "threshold_mode": args.threshold,
        "feature_files": {
            "ct": str(ct_path),
            "fused": str(fused_path),
            "pet_subset_source": str(pet_path) if pet_path.exists() else "derived_from_fused",
            "clinical": str(clinical_features_path) if clinical_features_path else "",
        },
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # comparison_metrics.csv
    comp_rows = []
    for name, obj in results.items():
        row = {"model": name, "n": int(obj["n"])}
        for k, v in obj["metrics"].items():
            if isinstance(v, (int, float)):
                row[k] = float(v)
        comp_rows.append(row)
    pd.DataFrame(comp_rows).to_csv(out_dir / "comparison_metrics.csv", index=False)

    def _pretty(metrics: Dict[str, object]) -> str:
        if "roc_auc" in metrics:
            keys = ["roc_auc", "pr_auc", "f1", "balanced_accuracy", "accuracy"]
        elif "roc_auc_ovr_macro" in metrics:
            keys = ["roc_auc_ovr_macro", "f1_macro", "balanced_accuracy", "accuracy"]
        else:
            keys = ["balanced_accuracy", "accuracy"]
        parts = []
        for k in keys:
            v = metrics.get(k, None)
            if isinstance(v, (int, float)):
                parts.append(f"{k}={v:.4f}")
        if isinstance(metrics.get("mean_threshold", None), (int, float)):
            parts.append(f"mean_threshold={metrics['mean_threshold']:.3f}")
        return ", ".join(parts)

    # Print
    print("\n=== Training complete ===")
    print(f"Output dir: {out_dir}")
    print(f"Labels: {labels_path}")
    print(f"CT-only FULL samples: {len(X_ct_full)}")
    print(f"PET subset samples: {len(X_ctpet)}")
    print(f"Features: CT={X_ct_full.shape[1]} | Fused={X_ctpet.shape[1]}")
    if args.add_clinical and df_clin is not None:
        print(f"Clinical features: {df_clin.shape[1]} numeric cols")
    print()

    for key in [
        "ct_only_full",
        "ct_only_pet_subset",
        "ct_pet_pet_subset",
        "clinical_only_full",
        "ct_clinical_full",
        "clinical_only_pet_subset",
        "ct_clinical_pet_subset",
    ]:
        if key in results:
            print(f"{key} (mean over folds):")
            print(" ", _pretty(results[key]["metrics"]))


if __name__ == "__main__":
    main()
