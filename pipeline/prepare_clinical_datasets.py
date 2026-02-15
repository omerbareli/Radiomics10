from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pipeline.config import Config, ensure_project_dirs


# ----------------------------
# Column normalization + aliases
# ----------------------------

def _clean_colname(c: str) -> str:
    c = str(c).strip()
    c = c.replace("\uFF2D", "M")  # full-width Ｍ -> M
    c = c.replace("\u2013", "-")  # en dash -> hyphen
    c = re.sub(r"\s+", " ", c)
    return c


def _keyify(c: str) -> str:
    """Aggressive normalization for matching headers despite Excel chaos."""
    c = _clean_colname(c).lower()
    c = c.replace("-", " ")
    c = re.sub(r"[^a-z0-9]+", "", c)
    return c


# Canonical column names used downstream
CANON = {
    "patient_id_src": "NewPatientID",
    "sex": "Sex",
    "age": "Age",
    "weight": "weight (kg)",
    "t_stage": "T-Stage",
    "n_stage": "N-Stage",
    "m_stage": "M-Stage",
    "grading": "Histopathological grading",
    "smoking": "Smoking History",
}

# Aliases: map canonical -> possible header variants
ALIASES: Dict[str, List[str]] = {
    CANON["patient_id_src"]: ["NewPatientID", "PatientID", "Patient ID", "ID", "New Patient ID"],
    CANON["sex"]: ["Sex", "Gender"],
    CANON["age"]: ["Age", "Age at diagnosis", "Age at Diagnosis", "Age (years)"],
    CANON["weight"]: ["weight (kg)", "Weight (kg)", "Weight(kg)", "Weight", "Body weight (kg)"],
    CANON["t_stage"]: ["T-Stage", "T stage", "T", "T Stage"],
    CANON["n_stage"]: ["N-Stage", "N stage", "N", "N Stage"],
    CANON["m_stage"]: ["M-Stage", "M stage", "M", "M Stage"],
    CANON["grading"]: ["Histopathological grading", "Histopathological Grading", "Grading", "Grade"],
    CANON["smoking"]: ["Smoking History", "Smoking", "Smoker", "Smoking status", "Smoking Status"],
}


def _resolve_col(df: pd.DataFrame, wanted: str) -> Optional[str]:
    """
    Resolve a column name from df.columns given a desired name.
    Tries:
      1) exact match on cleaned name
      2) keyify match
      3) alias set
    """
    if wanted in df.columns:
        return wanted

    cols = list(df.columns)
    cleaned_map = {_clean_colname(c): c for c in cols}
    if _clean_colname(wanted) in cleaned_map:
        return cleaned_map[_clean_colname(wanted)]

    key_map = {_keyify(c): c for c in cols}
    k = _keyify(wanted)
    if k in key_map:
        return key_map[k]

    # try aliases
    for alias in ALIASES.get(wanted, []):
        if alias in df.columns:
            return alias
        ak = _keyify(alias)
        if ak in key_map:
            return key_map[ak]

    return None


def _require_cols(df: pd.DataFrame, needed: Sequence[str]) -> Dict[str, str]:
    """
    Return mapping canonical_name -> actual_column_name in df.
    Raises if not found.
    """
    mapping: Dict[str, str] = {}
    missing: List[str] = []
    for w in needed:
        col = _resolve_col(df, w)
        if col is None:
            missing.append(w)
        else:
            mapping[w] = col
    if missing:
        raise RuntimeError(f"Missing required clinical columns: {missing}\nFound columns: {list(df.columns)}")
    return mapping


# ----------------------------
# ID normalization
# ----------------------------

def _norm_id_to_Lxxxx(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper()
    if not s or s in {"NAN", "NONE"}:
        return None

    m = re.search(r"(?i)(?:^|[^A-Z0-9])([A-Z])\s*[-_ ]*\s*(\d{1,6})(?:$|[^A-Z0-9])", s)
    if m:
        letter = m.group(1).upper()
        digits = m.group(2)
        return f"{letter}{digits.zfill(4)}"

    m = re.search(r"(\d{1,6})", s)
    if m:
        return f"A{m.group(1).zfill(4)}"

    return None


# ----------------------------
# Clinical parsing
# ----------------------------

def _to_num(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def _extract_stage_num(val) -> Optional[int]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip().upper().replace(" ", "")
    if not s or s in {"NA", "N/A", "NONE", "UNKNOWN", "X", "MX", "TX", "NX"}:
        return None
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


def _stage_bin(num: Optional[int], kind: str) -> Optional[int]:
    if num is None:
        return None
    kind = kind.upper()
    if kind == "T":
        return int(num >= 3)
    if kind == "N":
        return int(num >= 1)
    if kind == "M":
        return int(num >= 1)
    return None


def _parse_grading(val) -> Optional[int]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip().upper()
    if not s or s in {"NA", "N/A", "NONE", "UNKNOWN"}:
        return None

    roman = {"I": 1, "II": 2, "III": 3, "IV": 4}
    if s in roman:
        return roman[s]

    m = re.search(r"(\d+)", s)
    if m:
        try:
            g = int(m.group(1))
            if 0 <= g <= 10:
                return g
        except Exception:
            pass
    return None


def _make_binary_stage(series: pd.Series, kind: str) -> pd.Series:
    n = series.apply(_extract_stage_num)
    if kind.upper() == "T":
        return n.apply(lambda x: np.nan if x is None else int(x >= 3))
    if kind.upper() == "N":
        return n.apply(lambda x: np.nan if x is None else int(x >= 1))
    if kind.upper() == "M":
        return n.apply(lambda x: np.nan if x is None else int(x >= 1))
    raise ValueError(kind)


# ----------------------------
# Manifest helpers
# ----------------------------

def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


# ----------------------------
# De-duplication
# ----------------------------

def _dedup_by_patient(df: pd.DataFrame, patient_col: str) -> Tuple[pd.DataFrame, dict]:
    """
    Enforce 1 row per patient_id.
    Strategy:
      - If a date-like column exists, keep the latest date per patient.
      - Else keep the last occurrence in the file (usually latest export).
    """
    info = {
        "rows_in": int(df.shape[0]),
        "patients_in": int(df[patient_col].nunique(dropna=True)),
        "dup_patients": 0,
        "strategy": "",
        "date_col_used": "",
    }

    df2 = df.copy()
    df2["_row_order"] = np.arange(len(df2))

    # try to find a date column
    date_candidates = [c for c in df2.columns if re.search(r"(date|time)", str(c), re.IGNORECASE)]
    date_col = ""
    for c in date_candidates:
        # try parse; accept if at least some parse successfully
        parsed = pd.to_datetime(df2[c], errors="coerce")
        if parsed.notna().sum() >= max(3, int(0.2 * len(df2))):
            df2["_parsed_date"] = parsed
            date_col = c
            break

    dup_patients = df2[patient_col].duplicated(keep=False).sum()
    info["dup_patients"] = int(dup_patients)

    if date_col:
        info["strategy"] = "keep_latest_by_date"
        info["date_col_used"] = str(date_col)
        # sort: patient, parsed_date, row_order
        df2 = df2.sort_values(
            [patient_col, "_parsed_date", "_row_order"],
            ascending=[True, True, True],
            na_position="first",
        )
        out = df2.drop_duplicates(subset=[patient_col], keep="last").drop(columns=["_row_order", "_parsed_date"])
        return out, info

    info["strategy"] = "keep_last_row"
    out = df2.sort_values("_row_order").drop_duplicates(subset=[patient_col], keep="last").drop(columns=["_row_order"])
    return out, info


# ----------------------------
# Feature building with leakage guard
# ----------------------------

def _safe_get_dummies(series: pd.Series, prefix: str, dummy_na: bool = True) -> pd.DataFrame:
    """
    One-hot without 'nan' string artifacts:
      - preserve NaN as NaN
      - strip strings
      - treat empty/whitespace as NA
    """
    s = series.astype("string").str.strip()
    s = s.replace("", pd.NA)
    return pd.get_dummies(s, prefix=prefix, dummy_na=dummy_na)


def _build_clinical_features(
    df: pd.DataFrame,
    colmap: Dict[str, str],
    exclude_cols: set[str],
) -> pd.DataFrame:
    """
    Build clinical_features.{csv,parquet}.
    exclude_cols are canonical column names to exclude from features (leakage guard).
    """
    # base numeric features
    out = pd.DataFrame()
    out["patient_id"] = df["patient_id"].astype(str)

    # age/weight/sex always okay unless excluded
    if CANON["age"] not in exclude_cols:
        out["age_num"] = df[colmap[CANON["age"]]].apply(_to_num)
    if CANON["weight"] not in exclude_cols:
        out["weight_kg_num"] = df[colmap[CANON["weight"]]].apply(_to_num)
    if CANON["sex"] not in exclude_cols:
        sex = df[colmap[CANON["sex"]]].astype("string").str.strip().str.upper()
        out["sex_male"] = sex.map({"M": 1, "MALE": 1, "1": 1, "F": 0, "FEMALE": 0, "0": 0})

    # TNM (both numeric and bins)
    if CANON["t_stage"] not in exclude_cols:
        t = df[colmap[CANON["t_stage"]]]
        out["t_num"] = t.apply(_extract_stage_num)
        out["t_bin"] = out["t_num"].apply(lambda x: _stage_bin(x, "T"))
    if CANON["n_stage"] not in exclude_cols:
        n = df[colmap[CANON["n_stage"]]]
        out["n_num"] = n.apply(_extract_stage_num)
        out["n_bin"] = out["n_num"].apply(lambda x: _stage_bin(x, "N"))
    if CANON["m_stage"] not in exclude_cols:
        m = df[colmap[CANON["m_stage"]]]
        out["m_num"] = m.apply(_extract_stage_num)
        out["m_bin"] = out["m_num"].apply(lambda x: _stage_bin(x, "M"))

    # grading features (guard against leakage)
    if CANON["grading"] not in exclude_cols:
        out["grading_num"] = df[colmap[CANON["grading"]]].apply(_parse_grading)
        grade_oh = _safe_get_dummies(df[colmap[CANON["grading"]]], prefix="grade_raw", dummy_na=True)
    else:
        grade_oh = pd.DataFrame(index=df.index)

    # smoking one-hot (guard against leakage)
    if CANON["smoking"] not in exclude_cols:
        smoke_oh = _safe_get_dummies(df[colmap[CANON["smoking"]]], prefix="smoke", dummy_na=True)
    else:
        smoke_oh = pd.DataFrame(index=df.index)

    feats = pd.concat([out, smoke_oh, grade_oh], axis=1)

    # deterministic ordering
    cols = ["patient_id"] + sorted([c for c in feats.columns if c != "patient_id"])
    feats = feats[cols]
    return feats


# ----------------------------
# Labels
# ----------------------------

def _sanitize_target_name(target: str) -> str:
    s = str(target).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def _write_labels_for_target(
    df: pd.DataFrame,
    target_raw: str,
    out_dir: Path,
    binary_stage_labels: bool,
) -> Dict[str, Path]:
    """
    Returns dict of written files: {"primary": path, "binary": path?}
    """
    # resolve column
    col = _resolve_col(df, target_raw)
    if col is None:
        raise RuntimeError(f"Target '{target_raw}' not found in columns.")

    key_user = _sanitize_target_name(target_raw)


    key_col = _sanitize_target_name(col)



    labels = df[["patient_id", col]].copy().rename(columns={col: "label"})



    out_primary = out_dir / f"labels_{key_user}.csv"


    labels.to_csv(out_primary, index=False)



    # Also write an alias filename based on the resolved column name, if different.


    # This prevents silent mismatches like target_raw='M-Stage' but Excel header is just 'M'.


    if key_col != key_user:


        (out_dir / f"labels_{key_col}.csv").write_text(out_primary.read_text())



    out = {"primary": out_primary}

    if binary_stage_labels:
        t = col.upper().replace(" ", "").replace("_", "").replace("-", "")
        if t.startswith("T"):
            lab2 = labels.copy()
            lab2["label"] = _make_binary_stage(df[col], "T")
            out_bin = out_dir / f"labels_{key_user}_binary.csv"
            lab2.to_csv(out_bin, index=False)
            out["binary"] = out_bin
            if key_col != key_user:
                (out_dir / f"labels_{key_col}_binary.csv").write_text(out_bin.read_text())
        elif t.startswith("N"):
            lab2 = labels.copy()
            lab2["label"] = _make_binary_stage(df[col], "N")
            out_bin = out_dir / f"labels_{key_user}_binary.csv"
            lab2.to_csv(out_bin, index=False)
            out["binary"] = out_bin
            if key_col != key_user:
                (out_dir / f"labels_{key_col}_binary.csv").write_text(out_bin.read_text())
        elif t.startswith("M"):
            lab2 = labels.copy()
            lab2["label"] = _make_binary_stage(df[col], "M")
            out_bin = out_dir / f"labels_{key_user}_binary.csv"
            lab2.to_csv(out_bin, index=False)
            out["binary"] = out_bin
            if key_col != key_user:
                (out_dir / f"labels_{key_col}_binary.csv").write_text(out_bin.read_text())

    return out


# ----------------------------
# Callable API for train_models
# ----------------------------

@dataclass
class PrepOutputs:
    clinical_merged_csv: Path
    clinical_features_csv: Path
    clinical_features_parquet: Optional[Path]
    labels_written: Dict[str, Dict[str, Path]]
    dedup_info: dict
    excluded_from_features: List[str]


def prepare_clinical_datasets(
    cfg: Config,
    xlsx_path: Path,
    sheet: str,
    out_dir: Path,
    targets: Sequence[str],
    binary_stage_labels: bool,
    update_manifest: bool,
    exclude_from_features: Sequence[str],
) -> PrepOutputs:
    ensure_project_dirs(cfg)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Clinical XLSX not found: {xlsx_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    df.columns = [_clean_colname(c) for c in df.columns]

    # resolve required columns with aliasing
    needed = [
        CANON["patient_id_src"],
        CANON["sex"],
        CANON["age"],
        CANON["weight"],
        CANON["t_stage"],
        CANON["n_stage"],
        CANON["m_stage"],
        CANON["grading"],
        CANON["smoking"],
    ]
    colmap = _require_cols(df, needed)

    # normalize patient_id
    df["norm_id"] = df[colmap[CANON["patient_id_src"]]].apply(_norm_id_to_Lxxxx)
    df["patient_id"] = df["norm_id"].apply(lambda v: f"Lung_Dx-{v}" if v else None)
    df = df.dropna(subset=["patient_id"]).copy()

    # dedup (enforce 1 row per patient_id)
    df, dedup_info = _dedup_by_patient(df, "patient_id")

    # leakage guard: exclude any resolved target columns from features (canonical)
    exclude_set = {CANON_NAME for CANON_NAME in exclude_from_features if CANON_NAME in ALIASES or CANON_NAME in CANON.values()}
    # if targets include certain canonical columns by alias resolution, exclude them too
    resolved_targets = []
    for t in targets:
        col = _resolve_col(df, t)
        if col is not None:
            resolved_targets.append(col)

    # map resolved target column back to canonical keys when possible
    canon_by_actual = {colmap[k]: k for k in colmap}
    for col in resolved_targets:
        if col in canon_by_actual:
            exclude_set.add(canon_by_actual[col])

    # write merged clinical (after cleaning + dedup)
    clinical_merged = out_dir / "clinical_merged.csv"
    df.to_csv(clinical_merged, index=False)

    # features (with leakage exclusions)
    feats = _build_clinical_features(df, colmap=colmap, exclude_cols=exclude_set)
    feats_csv = out_dir / "clinical_features.csv"
    feats.to_csv(feats_csv, index=False)

    feats_parq: Optional[Path] = None
    try:
        feats_parq = out_dir / "clinical_features.parquet"
        feats.to_parquet(feats_parq, index=False)
    except Exception:
        feats_parq = None  # don’t crash pipeline if pyarrow missing

    # labels
    labels_written: Dict[str, Dict[str, Path]] = {}
    for t in targets:
        written = _write_labels_for_target(df, t, out_dir=out_dir, binary_stage_labels=binary_stage_labels)
        labels_written[str(t)] = written

    # manifest updates only for patients that truly exist in final feats
    if update_manifest:
        pid_set = set(feats["patient_id"].astype(str))
        # store label values per patient for each resolved target
        label_vals: Dict[str, Dict[str, Any]] = {}
        for t in targets:
            col = _resolve_col(df, t)
            if col is None:
                continue
            sub = df[["patient_id", col]].copy()
            label_vals[col] = dict(zip(sub["patient_id"].astype(str), sub[col].tolist()))

        for pid in sorted(pid_set):
            case_dir = cfg.CASES_DIR / pid
            if not case_dir.exists():
                continue
            manifest_path = case_dir / "manifest.json"
            m = _load_json(manifest_path)
            m.setdefault("paths", {})
            m.setdefault("clinical", {})

            m["paths"]["clinical_merged"] = str(clinical_merged)
            m["paths"]["clinical_features_csv"] = str(feats_csv)
            if feats_parq is not None:
                m["paths"]["clinical_features_parquet"] = str(feats_parq)

            m["clinical"]["source_xlsx"] = str(xlsx_path)
            m["clinical"]["sheet"] = str(sheet)
            m["clinical"]["updated_at"] = datetime.utcnow().isoformat() + "Z"
            m["clinical"]["dedup_info"] = dedup_info
            m["clinical"]["excluded_from_features"] = sorted(list(exclude_set))

            if targets:
                m["clinical"].setdefault("labels", {})
                for t in targets:
                    col = _resolve_col(df, t)
                    if col is None:
                        continue
                    m["clinical"]["labels"][col] = label_vals.get(col, {}).get(pid, None)

            _save_json(manifest_path, m)

    return PrepOutputs(
        clinical_merged_csv=clinical_merged,
        clinical_features_csv=feats_csv,
        clinical_features_parquet=feats_parq,
        labels_written=labels_written,
        dedup_info=dedup_info,
        excluded_from_features=sorted(list(exclude_set)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", type=str, default="Clinical Data")
    ap.add_argument("--xlsx", type=str, default="", help="Override cfg.CLINICAL_STATS_XLSX")
    ap.add_argument("--out-dir", type=str, default="", help="Default: data/datasets")

    ap.add_argument("--targets", type=str, default="",
                    help="Comma-separated target columns to write labels for (e.g. 'M-Stage,T-Stage').")
    ap.add_argument("--binary-stage-labels", action="store_true",
                    help="Also write *_binary for TNM targets.")

    # leakage / feature controls
    ap.add_argument("--exclude-from-features", type=str, default="",
                    help="Comma-separated canonical column names to exclude from features (e.g. 'Histopathological grading').")

    ap.add_argument("--no-update-manifest", dest="update_manifest", action="store_false", default=True)

    args = ap.parse_args()

    cfg = Config()
    ensure_project_dirs(cfg)

    xlsx = Path(args.xlsx) if args.xlsx.strip() else Path(cfg.CLINICAL_STATS_XLSX)
    out_dir = Path(args.out_dir) if args.out_dir.strip() else (cfg.DATA_DIR / "datasets")
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    exclude = [t.strip() for t in args.exclude_from_features.split(",") if t.strip()]

    out = prepare_clinical_datasets(
        cfg=cfg,
        xlsx_path=xlsx,
        sheet=args.sheet,
        out_dir=out_dir,
        targets=targets,
        binary_stage_labels=bool(args.binary_stage_labels),
        update_manifest=bool(args.update_manifest),
        exclude_from_features=exclude,
    )

    print("Wrote:")
    print(" -", out.clinical_merged_csv)
    print(" -", out.clinical_features_csv)
    if out.clinical_features_parquet is not None:
        print(" -", out.clinical_features_parquet)
    else:
        print(" - (parquet skipped: pyarrow/fastparquet not available)")

    for t, files in out.labels_written.items():
        print(f" - labels for {t}: {files.get('primary')}")
        if "binary" in files:
            print(f"   (binary): {files['binary']}")

    if out.dedup_info.get("dup_patients", 0) > 0:
        print("\nWARNING: Duplicate patients detected and deduplicated:")
        print(out.dedup_info)

    if out.excluded_from_features:
        print("\nExcluded-from-features (leakage guard / user exclusions):")
        print(out.excluded_from_features)


if __name__ == "__main__":
    main()
