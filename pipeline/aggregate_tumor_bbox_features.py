# pipeline/aggregate_tumor_bbox_features.py
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.config import Config, ensure_project_dirs, verify_write_access


CT_JSON_NAME = "tumor_bbox_ct_features.json"
PET_JSON_NAME = "tumor_bbox_pet_features.json"


# ----------------------------
# JSON + manifest helpers
# ----------------------------

def _load_json_with_error(path: Path) -> Tuple[Dict[str, Any], str]:
    """Return (data, err). On failure, data={}, err is a short string."""
    try:
        txt = path.read_text()
    except Exception as e:
        return {}, f"read_error: {repr(e)}"
    try:
        obj = json.loads(txt)
    except Exception as e:
        return {}, f"json_error: {repr(e)}"
    if not isinstance(obj, dict):
        return {}, "json_not_dict"
    return obj, ""


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str))


# ----------------------------
# Feature key filtering
# ----------------------------

def _filter_feature_keys(
    d: Dict[str, Any],
    allow_prefixes: Optional[List[str]] = None,
    deny_prefixes: Optional[List[str]] = None,
    deny_keys: Optional[set[str]] = None,
) -> Tuple[Dict[str, Any], dict]:
    """
    Filter dict keys before numeric coercion to avoid polluting feature space with metadata.

    - allow_prefixes: if provided, keep only keys that start with any prefix in the list.
    - deny_prefixes: always drop keys starting with any of these prefixes.
    - deny_keys: always drop these exact keys.
    """
    allow_prefixes = allow_prefixes or []
    deny_prefixes = deny_prefixes or []
    deny_keys = deny_keys or set()

    stats = {
        "n_in": int(len(d)),
        "n_out": 0,
        "n_dropped_allowlist": 0,
        "n_dropped_deny_prefix": 0,
        "n_dropped_deny_key": 0,
    }

    out: Dict[str, Any] = {}
    for k, v in d.items():
        if k in deny_keys:
            stats["n_dropped_deny_key"] += 1
            continue
        if any(k.startswith(p) for p in deny_prefixes):
            stats["n_dropped_deny_prefix"] += 1
            continue
        if allow_prefixes:
            if not any(k.startswith(p) for p in allow_prefixes):
                stats["n_dropped_allowlist"] += 1
                continue
        out[k] = v

    stats["n_out"] = int(len(out))
    return out, stats


# ----------------------------
# Feature conversion
# ----------------------------

def _to_numeric_series(d: Dict[str, Any]) -> Tuple[pd.Series, dict]:
    """
    Convert a dict of features to a numeric Series.
    - drops non-scalar values
    - coerces strings -> numeric when possible
    """
    stats = {
        "n_keys_in": int(len(d)),
        "n_none": 0,
        "n_numeric": 0,
        "n_bool": 0,
        "n_str": 0,
        "n_dropped_non_scalar": 0,
        "n_after_numeric_coerce": 0,
        "n_nan_after_coerce": 0,
    }

    clean: Dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            clean[k] = np.nan
            stats["n_none"] += 1
            continue

        if isinstance(v, bool):
            clean[k] = float(v)
            stats["n_bool"] += 1
            continue

        if isinstance(v, (int, float, np.integer, np.floating)):
            clean[k] = float(v)
            stats["n_numeric"] += 1
            continue

        if isinstance(v, str):
            clean[k] = v.strip()
            stats["n_str"] += 1
            continue

        stats["n_dropped_non_scalar"] += 1

    s = pd.Series(clean, dtype="object")
    s_num = pd.to_numeric(s, errors="coerce")
    stats["n_after_numeric_coerce"] = int(s_num.shape[0])
    stats["n_nan_after_coerce"] = int(s_num.isna().sum())
    return s_num, stats


# ----------------------------
# Collection helpers
# ----------------------------

def _collect_case_ids(cases_dir: Path) -> list[str]:
    return sorted([p.name for p in cases_dir.iterdir() if p.is_dir()])


def _build_feature_tables(
    cfg: Config,
    allow_prefixes: Optional[List[str]],
    deny_prefixes: Optional[List[str]],
    deny_keys: set[str],
    update_manifest: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_ids = _collect_case_ids(cfg.CASES_DIR)
    if not case_ids:
        raise RuntimeError(f"No case folders found under: {cfg.CASES_DIR}")

    ct_rows: Dict[str, pd.Series] = {}
    pet_rows: Dict[str, pd.Series] = {}
    report_rows: List[dict] = []

    for cid in case_ids:
        case_dir = cfg.CASES_DIR / cid
        feat_dir = case_dir / "features"
        ct_path = feat_dir / CT_JSON_NAME
        pet_path = feat_dir / PET_JSON_NAME
        manifest_path = case_dir / "manifest.json"

        has_ct = ct_path.exists()
        has_pet = pet_path.exists()

        ct_load_err = ""
        pet_load_err = ""

        ct_filter_stats = {}
        pet_filter_stats = {}
        ct_conv_stats = {}
        pet_conv_stats = {}

        n_ct = 0
        n_pet = 0
        n_ct_nan = 0
        n_pet_nan = 0

        if has_ct:
            ct_obj, ct_load_err = _load_json_with_error(ct_path)
            if not ct_load_err:
                ct_obj_f, ct_filter_stats = _filter_feature_keys(
                    ct_obj,
                    allow_prefixes=allow_prefixes,
                    deny_prefixes=deny_prefixes,
                    deny_keys=deny_keys,
                )
                ct_s, ct_conv_stats = _to_numeric_series(ct_obj_f)
                n_ct = int(ct_s.shape[0])
                n_ct_nan = int(ct_s.isna().sum())
                ct_rows[cid] = ct_s

        if has_pet:
            pet_obj, pet_load_err = _load_json_with_error(pet_path)
            if not pet_load_err:
                pet_obj_f, pet_filter_stats = _filter_feature_keys(
                    pet_obj,
                    allow_prefixes=allow_prefixes,
                    deny_prefixes=deny_prefixes,
                    deny_keys=deny_keys,
                )
                pet_s, pet_conv_stats = _to_numeric_series(pet_obj_f)
                n_pet = int(pet_s.shape[0])
                n_pet_nan = int(pet_s.isna().sum())
                pet_rows[cid] = pet_s

        # ---- Per-case manifest update (gated) ----
        if update_manifest:
            try:
                m = _load_manifest(manifest_path)
                m.setdefault("paths", {})
                m.setdefault("features", {})
                m.setdefault("status", {})
                m["updated_at"] = datetime.utcnow().isoformat() + "Z"
                m["paths"].setdefault("features_dir", str(feat_dir))

                agg = m["features"].setdefault("tumor_bbox_aggregation", {})
                agg["updated_at"] = datetime.utcnow().isoformat() + "Z"
                agg["ct_json"] = str(ct_path) if has_ct else ""
                agg["pet_json"] = str(pet_path) if has_pet else ""
                agg["has_ct_features_json"] = bool(has_ct)
                agg["has_pet_features_json"] = bool(has_pet)

                agg["ct_load_error"] = ct_load_err
                agg["pet_load_error"] = pet_load_err

                agg["ct_n_features"] = int(n_ct)
                agg["pet_n_features"] = int(n_pet)
                agg["ct_nan_count"] = int(n_ct_nan)
                agg["pet_nan_count"] = int(n_pet_nan)

                agg["ct_key_filter_stats"] = ct_filter_stats
                agg["pet_key_filter_stats"] = pet_filter_stats
                agg["ct_conversion_stats"] = ct_conv_stats
                agg["pet_conversion_stats"] = pet_conv_stats

                # compact status flags
                if has_ct and not ct_load_err and n_ct > 0:
                    ct_status = "ok"
                elif has_ct and ct_load_err:
                    ct_status = "broken_json"
                elif has_ct and not ct_load_err and n_ct == 0:
                    ct_status = "empty_after_filter"
                else:
                    ct_status = "missing"

                if has_pet and not pet_load_err and n_pet > 0:
                    pet_status = "ok"
                elif has_pet and pet_load_err:
                    pet_status = "broken_json"
                elif has_pet and not pet_load_err and n_pet == 0:
                    pet_status = "empty_after_filter"
                else:
                    pet_status = "missing"

                m["status"]["tumor_bbox_agg_ct"] = ct_status
                m["status"]["tumor_bbox_agg_pet"] = pet_status

                _save_manifest(manifest_path, m)
            except Exception:
                pass

        report_rows.append({
            "patient_id": cid,
            "has_ct_features": has_ct,
            "has_pet_features": has_pet,
            "ct_load_error": ct_load_err,
            "pet_load_error": pet_load_err,
            "ct_n_features": n_ct,
            "pet_n_features": n_pet,
            "ct_nan_count": n_ct_nan,
            "pet_nan_count": n_pet_nan,
            "ct_filter_dropped_allowlist": int(ct_filter_stats.get("n_dropped_allowlist", 0)) if isinstance(ct_filter_stats, dict) else 0,
            "pet_filter_dropped_allowlist": int(pet_filter_stats.get("n_dropped_allowlist", 0)) if isinstance(pet_filter_stats, dict) else 0,
            "ct_filter_dropped_deny_prefix": int(ct_filter_stats.get("n_dropped_deny_prefix", 0)) if isinstance(ct_filter_stats, dict) else 0,
            "pet_filter_dropped_deny_prefix": int(pet_filter_stats.get("n_dropped_deny_prefix", 0)) if isinstance(pet_filter_stats, dict) else 0,
            "ct_dropped_non_scalar": int(ct_conv_stats.get("n_dropped_non_scalar", 0)) if isinstance(ct_conv_stats, dict) else 0,
            "pet_dropped_non_scalar": int(pet_conv_stats.get("n_dropped_non_scalar", 0)) if isinstance(pet_conv_stats, dict) else 0,
            "ct_bool_count": int(ct_conv_stats.get("n_bool", 0)) if isinstance(ct_conv_stats, dict) else 0,
            "pet_bool_count": int(pet_conv_stats.get("n_bool", 0)) if isinstance(pet_conv_stats, dict) else 0,
        })

    df_ct = pd.DataFrame.from_dict(ct_rows, orient="index").sort_index()
    df_pet = pd.DataFrame.from_dict(pet_rows, orient="index").sort_index()
    df_ct.index.name = "patient_id"
    df_pet.index.name = "patient_id"

    df_ct = df_ct.add_prefix("ct__")
    df_pet = df_pet.add_prefix("pet__")

    all_ids = pd.Index(sorted(set(df_ct.index).union(set(df_pet.index))), name="patient_id")
    df_fused = pd.DataFrame(index=all_ids).join(df_ct, how="left").join(df_pet, how="left")

    # Deterministic columns
    df_ct = df_ct.reindex(sorted(df_ct.columns), axis=1)
    df_pet = df_pet.reindex(sorted(df_pet.columns), axis=1)
    df_fused = df_fused.reindex(sorted(df_fused.columns), axis=1)

    report_df = pd.DataFrame(report_rows).sort_values("patient_id")

    return df_ct, df_pet, df_fused, report_df


def _clean_for_ml(
    df: pd.DataFrame,
    drop_constant: bool = True,
    drop_sparse: bool = False,
    max_nan_frac: float = 0.95,
) -> Tuple[pd.DataFrame, dict]:
    info: Dict[str, Any] = {}

    df2 = df.replace([np.inf, -np.inf], np.nan)
    info["n_rows"] = int(df2.shape[0])
    info["n_cols_before"] = int(df2.shape[1])
    info["nan_total"] = int(df2.isna().sum().sum())

    dropped_sparse = []
    if drop_sparse and df2.shape[0] > 0:
        frac_nan = df2.isna().mean(axis=0)
        dropped_sparse = frac_nan[frac_nan > float(max_nan_frac)].index.tolist()
        df2 = df2.drop(columns=dropped_sparse)

    info["dropped_sparse_cols"] = int(len(dropped_sparse))
    info["max_nan_frac"] = float(max_nan_frac) if drop_sparse else None

    if drop_constant:
        nun = df2.nunique(dropna=True)
        const_cols = nun[nun <= 1].index.tolist()
        df2 = df2.drop(columns=const_cols)
        info["dropped_constant_cols"] = int(len(const_cols))
    else:
        info["dropped_constant_cols"] = 0

    info["n_cols_after"] = int(df2.shape[1])
    return df2, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-constant", action="store_true", help="Drop constant columns.")
    ap.add_argument("--drop-sparse", action="store_true", help="Drop columns with very high NaN fraction.")
    ap.add_argument("--max-nan-frac", type=float, default=0.95, help="Threshold for --drop-sparse.")
    ap.add_argument("--out-dir", type=str, default="", help="Override output dir (default: data/datasets).")

    # ✅ Correct: updates enabled by default; only negative flag exists.
    ap.add_argument(
        "--no-update-manifest",
        dest="update_manifest",
        action="store_false",
        default=True,
        help="Disable per-case manifest updates (default: updates are enabled).",
    )

    # Key filtering controls
    ap.add_argument("--allow-prefixes", type=str, default="",
                    help="Comma-separated allowlist prefixes. If set, only keys starting with these are kept.")
    ap.add_argument("--deny-prefixes", type=str, default="diagnostics_",
                    help="Comma-separated denylist prefixes (dropped even if allowed). Default drops diagnostics_.")
    ap.add_argument("--deny-keys", type=str, default="",
                    help="Comma-separated exact keys to drop.")

    args = ap.parse_args()

    cfg = Config()
    ensure_project_dirs(cfg)

    # Pre-flight permission check on mounted data directory
    verify_write_access(cfg.DATA_ROOT)

    out_dir = Path(args.out_dir) if args.out_dir.strip() else (cfg.DATA_DIR / "datasets")
    out_dir.mkdir(parents=True, exist_ok=True)

    allow_prefixes = [x.strip() for x in args.allow_prefixes.split(",") if x.strip()] or None
    deny_prefixes = [x.strip() for x in args.deny_prefixes.split(",") if x.strip()]
    deny_keys = {x.strip() for x in args.deny_keys.split(",") if x.strip()}

    df_ct, df_pet, df_fused, report_df = _build_feature_tables(
        cfg,
        allow_prefixes=allow_prefixes,
        deny_prefixes=deny_prefixes,
        deny_keys=deny_keys,
        update_manifest=bool(args.update_manifest),
    )

    df_ct_clean, ct_info = _clean_for_ml(
        df_ct, drop_constant=args.drop_constant, drop_sparse=args.drop_sparse, max_nan_frac=args.max_nan_frac
    )
    df_pet_clean, pet_info = _clean_for_ml(
        df_pet, drop_constant=args.drop_constant, drop_sparse=args.drop_sparse, max_nan_frac=args.max_nan_frac
    )
    df_fused_clean, fused_info = _clean_for_ml(
        df_fused, drop_constant=args.drop_constant, drop_sparse=args.drop_sparse, max_nan_frac=args.max_nan_frac
    )

    df_ct_clean.to_parquet(out_dir / "tumor_bbox_features_ct.parquet")
    df_pet_clean.to_parquet(out_dir / "tumor_bbox_features_pet.parquet")
    df_fused_clean.to_parquet(out_dir / "tumor_bbox_features_fused.parquet")
    report_df.to_csv(out_dir / "tumor_bbox_aggregation_report.csv", index=False)

    (out_dir / "tumor_bbox_aggregation_stats.json").write_text(json.dumps({
        "ct": ct_info,
        "pet": pet_info,
        "fused": fused_info,
        "key_filter": {
            "allow_prefixes": allow_prefixes or [],
            "deny_prefixes": deny_prefixes,
            "deny_keys": sorted(list(deny_keys)),
        },
        "counts": {
            "cases_total": int(report_df.shape[0]),
            "cases_with_ct_features_json": int(report_df["has_ct_features"].sum()),
            "cases_with_pet_features_json": int(report_df["has_pet_features"].sum()),
            "cases_ct_load_errors": int((report_df["ct_load_error"].astype(str) != "").sum()),
            "cases_pet_load_errors": int((report_df["pet_load_error"].astype(str) != "").sum()),
        },
        "update_manifest": bool(args.update_manifest),
    }, indent=2))

    print("Saved:")
    print(" -", out_dir / "tumor_bbox_features_ct.parquet")
    print(" -", out_dir / "tumor_bbox_features_pet.parquet")
    print(" -", out_dir / "tumor_bbox_features_fused.parquet")
    print(" -", out_dir / "tumor_bbox_aggregation_report.csv")
    print(" -", out_dir / "tumor_bbox_aggregation_stats.json")


if __name__ == "__main__":
    main()
