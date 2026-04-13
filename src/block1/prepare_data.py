"""
Step 1: Read UKB .rds files, define strict_healthy cohort, split, compute
B-spline basis, and save to parquet.

Usage:
    python -m src.block1.prepare_data

Output:
    data/processed/block1_data.parquet
    results/block1/cohort_stats.json
    results/block1/spline_config.json
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
from scipy.interpolate import BSpline
from sklearn.model_selection import train_test_split

from src.block1.config import (
    AGE_FIELD, ALL_IDP_FIELDS, ACTIVITY_FIELD, BMI_FIELD,
    DATA_DIR, EXCLUSION_ICD10_EXACT, EXCLUSION_ICD10_PREFIXES,
    PROCESSED_DIR, RDS_SOURCES, RESULTS_DIR, SEX_FIELD, SMOKING_FIELD,
    SPLINE_DEGREE, SPLINE_N_INTERIOR_KNOTS,
    TRAIN_RATIO, VAL_RATIO,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_rds_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read an .rds file and keep only *columns* (+ eid).  Free memory fast."""
    print(f"  Reading {path.name} …", end=" ", flush=True)
    result = pyreadr.read_r(str(path))
    df = result[None]  # pyreadr returns {None: DataFrame} for .rds
    keep = [c for c in ["eid"] + columns if c in df.columns]
    df = df[keep].copy()
    del result
    gc.collect()
    print(f"kept {len(keep)} cols, {len(df)} rows")
    return df


def get_excluded_eids(path: Path) -> set:
    """Read hesin_diag.rds and return eids with any exclusion ICD-10 code."""
    print(f"  Reading {path.name} for exclusions …", end=" ", flush=True)
    result = pyreadr.read_r(str(path))
    df = result[None]
    del result
    gc.collect()

    # Keep only eid + diag_icd10
    if "diag_icd10" not in df.columns:
        print("WARNING: diag_icd10 column not found, trying alternatives")
        icd_col = [c for c in df.columns if "icd10" in c.lower()]
        if not icd_col:
            print("ERROR: no ICD-10 column found in hesin_diag.rds")
            return set()
        diag_col = icd_col[0]
    else:
        diag_col = "diag_icd10"

    sub = df[["eid", diag_col]].dropna(subset=[diag_col]).copy()
    del df
    gc.collect()

    codes = sub[diag_col].astype(str).str.strip()

    # Match prefixes
    mask = pd.Series(False, index=sub.index)
    for prefix in EXCLUSION_ICD10_PREFIXES:
        mask |= codes.str.startswith(prefix)
    # Match exact codes
    for code in EXCLUSION_ICD10_EXACT:
        mask |= (codes == code)

    excluded = set(sub.loc[mask, "eid"].unique())
    del sub, codes, mask
    gc.collect()
    print(f"{len(excluded)} participants excluded")
    return excluded


def compute_bspline_basis(ages: np.ndarray, knots_interior: np.ndarray,
                          degree: int) -> np.ndarray:
    """Return B-spline design matrix (n_samples, n_basis)."""
    # Augment knots with boundary repeats
    t = np.concatenate([
        np.repeat(knots_interior[0], degree),
        knots_interior,
        np.repeat(knots_interior[-1], degree),
    ])
    n_basis = len(t) - degree - 1
    basis = np.zeros((len(ages), n_basis), dtype=np.float32)
    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        spl = BSpline(t, coeffs, degree, extrapolate=True)
        basis[:, i] = spl(ages)
    return basis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Read CMR IDPs ----
    print("[1/6] Reading CMR IDPs …")
    df_cmr = read_rds_columns(RDS_SOURCES["heart_mri"], ALL_IDP_FIELDS)

    # ---- 2. Read demographics ----
    print("[2/6] Reading demographics …")
    df_age = read_rds_columns(RDS_SOURCES["recruitment"], [AGE_FIELD])
    df_sex = read_rds_columns(RDS_SOURCES["population"], [SEX_FIELD])

    # ---- 3. Read covariates (BMI, smoking, activity) ----
    print("[3/6] Reading covariates …")
    df_phys = read_rds_columns(RDS_SOURCES["physical"], [BMI_FIELD])
    df_life = read_rds_columns(RDS_SOURCES["lifestyle"],
                               [SMOKING_FIELD, ACTIVITY_FIELD])

    # ---- 4. Merge all ----
    print("[4/6] Merging …")
    df = df_cmr.copy()
    del df_cmr; gc.collect()
    for right in [df_age, df_sex, df_phys, df_life]:
        df = df.merge(right, on="eid", how="left")
        del right
    gc.collect()

    # Rename for clarity
    rename_map = {
        AGE_FIELD: "age", SEX_FIELD: "sex",
        BMI_FIELD: "bmi", SMOKING_FIELD: "smoking",
        ACTIVITY_FIELD: "vigorous_activity",
    }
    df.rename(columns=rename_map, inplace=True)

    # --- Robust type conversion ---
    # R factors → pandas Categorical with string labels (e.g. "Male"/"Female").
    # pd.to_numeric("Male", errors="coerce") → NaN, which would kill all rows.
    # Fix: decode known string factors first, then generic numeric conversion.

    # Known string-label factor mappings (UKB R export conventions)
    _FACTOR_MAPS = {
        "sex": {"Female": 0, "female": 0, "0": 0,
                "Male": 1, "male": 1, "1": 1},
        "smoking": {"Never": 0, "Previous": 1, "Current": 2,
                    "Prefer not to answer": float("nan"),
                    "0": 0, "1": 1, "2": 2, "-3": float("nan")},
    }

    def _decode_factor(series: pd.Series, col_name: str) -> pd.Series:
        """Map known R factor labels to numeric values."""
        fmap = _FACTOR_MAPS.get(col_name)
        if fmap is None:
            return series
        # Convert to string for mapping (handles Categorical + object)
        str_vals = series.astype(str)
        unique_str = str_vals.unique()[:10]
        mapped = str_vals.map(fmap)
        n_mapped = mapped.notna().sum()
        n_total = len(series)
        print(f"    factor decode '{col_name}': "
              f"mapped {n_mapped}/{n_total}, "
              f"unique_str={list(unique_str)}")
        return mapped

    def safe_to_float32(series: pd.Series) -> pd.Series:
        """Convert any series to float32, handling R factors/categoricals."""
        if hasattr(series, "cat"):
            codes = series.cat.codes
            cats = series.cat.categories
            # If categories are already numeric-like, use them directly
            numeric_cats = pd.to_numeric(cats, errors="coerce")
            if numeric_cats.notna().all():
                result = numeric_cats[codes]
                result[codes == -1] = float("nan")
                return result.astype("float32")
        return pd.to_numeric(series, errors="coerce").astype("float32")

    # Decode known factors BEFORE generic numeric conversion
    for col_name in ["sex", "smoking", "vigorous_activity"]:
        if col_name in df.columns:
            df[col_name] = _decode_factor(df[col_name], col_name)

    for col in ALL_IDP_FIELDS + ["age", "bmi"]:
        if col in df.columns:
            df[col] = safe_to_float32(df[col])
    for col in ["sex", "smoking", "vigorous_activity"]:
        if col in df.columns:
            df[col] = safe_to_float32(df[col])

    # --- Diagnostic: show non-null counts before dropna ---
    required = ALL_IDP_FIELDS + ["age", "sex"]
    print("  Non-null counts for required columns:")
    for col in required:
        nn = df[col].notna().sum()
        dtype = df[col].dtype
        print(f"    {col:20s}  dtype={str(dtype):10s}  non-null={nn}")

    n_before = len(df)
    df.dropna(subset=required, inplace=True)
    print(f"  Dropped {n_before - len(df)} rows with missing IDP/age/sex "
          f"→ {len(df)} remaining")

    # ---- 5. Define strict_healthy ----
    print("[5/6] Defining strict_healthy cohort …")
    excluded_eids = get_excluded_eids(RDS_SOURCES["hesin_diag"])
    df["is_healthy"] = ~df["eid"].isin(excluded_eids)
    del excluded_eids; gc.collect()

    n_healthy = df["is_healthy"].sum()
    n_unhealthy = (~df["is_healthy"]).sum()
    print(f"  Healthy: {n_healthy}, Non-healthy: {n_unhealthy}")

    # ---- 6. Stratified split (healthy only) ----
    print("[6/6] Splitting …")
    healthy_idx = df.index[df["is_healthy"]].values

    # Create stratification bins: sex × 5-year age bin
    strat_labels = (
        df.loc[healthy_idx, "sex"].astype(int).astype(str)
        + "_"
        + (df.loc[healthy_idx, "age"] // 5).astype(int).astype(str)
    )

    # First split: train vs (val+test)
    val_test_ratio = VAL_RATIO + (1 - TRAIN_RATIO - VAL_RATIO)
    idx_train, idx_valtest, strat_train, strat_valtest = train_test_split(
        healthy_idx, strat_labels,
        test_size=val_test_ratio,
        random_state=42,
        stratify=strat_labels,
    )
    # Second split: val vs test
    val_frac = VAL_RATIO / val_test_ratio
    idx_val, idx_test = train_test_split(
        idx_valtest,
        test_size=1 - val_frac,
        random_state=42,
        stratify=strat_valtest,
    )

    df["split"] = "none"           # non-healthy
    df.loc[idx_train, "split"] = "train"
    df.loc[idx_val, "split"] = "val"
    df.loc[idx_test, "split"] = "test"

    print(f"  Train: {len(idx_train)}, Val: {len(idx_val)}, "
          f"Test: {len(idx_test)}, Non-healthy: {n_unhealthy}")

    # ---- 7. B-spline basis ----
    print("Computing B-spline basis …")
    train_ages = df.loc[idx_train, "age"].values
    age_min, age_max = float(train_ages.min()), float(train_ages.max())
    knots_interior = np.linspace(age_min, age_max,
                                 SPLINE_N_INTERIOR_KNOTS + 2)
    all_ages = df["age"].values.astype(np.float64)
    basis = compute_bspline_basis(all_ages, knots_interior, SPLINE_DEGREE)

    n_basis = basis.shape[1]
    basis_cols = [f"basis_{i}" for i in range(n_basis)]
    sex_vals = df["sex"].values.astype(np.float32)

    # Normative features: [basis, sex, sex*basis]
    sex_basis = basis * sex_vals[:, None]
    norm_feature_cols = basis_cols + ["sex_indicator"] + \
        [f"sex_basis_{i}" for i in range(n_basis)]

    for i, col in enumerate(basis_cols):
        df[col] = basis[:, i]
    df["sex_indicator"] = sex_vals
    for i in range(n_basis):
        df[f"sex_basis_{i}"] = sex_basis[:, i]

    del basis, sex_basis; gc.collect()

    # ---- 8. Save ----
    out_path = PROCESSED_DIR / "block1_data.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"Saved → {out_path}  ({len(df)} rows)")

    # Cohort stats
    stats = {
        "total_with_cmr_age_sex": int(len(df)),
        "n_healthy": int(n_healthy),
        "n_unhealthy": int(n_unhealthy),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "age_range": [age_min, age_max],
        "age_mean_train": float(train_ages.mean()),
        "age_std_train": float(train_ages.std()),
        "sex_ratio_train": float(
            df.loc[idx_train, "sex"].mean()  # fraction male
        ),
        "bmi_available": int(df["bmi"].notna().sum()),
        "smoking_available": int(df["smoking"].notna().sum()),
        "activity_available": int(df["vigorous_activity"].notna().sum()),
        "norm_feature_cols": norm_feature_cols,
    }
    stats_path = RESULTS_DIR / "cohort_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Cohort stats → {stats_path}")

    # Spline config (needed to reproduce basis for new data)
    spline_cfg = {
        "degree": SPLINE_DEGREE,
        "knots_interior": knots_interior.tolist(),
        "n_basis": n_basis,
        "age_min": age_min,
        "age_max": age_max,
    }
    spline_path = RESULTS_DIR / "spline_config.json"
    with open(spline_path, "w") as f:
        json.dump(spline_cfg, f, indent=2)
    print(f"Spline config → {spline_path}")

    print("Done.")


if __name__ == "__main__":
    main()
