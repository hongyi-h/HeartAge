"""
P0: CMR-side incident HF survival analysis — the gating test.

Question: Does Full Teacher deviation predict incident HF better than
baseline age-gap approaches, after adjusting for age, sex, BMI?

Pipeline:
  1. Extract imaging dates from UKB (Recruitment or heart_mri RDS)
  2. Extract incident HF events (I50) with dates from hesin + hesin_diag
  3. Extract death dates for censoring from Death.rds
  4. Join with Block 1 prediction CSVs
  5. Cox regression: compare C-indices of FT deviation vs BB/XGB/EN age-gaps
  6. Adjusted models (+ age, sex, BMI, HTN status)

Usage:
    python scripts/p0_survival_analysis.py

Output:
    results/block1/p0_survival_results.json
    results/block1/figures/km_by_deviation_tertile.png
    results/block1/figures/forest_plot.png
"""

import gc
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
from scipy import stats as sp_stats

# Survival analysis
try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.utils import concordance_index
except ImportError:
    print("ERROR: lifelines not installed. Run: pip install lifelines")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "UKB"
RESULTS_DIR = PROJECT_ROOT / "results" / "block1"
PRED_DIR = RESULTS_DIR / "predictions"
FIG_DIR = RESULTS_DIR / "figures"

# Ensure output dirs
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load UKB dates
# ---------------------------------------------------------------------------

def load_imaging_dates() -> pd.DataFrame:
    """Extract imaging visit dates from UKB.

    Tries heart_mri.rds first (p53_i2 = date of imaging assessment centre),
    then Recruitment.rds. Falls back to constructing approximate dates from
    age + birth year if needed.

    Returns: DataFrame with columns [eid, imaging_date].
    """
    print("[1] Loading imaging dates ...")

    # Strategy 1: heart_mri.rds — look for date fields
    hmri_path = DATA_DIR / "heart_mri.rds"
    if hmri_path.exists():
        print("  Reading heart_mri.rds ...")
        result = pyreadr.read_r(str(hmri_path))
        df = result[None]
        del result; gc.collect()

        # UKB imaging date fields: p53_i2 (date of attending assessment centre, instance 2)
        # Also check: p20252 (date of CMR), or any column with "date" in name
        date_candidates = [c for c in df.columns
                           if any(k in c.lower() for k in ["p53_i2", "p53_i3", "date"])]
        print(f"  Date candidates in heart_mri.rds: {date_candidates}")

        for col in ["p53_i2", "p53_i3"] + date_candidates:
            if col in df.columns:
                out = df[["eid", col]].dropna(subset=[col]).copy()
                out[col] = pd.to_datetime(out[col], errors="coerce")
                out = out.dropna(subset=[col])
                if len(out) > 1000:
                    out.rename(columns={col: "imaging_date"}, inplace=True)
                    print(f"  Found imaging dates in heart_mri.rds/{col}: "
                          f"{len(out)} records")
                    del df; gc.collect()
                    return out[["eid", "imaging_date"]]
        del df; gc.collect()

    # Strategy 2: Recruitment.rds — p53_i2
    recr_path = DATA_DIR / "Recruitment.rds"
    if recr_path.exists():
        print("  Reading Recruitment.rds ...")
        result = pyreadr.read_r(str(recr_path))
        df = result[None]
        del result; gc.collect()

        date_candidates = [c for c in df.columns if "p53" in c or "date" in c.lower()]
        print(f"  Date candidates in Recruitment.rds: {date_candidates}")

        for col in ["p53_i2", "p53_i3"] + date_candidates:
            if col in df.columns:
                out = df[["eid", col]].dropna(subset=[col]).copy()
                out[col] = pd.to_datetime(out[col], errors="coerce")
                out = out.dropna(subset=[col])
                if len(out) > 1000:
                    out.rename(columns={col: "imaging_date"}, inplace=True)
                    print(f"  Found imaging dates in Recruitment.rds/{col}: "
                          f"{len(out)} records")
                    del df; gc.collect()
                    return out[["eid", "imaging_date"]]
        del df; gc.collect()

    # Strategy 3: Approximate from age at imaging + birth year
    print("  WARNING: No exact imaging date found. Reconstructing from "
          "age + birth year ...")
    recr_path = DATA_DIR / "Recruitment.rds"
    pop_path = DATA_DIR / "Population_characteristics.rds"
    if recr_path.exists() and pop_path.exists():
        result = pyreadr.read_r(str(recr_path))
        df_age = result[None][["eid", "p21003_i2"]].dropna()
        del result; gc.collect()

        result = pyreadr.read_r(str(pop_path))
        df_pop = result[None]
        del result; gc.collect()

        # p34 = year of birth, p52 = month of birth
        birth_cols = [c for c in df_pop.columns if c in ["p34", "p52"]]
        df_birth = df_pop[["eid"] + birth_cols].copy()
        del df_pop; gc.collect()

        df_merged = df_age.merge(df_birth, on="eid", how="inner")

        if "p34" in df_merged.columns:
            age = pd.to_numeric(df_merged["p21003_i2"], errors="coerce")
            birth_year = pd.to_numeric(df_merged["p34"], errors="coerce")
            birth_month = 6  # default
            if "p52" in df_merged.columns:
                birth_month = pd.to_numeric(df_merged["p52"],
                                             errors="coerce").fillna(6)
            # Approximate imaging date
            imaging_year = birth_year + age
            df_merged["imaging_date"] = pd.to_datetime(
                dict(year=imaging_year.astype(int),
                     month=birth_month if isinstance(birth_month, int)
                     else birth_month.astype(int),
                     day=15),
                errors="coerce"
            )
            out = df_merged[["eid", "imaging_date"]].dropna()
            print(f"  Reconstructed imaging dates: {len(out)} records")
            return out

    raise RuntimeError("Cannot determine imaging dates from UKB data.")


def load_hf_events() -> pd.DataFrame:
    """Extract incident HF (I50*) events with dates from hesin + hesin_diag.

    Returns: DataFrame with columns [eid, hf_date] — first I50 per participant.
    """
    print("[2] Loading HF events from HES ...")

    # Load hesin_diag for ICD-10 codes
    diag_path = DATA_DIR / "hesin_diag.rds"
    result = pyreadr.read_r(str(diag_path))
    df_diag = result[None]
    del result; gc.collect()

    # Find the ICD-10 column
    if "diag_icd10" in df_diag.columns:
        diag_col = "diag_icd10"
    else:
        icd_cols = [c for c in df_diag.columns if "icd10" in c.lower()]
        if not icd_cols:
            raise RuntimeError("No ICD-10 column in hesin_diag.rds")
        diag_col = icd_cols[0]

    # Filter to I50 codes
    codes = df_diag[diag_col].astype(str).str.strip()
    hf_mask = codes.str.startswith("I50")
    df_hf = df_diag.loc[hf_mask, ["eid", "ins_index"]].copy()
    n_hf_records = len(df_hf)
    print(f"  Found {n_hf_records} HF diagnosis records "
          f"({df_hf['eid'].nunique()} unique participants)")
    del df_diag, codes; gc.collect()

    # Load hesin for episode dates
    hesin_path = DATA_DIR / "hesin.rds"
    result = pyreadr.read_r(str(hesin_path))
    df_hesin = result[None]
    del result; gc.collect()

    print(f"  hesin.rds columns: {list(df_hesin.columns)[:15]} ...")

    # Find date column: epistart > admidate > disdate
    date_col = None
    for candidate in ["epistart", "admidate", "disdate"]:
        if candidate in df_hesin.columns:
            date_col = candidate
            break

    if date_col is None:
        date_candidates = [c for c in df_hesin.columns if "date" in c.lower()
                           or "start" in c.lower()]
        if date_candidates:
            date_col = date_candidates[0]
        else:
            raise RuntimeError(f"No date column in hesin.rds. "
                               f"Columns: {list(df_hesin.columns)}")

    print(f"  Using date column: {date_col}")

    # Merge HF diagnoses with dates
    # ins_index links hesin_diag to hesin
    merge_keys = ["eid"]
    if "ins_index" in df_hesin.columns and "ins_index" in df_hf.columns:
        merge_keys.append("ins_index")

    df_hf = df_hf.merge(
        df_hesin[merge_keys + [date_col]].drop_duplicates(),
        on=merge_keys, how="left"
    )
    del df_hesin; gc.collect()

    df_hf[date_col] = pd.to_datetime(df_hf[date_col], errors="coerce")
    df_hf = df_hf.dropna(subset=[date_col])

    # Keep first HF event per participant
    df_hf = df_hf.sort_values(date_col).groupby("eid").first().reset_index()
    df_hf.rename(columns={date_col: "hf_date"}, inplace=True)

    print(f"  Unique participants with dated HF: {len(df_hf)}")
    return df_hf[["eid", "hf_date"]]


def load_death_dates() -> pd.DataFrame:
    """Load death dates for censoring."""
    print("[3] Loading death dates ...")
    death_path = DATA_DIR / "Death.rds"
    if not death_path.exists():
        print("  WARNING: Death.rds not found. Will use fixed censoring date.")
        return pd.DataFrame(columns=["eid", "death_date"])

    result = pyreadr.read_r(str(death_path))
    df = result[None]
    del result; gc.collect()

    print(f"  Death.rds columns: {list(df.columns)}")

    # Find date column
    date_col = None
    for candidate in ["p40000_i0", "date_of_death", "p40000"]:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        date_candidates = [c for c in df.columns if "date" in c.lower()
                           or "40000" in c]
        date_col = date_candidates[0] if date_candidates else None

    if date_col is None:
        print("  WARNING: No date column in Death.rds")
        return pd.DataFrame(columns=["eid", "death_date"])

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    out = df[["eid", date_col]].dropna(subset=[date_col]).copy()
    out.rename(columns={date_col: "death_date"}, inplace=True)

    # Keep earliest death date per person (if multiple instances)
    out = out.sort_values("death_date").groupby("eid").first().reset_index()
    print(f"  Deaths: {len(out)}")
    return out


def load_covariates() -> pd.DataFrame:
    """Load age, sex, BMI from existing block1_data.parquet or raw RDS."""
    print("[4] Loading covariates ...")

    # Try processed parquet first
    parquet_path = PROJECT_ROOT / "data" / "processed" / "block1_data.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path,
                             columns=["eid", "age", "sex", "bmi",
                                      "smoking", "is_healthy"])
        print(f"  From parquet: {len(df)} rows")
        return df

    # Otherwise build from raw RDS
    print("  Parquet not found, building from raw RDS ...")
    from src.block1.config import AGE_FIELD, SEX_FIELD, BMI_FIELD

    result = pyreadr.read_r(str(DATA_DIR / "Recruitment.rds"))
    df_age = result[None][["eid", AGE_FIELD]].dropna()
    df_age.rename(columns={AGE_FIELD: "age"}, inplace=True)
    del result; gc.collect()

    result = pyreadr.read_r(str(DATA_DIR / "Population_characteristics.rds"))
    df_sex = result[None][["eid", SEX_FIELD]].dropna()
    # Convert sex factor
    sex_map = {"Female": 0, "female": 0, "Male": 1, "male": 1,
               "0": 0, "1": 1}
    df_sex[SEX_FIELD] = df_sex[SEX_FIELD].astype(str).map(sex_map)
    df_sex.rename(columns={SEX_FIELD: "sex"}, inplace=True)
    del result; gc.collect()

    result = pyreadr.read_r(str(DATA_DIR / "Physical_measures.rds"))
    df_bmi = result[None][["eid", BMI_FIELD]].dropna()
    df_bmi.rename(columns={BMI_FIELD: "bmi"}, inplace=True)
    df_bmi["bmi"] = pd.to_numeric(df_bmi["bmi"], errors="coerce")
    del result; gc.collect()

    df = df_age.merge(df_sex, on="eid", how="inner")
    df = df.merge(df_bmi, on="eid", how="left")

    # is_healthy flag from hesin_diag
    from src.block1.config import (EXCLUSION_ICD10_PREFIXES,
                                   EXCLUSION_ICD10_EXACT)
    result = pyreadr.read_r(str(DATA_DIR / "hesin_diag.rds"))
    df_diag = result[None]
    del result; gc.collect()

    dcol = "diag_icd10"
    if dcol not in df_diag.columns:
        dcol = [c for c in df_diag.columns if "icd10" in c.lower()][0]
    codes = df_diag[dcol].astype(str).str.strip()
    mask = pd.Series(False, index=df_diag.index)
    for pfx in EXCLUSION_ICD10_PREFIXES:
        mask |= codes.str.startswith(pfx)
    for code in EXCLUSION_ICD10_EXACT:
        mask |= (codes == code)
    excluded = set(df_diag.loc[mask, "eid"].unique())
    df["is_healthy"] = ~df["eid"].isin(excluded)
    del df_diag; gc.collect()

    # HTN flag
    result = pyreadr.read_r(str(DATA_DIR / "hesin_diag.rds"))
    df_diag2 = result[None]
    del result; gc.collect()
    codes2 = df_diag2[dcol if dcol in df_diag2.columns
                       else "diag_icd10"].astype(str).str.strip()
    htn_mask = pd.Series(False, index=df_diag2.index)
    for pfx in ["I10", "I11", "I12", "I13", "I14", "I15"]:
        htn_mask |= codes2.str.startswith(pfx)
    htn_eids = set(df_diag2.loc[htn_mask, "eid"].unique())
    df["has_htn"] = df["eid"].isin(htn_eids)
    del df_diag2; gc.collect()

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["sex"] = pd.to_numeric(df["sex"], errors="coerce")
    print(f"  Built covariates: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# 2. Build survival dataset
# ---------------------------------------------------------------------------

def build_survival_df() -> pd.DataFrame:
    """Merge predictions + dates + events → survival-ready DataFrame."""

    # Load all components
    df_dates = load_imaging_dates()
    df_hf = load_hf_events()
    df_death = load_death_dates()
    df_cov = load_covariates()

    # Load predictions
    print("[5] Loading predictions ...")
    ft = pd.read_csv(PRED_DIR / "full_teacher_pred.csv")
    bb = pd.read_csv(PRED_DIR / "baseline_b_pred.csv")
    xgb = pd.read_csv(PRED_DIR / "baseline_a_xgb_pred.csv")
    en = pd.read_csv(PRED_DIR / "baseline_a_enet_pred.csv")

    # Rename to avoid collision
    ft.rename(columns={"structural_age": "ft_structural_age",
                        "deviation": "ft_deviation"}, inplace=True)
    bb.rename(columns={"predicted_age": "bb_predicted_age",
                        "deviation": "bb_age_gap"}, inplace=True)
    xgb.rename(columns={"predicted_age": "xgb_predicted_age",
                          "deviation": "xgb_age_gap"}, inplace=True)
    en.rename(columns={"predicted_age": "en_predicted_age",
                        "deviation": "en_age_gap"}, inplace=True)

    # Merge everything on eid
    # Start from predictions (they define our study cohort)
    df = ft[["eid", "ft_structural_age", "ft_deviation"] +
            [c for c in ft.columns if c.startswith("domain_")]]
    df = df.merge(bb[["eid", "bb_predicted_age", "bb_age_gap"]], on="eid", how="left")
    df = df.merge(xgb[["eid", "xgb_predicted_age", "xgb_age_gap"]], on="eid", how="left")
    df = df.merge(en[["eid", "en_predicted_age", "en_age_gap"]], on="eid", how="left")
    df = df.merge(df_cov, on="eid", how="left")
    df = df.merge(df_dates, on="eid", how="left")
    df = df.merge(df_hf, on="eid", how="left")
    df = df.merge(df_death, on="eid", how="left")

    # Compute FT age gap
    df["ft_age_gap"] = df["ft_structural_age"] - df["age"]

    # Administrative censoring date (latest follow-up in UKB ~2023-10-31)
    # Use max of observed event/death dates as proxy
    max_event = pd.NaT
    if df["hf_date"].notna().any():
        max_event = df["hf_date"].max()
    max_death = pd.NaT
    if "death_date" in df.columns and df["death_date"].notna().any():
        max_death = df["death_date"].max()
    admin_censor = max(d for d in [max_event, max_death] if pd.notna(d))
    print(f"  Administrative censoring date: {admin_censor}")

    # Drop participants without imaging date
    n_before = len(df)
    df = df.dropna(subset=["imaging_date"])
    print(f"  Dropped {n_before - len(df)} without imaging date → {len(df)}")

    # Define event and time
    # Incident HF = first I50 AFTER imaging date
    df["incident_hf"] = False
    has_hf = df["hf_date"].notna()
    df.loc[has_hf, "incident_hf"] = df.loc[has_hf, "hf_date"] > \
                                    df.loc[has_hf, "imaging_date"]

    # Prevalent HF (before imaging) — exclude
    prevalent_hf = has_hf & ~df["incident_hf"]
    n_prevalent = prevalent_hf.sum()
    print(f"  Prevalent HF (before imaging): {n_prevalent} — excluding")
    df = df[~prevalent_hf].copy()

    # End date: HF date (if incident), else death date, else admin censor
    df["end_date"] = admin_censor
    has_death = df["death_date"].notna()
    df.loc[has_death, "end_date"] = df.loc[has_death, "death_date"].clip(
        upper=admin_censor)
    df.loc[df["incident_hf"], "end_date"] = df.loc[
        df["incident_hf"], "hf_date"]

    # Follow-up time in years
    df["follow_up_years"] = (
        (df["end_date"] - df["imaging_date"]).dt.total_seconds()
        / (365.25 * 24 * 3600)
    )

    # Remove negative or zero follow-up
    valid = df["follow_up_years"] > 0
    n_invalid = (~valid).sum()
    if n_invalid > 0:
        print(f"  Removed {n_invalid} with non-positive follow-up")
    df = df[valid].copy()

    # Event indicator (binary)
    df["event"] = df["incident_hf"].astype(int)

    print(f"\n  Final cohort: {len(df)}")
    print(f"  Incident HF events: {df['event'].sum()} "
          f"({100 * df['event'].mean():.2f}%)")
    print(f"  Median follow-up: {df['follow_up_years'].median():.1f} years")
    print(f"  Age: {df['age'].mean():.1f} ± {df['age'].std():.1f}")

    return df


# ---------------------------------------------------------------------------
# 3. Analysis
# ---------------------------------------------------------------------------

def fit_cox_model(df: pd.DataFrame, time_col: str, event_col: str,
                  covariate_cols: list, label: str) -> dict:
    """Fit Cox PH model and return summary dict."""
    sub = df[[time_col, event_col] + covariate_cols].dropna().copy()

    # Standardize continuous covariates for comparable HRs
    for col in covariate_cols:
        if sub[col].nunique() > 2:
            mu, sigma = sub[col].mean(), sub[col].std()
            if sigma > 1e-8:
                sub[col] = (sub[col] - mu) / sigma

    cph = CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(sub, duration_col=time_col, event_col=event_col)

    # C-index
    c_idx = concordance_index(
        sub[time_col], -cph.predict_partial_hazard(sub), sub[event_col]
    )

    results = {
        "label": label,
        "n": len(sub),
        "n_events": int(sub[event_col].sum()),
        "c_index": round(c_idx, 4),
        "covariates": {},
    }
    for var in covariate_cols:
        row = cph.summary.loc[var]
        results["covariates"][var] = {
            "hr": round(float(np.exp(row["coef"])), 3),
            "hr_lower": round(float(np.exp(row["coef lower 95%"])), 3),
            "hr_upper": round(float(np.exp(row["coef upper 95%"])), 3),
            "p": float(row["p"]),
        }

    print(f"\n  [{label}]  C-index = {c_idx:.4f}  "
          f"(n={len(sub)}, events={int(sub[event_col].sum())})")
    for var, info in results["covariates"].items():
        sig = "***" if info["p"] < 0.001 else "**" if info["p"] < 0.01 \
            else "*" if info["p"] < 0.05 else ""
        print(f"    {var:25s}  HR={info['hr']:.3f} "
              f"[{info['hr_lower']:.3f}–{info['hr_upper']:.3f}]  "
              f"p={info['p']:.1e} {sig}")

    return results


def run_analysis(df: pd.DataFrame) -> dict:
    """Run all Cox models and comparisons."""
    results = {"models": []}

    print("\n" + "=" * 70)
    print("A. UNIVARIATE COX MODELS (single predictor)")
    print("=" * 70)

    # Univariate models
    predictors = {
        "FT_deviation": "ft_deviation",
        "FT_age_gap": "ft_age_gap",
        "BB_age_gap": "bb_age_gap",
        "XGB_age_gap": "xgb_age_gap",
        "EN_age_gap": "en_age_gap",
    }

    for name, col in predictors.items():
        if col in df.columns:
            r = fit_cox_model(df, "follow_up_years", "event",
                              [col], f"Univariate: {name}")
            results["models"].append(r)

    # Domain-specific deviations (FT only)
    domain_cols = [c for c in df.columns if c.startswith("domain_")]
    if domain_cols:
        r = fit_cox_model(df, "follow_up_years", "event",
                          domain_cols, "Univariate: FT all domains")
        results["models"].append(r)

    print("\n" + "=" * 70)
    print("B. ADJUSTED COX MODELS (+ age, sex, BMI)")
    print("=" * 70)

    base_covs = ["age", "sex"]
    if df["bmi"].notna().mean() > 0.8:
        base_covs.append("bmi")

    for name, col in predictors.items():
        if col in df.columns:
            r = fit_cox_model(df, "follow_up_years", "event",
                              [col] + base_covs,
                              f"Adjusted: {name}")
            results["models"].append(r)

    print("\n" + "=" * 70)
    print("C. FULLY ADJUSTED (+ age, sex, BMI, HTN)")
    print("=" * 70)

    if "has_htn" in df.columns:
        full_covs = base_covs + ["has_htn"]
        for name, col in [("FT_deviation", "ft_deviation"),
                          ("BB_age_gap", "bb_age_gap")]:
            if col in df.columns:
                r = fit_cox_model(df, "follow_up_years", "event",
                                  [col] + full_covs,
                                  f"Fully adjusted: {name}")
                results["models"].append(r)

    print("\n" + "=" * 70)
    print("D. FT DEVIATION + BB AGE_GAP (complementarity)")
    print("=" * 70)

    both_cols = ["ft_deviation", "bb_age_gap"] + base_covs
    if all(c in df.columns for c in both_cols):
        r = fit_cox_model(df, "follow_up_years", "event",
                          both_cols, "Joint: FT_deviation + BB_age_gap")
        results["models"].append(r)

    return results


def plot_km_by_tertile(df: pd.DataFrame, col: str, label: str,
                       filename: str):
    """Kaplan-Meier curves stratified by tertiles of a predictor."""
    sub = df.dropna(subset=[col, "follow_up_years", "event"]).copy()
    q33, q67 = sub[col].quantile([0.33, 0.67])
    sub["group"] = pd.cut(sub[col], bins=[-np.inf, q33, q67, np.inf],
                          labels=["Low", "Medium", "High"])

    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()
    colors = {"Low": "#2196F3", "Medium": "#FF9800", "High": "#F44336"}

    for group_name in ["Low", "Medium", "High"]:
        mask = sub["group"] == group_name
        n = mask.sum()
        events = sub.loc[mask, "event"].sum()
        kmf.fit(sub.loc[mask, "follow_up_years"],
                sub.loc[mask, "event"],
                label=f"{group_name} (n={n}, events={events})")
        kmf.plot_survival_function(ax=ax, color=colors[group_name])

    ax.set_xlabel("Follow-up (years)")
    ax.set_ylabel("HF-free survival")
    ax.set_title(f"KM survival by {label} tertiles")
    ax.legend(loc="lower left")
    ax.set_ylim(0.90, 1.0)

    # Log-rank test
    from lifelines.statistics import logrank_test
    high = sub["group"] == "High"
    low = sub["group"] == "Low"
    if high.sum() > 0 and low.sum() > 0:
        lr = logrank_test(
            sub.loc[high, "follow_up_years"],
            sub.loc[low, "follow_up_years"],
            sub.loc[high, "event"],
            sub.loc[low, "event"],
        )
        ax.text(0.02, 0.04, f"Log-rank (High vs Low): p={lr.p_value:.2e}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved: {FIG_DIR / filename}")


def plot_forest(results: dict, filename: str):
    """Forest plot comparing adjusted HRs of main predictor across models."""
    adjusted = [m for m in results["models"] if m["label"].startswith("Adjusted")]
    if not adjusted:
        return

    fig, ax = plt.subplots(figsize=(8, len(adjusted) * 0.6 + 1))
    y_positions = list(range(len(adjusted)))

    for i, m in enumerate(adjusted):
        # Get HR of the main predictor (first covariate)
        main_var = [v for v in m["covariates"]
                    if v not in ["age", "sex", "bmi", "has_htn"]][0]
        info = m["covariates"][main_var]
        hr = info["hr"]
        lo = info["hr_lower"]
        hi = info["hr_upper"]

        color = "#F44336" if info["p"] < 0.05 else "#9E9E9E"
        ax.plot([lo, hi], [i, i], color=color, linewidth=2)
        ax.plot(hr, i, "o", color=color, markersize=8)
        ax.text(hi + 0.02, i,
                f"HR={hr:.2f} [{lo:.2f}–{hi:.2f}], p={info['p']:.1e}",
                va="center", fontsize=8)

    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([m["label"].replace("Adjusted: ", "")
                        for m in adjusted], fontsize=9)
    ax.set_xlabel("Hazard Ratio (per SD)")
    ax.set_title("Adjusted Cox PH — Incident HF")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=150)
    plt.close()
    print(f"  Saved: {FIG_DIR / filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("P0: CMR-side incident HF survival analysis")
    print("=" * 70)

    # Build dataset
    df = build_survival_df()

    # Save intermediate for debugging
    df.to_csv(RESULTS_DIR / "p0_survival_cohort.csv", index=False)
    print(f"\n  Saved cohort → {RESULTS_DIR / 'p0_survival_cohort.csv'}")

    # Run analysis
    results = run_analysis(df)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: C-index comparison")
    print("=" * 70)
    summary = []
    for m in results["models"]:
        summary.append({
            "model": m["label"],
            "c_index": m["c_index"],
            "n": m["n"],
            "events": m["n_events"],
        })
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    results["summary"] = summary

    # Save
    out_path = RESULTS_DIR / "p0_survival_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults → {out_path}")

    # Plots
    print("\nGenerating plots ...")
    plot_km_by_tertile(df, "ft_deviation", "FT deviation",
                       "km_ft_deviation_tertile.png")
    plot_km_by_tertile(df, "ft_age_gap", "FT structural age gap",
                       "km_ft_age_gap_tertile.png")
    plot_km_by_tertile(df, "bb_age_gap", "Baseline B age gap",
                       "km_bb_age_gap_tertile.png")
    plot_forest(results, "forest_plot_adjusted.png")

    # Gate decision
    print("\n" + "=" * 70)
    print("GATE DECISION")
    print("=" * 70)
    adj_ft = [m for m in results["models"]
              if m["label"] == "Adjusted: FT_deviation"]
    adj_bb = [m for m in results["models"]
              if m["label"] == "Adjusted: BB_age_gap"]

    if adj_ft and adj_bb:
        c_ft = adj_ft[0]["c_index"]
        c_bb = adj_bb[0]["c_index"]
        delta = c_ft - c_bb
        print(f"  FT deviation C-index:  {c_ft:.4f}")
        print(f"  BB age-gap C-index:    {c_bb:.4f}")
        print(f"  Delta:                 {delta:+.4f}")
        if delta > 0:
            print("  → FT deviation WINS. Proceed to Block 2.")
        else:
            print("  → BB age-gap wins. Consider revising Block 1.")

        ft_hr = adj_ft[0]["covariates"].get("ft_deviation", {})
        bb_hr = adj_bb[0]["covariates"].get("bb_age_gap", {})
        if ft_hr and bb_hr:
            print(f"\n  FT deviation HR: {ft_hr['hr']:.3f} "
                  f"[{ft_hr['hr_lower']:.3f}–{ft_hr['hr_upper']:.3f}]  "
                  f"p={ft_hr['p']:.1e}")
            print(f"  BB age-gap HR:   {bb_hr['hr']:.3f} "
                  f"[{bb_hr['hr_lower']:.3f}–{bb_hr['hr_upper']:.3f}]  "
                  f"p={bb_hr['p']:.1e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
