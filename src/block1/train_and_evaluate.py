"""
Step 2: Train all Block 1 systems and evaluate.

Systems:
  1) Full Teacher   (domain encoders + normative head + structural-age head)
  2) Baseline A     (XGBoost + ElasticNet → chronological age)
  3) Baseline B     (MLP → chronological age, no manifold)
  4) Ablation       (Full Teacher + L_rec)

Usage:
    python -m src.block1.train_and_evaluate [--device cpu]

Output:
    results/block1/block1_results.json
    results/block1/predictions/*.csv
    results/block1/models/*.pt / *.pkl
    results/block1/figures/*.png
    results/block1/training_curves.json
"""

import argparse
import gc
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats as sp_stats
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset


def _to_np(t: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy, works even when numpy C bridge is broken."""
    try:
        return t.detach().cpu().numpy()
    except RuntimeError:
        return np.array(t.detach().cpu().tolist(), dtype=np.float32)

from src.block1.config import (
    ALL_IDP_FIELDS, IDP_DOMAINS, MODELS_DIR, FIGURES_DIR,
    PREDICTIONS_DIR, PROCESSED_DIR, RESULTS_DIR, TOTAL_IDP_DIM,
    TOTAL_LATENT_DIM, TRAIN_CFG,
)
from src.block1.models import (
    BaselineBMLP, FullTeacher,
    compute_deviation, compute_domain_scores,
    loss_age, loss_norm, loss_rank, loss_rec,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_prepared_data():
    """Load parquet, compute standardisation stats from healthy-train."""
    path = PROCESSED_DIR / "block1_data.parquet"
    df = pd.read_parquet(path)

    # Load cohort stats to get norm_feature_cols
    with open(RESULTS_DIR / "cohort_stats.json") as f:
        cstats = json.load(f)
    norm_feature_cols = cstats["norm_feature_cols"]

    # Standardise IDPs: fit on healthy train
    train_mask = (df["split"] == "train")
    idp_train = df.loc[train_mask, ALL_IDP_FIELDS].values.astype(np.float32)
    idp_mean = idp_train.mean(axis=0)
    idp_std = idp_train.std(axis=0)
    idp_std[idp_std < 1e-8] = 1.0  # avoid div-by-zero

    return df, norm_feature_cols, idp_mean, idp_std


def make_tensors(df, mask, norm_feature_cols, idp_mean, idp_std, device):
    """Create tensors for a subset of the DataFrame."""
    sub = df.loc[mask]
    idps = (sub[ALL_IDP_FIELDS].values.astype(np.float32) - idp_mean) / idp_std
    ages = sub["age"].values.astype(np.float32)
    sex = sub["sex"].values.astype(np.float32)
    norm_feat = sub[norm_feature_cols].values.astype(np.float32)

    return (
        torch.as_tensor(idps).to(device),
        torch.as_tensor(ages).to(device),
        torch.as_tensor(sex).to(device),
        torch.as_tensor(norm_feat).to(device),
    )


# ---------------------------------------------------------------------------
# Training: Full Teacher / Ablation
# ---------------------------------------------------------------------------

def train_teacher(df, norm_feature_cols, idp_mean, idp_std,
                  device, use_rec=False, tag="full_teacher"):
    """Train the FullTeacher model with early stopping on val loss."""
    cfg = TRAIN_CFG
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    n_norm_feat = len(norm_feature_cols)
    model = FullTeacher(n_norm_feat, use_rec=use_rec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["max_epochs"])

    # Data
    idps_tr, age_tr, sex_tr, nf_tr = make_tensors(
        df, df["split"] == "train", norm_feature_cols, idp_mean, idp_std, device)
    idps_va, age_va, sex_va, nf_va = make_tensors(
        df, df["split"] == "val", norm_feature_cols, idp_mean, idp_std, device)

    loader_tr = DataLoader(
        TensorDataset(idps_tr, age_tr, nf_tr),
        batch_size=cfg["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "l_norm": [], "l_age": [],
               "l_rank": []}
    if use_rec:
        history["l_rec"] = []

    print(f"\nTraining {tag} …")
    for epoch in range(cfg["max_epochs"]):
        model.train()
        epoch_losses = {"total": 0, "norm": 0, "age": 0, "rank": 0, "rec": 0}
        n_batches = 0

        for batch_idps, batch_age, batch_nf in loader_tr:
            z, mu, lv, sa = model(batch_idps, batch_nf)

            ln = loss_norm(z, mu, lv)
            la = loss_age(sa, batch_age)
            lr_ = loss_rank(sa, batch_age)
            total = cfg["w_norm"] * ln + cfg["w_age"] * la + cfg["w_rank"] * lr_

            if use_rec:
                lrec = loss_rec(z, model, batch_idps)
                total = total + cfg["w_rec"] * lrec
                epoch_losses["rec"] += lrec.item()

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_losses["total"] += total.item()
            epoch_losses["norm"] += ln.item()
            epoch_losses["age"] += la.item()
            epoch_losses["rank"] += lr_.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            z_v, mu_v, lv_v, sa_v = model(idps_va, nf_va)
            vl_norm = loss_norm(z_v, mu_v, lv_v)
            vl_age = loss_age(sa_v, age_va)
            vl_rank = loss_rank(sa_v, age_va)
            val_loss = (cfg["w_norm"] * vl_norm + cfg["w_age"] * vl_age
                        + cfg["w_rank"] * vl_rank).item()

        avg_train = epoch_losses["total"] / max(n_batches, 1)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss)
        history["l_norm"].append(epoch_losses["norm"] / max(n_batches, 1))
        history["l_age"].append(epoch_losses["age"] / max(n_batches, 1))
        history["l_rank"].append(epoch_losses["rank"] / max(n_batches, 1))
        if use_rec:
            history["l_rec"].append(epoch_losses["rec"] / max(n_batches, 1))

        if epoch % 20 == 0 or epoch == cfg["max_epochs"] - 1:
            print(f"  epoch {epoch:3d}  train={avg_train:.4f}  "
                  f"val={val_loss:.4f}  "
                  f"norm={history['l_norm'][-1]:.4f}  "
                  f"age={history['l_age'][-1]:.4f}")

        # Early stopping
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()

    # Save
    ckpt_path = MODELS_DIR / f"{tag}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "idp_mean": idp_mean,
        "idp_std": idp_std,
        "norm_feature_cols": norm_feature_cols,
    }, ckpt_path)

    return model, history


# ---------------------------------------------------------------------------
# Training: Baseline A (XGBoost / ElasticNet)
# ---------------------------------------------------------------------------

def train_baseline_a(df, idp_mean, idp_std):
    """Train ElasticNet on standardised IDPs + sex → age.
    XGBoost attempted first; falls back to ElasticNet if xgboost unavailable.
    """
    train_mask = df["split"] == "train"
    X_tr = (df.loc[train_mask, ALL_IDP_FIELDS].values - idp_mean) / idp_std
    sex_tr = df.loc[train_mask, "sex"].values.reshape(-1, 1)
    X_tr = np.hstack([X_tr, sex_tr]).astype(np.float32)
    y_tr = df.loc[train_mask, "age"].values.astype(np.float32)

    models_a = {}

    # ElasticNet (always)
    print("\nTraining Baseline A: ElasticNet …")
    enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5,
                        max_iter=2000, n_jobs=-1, random_state=42)
    enet.fit(X_tr, y_tr)
    models_a["enet"] = enet
    import joblib
    joblib.dump(enet, MODELS_DIR / "baseline_a_enet.pkl")
    print(f"  ElasticNet alpha={enet.alpha_:.4f}, l1_ratio={enet.l1_ratio_:.2f}")

    # XGBoost (best effort)
    try:
        import xgboost as xgb
        print("Training Baseline A: XGBoost …")
        xgb_model = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, tree_method="hist",
        )
        xgb_model.fit(X_tr, y_tr,
                       eval_set=[(X_tr, y_tr)], verbose=False)
        models_a["xgb"] = xgb_model
        joblib.dump(xgb_model, MODELS_DIR / "baseline_a_xgb.pkl")
        print("  XGBoost trained.")
    except ImportError:
        print("  xgboost not installed — skipping XGBoost baseline")

    return models_a


# ---------------------------------------------------------------------------
# Training: Baseline B (MLP, no manifold)
# ---------------------------------------------------------------------------

def train_baseline_b(df, idp_mean, idp_std, device):
    """Train BaselineBMLP: IDPs + sex → predicted age."""
    cfg = TRAIN_CFG
    torch.manual_seed(cfg["seed"])

    model = BaselineBMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["max_epochs"])

    def _pack(mask):
        sub = df.loc[mask]
        idps = (sub[ALL_IDP_FIELDS].values.astype(np.float32) - idp_mean) / idp_std
        sex = sub["sex"].values.astype(np.float32).reshape(-1, 1)
        x = np.hstack([idps, sex])
        y = sub["age"].values.astype(np.float32)
        return (torch.as_tensor(x).to(device),
                torch.as_tensor(y).to(device))

    x_tr, y_tr = _pack(df["split"] == "train")
    x_va, y_va = _pack(df["split"] == "val")

    loader = DataLoader(TensorDataset(x_tr, y_tr),
                        batch_size=cfg["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(cfg["seed"]))

    best_val = float("inf")
    patience_ctr = 0
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    print("\nTraining Baseline B: MLP …")
    for epoch in range(cfg["max_epochs"]):
        model.train()
        total_loss = 0; nb = 0
        for bx, by in loader:
            pred = model(bx)
            loss = F.smooth_l1_loss(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); nb += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = F.smooth_l1_loss(model(x_va), y_va).item()

        history["train_loss"].append(total_loss / max(nb, 1))
        history["val_loss"].append(val_loss)

        if epoch % 50 == 0:
            print(f"  epoch {epoch:3d}  train={history['train_loss'][-1]:.4f}  "
                  f"val={val_loss:.4f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            patience_ctr = 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    torch.save({"state_dict": model.state_dict(),
                "idp_mean": idp_mean, "idp_std": idp_std},
               MODELS_DIR / "baseline_b.pt")
    return model, history


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_teacher(model, df, norm_feature_cols, idp_mean, idp_std, device):
    """Apply FullTeacher to the whole dataframe; return predictions df."""
    idps_all = (df[ALL_IDP_FIELDS].values.astype(np.float32) - idp_mean) / idp_std
    nf_all = df[norm_feature_cols].values.astype(np.float32)

    # Process in chunks to save memory
    CHUNK = 4096
    struct_ages, deviations = [], []
    domain_score_dict = {d: [] for d in IDP_DOMAINS}

    model.eval()
    for start in range(0, len(df), CHUNK):
        end = min(start + CHUNK, len(df))
        idps_t = torch.as_tensor(idps_all[start:end].copy()).to(device)
        nf_t = torch.as_tensor(nf_all[start:end].copy()).to(device)

        z, mu, lv, sa = model(idps_t, nf_t)
        dev = compute_deviation(z, mu, lv)
        ds = compute_domain_scores(z, mu, lv)

        struct_ages.append(_to_np(sa))
        deviations.append(_to_np(dev))
        for d in IDP_DOMAINS:
            domain_score_dict[d].append(_to_np(ds[d]))

    pred_df = pd.DataFrame({
        "eid": df["eid"].values,
        "structural_age": np.concatenate(struct_ages),
        "deviation": np.concatenate(deviations),
    })
    for d in IDP_DOMAINS:
        pred_df[f"domain_{d}"] = np.concatenate(domain_score_dict[d])

    return pred_df


@torch.no_grad()
def eval_baseline_b(model, df, idp_mean, idp_std, device):
    """Apply BaselineBMLP to all data."""
    idps = (df[ALL_IDP_FIELDS].values.astype(np.float32) - idp_mean) / idp_std
    sex = df["sex"].values.astype(np.float32).reshape(-1, 1)
    x = np.hstack([idps, sex])

    CHUNK = 4096
    preds = []
    model.eval()
    for start in range(0, len(df), CHUNK):
        end = min(start + CHUNK, len(df))
        xt = torch.as_tensor(x[start:end].copy()).to(device)
        preds.append(_to_np(model(xt)))

    pred_age = np.concatenate(preds)
    return pd.DataFrame({
        "eid": df["eid"].values,
        "predicted_age": pred_age,
        "deviation": pred_age - df["age"].values.astype(np.float32),
    })


def eval_baseline_a_model(model, df, idp_mean, idp_std, label):
    """Apply a sklearn model to all data."""
    idps = (df[ALL_IDP_FIELDS].values.astype(np.float32) - idp_mean) / idp_std
    sex = df["sex"].values.astype(np.float32).reshape(-1, 1)
    x = np.hstack([idps, sex])
    pred_age = model.predict(x).astype(np.float32)
    return pd.DataFrame({
        "eid": df["eid"].values,
        "predicted_age": pred_age,
        "deviation": pred_age - df["age"].values.astype(np.float32),
    })


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calibration_metrics(pred_age, true_age):
    """MAE, RMSE, calibration slope/intercept, R²."""
    mae = float(np.mean(np.abs(pred_age - true_age)))
    rmse = float(np.sqrt(np.mean((pred_age - true_age) ** 2)))
    slope, intercept, r, p, se = sp_stats.linregress(pred_age, true_age)
    r2 = float(r2_score(true_age, pred_age))
    return {
        "mae": mae, "rmse": rmse,
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "r2": r2,
    }


def deviation_stats(dev_values):
    """Mean, std (variance proxy) of deviation."""
    return {
        "mean": float(np.mean(dev_values)),
        "std": float(np.std(dev_values)),
        "variance": float(np.var(dev_values)),
        "median": float(np.median(dev_values)),
    }


def risk_factor_corr(dev_values, covariate, name):
    """Spearman correlation between deviation and a covariate."""
    mask = np.isfinite(covariate) & np.isfinite(dev_values)
    if mask.sum() < 30:
        return {"r": None, "p": None, "n": int(mask.sum()), "name": name}
    r, p = sp_stats.spearmanr(dev_values[mask], covariate[mask])
    return {"r": float(r), "p": float(p), "n": int(mask.sum()), "name": name}


def evaluate_system(pred_df, df, system_name, age_col="structural_age",
                    dev_col="deviation"):
    """Compute all Block 1 metrics for one system."""
    metrics = {}

    # --- Healthy test ---
    test_mask = (df["split"] == "test").values
    test_pred = pred_df.loc[test_mask]
    test_true_age = df.loc[test_mask, "age"].values.astype(np.float32)

    if age_col in test_pred.columns:
        metrics["test_healthy_calibration"] = calibration_metrics(
            test_pred[age_col].values, test_true_age)

    metrics["test_healthy_deviation"] = deviation_stats(
        test_pred[dev_col].values)

    # --- Healthy train (for comparison) ---
    train_mask = (df["split"] == "train").values
    metrics["train_healthy_deviation"] = deviation_stats(
        pred_df.loc[train_mask, dev_col].values)

    # --- Non-healthy ---
    nh_mask = (~df["is_healthy"]).values
    if nh_mask.sum() > 0:
        nh_pred = pred_df.loc[nh_mask]
        metrics["non_healthy_deviation"] = deviation_stats(
            nh_pred[dev_col].values)

        # Risk factor correlations
        nh_dev = nh_pred[dev_col].values.astype(np.float64)

        # BMI
        bmi = df.loc[nh_mask, "bmi"].values.astype(np.float64)
        metrics["corr_deviation_bmi"] = risk_factor_corr(nh_dev, bmi, "BMI")

        # Smoking (current=2 → 1, else → 0)
        smoking_raw = df.loc[nh_mask, "smoking"].values.astype(np.float64)
        smoking_binary = (smoking_raw == 2).astype(np.float64)
        smoking_binary[np.isnan(smoking_raw)] = np.nan
        metrics["corr_deviation_smoking"] = risk_factor_corr(
            nh_dev, smoking_binary, "current_smoking")

        # Physical inactivity (0 days vigorous → 1, else → 0)
        act_raw = df.loc[nh_mask, "vigorous_activity"].values.astype(np.float64)
        # code -1 (don't know) and -3 (prefer not to say) as NaN
        act_clean = np.where(act_raw < 0, np.nan, act_raw)
        inactivity = (act_clean == 0).astype(np.float64)
        inactivity[np.isnan(act_clean)] = np.nan
        metrics["corr_deviation_inactivity"] = risk_factor_corr(
            nh_dev, inactivity, "physical_inactivity")

    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_calibration(pred_age, true_age, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pred_age, true_age, s=1, alpha=0.3)
    lims = [min(pred_age.min(), true_age.min()) - 2,
            max(pred_age.max(), true_age.max()) + 2]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("Predicted structural age")
    ax.set_ylabel("Chronological age")
    ax.set_title(title)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_deviation_distributions(dev_healthy, dev_unhealthy, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(dev_healthy, bins=50, alpha=0.6, label="Healthy", density=True)
    if len(dev_unhealthy) > 0:
        ax.hist(dev_unhealthy, bins=50, alpha=0.6, label="Non-healthy",
                density=True)
    ax.set_xlabel("Deviation from manifold")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curves(histories: dict, out_path):
    fig, axes = plt.subplots(1, len(histories), figsize=(5 * len(histories), 4))
    if len(histories) == 1:
        axes = [axes]
    for ax, (name, h) in zip(axes, histories.items()):
        ax.plot(h["train_loss"], label="train")
        ax.plot(h["val_loss"], label="val")
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    for d in [MODELS_DIR, FIGURES_DIR, PREDICTIONS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("Loading prepared data …")
    df, norm_feature_cols, idp_mean, idp_std = load_prepared_data()
    print(f"  {len(df)} total, "
          f"{(df['split']=='train').sum()} train, "
          f"{(df['split']=='val').sum()} val, "
          f"{(df['split']=='test').sum()} test, "
          f"{(~df['is_healthy']).sum()} non-healthy")

    all_results = {}
    all_histories = {}

    # ================================================================
    # 1. Full Teacher
    # ================================================================
    teacher, hist_t = train_teacher(
        df, norm_feature_cols, idp_mean, idp_std, device,
        use_rec=False, tag="full_teacher")
    all_histories["full_teacher"] = hist_t

    pred_teacher = eval_teacher(teacher, df, norm_feature_cols,
                                idp_mean, idp_std, device)
    pred_teacher.to_csv(PREDICTIONS_DIR / "full_teacher_pred.csv", index=False)

    all_results["full_teacher"] = evaluate_system(
        pred_teacher, df, "full_teacher",
        age_col="structural_age", dev_col="deviation")

    # Calibration plot (healthy test)
    test_m = (df["split"] == "test").values
    plot_calibration(
        pred_teacher.loc[test_m, "structural_age"].values,
        df.loc[test_m, "age"].values,
        "Full Teacher — calibration (healthy test)",
        FIGURES_DIR / "calibration_full_teacher.png")

    # Deviation distribution
    healthy_m = df["is_healthy"].values
    plot_deviation_distributions(
        pred_teacher.loc[healthy_m, "deviation"].values,
        pred_teacher.loc[~healthy_m, "deviation"].values,
        "Full Teacher — deviation", FIGURES_DIR / "deviation_full_teacher.png")

    del teacher; gc.collect()

    # ================================================================
    # 2. Baseline A (ElasticNet / XGBoost)
    # ================================================================
    models_a = train_baseline_a(df, idp_mean, idp_std)
    for label, model_a in models_a.items():
        tag = f"baseline_a_{label}"
        pred_a = eval_baseline_a_model(model_a, df, idp_mean, idp_std, label)
        pred_a.to_csv(PREDICTIONS_DIR / f"{tag}_pred.csv", index=False)
        all_results[tag] = evaluate_system(
            pred_a, df, tag, age_col="predicted_age", dev_col="deviation")

        plot_calibration(
            pred_a.loc[test_m, "predicted_age"].values,
            df.loc[test_m, "age"].values,
            f"Baseline A ({label}) — calibration",
            FIGURES_DIR / f"calibration_{tag}.png")

    del models_a; gc.collect()

    # ================================================================
    # 3. Baseline B (MLP)
    # ================================================================
    model_b, hist_b = train_baseline_b(df, idp_mean, idp_std, device)
    all_histories["baseline_b"] = hist_b

    pred_b = eval_baseline_b(model_b, df, idp_mean, idp_std, device)
    pred_b.to_csv(PREDICTIONS_DIR / "baseline_b_pred.csv", index=False)
    all_results["baseline_b"] = evaluate_system(
        pred_b, df, "baseline_b", age_col="predicted_age", dev_col="deviation")

    plot_calibration(
        pred_b.loc[test_m, "predicted_age"].values,
        df.loc[test_m, "age"].values,
        "Baseline B (MLP) — calibration",
        FIGURES_DIR / "calibration_baseline_b.png")

    del model_b; gc.collect()

    # ================================================================
    # 4. Ablation: Full Teacher + L_rec
    # ================================================================
    teacher_rec, hist_rec = train_teacher(
        df, norm_feature_cols, idp_mean, idp_std, device,
        use_rec=True, tag="ablation_rec")
    all_histories["ablation_rec"] = hist_rec

    pred_rec = eval_teacher(teacher_rec, df, norm_feature_cols,
                            idp_mean, idp_std, device)
    pred_rec.to_csv(PREDICTIONS_DIR / "ablation_rec_pred.csv", index=False)
    all_results["ablation_rec"] = evaluate_system(
        pred_rec, df, "ablation_rec",
        age_col="structural_age", dev_col="deviation")

    del teacher_rec; gc.collect()

    # ================================================================
    # 5. Comparative metrics
    # ================================================================
    print("\nComputing comparative metrics …")
    comparisons = {}

    # --- 5a. Proper comparison: AUROC of deviation for healthy vs non-healthy ---
    # Both systems should produce a deviation that separates healthy from non-healthy.
    # Use AUROC as an apples-to-apples metric (scale-invariant).
    from sklearn.metrics import roc_auc_score

    full_mask = df["is_healthy"].values  # True = healthy
    # For FT: deviation = Mahalanobis (higher = more abnormal)
    dev_ft_all = pred_teacher["deviation"].values
    # For BB: use |age residual| as deviation proxy (higher = more abnormal)
    dev_bb_all = np.abs(pred_b["deviation"].values)

    # Labels: 1 = non-healthy (the positive class we want to detect)
    labels_all = (~df["is_healthy"]).values.astype(int)

    auroc_ft = float(roc_auc_score(labels_all, dev_ft_all))
    auroc_bb = float(roc_auc_score(labels_all, dev_bb_all))
    comparisons["discrimination_auroc"] = {
        "full_teacher_auroc": auroc_ft,
        "baseline_b_auroc": auroc_bb,
        "difference": auroc_ft - auroc_bb,
        "note": "AUROC for classifying healthy vs non-healthy using deviation"
    }

    # --- 5b. Deviation variance on healthy test (same-scale via z-scoring) ---
    test_m_only = (df["split"] == "test").values
    dev_ft_test = pred_teacher.loc[test_m_only, "deviation"].values
    dev_bb_test = np.abs(pred_b.loc[test_m_only, "deviation"].values)
    # Z-score each to unit-free scale, then compare CoV
    comparisons["deviation_coefficient_of_variation"] = {
        "full_teacher_cv": float(np.std(dev_ft_test) / (np.mean(dev_ft_test) + 1e-8)),
        "baseline_b_cv": float(np.std(dev_bb_test) / (np.mean(dev_bb_test) + 1e-8)),
        "note": "Coefficient of variation (std/mean), scale-invariant"
    }

    # --- 5c. Risk factor correlation comparison (Steiger test for dependent r) ---
    nh_mask = (~df["is_healthy"]).values
    if nh_mask.sum() > 100:
        bmi_nh = df.loc[nh_mask, "bmi"].values.astype(np.float64)
        valid = np.isfinite(bmi_nh)
        dev_ft_nh = pred_teacher.loc[nh_mask, "deviation"].values.astype(np.float64)
        dev_bb_nh = np.abs(pred_b.loc[nh_mask, "deviation"].values).astype(np.float64)

        if valid.sum() > 30:
            r_ft, _ = sp_stats.spearmanr(dev_ft_nh[valid], bmi_nh[valid])
            r_bb, _ = sp_stats.spearmanr(dev_bb_nh[valid], bmi_nh[valid])
            # Correlation between the two deviation measures (needed for Steiger)
            r_12, _ = sp_stats.spearmanr(dev_ft_nh[valid], dev_bb_nh[valid])
            n_v = int(valid.sum())

            # Steiger (1980) test for dependent correlations
            r_mean_sq = ((r_ft + r_bb) / 2) ** 2
            denom = (1 - r_12) * (1 - r_mean_sq) * 2
            if abs(denom) > 1e-10:
                # Simplified Steiger formula
                z_ft = np.arctanh(r_ft)
                z_bb = np.arctanh(r_bb)
                # Correct SE for dependent correlations
                f_r12 = (1 - r_12) / (2 * (1 - r_mean_sq))
                se = np.sqrt((2 * (1 - r_12)) / (n_v * f_r12 + 1e-10))
                z_diff = (z_ft - z_bb) / (se + 1e-10)
                p_steiger = 2 * (1 - sp_stats.norm.cdf(abs(z_diff)))
            else:
                z_diff = 0.0
                p_steiger = 1.0

            comparisons["bmi_corr_steiger"] = {
                "r_full_teacher": float(r_ft),
                "r_baseline_b": float(r_bb),
                "r_between_deviations": float(r_12),
                "z_diff": float(z_diff),
                "p_value": float(p_steiger),
                "n": n_v,
                "note": "Steiger test for dependent correlations (correct for paired data)"
            }

    all_results["comparisons"] = comparisons

    # ================================================================
    # 6. Save everything
    # ================================================================
    results_path = RESULTS_DIR / "block1_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults → {results_path}")

    # Training histories
    curves_path = RESULTS_DIR / "training_curves.json"
    with open(curves_path, "w") as f:
        json.dump(all_histories, f, indent=2)
    print(f"Training curves → {curves_path}")

    # Plot all training curves
    plot_training_curves(all_histories, FIGURES_DIR / "training_curves.png")

    # ================================================================
    # 7. Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("Block 1 — Summary")
    print("=" * 60)
    for sys_name in ["full_teacher", "baseline_a_enet", "baseline_b",
                     "ablation_rec"]:
        if sys_name not in all_results:
            continue
        r = all_results[sys_name]
        cal_key = "test_healthy_calibration"
        if cal_key in r:
            c = r[cal_key]
            print(f"\n{sys_name}:")
            print(f"  MAE={c['mae']:.2f}  RMSE={c['rmse']:.2f}  "
                  f"slope={c['calibration_slope']:.3f}  R²={c['r2']:.3f}")
        dev = r.get("test_healthy_deviation", {})
        print(f"  Deviation var={dev.get('variance','?'):.4f}  "
              f"mean={dev.get('mean','?'):.4f}")

    comp = all_results.get("comparisons", {})
    disc = comp.get("discrimination_auroc", {})
    if disc:
        print(f"\nDeviation AUROC (healthy vs non-healthy): "
              f"FT={disc.get('full_teacher_auroc', '?'):.4f} vs "
              f"BB={disc.get('baseline_b_auroc', '?'):.4f}")
    steiger = comp.get("bmi_corr_steiger", {})
    if steiger:
        print(f"BMI correlation (Steiger): "
              f"FT r={steiger.get('r_full_teacher', '?'):.4f} vs "
              f"BB r={steiger.get('r_baseline_b', '?'):.4f}  "
              f"p={steiger.get('p_value', '?'):.4g}")

    print(f"\nAll results saved to {RESULTS_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
