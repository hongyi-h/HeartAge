"""
Step 3 (optional): Bootstrap stability analysis for the Full Teacher.

Resamples healthy training data N times, retrains the teacher each time,
and reports the SD of structural_age predictions on the test set.

Usage:
    python -m src.block1.bootstrap [--n_bootstrap 50] [--device cpu]

Output:
    results/block1/bootstrap_stability.json
"""

import argparse
import json
import gc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.block1.config import (
    ALL_IDP_FIELDS, PROCESSED_DIR, RESULTS_DIR, TRAIN_CFG,
    TOTAL_LATENT_DIM,
)
from src.block1.models import (
    FullTeacher, loss_age, loss_norm, loss_rank,
)
from src.block1.train_and_evaluate import load_prepared_data, make_tensors


def train_teacher_fast(idps_tr, age_tr, nf_tr, idps_va, age_va, nf_va,
                       n_norm_feat, device, seed, max_epochs=150):
    """Lighter training loop for bootstrap (fewer epochs, no history)."""
    cfg = TRAIN_CFG
    torch.manual_seed(seed)

    model = FullTeacher(n_norm_feat, use_rec=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])

    loader = DataLoader(
        TensorDataset(idps_tr, age_tr, nf_tr),
        batch_size=cfg["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    best_val = float("inf")
    patience = 20
    cnt = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        for bi, ba, bn in loader:
            z, mu, lv, sa = model(bi, bn)
            loss = (cfg["w_norm"] * loss_norm(z, mu, lv)
                    + cfg["w_age"] * loss_age(sa, ba)
                    + cfg["w_rank"] * loss_rank(sa, ba))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            z_v, mu_v, lv_v, sa_v = model(idps_va, nf_va)
            vl = (cfg["w_norm"] * loss_norm(z_v, mu_v, lv_v)
                  + cfg["w_age"] * loss_age(sa_v, age_va)
                  + cfg["w_rank"] * loss_rank(sa_v, age_va)).item()

        if vl < best_val - 1e-5:
            best_val = vl
            cnt = 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            cnt += 1
            if cnt >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_bootstrap", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"Bootstrap stability: {args.n_bootstrap} resamples")

    df, norm_feature_cols, idp_mean, idp_std = load_prepared_data()
    n_norm_feat = len(norm_feature_cols)

    # Fixed test tensors
    test_mask = df["split"] == "test"
    idps_te, age_te, sex_te, nf_te = make_tensors(
        df, test_mask, norm_feature_cols, idp_mean, idp_std, device)
    n_test = int(test_mask.sum())

    # Fixed val tensors
    val_mask = df["split"] == "val"
    idps_va, age_va, sex_va, nf_va = make_tensors(
        df, val_mask, norm_feature_cols, idp_mean, idp_std, device)

    # Train-set data (will be resampled)
    train_mask = df["split"] == "train"
    idps_tr_full, age_tr_full, sex_tr_full, nf_tr_full = make_tensors(
        df, train_mask, norm_feature_cols, idp_mean, idp_std, device)
    n_train = int(train_mask.sum())

    # Store test-set structural_age from each bootstrap
    struct_ages = np.zeros((args.n_bootstrap, n_test), dtype=np.float32)

    for b in range(args.n_bootstrap):
        print(f"  Bootstrap {b+1}/{args.n_bootstrap} …", end=" ", flush=True)

        # Resample training indices with replacement
        rng = np.random.RandomState(TRAIN_CFG["seed"] + b)
        boot_idx = torch.from_numpy(
            rng.randint(0, n_train, size=n_train)).long()

        idps_boot = idps_tr_full[boot_idx]
        age_boot = age_tr_full[boot_idx]
        nf_boot = nf_tr_full[boot_idx]

        model = train_teacher_fast(
            idps_boot, age_boot, nf_boot,
            idps_va, age_va, nf_va,
            n_norm_feat, device,
            seed=TRAIN_CFG["seed"] + b,
        )

        with torch.no_grad():
            _, _, _, sa_test = model(idps_te, nf_te)
            struct_ages[b] = sa_test.cpu().numpy()

        del model
        gc.collect()
        print(f"done (SA mean={struct_ages[b].mean():.1f})")

    # Compute stability
    mean_sa = struct_ages.mean(axis=0)
    std_sa = struct_ages.std(axis=0)

    result = {
        "n_bootstrap": args.n_bootstrap,
        "structural_age_sd_per_subject_mean": float(std_sa.mean()),
        "structural_age_sd_per_subject_median": float(np.median(std_sa)),
        "structural_age_sd_per_subject_max": float(std_sa.max()),
        "structural_age_sd_per_subject_p95": float(np.percentile(std_sa, 95)),
        "pass_criterion": bool(std_sa.mean() < 1.5),
        "criterion": "mean SD < 1.5 years",
    }

    out_path = RESULTS_DIR / "bootstrap_stability.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults → {out_path}")
    print(f"  Mean SD   = {result['structural_age_sd_per_subject_mean']:.3f} years")
    print(f"  Median SD = {result['structural_age_sd_per_subject_median']:.3f} years")
    print(f"  Pass: {result['pass_criterion']} (criterion: {result['criterion']})")


if __name__ == "__main__":
    main()
