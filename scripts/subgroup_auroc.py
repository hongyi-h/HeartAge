"""Subgroup AUROC analysis for Block 1 deviation."""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from src.block1.config import EXCLUSION_ICD10_PREFIXES, EXCLUSION_ICD10_EXACT

ft = pd.read_csv("results/block1/predictions/full_teacher_pred.csv")
bb = pd.read_csv("results/block1/predictions/baseline_b_pred.csv")
diag = pd.read_csv("results/block1/hesin_diag_study.csv")

icd_col = [c for c in diag.columns if c != 'eid'][0]
codes = diag[icd_col].dropna().astype(str).str.strip()
diag = diag.loc[codes.index].copy()
diag['code'] = codes.values

groups = {
    'HTN': ['I10','I11','I12','I13','I14','I15'],
    'DM': ['E10','E11','E12','E13','E14'],
    'AF': ['I48'],
    'MI': ['I21','I22'],
    'HF': ['I50'],
    'CM': ['I42','I43'],
    'Valve': ['I05','I06','I07','I08','I09','I34','I35','I36','I37'],
    'CKD': ['N183','N184','N185'],
}

all_study_eids = set(ft['eid'].values)
group_eids = {}
for gname, pfxs in groups.items():
    mask = pd.Series(False, index=diag.index)
    for p in pfxs:
        mask |= diag['code'].str.startswith(p)
    group_eids[gname] = set(diag.loc[mask, 'eid'].unique()) & all_study_eids

all_excl_mask = pd.Series(False, index=diag.index)
for prefix in EXCLUSION_ICD10_PREFIXES:
    all_excl_mask |= diag['code'].str.startswith(prefix)
for code in EXCLUSION_ICD10_EXACT:
    all_excl_mask |= (diag['code'] == code)
all_nh = set(diag.loc[all_excl_mask, 'eid'].unique()) & all_study_eids
healthy_eids = all_study_eids - all_nh

print(f"Healthy: {len(healthy_eids)}, Non-healthy: {len(all_nh)}")
print()

# Composition
print("=== Disease composition ===")
for g in ['HTN','DM','AF','MI','HF','CM','Valve','CKD']:
    n = len(group_eids[g])
    pct = n/len(all_nh)*100
    print(f"  {g:10s}: {n:5d} ({pct:5.1f}%)")
print()

# Subgroup AUROC
print("=== Subgroup AUROC ===")
ft_dev = ft.set_index('eid')['deviation']
bb_dev_abs = bb.set_index('eid')['deviation'].abs()

for g in ['HTN','DM','AF','MI','HF','CM','Valve','CKD']:
    ge = group_eids[g]
    if len(ge) < 30:
        continue
    eids = sorted(healthy_eids | ge)
    labels = np.array([0 if e in healthy_eids else 1 for e in eids])
    d_ft = np.array([ft_dev.get(e, np.nan) for e in eids])
    d_bb = np.array([bb_dev_abs.get(e, np.nan) for e in eids])
    v = np.isfinite(d_ft) & np.isfinite(d_bb)
    a_ft = roc_auc_score(labels[v], d_ft[v])
    a_bb = roc_auc_score(labels[v], d_bb[v])
    h_d = d_ft[v & (labels==0)]
    s_d = d_ft[v & (labels==1)]
    cd = (s_d.mean() - h_d.mean()) / np.sqrt((h_d.std()**2 + s_d.std()**2) / 2)
    print(f"  {g:10s} n={len(ge):5d}  FT={a_ft:.3f}  BB={a_bb:.3f}  diff={a_ft-a_bb:+.3f}  d={cd:.3f}")
print()

# Multimorbidity
print("=== Multimorbidity dose-response ===")
eid_cnt = {}
for e in all_nh:
    eid_cnt[e] = sum(1 for ge in group_eids.values() if e in ge)
for nd in range(1, 6):
    es = [e for e, c in eid_cnt.items() if c == nd]
    ds = [ft_dev.get(e, np.nan) for e in es]
    ds = [d for d in ds if np.isfinite(d)]
    if len(ds) > 10:
        print(f"  {nd} groups: n={len(ds)}, dev={np.mean(ds):.3f} +/- {np.std(ds):.3f}")
print()

# Domain deviations for HF
print("=== HF domain deviations ===")
ft_idx = ft.set_index('eid')
hf = group_eids['HF']
for dom in ['domain_LV','domain_RV','domain_Atrial','domain_Aortic','domain_Mechanics']:
    hv = ft_idx.loc[ft_idx.index.isin(healthy_eids), dom].values
    sv = ft_idx.loc[ft_idx.index.isin(hf), dom].values
    d = (sv.mean() - hv.mean()) / np.sqrt((hv.std()**2 + sv.std()**2) / 2)
    print(f"  {dom:20s}: H={hv.mean():.3f} HF={sv.mean():.3f} d={d:.3f}")
