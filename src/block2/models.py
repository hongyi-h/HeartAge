"""
Model definitions for Block 2: ECG Student distillation.

  - ResNet1DEncoder    (12-lead ECG → embedding)
  - ConceptBottleneck  (embedding → 12 named concepts)
  - MainHead           (concepts → structural_heart_age)
  - FullStudent        (encoder + bottleneck + head)
  - NoBnStudent        (encoder + direct head, no concept layer)
  - ChronoStudent      (encoder + head, trained on chronological age)
  - DirectOutcomeStudent (encoder + head, trained on HF outcome)

Derived scores (not trained):
  - remodeling_burden, perturbation_index, scope_uncertainty
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.block2.config import (
    ENCODER_CFG, STUDENT_CFG, N_CONCEPTS,
    STRUCTURAL_CONCEPT_IDX, RHYTHM_CONCEPT_IDX, QUALITY_CONCEPT_IDX,
    PERTURBATION_WEIGHTS, SCOPE_WEIGHTS,
    PROJECT_ROOT,
)

# Path to pretrained encoder checkpoint (from MAE pretraining)
PRETRAIN_CKPT = PROJECT_ROOT / "results" / "block2" / "pretrain" / "encoder_pretrained.pt"


def load_pretrained_encoder(model: nn.Module, ckpt_path: Path = None) -> bool:
    """Load MAE-pretrained weights into a student model's encoder.

    Looks for `model.encoder` attribute and loads matching keys.
    Returns True if weights were loaded, False if checkpoint not found.
    """
    ckpt_path = ckpt_path or PRETRAIN_CKPT
    if not ckpt_path.exists():
        return False
    if not hasattr(model, "encoder"):
        return False

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["encoder_state_dict"]
    missing, unexpected = model.encoder.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] Pretrained encoder missing keys: {missing}")
    if unexpected:
        print(f"  [WARN] Pretrained encoder unexpected keys: {unexpected}")
    return True


# ---------------------------------------------------------------------------
# ResNet-1D ECG Encoder
# ---------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    """Residual block: Conv1D → BN → ReLU → Conv1D → BN + skip."""
    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class DownsampleBlock(nn.Module):
    """Downsample: increase channels + reduce temporal resolution."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int, dropout: float):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               stride=stride, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1, stride=stride)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class ResNet1DEncoder(nn.Module):
    """1D ResNet encoder for 12-lead ECG → fixed-size embedding.

    Input:  (B, 12, 5000)  — 12 leads, 10s at 500Hz
    Output: (B, embedding_dim)
    """
    def __init__(self, in_channels: int = None, base_filters: int = None,
                 n_blocks: int = None, kernel_size: int = None,
                 stride: int = None, dropout: float = None,
                 embedding_dim: int = None):
        super().__init__()
        cfg = ENCODER_CFG
        in_channels = in_channels or cfg["in_channels"]
        base_filters = base_filters or cfg["base_filters"]
        n_blocks = n_blocks or cfg["n_blocks"]
        kernel_size = kernel_size or cfg["kernel_size"]
        stride = stride or cfg["stride"]
        dropout = dropout or cfg["dropout"]
        embedding_dim = embedding_dim or cfg["embedding_dim"]

        # Initial projection
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=15, padding=7),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual blocks with progressive downsampling
        blocks = []
        ch = base_filters
        for i in range(n_blocks):
            next_ch = ch * 2 if i > 0 else ch
            if next_ch != ch:
                blocks.append(DownsampleBlock(ch, next_ch, kernel_size,
                                              stride, dropout))
            else:
                blocks.append(DownsampleBlock(ch, next_ch, kernel_size,
                                              stride, dropout))
            blocks.append(ResBlock1D(next_ch, kernel_size, dropout))
            ch = next_ch
        self.blocks = nn.Sequential(*blocks)

        # Global average pooling + projection
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(ch, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 12, 5000) → (B, embedding_dim)"""
        out = self.stem(x)
        out = self.blocks(out)
        out = self.pool(out).squeeze(-1)
        return self.proj(out)


# ---------------------------------------------------------------------------
# Concept Bottleneck
# ---------------------------------------------------------------------------

class ConceptBottleneck(nn.Module):
    """Embedding → 12 named concept activations.

    Each concept has its own small MLP head from the shared embedding.
    Structural concepts (0-4) are supervised by teacher domain_scores.
    Rhythm concepts (5-8) by PTB-XL labels.
    Quality concepts (9-11) by rule-based QC scores.
    """
    def __init__(self, embedding_dim: int = None, n_concepts: int = None,
                 hidden_dim: int = None):
        super().__init__()
        cfg = STUDENT_CFG
        embedding_dim = embedding_dim or cfg["embedding_dim"]
        n_concepts = n_concepts or cfg["n_concepts"]
        hidden_dim = hidden_dim or cfg["concept_hidden"]

        self.concept_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(n_concepts)
        ])

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """embedding: (B, D) → concepts: (B, N_CONCEPTS)"""
        return torch.cat([head(embedding) for head in self.concept_heads],
                         dim=-1)


# ---------------------------------------------------------------------------
# Main Head (concepts → structural_heart_age)
# ---------------------------------------------------------------------------

class MainHead(nn.Module):
    """Concept activations → scalar structural_heart_age."""
    def __init__(self, n_concepts: int = None, hidden_dims: list = None):
        super().__init__()
        cfg = STUDENT_CFG
        n_concepts = n_concepts or cfg["n_concepts"]
        hidden_dims = hidden_dims or cfg["main_head_hidden"]

        layers = []
        prev = n_concepts
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        """concepts: (B, N_CONCEPTS) → structural_age: (B,)"""
        return self.net(concepts).squeeze(-1)


# ---------------------------------------------------------------------------
# Full Student (encoder + concept bottleneck + main head)
# ---------------------------------------------------------------------------

class FullStudent(nn.Module):
    """ECG → embedding → 12 concepts → structural_heart_age.

    Forward returns:
        embedding    (B, D)           — encoder output
        concepts     (B, N_CONCEPTS)  — concept activations
        struct_age   (B,)             — predicted structural heart age
    """
    def __init__(self):
        super().__init__()
        self.encoder = ResNet1DEncoder()
        self.bottleneck = ConceptBottleneck()
        self.main_head = MainHead()

    def forward(self, ecg: torch.Tensor):
        embedding = self.encoder(ecg)
        concepts = self.bottleneck(embedding)
        struct_age = self.main_head(concepts)
        return embedding, concepts, struct_age

    def predict_with_uncertainty(self, ecg: torch.Tensor,
                                 n_samples: int = None) -> dict:
        """MC dropout forward passes for scope_uncertainty."""
        n_samples = n_samples or STUDENT_CFG["mc_samples"]
        self.train()  # enable dropout
        ages = []
        concept_list = []
        with torch.no_grad():
            for _ in range(n_samples):
                emb = self.encoder(ecg)
                concepts = self.bottleneck(emb)
                sa = self.main_head(concepts)
                ages.append(sa)
                concept_list.append(concepts)
        self.eval()

        ages = torch.stack(ages, dim=0)       # (S, B)
        concepts = torch.stack(concept_list)   # (S, B, C)
        return {
            "mean_age": ages.mean(dim=0),
            "std_age": ages.std(dim=0),
            "mean_concepts": concepts.mean(dim=0),
            "mc_var": ages.var(dim=0),
        }


# ---------------------------------------------------------------------------
# No-Bottleneck Student (encoder + direct head)
# ---------------------------------------------------------------------------

class NoBnStudent(nn.Module):
    """ECG → embedding → structural_heart_age (no concept layer)."""
    def __init__(self):
        super().__init__()
        cfg = STUDENT_CFG
        self.encoder = ResNet1DEncoder()
        hidden_dims = cfg["main_head_hidden"]
        layers = []
        prev = cfg["embedding_dim"]
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, ecg: torch.Tensor):
        embedding = self.encoder(ecg)
        struct_age = self.head(embedding).squeeze(-1)
        return embedding, struct_age


# ---------------------------------------------------------------------------
# Chrono-Age Student (same architecture, different target)
# ---------------------------------------------------------------------------

class ChronoStudent(nn.Module):
    """ECG → embedding → predicted chronological age."""
    def __init__(self):
        super().__init__()
        cfg = STUDENT_CFG
        self.encoder = ResNet1DEncoder()
        hidden_dims = cfg["main_head_hidden"]
        layers = []
        prev = cfg["embedding_dim"]
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, ecg: torch.Tensor):
        embedding = self.encoder(ecg)
        pred_age = self.head(embedding).squeeze(-1)
        return embedding, pred_age


# ---------------------------------------------------------------------------
# Direct-Outcome Student (outcome leakage check)
# ---------------------------------------------------------------------------

class DirectOutcomeStudent(nn.Module):
    """ECG → embedding → P(incident HF). Binary classification comparator."""
    def __init__(self):
        super().__init__()
        cfg = STUDENT_CFG
        self.encoder = ResNet1DEncoder()
        hidden_dims = cfg["main_head_hidden"]
        layers = []
        prev = cfg["embedding_dim"]
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, ecg: torch.Tensor):
        embedding = self.encoder(ecg)
        logit = self.head(embedding).squeeze(-1)
        return embedding, logit


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def loss_distill(pred_age: torch.Tensor,
                 teacher_age: torch.Tensor) -> torch.Tensor:
    """MSE distillation loss: student structural_age vs teacher structural_age."""
    return F.mse_loss(pred_age, teacher_age)


def loss_rank_ecg(pred_age: torch.Tensor,
                  true_age: torch.Tensor) -> torch.Tensor:
    """Soft pairwise ranking: older → higher structural age (same as Block 1)."""
    n = len(true_age)
    if n < 2:
        return torch.tensor(0.0, device=pred_age.device)
    idx_i, idx_j = torch.triu_indices(n, n, offset=1)
    # Subsample pairs for efficiency (max 2048 pairs per batch)
    if len(idx_i) > 2048:
        perm = torch.randperm(len(idx_i), device=pred_age.device)[:2048]
        idx_i, idx_j = idx_i[perm], idx_j[perm]
    age_diff = true_age[idx_i] - true_age[idx_j]
    pred_diff = pred_age[idx_i] - pred_age[idx_j]
    signs = torch.sign(age_diff)
    mask = signs != 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_age.device)
    return -F.logsigmoid(signs[mask] * pred_diff[mask]).mean()


def loss_concept_mse(pred_concepts: torch.Tensor,
                     target_values: torch.Tensor,
                     concept_idx: list) -> torch.Tensor:
    """MSE loss for a subset of concepts against supervision targets."""
    pred = pred_concepts[:, concept_idx]
    return F.mse_loss(pred, target_values)


def loss_concept_bce(pred_concepts: torch.Tensor,
                     target_labels: torch.Tensor,
                     concept_idx: list) -> torch.Tensor:
    """Binary cross-entropy for binary concept labels (rhythm/quality)."""
    pred = pred_concepts[:, concept_idx]
    return F.binary_cross_entropy_with_logits(pred, target_labels)


# ---------------------------------------------------------------------------
# Derived scores (not trained — computed from concept activations)
# ---------------------------------------------------------------------------

def compute_remodeling_burden(concepts: torch.Tensor) -> torch.Tensor:
    """remodeling_burden = sqrt(mean(structural_concepts^2))"""
    struct = concepts[:, STRUCTURAL_CONCEPT_IDX]
    return (struct ** 2).mean(dim=1).sqrt()


def compute_perturbation_index(concepts: torch.Tensor) -> torch.Tensor:
    """Weighted combination of rhythm/quality concepts indicating transient state.

    perturbation = w1*(1-sinus_conf) + w2*|QRS-norm| + w3*ST_dev + w4*(1-SNR)
    """
    w = PERTURBATION_WEIGHTS
    sinus = torch.sigmoid(concepts[:, 5])   # concept 5: sinus_rhythm_conf
    qrs = torch.abs(concepts[:, 6])         # concept 6: QRS_duration_norm
    st = torch.abs(concepts[:, 7])          # concept 7: ST_deviation_score
    snr = torch.sigmoid(concepts[:, 9])     # concept 9: signal_noise_ratio

    return (w["w_sinus"] * (1 - sinus)
            + w["w_qrs"] * qrs
            + w["w_st"] * st
            + w["w_snr"] * (1 - snr))


def compute_scope_uncertainty(mc_var: torch.Tensor,
                              embedding: torch.Tensor,
                              train_mean: torch.Tensor = None,
                              train_cov_inv: torch.Tensor = None,
                              quality_concepts: torch.Tensor = None,
                              ) -> torch.Tensor:
    """scope_uncertainty = alpha*MC_var + beta*Mahalanobis + gamma*quality_penalty.

    Args:
        mc_var: (B,) MC dropout variance of structural_age
        embedding: (B, D) encoder output
        train_mean: (D,) mean embedding from training set
        train_cov_inv: (D, D) inverse covariance of training embeddings
        quality_concepts: (B, 3) quality concept activations (concepts 9-11)
    """
    w = SCOPE_WEIGHTS
    score = w["alpha_mc"] * mc_var

    # Mahalanobis distance
    if train_mean is not None and train_cov_inv is not None:
        diff = embedding - train_mean.unsqueeze(0)
        mahal = (diff @ train_cov_inv * diff).sum(dim=1).sqrt()
        score = score + w["beta_mahal"] * mahal

    # Quality penalty: low quality → high uncertainty
    if quality_concepts is not None:
        quality_score = torch.sigmoid(quality_concepts).mean(dim=1)
        score = score + w["gamma_miss"] * (1 - quality_score)

    return score
