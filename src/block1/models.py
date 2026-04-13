"""
Model definitions for Block 1:
  - FullTeacher  (domain encoders + normative head + structural-age head)
  - BaselineBMLP (single MLP → age, no manifold)
  - Decoder      (z → reconstructed IDPs, for ablation L_rec)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.block1.config import IDP_DOMAINS, TOTAL_LATENT_DIM, ENCODER_HIDDEN_MIN, MONO_HIDDEN, BASELINE_B_HIDDEN, TOTAL_IDP_DIM


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DomainEncoder(nn.Module):
    """Linear → GELU → Linear for one anatomical domain."""
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        hidden = max(input_dim * 2, ENCODER_HIDDEN_MIN)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NormativeHead(nn.Module):
    """Map pre-computed normative features → μ(age,sex), log_var(age,sex)
    per latent dimension.  Pure linear — non-linearity is in the spline basis.
    """
    def __init__(self, n_features: int, latent_dim: int):
        super().__init__()
        self.mu_layer = nn.Linear(n_features, latent_dim)
        self.logvar_layer = nn.Linear(n_features, latent_dim)
        # Initialise log_var bias ≈ 0 → var ≈ 1
        nn.init.zeros_(self.logvar_layer.weight)
        nn.init.zeros_(self.logvar_layer.bias)

    def forward(self, norm_features: torch.Tensor):
        mu = self.mu_layer(norm_features)
        log_var = self.logvar_layer(norm_features)
        return mu, log_var


class StructuralAgeHead(nn.Module):
    """Small MLP: z → scalar structural_age.
    Ranking loss (L_rank) enforces monotonicity in the age direction.
    """
    def __init__(self, latent_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers = []
        prev = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class Decoder(nn.Module):
    """z → reconstructed IDPs (for ablation L_rec only)."""
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# Full Teacher
# ---------------------------------------------------------------------------

class FullTeacher(nn.Module):
    """CMR age-conditional structural manifold teacher.

    Forward returns:
        z            (B, 12) – latent embeddings from domain encoders
        mu           (B, 12) – normative mean per latent dim
        log_var      (B, 12) – normative log-variance
        struct_age   (B,)    – predicted structural age
    """
    def __init__(self, norm_feature_dim: int, use_rec: bool = False):
        super().__init__()
        self.use_rec = use_rec

        # Domain encoders — one per domain (ordered by DOMAIN_NAMES)
        self.encoders = nn.ModuleDict()
        self._domain_input_slices = {}
        self._domain_latent_slices = {}
        latent_offset = 0
        input_offset = 0
        for dname, dinfo in IDP_DOMAINS.items():
            n_in = len(dinfo["fields"])
            n_lat = dinfo["latent_dim"]
            self.encoders[dname] = DomainEncoder(n_in, n_lat)
            self._domain_input_slices[dname] = (input_offset, input_offset + n_in)
            self._domain_latent_slices[dname] = (latent_offset, latent_offset + n_lat)
            input_offset += n_in
            latent_offset += n_lat

        # Normative head
        self.normative = NormativeHead(norm_feature_dim, TOTAL_LATENT_DIM)

        # Structural-age head
        self.age_head = StructuralAgeHead(TOTAL_LATENT_DIM, MONO_HIDDEN)

        # Optional decoder (ablation)
        if use_rec:
            self.decoder = Decoder(TOTAL_LATENT_DIM, TOTAL_IDP_DIM)

    def encode(self, idps: torch.Tensor) -> torch.Tensor:
        """27 IDPs → z ∈ ℝ¹² via domain-wise encoders."""
        parts = []
        for dname in IDP_DOMAINS:
            s, e = self._domain_input_slices[dname]
            parts.append(self.encoders[dname](idps[:, s:e]))
        return torch.cat(parts, dim=1)

    def forward(self, idps: torch.Tensor, norm_features: torch.Tensor):
        z = self.encode(idps)
        mu, log_var = self.normative(norm_features)
        struct_age = self.age_head(z)
        return z, mu, log_var, struct_age

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        assert self.use_rec, "Decoder not enabled"
        return self.decoder(z)


# ---------------------------------------------------------------------------
# Baseline B — MLP, no manifold
# ---------------------------------------------------------------------------

class BaselineBMLP(nn.Module):
    """27 IDPs + sex → predicted chronological age.
    No domain encoders, no normative head.
    """
    def __init__(self, input_dim: int = TOTAL_IDP_DIM + 1,
                 hidden_dims: list[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = BASELINE_B_HIDDEN
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x = concat(standardized_idps, sex)  → predicted_age (B,)"""
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def loss_norm(z: torch.Tensor, mu: torch.Tensor,
              log_var: torch.Tensor) -> torch.Tensor:
    """Gaussian NLL: normative head learns to describe encoder output.
    z is detached so L_norm gradients only flow to the normative head,
    preventing the encoder from collapsing z onto μ.
    log_var is clamped to [-5, 5] to bound σ²."""
    z_sg = z.detach()  # stop-gradient: encoder is NOT trained by L_norm
    log_var = log_var.clamp(-5, 5)
    var = log_var.exp()
    return 0.5 * (log_var + (z_sg - mu) ** 2 / var).mean()


def loss_age(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Huber loss for structural-age calibration."""
    return F.smooth_l1_loss(pred, target)


def loss_rank(struct_age: torch.Tensor,
              true_age: torch.Tensor) -> torch.Tensor:
    """Soft pairwise ranking: older ⇒ higher structural age."""
    n = len(true_age)
    if n < 2:
        return torch.tensor(0.0, device=struct_age.device)
    idx_i, idx_j = torch.triu_indices(n, n, offset=1)
    age_diff = true_age[idx_i] - true_age[idx_j]     # positive when i older
    struct_diff = struct_age[idx_i] - struct_age[idx_j]
    signs = torch.sign(age_diff)
    mask = signs != 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=struct_age.device)
    return -F.logsigmoid(signs[mask] * struct_diff[mask]).mean()


def loss_rec(z: torch.Tensor, model: FullTeacher,
             idps_std: torch.Tensor) -> torch.Tensor:
    """L1 reconstruction loss (ablation only)."""
    recon = model.reconstruct(z)
    return F.l1_loss(recon, idps_std)


def compute_deviation(z: torch.Tensor, mu: torch.Tensor,
                      log_var: torch.Tensor) -> torch.Tensor:
    """Per-sample deviation from manifold (scalar).
    deviation = sqrt(mean_k((z_k - μ_k)² / σ²_k))
    log_var clamped to match training constraint.
    """
    log_var = log_var.clamp(-5, 5)
    var = log_var.exp()
    return ((z - mu) ** 2 / var).mean(dim=1).sqrt()


def compute_domain_scores(z: torch.Tensor, mu: torch.Tensor,
                          log_var: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-domain deviation scores."""
    log_var = log_var.clamp(-5, 5)
    var = log_var.exp()
    scores = {}
    offset = 0
    for dname, dinfo in IDP_DOMAINS.items():
        d = dinfo["latent_dim"]
        s, e = offset, offset + d
        domain_dev = ((z[:, s:e] - mu[:, s:e]) ** 2 / var[:, s:e]).mean(dim=1).sqrt()
        scores[dname] = domain_dev
        offset = e
    return scores
