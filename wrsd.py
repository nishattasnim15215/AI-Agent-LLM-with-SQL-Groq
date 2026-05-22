"""
================================================================================
Wavenumber-Resolved Spectral Diffusion (WRSD) for 2D Turbulence
================================================================================

Calibrated, physics-aware unconditional generation of forced 2D Kolmogorov flow
with a wavenumber-resolved spectral gating bottleneck and conformal UQ.

Research framing
----------------
This codebase trains a U-Net denoising diffusion model on snapshots of forced
2D turbulence and evaluates it as an *unconditional* generative surrogate
conditioned only on a Reynolds-number regime label. Contributions:

  (1) A wavenumber-resolved gating bottleneck (WRSD) that applies a learned
      per-channel, per-radial-wavenumber multiplier to the FFT of the U-Net's
      bottleneck features. The gate is a true frequency-selective filter that
      can both amplify and attenuate spectral bands.

  (2) A clean architectural ablation (gate-only, loss-only, both) that
      decomposes the gain by component, showing they target different physics
      diagnostics (spectral cascade vs. small-scale stats) and combine.

  (3) Calibrated uncertainty quantification via one-parameter conformal
      recalibration of the per-pixel ensemble interval, with calibration and
      test splits kept disjoint.

Scope and explicit limitations
------------------------------
  * 2D Kolmogorov flow at 128 x 128 by default; 3D, higher Re, and unstructured
    geometries are flagged as future work.
  * The conformal recalibration is one-parameter scalar rescaling, not split
    conformal; no finite-sample marginal coverage guarantee is claimed.
  * This is NOT a next-step PDE surrogate. Conditional one-step predictors
    (PDE-Refiner, FNO with rollout) solve a different and easier task with the
    previous state given, and are not directly comparable.

Reproducibility and CLI
-----------------------
  python wrsd.py --smoke          # ~2 minute smoke test
  python wrsd.py                  # full run: 128^2, 5 seeds, 6 variants
  python wrsd.py --grid 64        # legacy 64^2 sweep
================================================================================
"""

import time
import math
import hashlib
import argparse
import warnings
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# Targeted warning suppression instead of a global ignore.
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda.amp")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*meshgrid.*indexing.*")


# ==============================================================================
# SECTION 1. CONFIGURATION
# ==============================================================================

class Config:
    """All experiment hyperparameters. Modify via CLI flags, not by editing."""
    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_DIR     = PROJECT_ROOT / "Dataset"
    RESULTS_DIR  = PROJECT_ROOT / "Results"
    CKPT_DIR     = PROJECT_ROOT / "Checkpoints"
    FIG_DIR      = RESULTS_DIR / "Figures"
    CSV_DIR      = RESULTS_DIR / "Tables"

    # --- DNS / dataset ---
    GRID                  = 128
    DOMAIN                = 2.0 * math.pi
    NU_LIST               = [0.005, 0.01]      # two regimes; both have inertial range at 128^2
    FORCING_K             = 4
    ALPHA_DRAG            = 0.1
    DT                    = 1e-3
    SPINUP_STEPS          = 4000
    SNAPSHOT_INTERVAL     = 200                # DNS dt's between training snapshots (decorrelation)
    SNAPSHOTS_PER_REGIME  = 1000
    USE_DEALIASED_JACOBIAN = True              # use full 3/2-rule zero-padding in the Jacobian

    # --- Diffusion schedule (EDM-style x_0-prediction parameterization) ---
    SIGMA_MIN  = 0.002
    SIGMA_MAX  = 80.0
    SIGMA_DATA = 1.0
    RHO        = 7.0
    P_MEAN     = -1.2
    P_STD      = 1.2

    # --- Training ---
    BATCH_SIZE     = 32
    EPOCHS         = 120
    LR             = 2e-4
    WEIGHT_DECAY   = 1e-5
    EMA_DECAY      = 0.9999                    # bumped from 0.999 (standard for diffusion)
    USE_AMP        = True
    NUM_WORKERS    = 2
    BASE_CHANNELS  = 48
    SEEDS          = [13, 29, 47, 71, 89]      # n=5 for bootstrap CIs

    # --- Physics-informed loss weights (applied to predicted x_0, not noise) ---
    LAMBDA_ENSTROPHY = 0.05
    LAMBDA_SPECTRAL  = 0.05

    # --- Train / val / test split fractions (data hygiene) ---
    # A FIXED test holdout is reserved before any training (seed 0). The
    # remaining train pool is then split per-seed into inner-train and
    # inner-val (the latter is only used for training-time monitoring,
    # never for reported numbers).
    TEST_FRACTION       = 0.15
    INNER_VAL_FRACTION  = 0.10

    # --- Evaluation / sampling ---
    N_EVAL_SAMPLES  = 256                      # capped at test-set size
    N_ENSEMBLE      = 32
    N_SAMPLE_STEPS  = 50
    UQ_CAL_FRACTION = 0.5                      # 50/50 calibration/test split of UQ samples

    # --- Statistics ---
    BOOTSTRAP_RESAMPLES = 2000
    BOOTSTRAP_CI        = 0.95

    @classmethod
    def setup(cls):
        for d in (cls.DATA_DIR, cls.RESULTS_DIR, cls.CKPT_DIR, cls.FIG_DIR, cls.CSV_DIR):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def dataset_hash(cls):
        """Stable hash of every config field that affects the cached DNS data.
        Changing any of these invalidates the cache automatically (fixes the
        incomplete-cache-key bug from the previous version)."""
        keys = ("GRID", "DOMAIN", "NU_LIST", "FORCING_K", "ALPHA_DRAG", "DT",
                "SPINUP_STEPS", "SNAPSHOT_INTERVAL", "SNAPSHOTS_PER_REGIME",
                "USE_DEALIASED_JACOBIAN")
        payload = repr({k: getattr(cls, k) for k in keys})
        return hashlib.sha1(payload.encode()).hexdigest()[:10]


def configure_device():
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"[Device] CUDA: {n} GPU(s).")
        for i in range(n):
            print(f"         GPU {i}: {torch.cuda.get_device_name(i)}")
        return torch.device("cuda"), n
    print("[Device] No CUDA detected; using CPU.")
    return torch.device("cpu"), 0


# ==============================================================================
# SECTION 2. PSEUDO-SPECTRAL DNS OF FORCED 2D KOLMOGOROV FLOW
# ==============================================================================

class KolmogorovDNS:
    """Pseudo-spectral RK4 solver for 2D Navier-Stokes in vorticity form with
    Kolmogorov forcing  f = -k_f sin(k_f y) e_x  and linear (Ekman) drag.

    Convention. axis 0 ~ x, axis 1 ~ y. The vorticity forcing is the curl of f,
    which is k_f^2 cos(k_f y) -- a function of y, varying along axis 1.

    Dealiasing. When USE_DEALIASED_JACOBIAN is True (default), the convective
    term u_j * d_j omega is evaluated with full 3/2-rule zero-padding (Orszag
    1971): the operands are zero-padded in Fourier space to a 3N/2 grid before
    real-space multiplication, then FFT'd back and truncated to the original
    N grid. This eliminates aliasing of high-k product modes into the resolved
    range, which the previous output-only 2/3 truncation did not.
    """

    def __init__(self, N, L, nu, kf, alpha, dt, device, dealias_jacobian=True):
        self.N, self.L, self.nu, self.kf, self.alpha, self.dt = N, L, nu, kf, alpha, dt
        self.device = device
        self.dealias_jacobian = dealias_jacobian

        # Wavenumber grid (integer for L=2*pi)
        k = torch.fft.fftfreq(N, d=L / (N * 2.0 * math.pi)).to(device)
        kx, ky = torch.meshgrid(k, k, indexing="ij")
        self.kx, self.ky = kx, ky
        self.k2 = kx ** 2 + ky ** 2

        # Inverse Laplacian: 0 at DC, 1/k^2 elsewhere. The forward k^2 stays
        # zero at DC, so dissipation -nu*k^2*w_hat correctly does not act on
        # the DC mode even if numerical noise produces nonzero amplitude
        # there. This fixes the previous "k2[0,0] = 1" hack which over-
        # dissipated the DC mode.
        self.inv_k2 = torch.where(self.k2 > 0,
                                  1.0 / self.k2.clamp(min=1e-30),
                                  torch.zeros_like(self.k2))

        # 2/3-rule output truncation mask (kept even when 3/2 padding is used)
        kmax = N // 3
        self.dealias = ((kx.abs() <= kmax) & (ky.abs() <= kmax)).float()

        # Padded grid for 3/2-rule dealiasing
        self.N_pad = (3 * N) // 2

        # Vorticity forcing = curl of f = -k_f sin(k_f y) e_x  -->  k_f^2 cos(k_f y)
        x = torch.linspace(0, L, N + 1, device=device)[:-1]
        _, Y = torch.meshgrid(x, x, indexing="ij")
        self.forcing = (kf ** 2) * torch.cos(kf * Y)
        self.forcing_hat = torch.fft.fft2(self.forcing)

    # --- 3/2-rule padded multiplication --------------------------------------
    @staticmethod
    def _pad_to(x_hat, N_pad):
        """Zero-pad a Fourier-domain tensor (..., N, N) into (..., N_pad, N_pad)
        preserving the FFT layout (positive freqs at the start, negative at
        the end). Scaling factor (N_pad/N)^2 is applied so that real-space
        IFFT on the padded grid reproduces the unpadded real-space field."""
        *batch, N, _ = x_hat.shape
        scale = (N_pad / N) ** 2
        out = x_hat.new_zeros(*batch, N_pad, N_pad)
        half = N // 2
        # Positive freqs: rows/cols [0:half+1]
        # Negative freqs: rows/cols [N - half + 1 : N] in source, [N_pad - half + 1 : N_pad] in dest
        # Standard FFT layout: indices 0..N/2 are non-negative, N/2+1..N-1 are negative
        # We pad by inserting zeros in the middle.
        out[..., :half + 1, :half + 1] = x_hat[..., :half + 1, :half + 1]
        out[..., :half + 1, N_pad - half + 1:] = x_hat[..., :half + 1, half + 1:]
        out[..., N_pad - half + 1:, :half + 1] = x_hat[..., half + 1:, :half + 1]
        out[..., N_pad - half + 1:, N_pad - half + 1:] = x_hat[..., half + 1:, half + 1:]
        return out * scale

    @staticmethod
    def _truncate_to(x_hat_pad, N):
        """Inverse of _pad_to: take an (N_pad, N_pad) Fourier tensor and
        return an (N, N) tensor by keeping only the non-aliased modes."""
        *batch, N_pad, _ = x_hat_pad.shape
        scale = (N / N_pad) ** 2
        out = x_hat_pad.new_zeros(*batch, N, N)
        half = N // 2
        out[..., :half + 1, :half + 1] = x_hat_pad[..., :half + 1, :half + 1]
        out[..., :half + 1, half + 1:] = x_hat_pad[..., :half + 1, N_pad - half + 1:]
        out[..., half + 1:, :half + 1] = x_hat_pad[..., N_pad - half + 1:, :half + 1]
        out[..., half + 1:, half + 1:] = x_hat_pad[..., N_pad - half + 1:, N_pad - half + 1:]
        return out * scale

    def _product_dealiased(self, a_hat, b_hat):
        """(a * b) computed with 3/2-rule zero padding, returned in unpadded
        Fourier space. Removes aliasing introduced by point-wise multiplication."""
        a_pad = self._pad_to(a_hat, self.N_pad)
        b_pad = self._pad_to(b_hat, self.N_pad)
        a = torch.fft.ifft2(a_pad).real
        b = torch.fft.ifft2(b_pad).real
        return self._truncate_to(torch.fft.fft2(a * b), self.N)

    # --- Right-hand side of d omega_hat / dt ---------------------------------
    def _jacobian_hat(self, w_hat):
        psi_hat = -w_hat * self.inv_k2
        u_hat   = 1j * self.ky * psi_hat
        v_hat   = -1j * self.kx * psi_hat
        wx_hat  = 1j * self.kx * w_hat
        wy_hat  = 1j * self.ky * w_hat
        if self.dealias_jacobian:
            uwx = self._product_dealiased(u_hat, wx_hat)
            vwy = self._product_dealiased(v_hat, wy_hat)
            return (uwx + vwy) * self.dealias
        # Legacy: output-only 2/3 truncation (kept for ablation reproducibility)
        u  = torch.fft.ifft2(u_hat ).real
        v  = torch.fft.ifft2(v_hat ).real
        wx = torch.fft.ifft2(wx_hat).real
        wy = torch.fft.ifft2(wy_hat).real
        return torch.fft.fft2(u * wx + v * wy) * self.dealias

    def _rhs(self, w_hat):
        return (-self._jacobian_hat(w_hat)
                - self.nu * self.k2 * w_hat
                - self.alpha * w_hat
                + self.forcing_hat)

    def step_rk4(self, w_hat):
        k1 = self._rhs(w_hat)
        k2 = self._rhs(w_hat + 0.5 * self.dt * k1)
        k3 = self._rhs(w_hat + 0.5 * self.dt * k2)
        k4 = self._rhs(w_hat + self.dt * k3)
        return w_hat + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _init_field(self, seed):
        g = torch.Generator(device=self.device).manual_seed(seed)
        w = 0.5 * torch.randn(self.N, self.N, generator=g, device=self.device)
        w_hat = torch.fft.fft2(w) * self.dealias
        w_hat[0, 0] = 0.0
        return w_hat

    def simulate(self, n_snapshots, snapshot_interval, spinup_steps, seed):
        w_hat = self._init_field(seed)
        for _ in range(spinup_steps):
            w_hat = self.step_rk4(w_hat)
        snaps = torch.zeros(n_snapshots, self.N, self.N, device=self.device)
        for i in range(n_snapshots):
            for _ in range(snapshot_interval):
                w_hat = self.step_rk4(w_hat)
            snaps[i] = torch.fft.ifft2(w_hat).real
        return snaps

    # --- Reynolds-number diagnostics -----------------------------------------
    def diagnostics(self, omega):
        """Integral-scale diagnostics from a (T, H, W) vorticity tensor.

        Taylor microscale in 2D. We use the standard 2D definition
            lambda = sqrt( 2 * <u^2> / <omega^2> )
        which is equivalent to lambda^2 = <u^2>/<(d_x u)^2> under 2D
        isotropy and incompressibility. The previous version used the 3D
        factor (5) and over-reported Re_lambda by sqrt(5/2) ~ 1.58.
        """
        if omega.ndim == 2:
            omega = omega.unsqueeze(0)
        w_hat   = torch.fft.fft2(omega)
        psi_hat = -w_hat * self.inv_k2
        u =  torch.fft.ifft2( 1j * self.ky * psi_hat).real
        v =  torch.fft.ifft2(-1j * self.kx * psi_hat).real
        u_rms     = torch.sqrt((u ** 2 + v ** 2).mean()).item()
        enstrophy = (omega ** 2).mean().item()
        # 2D Taylor microscale (FIX: 2 not 5)
        lam   = math.sqrt(2.0 * u_rms ** 2 / max(enstrophy, 1e-12))
        Ek, kvals = compute_radial_spectrum(omega)
        k = kvals.float().clamp(min=1.0)
        Ek_mean = Ek.mean(0)
        L_int = math.pi * (Ek_mean / k).sum().item() / max(Ek_mean.sum().item(), 1e-12)
        return {
            "u_rms":     u_rms,
            "enstrophy": enstrophy,
            "L_int":     L_int,
            "lambda":    lam,
            "Re_int":    u_rms * L_int / self.nu,
            "Re_lam":    u_rms * lam   / self.nu,
        }


# ==============================================================================
# SECTION 3. SPECTRAL DIAGNOSTICS
# ==============================================================================
#
# Conventions used throughout.
# --------------------------
# - "Energy spectrum" E(k) is the radially-binned sum of |omega_hat(k)|^2 (the
#   vorticity power spectrum), which is proportional to the enstrophy spectrum.
#   We use the term "energy spectrum" in keeping with the radial-spectrum
#   convention common in the diffusion-surrogate literature. The Kraichnan
#   k^-3 reference in DNS validation applies to this quantity.
# - True ENERGY FLUX Pi_E(K) and ENSTROPHY FLUX Pi_Z(K) are kept as separate
#   metrics with explicit names (compute_energy_flux, compute_enstrophy_flux),
#   following Boffetta and Ecke (Annu. Rev. Fluid Mech. 2012) sign convention:
#     Pi_E(K) := - sum_{|k|<=K} T_E(k),  with Pi_E(K < k_f) < 0 = inverse energy cascade
#     Pi_Z(K) := - sum_{|k|<=K} T_Z(k),  with Pi_Z(K > k_f) > 0 = forward enstrophy cascade
# - "Structure functions" S_p(r) in this codebase are vorticity structure
#   functions, computed on the field the diffusion model generates. These are
#   distinct from the classical velocity structure functions of the Kolmogorov
#   theory. Metric names carry the prefix `vorticity_` to make this explicit.

def compute_radial_spectrum(field):
    """Radially-binned vorticity power spectrum, orthonormal FFT.  Vectorized
    scatter_add over the batch dimension."""
    if field.ndim == 2:
        field = field.unsqueeze(0)
    B, H, W = field.shape
    Fh = torch.fft.fft2(field, norm="ortho")
    E2 = (Fh.abs() ** 2)                                          # (B, H, W)
    ky = torch.fft.fftfreq(H, d=1.0 / H).to(field.device)
    kx = torch.fft.fftfreq(W, d=1.0 / W).to(field.device)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    K = torch.sqrt(KX ** 2 + KY ** 2)
    kmax = int(K.max().item()) + 1
    bins = torch.arange(0, kmax + 1, device=field.device)
    K_round = K.round().long().clamp(0, kmax - 1).flatten()
    K_idx = K_round.unsqueeze(0).expand(B, -1)
    Ek = torch.zeros(B, kmax, device=field.device).scatter_add_(1, K_idx, E2.view(B, -1))
    return Ek, bins[:kmax]


def compute_dealiased_spectrum(field, kmax_frac=0.66):
    Ek, kvals = compute_radial_spectrum(field)
    k_cut = int(kmax_frac * kvals.max().item())
    mask = kvals <= k_cut
    return Ek[:, mask], kvals[mask]


def compute_vorticity_structure_function(field, order, r_max=None):
    """Structure function of a scalar field (the vorticity output of the model).
    Averages over shifts along the two grid axes (a discrete isotropy proxy).
    The classical velocity structure function is a different quantity."""
    if field.ndim == 4:
        field = field.squeeze(1)
    B, H, W = field.shape
    if r_max is None:
        r_max = H // 3
    rs = list(range(1, r_max + 1))
    S = torch.zeros(B, len(rs), device=field.device)
    for i, r in enumerate(rs):
        d1 = (field - torch.roll(field, shifts=r, dims=-1)).abs() ** order
        d2 = (field - torch.roll(field, shifts=r, dims=-2)).abs() ** order
        S[:, i] = 0.5 * (d1.mean(dim=(-1, -2)) + d2.mean(dim=(-1, -2)))
    return S, torch.tensor(rs, device=field.device, dtype=torch.float32)


def compute_integral_length(field):
    if field.ndim == 4:
        field = field.squeeze(1)
    Ek, kvals = compute_radial_spectrum(field)
    k = kvals.float().clamp(min=1.0)
    return math.pi * (Ek / k).sum(dim=1) / Ek.sum(dim=1).clamp(min=1e-12)


# --- Energy and enstrophy flux (separate quantities, separate functions) ----

def _make_k_grid(H, W, device):
    kx = torch.fft.fftfreq(H, d=1.0 / H).to(device)
    ky = torch.fft.fftfreq(W, d=1.0 / W).to(device)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    K  = torch.sqrt(KX ** 2 + KY ** 2)
    K2 = (KX ** 2 + KY ** 2).clamp(min=1e-12)
    return KX, KY, K, K2


def compute_energy_flux(field):
    """True spectral energy flux Pi_E(K) for 2D incompressible NS in vorticity
    form. Sign convention: Pi_E(K) > 0 means net energy flux from {|k|<=K} to
    higher wavenumbers (forward); Pi_E(K) < 0 = inverse energy cascade.

    Implementation. Recover velocity from vorticity via streamfunction, evaluate
    the velocity-nonlinear term (u . grad) u in real space, lift to the energy
    transfer T_E(k) = -Re<u_hat^*(k) . ((u.grad)u)_hat(k)> summed over
    components, and define Pi_E(K) = -sum_{|k|<=K} T_E(k)."""
    if field.ndim == 4:
        field = field.squeeze(1)
    B, H, W = field.shape
    KX, KY, K, K2 = _make_k_grid(H, W, field.device)

    w_hat   = torch.fft.fft2(field, norm="ortho")
    psi_hat = -w_hat / K2.unsqueeze(0)
    u_hat   =  1j * KY * psi_hat
    v_hat   = -1j * KX * psi_hat
    u  = torch.fft.ifft2(u_hat , norm="ortho").real
    v  = torch.fft.ifft2(v_hat , norm="ortho").real
    ux = torch.fft.ifft2( 1j * KX * u_hat, norm="ortho").real
    uy = torch.fft.ifft2( 1j * KY * u_hat, norm="ortho").real
    vx = torch.fft.ifft2( 1j * KX * v_hat, norm="ortho").real
    vy = torch.fft.ifft2( 1j * KY * v_hat, norm="ortho").real
    adv_u_hat = torch.fft.fft2(u * ux + v * uy, norm="ortho")
    adv_v_hat = torch.fft.fft2(u * vx + v * vy, norm="ortho")

    # Per-mode energy transfer: T_E(k) = -Re[u*.adv_u_hat + v*.adv_v_hat]
    T_E = -(u_hat.conj() * adv_u_hat + v_hat.conj() * adv_v_hat).real   # (B, H, W)

    kmax = int(K.max().item()) + 1
    K_round = K.round().long().clamp(0, kmax - 1).flatten()
    T_shell = torch.zeros(B, kmax, device=field.device).scatter_add_(
        1, K_round.unsqueeze(0).expand(B, -1), T_E.view(B, -1)
    )
    Pi_E = -T_shell.cumsum(dim=1)
    return Pi_E, torch.arange(kmax, device=field.device, dtype=torch.float32)


def compute_enstrophy_flux(field):
    """Spectral enstrophy flux Pi_Z(K) for 2D NS, vorticity form. Sign
    convention: Pi_Z(K) > 0 = forward enstrophy cascade out of {|k|<=K}."""
    if field.ndim == 4:
        field = field.squeeze(1)
    B, H, W = field.shape
    KX, KY, K, K2 = _make_k_grid(H, W, field.device)
    w_hat = torch.fft.fft2(field, norm="ortho")
    psi_hat = -w_hat / K2.unsqueeze(0)
    u = torch.fft.ifft2( 1j * KY * psi_hat, norm="ortho").real
    v = torch.fft.ifft2(-1j * KX * psi_hat, norm="ortho").real
    nl = (u * torch.fft.ifft2(1j * KX * w_hat, norm="ortho").real +
          v * torch.fft.ifft2(1j * KY * w_hat, norm="ortho").real)
    nl_hat = torch.fft.fft2(nl, norm="ortho")
    # T_Z(k) = -Re[w*.nl_hat]
    T_Z = -(w_hat.conj() * nl_hat).real
    kmax = int(K.max().item()) + 1
    K_round = K.round().long().clamp(0, kmax - 1).flatten()
    T_shell = torch.zeros(B, kmax, device=field.device).scatter_add_(
        1, K_round.unsqueeze(0).expand(B, -1), T_Z.view(B, -1)
    )
    Pi_Z = -T_shell.cumsum(dim=1)
    return Pi_Z, torch.arange(kmax, device=field.device, dtype=torch.float32)


def fit_inertial_slope(Ek_mean, k_lo, k_hi):
    """Least-squares fit of log10(E) vs log10(k) over [k_lo, k_hi].
    Returns (slope, intercept, R^2)."""
    k = torch.arange(Ek_mean.shape[0]).float()
    mask = (k >= k_lo) & (k <= k_hi) & (Ek_mean > 1e-12)
    if mask.sum() < 4:
        return float("nan"), float("nan"), float("nan")
    logk = torch.log10(k[mask] + 1e-8)
    logE = torch.log10(Ek_mean[mask] + 1e-12)
    A = torch.stack([logk, torch.ones_like(logk)], dim=1)
    sol, *_ = torch.linalg.lstsq(A, logE.unsqueeze(1))
    slope, intercept = sol[0, 0].item(), sol[1, 0].item()
    pred = slope * logk + intercept
    ss_res = ((logE - pred) ** 2).sum().item()
    ss_tot = ((logE - logE.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return slope, intercept, r2


# ==============================================================================
# SECTION 4. DATASET GENERATION + DNS VALIDATION
# ==============================================================================

def generate_dataset(cfg, device):
    """Generate (or load from cache) the DNS dataset. Cache key includes every
    parameter that affects the data (fix: previously only GRID + NU_LIST)."""
    cache = cfg.DATA_DIR / f"kolmogorov_{cfg.dataset_hash()}.pt"
    if cache.exists():
        print(f"[Data] Cached dataset found: {cache.name}")
        return torch.load(cache, map_location="cpu")

    print(f"[Data] Generating Kolmogorov flow at N={cfg.GRID}, {len(cfg.NU_LIST)} regimes.")
    print(f"[Data] Dataset hash: {cfg.dataset_hash()}")
    all_fields, all_re, diagnostics = [], [], []
    for r_idx, nu in enumerate(cfg.NU_LIST):
        print(f"[Data]   Regime {r_idx+1}/{len(cfg.NU_LIST)}  nu={nu}")
        dns = KolmogorovDNS(cfg.GRID, cfg.DOMAIN, nu, cfg.FORCING_K, cfg.ALPHA_DRAG,
                            cfg.DT, device, dealias_jacobian=cfg.USE_DEALIASED_JACOBIAN)
        snaps = dns.simulate(cfg.SNAPSHOTS_PER_REGIME, cfg.SNAPSHOT_INTERVAL,
                             cfg.SPINUP_STEPS, seed=1000 + r_idx)
        diag = dns.diagnostics(snaps[-256:])
        diag["regime"] = r_idx; diag["nu"] = nu
        diagnostics.append(diag)
        all_fields.append(snaps.cpu())
        all_re.append(torch.full((cfg.SNAPSHOTS_PER_REGIME,), float(r_idx)))
        print(f"[Data]    u_rms={diag['u_rms']:.3f}  L_int={diag['L_int']:.3f}  "
              f"lambda={diag['lambda']:.3f}  Re_int={diag['Re_int']:.1f}  "
              f"Re_lam={diag['Re_lam']:.1f}")

    fields = torch.cat(all_fields, dim=0)
    re_labels = torch.cat(all_re, dim=0).long()
    mean, std = fields.mean(), fields.std()
    fields_norm = (fields - mean) / (std + 1e-8)
    payload = {
        "fields":      fields_norm.unsqueeze(1).float(),
        "re_labels":   re_labels,
        "raw_mean":    mean.item(),
        "raw_std":     std.item(),
        "nu_list":     cfg.NU_LIST,
        "grid":        cfg.GRID,
        "diagnostics": diagnostics,
        "hash":        cfg.dataset_hash(),
    }
    torch.save(payload, cache)
    print(f"[Data] Saved -> {cache.name}")
    return payload


def validate_dns(cfg, payload, device):
    """Energy spectra with per-regime slope fits AND per-regime k^-3 reference
    lines (fix: previous version anchored the reference to a single regime)."""
    print("[DNS]  Validating energy spectra and fitting inertial-range slopes.")
    fields = payload["fields"].to(device) * payload["raw_std"] + payload["raw_mean"]
    re_labels = payload["re_labels"]
    rows = []
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    cmap = plt.get_cmap("viridis")
    for r in range(len(cfg.NU_LIST)):
        sel = (re_labels == r)
        Ek, kvals = compute_dealiased_spectrum(fields[sel].squeeze(1)[:256])
        Ek_mean_t = Ek.mean(0)
        Ek_mean = Ek_mean_t.cpu().numpy()
        k = kvals.cpu().numpy()
        # Inertial range: from k_f+2 to ~k_max/4
        k_lo = cfg.FORCING_K + 2
        k_hi = max(k_lo + 3, int(kvals.max().item()) // 4)
        slope, intercept, r2 = fit_inertial_slope(Ek_mean_t, k_lo, k_hi)
        diag = payload["diagnostics"][r]
        rows.append({"regime": r, "nu": cfg.NU_LIST[r],
                     "Re_int": diag["Re_int"], "Re_lam": diag["Re_lam"],
                     "L_int": diag["L_int"], "lambda": diag["lambda"],
                     "fit_k_lo": int(k_lo), "fit_k_hi": int(k_hi),
                     "fit_slope": slope, "fit_R2": r2})
        nz = (k > 0) & (Ek_mean > 0)
        color = cmap(0.2 + 0.6 * r / max(1, len(cfg.NU_LIST) - 1))
        ax.loglog(k[nz], Ek_mean[nz], color=color, lw=1.8,
                  label=(f"nu={cfg.NU_LIST[r]}  Re_int={diag['Re_int']:.0f}  "
                         f"slope={slope:.2f}  (R^2={r2:.2f})"))
        # Empirical fit line in the fit window
        kk = np.array([k_lo, k_hi], dtype=float)
        ax.loglog(kk, 10 ** (slope * np.log10(kk) + intercept),
                  color=color, ls=":", lw=1.2)
        # PER-REGIME k^-3 reference (fix: previously anchored to one regime only)
        if k_lo < len(Ek_mean) and Ek_mean[k_lo] > 0:
            amp_ref = Ek_mean[k_lo] * (k_lo ** 3)
            k_ref = np.linspace(k_lo, max(k_lo + 4, kvals.max().item() // 2), 30)
            ax.loglog(k_ref, amp_ref * k_ref ** (-3.0), color=color, ls="--",
                      lw=0.8, alpha=0.55,
                      label=("Kraichnan $k^{-3}$ (per regime)" if r == 0 else None))
    ax.axvline(cfg.FORCING_K, color="orange", ls=":", lw=1.0, label=f"$k_f={cfg.FORCING_K}$")
    ax.set_xlabel("k"); ax.set_ylabel("E(k)")
    ax.set_title(f"DNS Energy Spectra and Inertial-Range Slope Fits  (N={cfg.GRID})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(cfg.FIG_DIR / "dns_validation.png", dpi=140); plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(cfg.CSV_DIR / "dns_validation.csv", index=False)
    print("[DNS]  Slope fits:\n" + df.to_string(index=False))


class TurbulenceDataset(Dataset):
    def __init__(self, fields, re_labels):
        self.fields = fields
        self.re_labels = re_labels
    def __len__(self):
        return self.fields.shape[0]
    def __getitem__(self, i):
        return self.fields[i], self.re_labels[i]


def prepare_splits(full_ds, cfg):
    """Fix for train/test contamination. Holds out a FIXED test set using
    seed 0 BEFORE any training begins. Returns (train_pool_indices, test_indices),
    both deterministic across runs. Per-seed inner train/val splits are then
    done inside train_one_model on the train pool only -- the test set is
    NEVER touched during training, by anything, for any seed."""
    N = len(full_ds)
    n_test = max(1, int(round(cfg.TEST_FRACTION * N)))
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g).tolist()
    test_indices  = perm[:n_test]
    train_indices = perm[n_test:]
    return train_indices, test_indices


# ==============================================================================
# SECTION 5. EDM-STYLE DIFFUSION SCHEDULE (x_0-prediction parameterization)
# ==============================================================================
#
# Following Karras et al. 2022 ("Elucidating the Design Space of Diffusion-
# Based Generative Models"), we use the EDM preconditioning that wraps the
# network output f into  D(x) = c_skip*x + c_out*f, which IS the predicted
# clean sample x_0 (not v). Training loss is MSE on (D(x) - x_0) weighted by
# the EDM (sigma^2 + sigma_data^2)/(sigma*sigma_data)^2 schedule. This is
# x_0-prediction with EDM weighting -- NOT v-prediction (which would target
# v = sigma*epsilon - sigma_data*x).

class EDMSchedule:
    def __init__(self, sigma_min, sigma_max, sigma_data, rho, p_mean, p_std, device):
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.sigma_data, self.rho      = sigma_data, rho
        self.p_mean, self.p_std        = p_mean, p_std
        self.device = device

    def sample_sigma_train(self, B):
        eps = torch.randn(B, device=self.device)
        return (eps * self.p_std + self.p_mean).exp().clamp(self.sigma_min, self.sigma_max)

    def build_inference_schedule(self, n_steps):
        i = torch.arange(n_steps, device=self.device, dtype=torch.float64)
        t = (self.sigma_max ** (1 / self.rho) +
             i / (n_steps - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        return torch.cat([t, torch.zeros(1, device=self.device, dtype=torch.float64)]).float()

    def preconditioning(self, sigma):
        sd = self.sigma_data
        c_skip  = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out   = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in    = 1.0 / (sigma ** 2 + sd ** 2).sqrt()
        c_noise = 0.25 * sigma.log()
        return c_skip, c_out, c_in, c_noise

    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2


# ==============================================================================
# SECTION 6. COMMON MODEL COMPONENTS
# ==============================================================================

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ConditionEmbedding(nn.Module):
    def __init__(self, dim, n_re):
        super().__init__()
        self.t_embed  = SinusoidalEmbedding(dim)
        self.re_embed = nn.Embedding(n_re, dim)
        self.mlp = nn.Sequential(nn.Linear(dim * 2, dim * 4), nn.SiLU(),
                                 nn.Linear(dim * 4, dim))
    def forward(self, c_noise, re):
        return self.mlp(torch.cat([self.t_embed(c_noise), self.re_embed(re)], dim=1))


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class DownsampleBlock(nn.Module):
    def __init__(self, ch):
        super().__init__(); self.op = nn.Conv2d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x): return self.op(x)


class UpsampleBlock(nn.Module):
    def __init__(self, ch):
        super().__init__(); self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x): return self.op(x)


# ==============================================================================
# SECTION 7. BOTTLENECK VARIANTS
# ==============================================================================

class WavenumberResolvedGate(nn.Module):
    """Per-channel, per-radial-wavenumber gating in Fourier space. Identity
    at initialization (scale = 0); the (2*sigmoid - 1) mapping allows the
    gate to both amplify and attenuate spectral components."""
    def __init__(self, channels, emb_dim, n_radial_bins=16, rank=2):
        super().__init__()
        self.C        = channels
        self.n_bins   = n_radial_bins
        self.rank     = rank
        out_dim = rank * channels + rank * n_radial_bins + channels
        self.gate_mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim), nn.SiLU(),
            nn.Linear(2 * emb_dim, out_dim),
        )
        self.scale = nn.Parameter(torch.zeros(1))

    def _bin_indices(self, H, W, device):
        kx = torch.fft.fftfreq(W, d=1.0 / W).to(device)
        ky = torch.fft.fftfreq(H, d=1.0 / H).to(device)
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        K = torch.sqrt(KX ** 2 + KY ** 2)
        return (K / (K.max() + 1e-8) * (self.n_bins - 1)).round().long().clamp(0, self.n_bins - 1)

    def forward(self, x, emb):
        B, C, H, W = x.shape
        out = self.gate_mlp(emb)
        a = out[:, : self.rank * C].view(B, self.rank, C)
        b = out[:, self.rank * C : self.rank * (C + self.n_bins)].view(B, self.rank, self.n_bins)
        d = out[:, self.rank * (C + self.n_bins) :]
        gate_ck = torch.einsum("brc,brk->bck", a, b) + d.unsqueeze(-1)
        gate_ck = torch.sigmoid(gate_ck)
        gate_map = gate_ck[:, :, self._bin_indices(H, W, x.device)]
        x_hat = torch.fft.fft2(x, norm="ortho")
        x_hat = x_hat * (1.0 + self.scale * (2.0 * gate_map - 1.0))
        return torch.fft.ifft2(x_hat, norm="ortho").real


class SqueezeExciteGate(nn.Module):
    """Hu et al. (2018) Squeeze-and-Excite as a matched-parameter classical
    control. The conditioning embedding is appended to the squeezed descriptor."""
    def __init__(self, channels, emb_dim):
        super().__init__()
        hidden = max(channels // 2, 8)
        self.gate_mlp = nn.Sequential(
            nn.Linear(channels + emb_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )
    def forward(self, x, emb):
        z = x.mean(dim=(-1, -2))
        g = self.gate_mlp(torch.cat([z, emb], dim=1))
        return x * g.unsqueeze(-1).unsqueeze(-1)


class FNOBlock(nn.Module):
    """Single Fourier Neural Operator block on the first `modes` modes per axis."""
    def __init__(self, channels, modes=8):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (channels * channels)
        self.weight = nn.Parameter(scale * torch.randn(channels, channels, modes, modes, 2))
        self.conv = nn.Conv2d(channels, channels, 1)
    def _cmul(self, x_ft, w):
        w_c = torch.complex(w[..., 0], w[..., 1])
        return torch.einsum("bixy,ioxy->boxy", x_ft, w_c)
    def forward(self, x, emb=None):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, C, H, W // 2 + 1, device=x.device, dtype=torch.complex64)
        m = min(self.modes, H // 2, W // 2 + 1)
        out_ft[:, :,  :m, :m] = self._cmul(x_ft[:, :,  :m, :m], self.weight[:, :, :m, :m])
        out_ft[:, :, -m:, :m] = self._cmul(x_ft[:, :, -m:, :m], self.weight[:, :, :m, :m])
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho") + self.conv(x)


# ==============================================================================
# SECTION 8. UNIFIED U-NET DENOISER
# ==============================================================================

class TurbulenceDenoiser(nn.Module):
    _BOTTLENECK_OF = {
        "vanilla":        "none",
        "se":             "se",
        "fno":            "fno",
        "wrsd":           "wrsd",
        "wrsd_arch_only": "wrsd",
        "wrsd_loss_only": "none",
    }
    VARIANTS = tuple(_BOTTLENECK_OF.keys())

    def __init__(self, variant, base_ch=48, n_re=2, emb_dim=128, fno_modes=8):
        super().__init__()
        assert variant in self.VARIANTS, f"unknown variant {variant!r}"
        self.variant = variant
        self._arch = self._BOTTLENECK_OF[variant]
        self.emb = ConditionEmbedding(emb_dim, n_re)
        self.in_conv = nn.Conv2d(1, base_ch, 3, padding=1)
        self.b1 = ResidualBlock(base_ch,     base_ch,     emb_dim)
        self.d1 = DownsampleBlock(base_ch)
        self.b2 = ResidualBlock(base_ch,     base_ch * 2, emb_dim)
        self.d2 = DownsampleBlock(base_ch * 2)
        self.b3 = ResidualBlock(base_ch * 2, base_ch * 4, emb_dim)
        bch = base_ch * 4
        if   self._arch == "none": self.bottleneck = nn.Identity()
        elif self._arch == "se":   self.bottleneck = SqueezeExciteGate(bch, emb_dim)
        elif self._arch == "fno":  self.bottleneck = FNOBlock(bch, modes=fno_modes)
        elif self._arch == "wrsd": self.bottleneck = WavenumberResolvedGate(bch, emb_dim)
        self.u2 = UpsampleBlock(base_ch * 4)
        self.b4 = ResidualBlock(base_ch * 4 + base_ch * 2, base_ch * 2, emb_dim)
        self.u1 = UpsampleBlock(base_ch * 2)
        self.b5 = ResidualBlock(base_ch * 2 + base_ch,     base_ch,     emb_dim)
        self.out_conv = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, x, c_noise, re):
        emb = self.emb(c_noise, re)
        h1 = self.b1(self.in_conv(x), emb)
        h2 = self.b2(self.d1(h1), emb)
        h3 = self.b3(self.d2(h2), emb)
        if self._arch in ("se", "wrsd"):
            h3 = h3 + self.bottleneck(h3, emb)
        elif self._arch == "fno":
            h3 = h3 + self.bottleneck(h3)
        u2 = self.b4(torch.cat([self.u2(h3), h2], dim=1), emb)
        u1 = self.b5(torch.cat([self.u1(u2), h1], dim=1), emb)
        return self.out_conv(u1)


def denoise_preconditioned(model, x, sigma, re, sched):
    c_skip, c_out, c_in, c_noise = sched.preconditioning(sigma)
    c_skip_b = c_skip.view(-1, 1, 1, 1)
    c_out_b  = c_out .view(-1, 1, 1, 1)
    c_in_b   = c_in  .view(-1, 1, 1, 1)
    f = model(c_in_b * x, c_noise, re)
    return c_skip_b * x + c_out_b * f


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==============================================================================
# SECTION 9. PHYSICS-INFORMED LOSSES (computed on predicted x_0)
# ==============================================================================

def loss_enstrophy(x0_pred, x0_true):
    return F.l1_loss(x0_pred.pow(2).mean(dim=(-1, -2)),
                     x0_true.pow(2).mean(dim=(-1, -2)))


def loss_spectral(x0_pred, x0_true):
    Fp = torch.fft.fft2(x0_pred, norm="ortho").abs()
    Ft = torch.fft.fft2(x0_true, norm="ortho").abs()
    return F.l1_loss(Fp, Ft)


# ==============================================================================
# SECTION 10. EMA AND CHECKPOINTING
# ==============================================================================

class EMA:
    """Exponential-moving-average shadow weights with explicit warnings on
    missing/unexpected keys at load time (fix for silent corruption when the
    architecture changes between save and load)."""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
    def state_dict(self):
        return {n: v.cpu() for n, v in self.shadow.items()}
    def load_state_dict(self, state):
        missing    = [n for n in self.shadow if n not in state]
        unexpected = [n for n in state    if n not in self.shadow]
        if missing:
            print(f"[EMA]  Warning: {len(missing)} shadow params missing from checkpoint "
                  f"(first 3: {missing[:3]}). Initializing from current model weights.")
        if unexpected:
            print(f"[EMA]  Warning: {len(unexpected)} checkpoint keys not in shadow "
                  f"(first 3: {unexpected[:3]}). Ignoring.")
        for n in self.shadow:
            if n in state:
                self.shadow[n] = state[n].to(self.shadow[n].device)
    @torch.no_grad()
    def copy_to(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].to(p.device))


def ckpt_path(cfg, variant, seed):
    return cfg.CKPT_DIR / f"{variant}_seed{seed}.pt"


def save_ckpt(path, model, optim, ema, epoch, history, compute):
    target = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({"model": target.state_dict(), "optim": optim.state_dict(),
                "ema": ema.state_dict(), "epoch": epoch, "history": history,
                "compute": compute}, path)


def load_ckpt(path, model, optim, ema):
    if not path.exists():
        return 0, [], {}
    state = torch.load(path, map_location="cpu")
    target = model.module if isinstance(model, nn.DataParallel) else model
    target.load_state_dict(state["model"])
    optim.load_state_dict(state["optim"])
    if "ema" in state:
        ema.load_state_dict(state["ema"])
    return state["epoch"], state.get("history", []), state.get("compute", {})


# ==============================================================================
# SECTION 11. TRAINING LOOP
# ==============================================================================

def train_one_model(variant, seed, cfg, train_pool, device, n_gpus, use_physics):
    """Train one (variant, seed). The argument `train_pool` is the held-out-test-
    excluded portion of the dataset. We split train_pool into inner-train and
    inner-val with a per-seed generator; inner-val is used ONLY for training-time
    monitoring (val_mse logging), never for reported metrics."""
    print(f"\n[Train] === variant={variant.upper()}  seed={seed}  "
          f"phys={'on' if use_physics else 'off'} ===")
    torch.manual_seed(seed); np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    N = len(train_pool)
    n_val = max(1, int(round(cfg.INNER_VAL_FRACTION * N)))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g).tolist()
    inner_val_idx = perm[:n_val]
    inner_tr_idx  = perm[n_val:]
    train_ds = Subset(train_pool, inner_tr_idx)
    val_ds   = Subset(train_pool, inner_val_idx)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False,
                              num_workers=cfg.NUM_WORKERS, pin_memory=(device.type == "cuda"))

    model = TurbulenceDenoiser(variant, base_ch=cfg.BASE_CHANNELS,
                               n_re=len(cfg.NU_LIST)).to(device)
    n_params = count_parameters(model)
    print(f"[Train] Parameters: {n_params/1e6:.3f} M")
    ema = EMA(model, decay=cfg.EMA_DECAY)
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"[Train] DataParallel over {n_gpus} GPUs.")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    warmup = max(1, cfg.EPOCHS // 20)
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        prog = (epoch - warmup) / max(1, cfg.EPOCHS - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    diff_sched = EDMSchedule(cfg.SIGMA_MIN, cfg.SIGMA_MAX, cfg.SIGMA_DATA, cfg.RHO,
                             cfg.P_MEAN, cfg.P_STD, device)
    scaler = GradScaler(enabled=cfg.USE_AMP and device.type == "cuda")

    cpath = ckpt_path(cfg, variant, seed)
    start_epoch, history, prior_compute = load_ckpt(cpath, model, optim, ema)
    if start_epoch > 0:
        for _ in range(start_epoch):
            lr_sched.step()
        print(f"[Train] Resumed from epoch {start_epoch}.")

    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, cfg.EPOCHS):
        model.train()
        ep_loss = ep_mse = ep_phys = 0.0; nb = 0
        for x, re in train_loader:
            x  = x .to(device, non_blocking=True)
            re = re.to(device, non_blocking=True)
            B = x.shape[0]
            sigma = diff_sched.sample_sigma_train(B)
            noise = torch.randn_like(x) * sigma.view(-1, 1, 1, 1)
            x_noisy = x + noise
            optim.zero_grad(set_to_none=True)
            ctx = autocast() if (cfg.USE_AMP and device.type == "cuda") else nullcontext()
            with ctx:
                x0_pred = denoise_preconditioned(model, x_noisy, sigma, re, diff_sched)
                w = diff_sched.loss_weight(sigma).view(-1, 1, 1, 1)
                mse = (w * (x0_pred - x) ** 2).mean()
                if use_physics:
                    L_ens  = loss_enstrophy(x0_pred, x)
                    L_spec = loss_spectral (x0_pred, x)
                    loss = mse + cfg.LAMBDA_ENSTROPHY * L_ens + cfg.LAMBDA_SPECTRAL * L_spec
                    phys = (L_ens + L_spec).item()
                else:
                    loss = mse; phys = 0.0
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update()
            ema.update(model.module if isinstance(model, nn.DataParallel) else model)
            ep_loss += loss.item(); ep_mse += mse.item(); ep_phys += phys; nb += 1

        model.eval()
        v_mse, v_n = 0.0, 0
        with torch.no_grad():
            for x, re in val_loader:
                x, re = x.to(device), re.to(device); B = x.shape[0]
                sigma = diff_sched.sample_sigma_train(B)
                noise = torch.randn_like(x) * sigma.view(-1, 1, 1, 1)
                x0_pred = denoise_preconditioned(model, x + noise, sigma, re, diff_sched)
                v_mse += ((x0_pred - x) ** 2).mean().item(); v_n += 1

        rec = {"epoch": epoch + 1, "train_loss": ep_loss / nb,
               "train_mse": ep_mse / nb, "train_phys": ep_phys / nb,
               "val_mse":   v_mse / max(v_n, 1), "lr": optim.param_groups[0]["lr"]}
        history.append(rec)
        print(f"[Train] ep {epoch+1:03d}/{cfg.EPOCHS}  loss={rec['train_loss']:.4f}  "
              f"val_mse={rec['val_mse']:.4f}  phys={rec['train_phys']:.4f}  "
              f"lr={rec['lr']:.2e}")
        sess_secs = time.time() - t0
        sess_mem  = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0
        compute = {
            "n_params":     n_params,
            "train_time_s": prior_compute.get("train_time_s", 0.0) + sess_secs,
            "peak_mem_mb":  max(prior_compute.get("peak_mem_mb", 0.0), sess_mem),
        }
        save_ckpt(cpath, model, optim, ema, epoch + 1, history, compute)
        lr_sched.step()

    sess_secs = time.time() - t0
    sess_mem  = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0
    if start_epoch >= cfg.EPOCHS and prior_compute:
        train_time = float(prior_compute.get("train_time_s", sess_secs))
        peak_mem   = float(prior_compute.get("peak_mem_mb",  sess_mem))
    else:
        train_time = prior_compute.get("train_time_s", 0.0) + sess_secs
        peak_mem   = max(prior_compute.get("peak_mem_mb",   0.0), sess_mem)
    print(f"[Train] done in {train_time:.1f}s  peak_mem={peak_mem:.1f} MB")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ema.copy_to(model.module if isinstance(model, nn.DataParallel) else model)
    return model, diff_sched, history, {"n_params": n_params, "train_time_s": train_time,
                                         "peak_mem_mb": peak_mem}


# ==============================================================================
# SECTION 12. DPM-SOLVER++ 2M SAMPLER  (with optional deterministic noise seed)
# ==============================================================================

@torch.no_grad()
def sample_dpmpp_2m(model, sched, n, re_label, image_size, device,
                    n_steps=50, clip_x0=8.0, noise_seed=None):
    """If `noise_seed` is given, the initial noise is drawn from a seeded RNG.
    Used for paired-sample comparisons across variants -- all variants then
    see the same initial latent (fix for vorticity-figure non-pairing)."""
    target = model.module if isinstance(model, nn.DataParallel) else model
    target.eval()
    sigmas = sched.build_inference_schedule(n_steps)
    if noise_seed is None:
        x0 = torch.randn(n, 1, image_size, image_size, device=device)
    else:
        g = torch.Generator(device=device).manual_seed(int(noise_seed))
        x0 = torch.randn(n, 1, image_size, image_size, device=device, generator=g)
    x = x0 * sigmas[0]
    re = torch.full((n,), re_label, dtype=torch.long, device=device)
    old = None
    for i in range(n_steps):
        sigma      = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_b = torch.full((n,), sigma.item(), device=device)
        denoised = denoise_preconditioned(target, x, sigma_b, re, sched).clamp(-clip_x0, clip_x0)
        if sigma_next == 0:
            x = denoised
        else:
            t      = -sigma.log()
            t_next = -sigma_next.log()
            h = t_next - t
            if old is None or i == n_steps - 1:
                x = (sigma_next / sigma) * x - (-h).expm1() * denoised
            else:
                h_prev = t - (-sigmas[i - 1].log())
                r = h_prev / h
                D = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old
                x = (sigma_next / sigma) * x - (-h).expm1() * D
            old = denoised
    return x


# ==============================================================================
# SECTION 13. EVALUATION METRICS
# ==============================================================================

def metric_lsd_aggregate(pred, true):
    """Population log-spectral distance: aggregate the per-snapshot dealiased
    spectra into ensemble mean spectra, then mean_k |log10 E_pred(k) - log10
    E_true(k)|. The canonical metric for generative turbulence (the snapshot-
    to-snapshot version is improperly paired for an unconditional generator)."""
    Ep, _ = compute_dealiased_spectrum(pred.squeeze(1))
    Et, _ = compute_dealiased_spectrum(true.squeeze(1))
    Ep = Ep.mean(0).clamp(min=1e-12)
    Et = Et.mean(0).clamp(min=1e-12)
    return (torch.log10(Ep) - torch.log10(Et)).abs().mean().item()


def metric_lsd_per_snapshot_diag(pred_one, true_one):
    """DIAGNOSTIC ONLY (improperly paired). Not in headline or stat tests."""
    Ep, _ = compute_dealiased_spectrum(pred_one.squeeze(1))
    Et, _ = compute_dealiased_spectrum(true_one.squeeze(1))
    Ep = Ep.squeeze(0).clamp(min=1e-12)
    Et = Et.squeeze(0).clamp(min=1e-12)
    return (torch.log10(Ep) - torch.log10(Et)).abs().mean().item()


def metric_vorticity_structure_log_rmse(pred, true, order):
    """Log-RMSE of the vorticity structure function (not the velocity SF)."""
    Sp, _ = compute_vorticity_structure_function(pred, order)
    St, _ = compute_vorticity_structure_function(true, order)
    Sp_m = Sp.mean(0).clamp(min=1e-12)
    St_m = St.mean(0).clamp(min=1e-12)
    return ((torch.log10(Sp_m) - torch.log10(St_m)) ** 2).mean().sqrt().item()


def metric_integral_length_rel_err(pred, true):
    Lp = compute_integral_length(pred).mean()
    Lt = compute_integral_length(true).mean()
    return ((Lp - Lt).abs() / (Lt + 1e-8)).item()


def metric_energy_flux_rmse(pred, true):
    """RMSE on the true energy flux Pi_E(K), normalized by max|Pi_E_DNS|."""
    Pp, _ = compute_energy_flux(pred)
    Pt, _ = compute_energy_flux(true)
    Pp_m, Pt_m = Pp.mean(0), Pt.mean(0)
    norm = Pt_m.abs().max().clamp(min=1e-8)
    return ((Pp_m - Pt_m) ** 2).mean().sqrt().item() / norm.item()


def metric_enstrophy_flux_rmse(pred, true):
    """RMSE on the enstrophy flux Pi_Z(K), normalized by max|Pi_Z_DNS|."""
    Pp, _ = compute_enstrophy_flux(pred)
    Pt, _ = compute_enstrophy_flux(true)
    Pp_m, Pt_m = Pp.mean(0), Pt.mean(0)
    norm = Pt_m.abs().max().clamp(min=1e-8)
    return ((Pp_m - Pt_m) ** 2).mean().sqrt().item() / norm.item()


def metric_inverse_energy_cascade_recovery(pred, true, kf=4):
    """Fraction of the DNS inverse energy cascade signal recovered, in [0, 200%].

    Definition. In 2D forced turbulence the energy flux Pi_E(K) is negative at
    K < k_f (inverse cascade). We define
        inv_E(field) := sum_{0 < K < k_f} max(-Pi_E(K), 0)
    and report 100 * inv_E(pred) / inv_E(true). The output is clipped to
    [0, 200%] to keep figures readable; a value of 200% (rare) indicates an
    over-prediction by 2x or more and should be inspected in the raw CSV."""
    Pp, kp = compute_energy_flux(pred)
    Pt, kt = compute_energy_flux(true)
    Pp_m, Pt_m = Pp.mean(0), Pt.mean(0)
    mask = (kt < kf) & (kt > 0)
    if mask.sum() == 0 or Pt_m[mask].clamp(max=0).abs().sum() < 1e-8:
        return 0.0
    inv_p = (-Pp_m[mask]).clamp(min=0).sum()
    inv_t = (-Pt_m[mask]).clamp(min=0).sum()
    return 100.0 * (inv_p / inv_t.clamp(min=1e-8)).clamp(min=0.0, max=2.0).item()


def metric_forward_enstrophy_cascade_recovery(pred, true, kf=4):
    """Fraction of the DNS forward enstrophy cascade recovered, in [0, 200%].

    Definition. Pi_Z(K) > 0 at K > k_f indicates forward enstrophy cascade. We
    define  fwd_Z(field) := sum_{K > k_f} max(Pi_Z(K), 0)  and report ratio.
    Same clipping caveat as above."""
    Pp, kp = compute_enstrophy_flux(pred)
    Pt, kt = compute_enstrophy_flux(true)
    Pp_m, Pt_m = Pp.mean(0), Pt.mean(0)
    mask = (kt > kf)
    if mask.sum() == 0 or Pt_m[mask].clamp(min=0).sum() < 1e-8:
        return 0.0
    fwd_p = Pp_m[mask].clamp(min=0).sum()
    fwd_t = Pt_m[mask].clamp(min=0).sum()
    return 100.0 * (fwd_p / fwd_t.clamp(min=1e-8)).clamp(min=0.0, max=2.0).item()


# --- UQ metrics --------------------------------------------------------------

def metric_crps_ensemble(ens, target):
    """Continuous Ranked Probability Score. Memory-bounded version: batches
    the M x M pairwise-difference sum over the ensemble dimension instead of
    materializing the full (M, M, N) tensor (fix for memory blow-up at large N
    or large M)."""
    e = ens.cpu().numpy().astype(np.float32)
    y = target.cpu().numpy().astype(np.float32)
    M = e.shape[0]
    e_flat = e.reshape(M, -1)
    y_flat = y.reshape(-1)
    N = e_flat.shape[1]
    t1 = np.abs(e_flat - y_flat[None, :]).mean(0)
    t2 = np.zeros(N, dtype=np.float64)
    for i in range(M):
        t2 += np.abs(e_flat[i:i+1] - e_flat).mean(0)   # mean over j
    t2 *= 0.5 / M
    return float((t1 - t2).mean())


def metric_coverage_band(ens, target, lo_pct=5, hi_pct=95):
    e = ens.cpu().numpy(); y = target.cpu().numpy()
    lo = np.percentile(e, lo_pct, axis=0)
    hi = np.percentile(e, hi_pct, axis=0)
    return float(((y >= lo) & (y <= hi)).mean())


def calibrate_alpha(ens_cal, y_cal, target_cov=0.90, grid=None):
    if grid is None:
        grid = np.linspace(0.1, 4.0, 79)
    e = ens_cal.cpu().numpy(); y = y_cal.cpu().numpy()
    lo  = np.percentile(e, 5,  axis=0)
    hi  = np.percentile(e, 95, axis=0)
    med = np.percentile(e, 50, axis=0)
    half_lo = np.maximum(med - lo, 0.0)
    half_hi = np.maximum(hi - med, 0.0)
    best_a, best_gap = 1.0, float("inf")
    for a in grid:
        cov = ((y >= med - a * half_lo) & (y <= med + a * half_hi)).mean()
        gap = abs(cov - target_cov)
        if gap < best_gap:
            best_a, best_gap = float(a), float(gap)
    return best_a


def coverage_calibrated(ens, target, alpha):
    e = ens.cpu().numpy(); y = target.cpu().numpy()
    lo  = np.percentile(e, 5,  axis=0)
    hi  = np.percentile(e, 95, axis=0)
    med = np.percentile(e, 50, axis=0)
    half_lo = np.maximum(med - lo, 0.0)
    half_hi = np.maximum(hi - med, 0.0)
    return float(((y >= med - alpha * half_lo) & (y <= med + alpha * half_hi)).mean())


# ==============================================================================
# SECTION 14. EVALUATION ROUTINE  (test-set only)
# ==============================================================================

def evaluate(model, sched, test_ds, cfg, device, variant, seed):
    """Evaluate on the FIXED test holdout (never seen by any training run)."""
    print(f"[Eval] {variant}/seed{seed}")
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    all_pred, all_true, all_re = [], [], []
    with torch.no_grad():
        for x, re in loader:
            x, re = x.to(device), re.to(device)
            for r in range(len(cfg.NU_LIST)):
                idx = (re == r)
                if idx.sum() == 0: continue
                n = int(idx.sum().item())
                gens = sample_dpmpp_2m(model, sched, n, r, cfg.GRID, device,
                                       n_steps=cfg.N_SAMPLE_STEPS)
                all_pred.append(gens.cpu())
                all_true.append(x[idx].cpu())
                all_re.append(torch.full((n,), r))
            if sum(p.shape[0] for p in all_pred) >= cfg.N_EVAL_SAMPLES:
                break
    pred   = torch.cat(all_pred, dim=0)[:cfg.N_EVAL_SAMPLES]
    true   = torch.cat(all_true, dim=0)[:cfg.N_EVAL_SAMPLES]
    re_arr = torch.cat(all_re,   dim=0)[:cfg.N_EVAL_SAMPLES]

    # Per-snapshot DIAGNOSTICS (all suffixed _diag; not in headline / stat tests)
    per_sample = []
    for i in range(pred.shape[0]):
        per_sample.append({
            "variant": variant, "seed": seed, "re_idx": int(re_arr[i].item()),
            "mse_diag":                       float(F.mse_loss(pred[i], true[i]).item()),
            "lsd_per_snapshot_diag":          metric_lsd_per_snapshot_diag(pred[i:i+1], true[i:i+1]),
            "vorticity_S2_log_rmse_diag":     metric_vorticity_structure_log_rmse(pred[i:i+1], true[i:i+1], order=2),
            "vorticity_S3_log_rmse_diag":     metric_vorticity_structure_log_rmse(pred[i:i+1], true[i:i+1], order=3),
        })

    # Aggregate / population metrics (the headline)
    aggregates = {
        "lsd_aggregate":                metric_lsd_aggregate(pred, true),
        "vorticity_S2_log_rmse":        metric_vorticity_structure_log_rmse(pred, true, order=2),
        "vorticity_S3_log_rmse":        metric_vorticity_structure_log_rmse(pred, true, order=3),
        "integral_length_rel_err":      metric_integral_length_rel_err(pred, true),
        "energy_flux_rmse":             metric_energy_flux_rmse(pred, true),
        "enstrophy_flux_rmse":          metric_enstrophy_flux_rmse(pred, true),
        "inverse_energy_cascade_recovery_pct":    metric_inverse_energy_cascade_recovery(pred, true, kf=cfg.FORCING_K),
        "forward_enstrophy_cascade_recovery_pct": metric_forward_enstrophy_cascade_recovery(pred, true, kf=cfg.FORCING_K),
    }

    per_regime = []
    for r in range(len(cfg.NU_LIST)):
        mask = (re_arr == r)
        if mask.sum() < 8:
            continue
        pr = pred[mask]; tr = true[mask]
        per_regime.append({
            "variant": variant, "seed": seed, "re_idx": r, "nu": cfg.NU_LIST[r],
            "n_samples":                              int(mask.sum().item()),
            "lsd_aggregate":                          metric_lsd_aggregate(pr, tr),
            "vorticity_S2_log_rmse":                  metric_vorticity_structure_log_rmse(pr, tr, order=2),
            "vorticity_S3_log_rmse":                  metric_vorticity_structure_log_rmse(pr, tr, order=3),
            "integral_length_rel_err":                metric_integral_length_rel_err(pr, tr),
            "energy_flux_rmse":                       metric_energy_flux_rmse(pr, tr),
            "enstrophy_flux_rmse":                    metric_enstrophy_flux_rmse(pr, tr),
            "inverse_energy_cascade_recovery_pct":    metric_inverse_energy_cascade_recovery(pr, tr, kf=cfg.FORCING_K),
            "forward_enstrophy_cascade_recovery_pct": metric_forward_enstrophy_cascade_recovery(pr, tr, kf=cfg.FORCING_K),
        })

    print(f"[Eval] LSD_agg={aggregates['lsd_aggregate']:.4f}  "
          f"S2={aggregates['vorticity_S2_log_rmse']:.4f}  "
          f"EnergyFluxRMSE={aggregates['energy_flux_rmse']:.4f}  "
          f"EnstrophyFluxRMSE={aggregates['enstrophy_flux_rmse']:.4f}  "
          f"InvE={aggregates['inverse_energy_cascade_recovery_pct']:.1f}%  "
          f"FwdZ={aggregates['forward_enstrophy_cascade_recovery_pct']:.1f}%")

    # --- Uncertainty quantification with conformal recalibration ---
    n_uq = min(64, pred.shape[0])
    true_uq = true[:n_uq].to(device)
    re_uq   = re_arr[:n_uq].to(device)
    ens = torch.zeros(cfg.N_ENSEMBLE, n_uq, 1, cfg.GRID, cfg.GRID)
    with torch.no_grad():
        for m in range(cfg.N_ENSEMBLE):
            buf = torch.zeros(n_uq, 1, cfg.GRID, cfg.GRID)
            for r in range(len(cfg.NU_LIST)):
                idx = (re_uq == r)
                if idx.sum() == 0: continue
                gens = sample_dpmpp_2m(model, sched, int(idx.sum().item()), r,
                                       cfg.GRID, device, n_steps=cfg.N_SAMPLE_STEPS)
                buf[idx.nonzero().flatten().cpu()] = gens.cpu()
            ens[m] = buf

    n_cal = max(2, int(cfg.UQ_CAL_FRACTION * n_uq))
    if n_uq - n_cal >= 2:
        ens_cal, ens_tst = ens[:, :n_cal], ens[:, n_cal:]
        y_cal, y_tst = true_uq[:n_cal].cpu(), true_uq[n_cal:].cpu()
        alpha = calibrate_alpha(ens_cal, y_cal, target_cov=0.90)
        aggregates.update({
            "crps_test":              metric_crps_ensemble(ens_tst, y_tst),
            "coverage_90_raw_test":   metric_coverage_band(ens_tst, y_tst),
            "coverage_90_calibrated": coverage_calibrated(ens_tst, y_tst, alpha),
            "coverage_alpha":         alpha,
        })
        print(f"[Eval]  UQ: alpha={alpha:.3f}  raw_cov={aggregates['coverage_90_raw_test']:.3f}  "
              f"cal_cov={aggregates['coverage_90_calibrated']:.3f}  "
              f"CRPS={aggregates['crps_test']:.4f}")
    else:
        for k in ("crps_test", "coverage_90_raw_test", "coverage_90_calibrated", "coverage_alpha"):
            aggregates[k] = float("nan")

    return aggregates, per_sample, per_regime, pred, true, re_arr


# ==============================================================================
# SECTION 15. SEED-LEVEL STATISTICAL ANALYSIS
# ==============================================================================

def bootstrap_ci(values, n_resamples=2000, ci=0.95, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    if len(vals) < 2:
        return float(vals.mean() if len(vals) else float("nan")), float("nan"), float("nan")
    samples = rng.choice(vals, size=(n_resamples, len(vals)), replace=True).mean(axis=1)
    lo = np.percentile(samples, 100 * (1 - ci) / 2)
    hi = np.percentile(samples, 100 * (1 + ci) / 2)
    return float(vals.mean()), float(lo), float(hi)


def seed_level_paired_tests(summary_df, target, baseline, metrics, cfg):
    rows = []
    for m in metrics:
        if m not in summary_df.columns:
            continue
        a_df = summary_df[summary_df["variant"] == target  ].set_index("seed")[m]
        b_df = summary_df[summary_df["variant"] == baseline].set_index("seed")[m]
        common = sorted(set(a_df.index) & set(b_df.index))
        if len(common) < 3:
            continue
        a = a_df.loc[common].values.astype(float)
        b = b_df.loc[common].values.astype(float)
        diff = a - b
        mean_d, lo_d, hi_d = bootstrap_ci(diff, cfg.BOOTSTRAP_RESAMPLES, cfg.BOOTSTRAP_CI, seed=12345)
        try:
            t_stat, t_p = stats.ttest_rel(a, b)
        except Exception:
            t_stat, t_p = float("nan"), float("nan")
        try:
            w_stat, w_p = stats.wilcoxon(a, b)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        cohen_d = float(diff.mean() / (diff.std(ddof=1) + 1e-12)) if len(diff) > 1 else float("nan")
        rows.append({
            "comparison":    f"{target}_vs_{baseline}",
            "metric":        m,
            "n_seeds":       int(len(common)),
            "target_mean":   float(a.mean()),
            "baseline_mean": float(b.mean()),
            "diff_mean":     mean_d,
            "diff_ci_lo":    lo_d,
            "diff_ci_hi":    hi_d,
            "t_p":           float(t_p),
            "wilcoxon_p":    float(w_p),
            "cohen_d":       cohen_d,
        })
    return pd.DataFrame(rows)


def aggregate_with_bootstrap_ci(summary_df, metrics, cfg):
    rows = []
    for variant, grp in summary_df.groupby("variant"):
        row = {"variant": variant, "n_seeds": int(len(grp))}
        for m in metrics:
            if m not in grp.columns:
                continue
            mean, lo, hi = bootstrap_ci(grp[m].dropna().values,
                                        cfg.BOOTSTRAP_RESAMPLES, cfg.BOOTSTRAP_CI,
                                        seed=hash((variant, m)) & 0xFFFF)
            row[f"{m}_mean"]  = mean
            row[f"{m}_ci_lo"] = lo
            row[f"{m}_ci_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values("variant").reset_index(drop=True)


# ==============================================================================
# SECTION 16. VISUALIZATION
# ==============================================================================

VARIANT_COLORS = {
    "vanilla":        "#888888",
    "se":             "#7B7BC6",
    "fno":            "#5BAF6A",
    "wrsd":           "#C9534D",
    "wrsd_arch_only": "#F0A040",
    "wrsd_loss_only": "#7050C0",
}
VARIANT_ORDER = ["vanilla", "se", "fno", "wrsd_arch_only", "wrsd_loss_only", "wrsd"]


def plot_training_curves(all_histories, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    by_variant = {}
    for (variant, seed), hist in all_histories.items():
        by_variant.setdefault(variant, []).append(pd.DataFrame(hist))
    for variant, dfs in by_variant.items():
        if variant not in VARIANT_COLORS:
            continue
        # Guard against runs of unequal length
        min_e = min(len(d) for d in dfs)
        epochs = dfs[0]["epoch"].values[:min_e]
        tl = np.stack([d["train_loss"].values[:min_e] for d in dfs], axis=0)
        vm = np.stack([d["val_mse"   ].values[:min_e] for d in dfs], axis=0)
        for ax, arr in [(axes[0], tl), (axes[1], vm)]:
            m  = arr.mean(0); sd = arr.std(0)
            color = VARIANT_COLORS.get(variant, "#999")
            ax.plot(epochs, m, color=color, lw=1.8, label=variant)
            ax.fill_between(epochs, m - sd, m + sd, color=color, alpha=0.18)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train loss")
    axes[0].set_title(f"Training loss (mean +/- std over {len(cfg.SEEDS)} seeds)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val MSE")
    axes[1].set_title(f"Validation MSE (mean +/- std over {len(cfg.SEEDS)} seeds)")
    axes[0].legend(fontsize=9); axes[1].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(cfg.FIG_DIR / "training_curves.png", dpi=140); plt.close(fig)


def plot_dealiased_spectra(samples_per_variant, true_samples, cfg):
    fig, ax = plt.subplots(figsize=(7, 5))
    Et, kvals = compute_dealiased_spectrum(true_samples.squeeze(1))
    k = kvals.cpu().numpy()
    ax.loglog(k[1:], Et.mean(0).cpu().numpy()[1:], "k-", lw=2.6, label="DNS (dealiased)")
    for v in VARIANT_ORDER:
        if v not in samples_per_variant: continue
        Ep, _ = compute_dealiased_spectrum(samples_per_variant[v].squeeze(1))
        ax.loglog(k[1:], Ep.mean(0).cpu().numpy()[1:], "--", lw=1.6,
                  color=VARIANT_COLORS[v], alpha=0.9, label=v)
    ax.axvspan(cfg.FORCING_K, max(cfg.GRID // 4, cfg.FORCING_K + 4), alpha=0.15,
               color="orange", label="inertial range")
    ax.set_xlabel("k"); ax.set_ylabel("E(k)")
    ax.set_title("Dealiased Energy Spectra: Generated vs DNS")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(cfg.FIG_DIR / "spectra_dealiased.png", dpi=140); plt.close(fig)


def plot_structure_functions(samples_per_variant, true_samples, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    S2_t, rs = compute_vorticity_structure_function(true_samples, order=2)
    S3_t, _  = compute_vorticity_structure_function(true_samples, order=3)
    r_np = rs.cpu().numpy()
    axes[0].loglog(r_np, S2_t.mean(0).cpu().numpy(), "k-", lw=2.6, label="DNS")
    axes[1].loglog(r_np, S3_t.mean(0).cpu().numpy(), "k-", lw=2.6, label="DNS")
    for v in VARIANT_ORDER:
        if v not in samples_per_variant: continue
        s = samples_per_variant[v]
        S2, _ = compute_vorticity_structure_function(s, order=2)
        S3, _ = compute_vorticity_structure_function(s, order=3)
        axes[0].loglog(r_np, S2.mean(0).cpu().numpy(), "--", lw=1.5,
                       color=VARIANT_COLORS[v], alpha=0.9, label=v)
        axes[1].loglog(r_np, S3.mean(0).cpu().numpy(), "--", lw=1.5,
                       color=VARIANT_COLORS[v], alpha=0.9, label=v)
    axes[0].set_xlabel("r"); axes[0].set_ylabel(r"$S_2^{\omega}(r)$")
    axes[0].set_title("Vorticity 2nd-order structure function")
    axes[1].set_xlabel("r"); axes[1].set_ylabel(r"$|S_3^{\omega}(r)|$")
    axes[1].set_title("Vorticity 3rd-order structure function")
    axes[0].legend(fontsize=9); axes[1].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(cfg.FIG_DIR / "structure_functions.png", dpi=140); plt.close(fig)


def plot_flux_panels(samples_by_variant_regime, true_by_regime, cfg):
    """One row per flux (energy / enstrophy), one column per regime."""
    n_re = len(cfg.NU_LIST)
    fig, axes = plt.subplots(2, n_re, figsize=(4.6 * n_re, 8.0), squeeze=False)
    for row, (flux_fn, label) in enumerate([(compute_energy_flux,    "Energy"),
                                            (compute_enstrophy_flux, "Enstrophy")]):
        for r_idx in range(n_re):
            ax = axes[row, r_idx]
            true = true_by_regime.get(r_idx)
            if true is None or len(true) == 0:
                continue
            Pt, k = flux_fn(true)
            Pt_m = Pt.mean(0)
            norm = Pt_m.abs().max().clamp(min=1e-8)
            ax.semilogx(k.cpu().numpy(), (Pt_m / norm).cpu().numpy(),
                        "k-", lw=2.6, label="DNS")
            for v in VARIANT_ORDER:
                if v not in samples_by_variant_regime: continue
                sr = samples_by_variant_regime[v].get(r_idx)
                if sr is None or len(sr) == 0: continue
                Pp, _ = flux_fn(sr)
                ax.semilogx(k.cpu().numpy(), (Pp.mean(0) / norm).cpu().numpy(),
                            "--", lw=1.5, color=VARIANT_COLORS[v], alpha=0.9, label=v)
            ax.axvline(cfg.FORCING_K, color="orange", ls=":", lw=1.0,
                       label="$k_f$" if (row == 0 and r_idx == 0) else None)
            ax.axhline(0, color="gray", lw=0.6)
            ax.set_xlabel("k")
            if r_idx == 0:
                ax.set_ylabel(rf"$\Pi_{{{label[0]}}}(k)\,/\,|\Pi_{{DNS}}|_{{\max}}$")
                if row == 0:
                    ax.legend(fontsize=8)
            ax.set_title(rf"{label} flux  -  regime {r_idx}  $\nu={cfg.NU_LIST[r_idx]}$")
    fig.suptitle("Normalized energy (top) and enstrophy (bottom) spectral fluxes", y=1.02)
    fig.tight_layout(); fig.savefig(cfg.FIG_DIR / "fluxes.png", dpi=140); plt.close(fig)


def plot_headline_bars_with_ci(agg_ci_df, cfg):
    metrics = ["lsd_aggregate", "vorticity_S2_log_rmse", "vorticity_S3_log_rmse",
               "integral_length_rel_err", "energy_flux_rmse", "enstrophy_flux_rmse",
               "inverse_energy_cascade_recovery_pct", "forward_enstrophy_cascade_recovery_pct"]
    metrics = [m for m in metrics if f"{m}_mean" in agg_ci_df.columns]
    n = len(metrics)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.0 * ((n + 1) // 2), 7.5))
    axes = axes.flatten()
    df = agg_ci_df.set_index("variant").reindex(
        [v for v in VARIANT_ORDER if v in agg_ci_df["variant"].values])
    for i, m in enumerate(metrics):
        means = df[f"{m}_mean"].values
        lo = df[f"{m}_ci_lo"].values
        hi = df[f"{m}_ci_hi"].values
        err_lo = np.maximum(means - lo, 0); err_hi = np.maximum(hi - means, 0)
        colors = [VARIANT_COLORS.get(v, "#999") for v in df.index]
        axes[i].bar(df.index, means, yerr=[err_lo, err_hi], capsize=4, color=colors)
        axes[i].set_title(m, fontsize=9); axes[i].tick_params(axis="x", rotation=30, labelsize=8)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Headline metrics: per-variant mean with 95% bootstrap CI (n={len(cfg.SEEDS)} seeds)",
                 y=1.02)
    fig.tight_layout(); fig.savefig(cfg.FIG_DIR / "headline_bars.png", dpi=140); plt.close(fig)


def plot_pareto(summary_df, compute_df, agg_ci_df, cfg):
    """Bootstrap CIs on the y-axis (consistent with the headline figure)."""
    metrics = ["lsd_aggregate", "vorticity_S2_log_rmse", "energy_flux_rmse"]
    merged = summary_df.merge(compute_df[["variant", "seed", "n_params"]],
                              on=["variant", "seed"])
    params_per_v = merged.groupby("variant")["n_params"].mean()
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.0))
    for i, m in enumerate(metrics):
        ax = axes[i]
        for variant in [v for v in VARIANT_ORDER if v in agg_ci_df["variant"].values]:
            row = agg_ci_df[agg_ci_df["variant"] == variant].iloc[0]
            if f"{m}_mean" not in row: continue
            mean = row[f"{m}_mean"]
            lo, hi = row[f"{m}_ci_lo"], row[f"{m}_ci_hi"]
            err_lo = max(mean - lo, 0); err_hi = max(hi - mean, 0)
            ax.errorbar(params_per_v[variant] / 1e6, mean,
                        yerr=[[err_lo], [err_hi]], fmt="o", markersize=10, capsize=5,
                        color=VARIANT_COLORS.get(variant, "#999"), label=variant)
        ax.set_xlabel("Parameters (M)"); ax.set_ylabel(m)
        ax.set_title(m, fontsize=10)
        if i == 0:
            ax.legend(fontsize=9)
    fig.suptitle(f"Parameter / Performance Pareto Frontier (95% bootstrap CIs)", y=1.02)
    fig.tight_layout(); fig.savefig(cfg.FIG_DIR / "pareto.png", dpi=140); plt.close(fig)


def plot_uq_panel(summary_df, cfg):
    cols = [c for c in ["crps_test", "coverage_90_raw_test",
                        "coverage_90_calibrated", "coverage_alpha"]
            if c in summary_df.columns]
    if not cols:
        return
    agg = summary_df.groupby("variant")[cols].agg(["mean", "std"])
    variants = [v for v in VARIANT_ORDER if v in agg.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(variants)); w = 0.27
    crps = agg.loc[variants, ("crps_test",              "mean")].values
    rawc = agg.loc[variants, ("coverage_90_raw_test",   "mean")].values
    calc = agg.loc[variants, ("coverage_90_calibrated", "mean")].values
    crps_e = agg.loc[variants, ("crps_test",              "std")].fillna(0).values
    rawc_e = agg.loc[variants, ("coverage_90_raw_test",   "std")].fillna(0).values
    calc_e = agg.loc[variants, ("coverage_90_calibrated", "std")].fillna(0).values
    ax.bar(x - w, crps, w, yerr=crps_e, capsize=4, label="CRPS (lower better)", color="#C9534D")
    ax.bar(x,     rawc, w, yerr=rawc_e, capsize=4, label="Coverage@90% (raw)",  color="#5BAF6A")
    ax.bar(x + w, calc, w, yerr=calc_e, capsize=4, label="Coverage@90% (calibrated)", color="#2E7D43")
    ax.axhline(0.90, color="k", ls="--", lw=0.8, label="target = 0.90")
    ax.set_xticks(x); ax.set_xticklabels(variants, rotation=15)
    ax.set_title(f"Uncertainty Quantification (M={cfg.N_ENSEMBLE} ensemble, 50/50 cal/test)")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(cfg.FIG_DIR / "uq_panel.png", dpi=140); plt.close(fig)


def plot_vorticity_panel(paired_samples_per_variant, true_sample, cfg):
    """Paired-noise comparison. Every variant is sampled from the SAME initial
    latent noise (fix: previously the figure showed independent samples and
    invited a false equivalence). DNS is shown for orientation; the comparison
    is across the variant panels."""
    H = true_sample.shape[-1]
    crop = H // 4
    cx, cy = H // 2, H // 2
    variants = [v for v in VARIANT_ORDER if v in paired_samples_per_variant]
    n_cols = 1 + len(variants)
    fig, axes = plt.subplots(2, n_cols, figsize=(2.4 * n_cols, 5.0))
    vmax = float(true_sample.abs().max().item())
    titles = ["DNS reference"] + [f"{v}\n(shared noise)" for v in variants]
    fields_top = [true_sample] + [paired_samples_per_variant[v] for v in variants]
    for col, (t, f) in enumerate(zip(titles, fields_top)):
        ax = axes[0, col]
        ax.imshow(f[0].cpu().numpy() if f.ndim == 3 else f[0, 0].cpu().numpy(),
                  cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        rect = plt.Rectangle((cy - crop // 2, cx - crop // 2), crop, crop,
                             linewidth=1.2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
    for col, (t, f) in enumerate(zip(titles, fields_top)):
        ax = axes[1, col]
        arr = f[0].cpu().numpy() if f.ndim == 3 else f[0, 0].cpu().numpy()
        zoom = arr[cx - crop // 2:cx + crop // 2, cy - crop // 2:cy + crop // 2]
        ax.imshow(zoom, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title("zoom", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Paired vorticity samples (shared latent noise across variants)", y=1.02)
    fig.tight_layout(); fig.savefig(cfg.FIG_DIR / "vorticity_samples.png", dpi=140); plt.close(fig)


# ==============================================================================
# SECTION 17. MAIN
# ==============================================================================

HEADLINE_PHYSICS = ["lsd_aggregate", "vorticity_S2_log_rmse", "vorticity_S3_log_rmse",
                    "integral_length_rel_err", "energy_flux_rmse", "enstrophy_flux_rmse",
                    "inverse_energy_cascade_recovery_pct",
                    "forward_enstrophy_cascade_recovery_pct"]
HEADLINE_UQ      = ["crps_test", "coverage_90_raw_test",
                    "coverage_90_calibrated", "coverage_alpha"]
ALL_HEADLINE     = HEADLINE_PHYSICS + HEADLINE_UQ
PHYSICS_VARIANTS = {"wrsd", "wrsd_loss_only"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="~2 minute smoke test for end-to-end correctness")
    p.add_argument("--variants", nargs="+", default=list(TurbulenceDenoiser.VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--grid", type=int, default=None)
    p.add_argument("--snapshots", type=int, default=None)
    p.add_argument("--sample_steps", type=int, default=None)
    return p.parse_args()


def apply_cli_overrides(cfg, args):
    if args.smoke:
        cfg.GRID = 32
        cfg.EPOCHS = 1; cfg.SEEDS = [13, 29]
        cfg.SNAPSHOTS_PER_REGIME = 80
        cfg.SPINUP_STEPS = 200; cfg.SNAPSHOT_INTERVAL = 20
        cfg.BATCH_SIZE = 8
        cfg.N_EVAL_SAMPLES = 24; cfg.N_ENSEMBLE = 4; cfg.N_SAMPLE_STEPS = 20
        cfg.BOOTSTRAP_RESAMPLES = 200
        # Smoke test only verifies pipeline wiring; the 3/2-rule Jacobian
        # is correct but slow on CPU. Real runs leave it enabled (default).
        cfg.USE_DEALIASED_JACOBIAN = False
    if args.grid is not None:         cfg.GRID = args.grid
    if args.seeds is not None:        cfg.SEEDS = args.seeds
    if args.epochs is not None:       cfg.EPOCHS = args.epochs
    if args.snapshots is not None:    cfg.SNAPSHOTS_PER_REGIME = args.snapshots
    if args.sample_steps is not None: cfg.N_SAMPLE_STEPS = args.sample_steps


def main():
    args = parse_args()
    Config.setup()
    apply_cli_overrides(Config, args)

    print("=" * 80)
    print(" Wavenumber-Resolved Spectral Diffusion (WRSD) for 2D Turbulence")
    print(f" Grid={Config.GRID}^2  Regimes (nu)={Config.NU_LIST}  Seeds={Config.SEEDS}")
    print(f" Epochs={Config.EPOCHS}  Variants={args.variants}")
    print(f" 3/2-rule Jacobian dealiasing: {Config.USE_DEALIASED_JACOBIAN}")
    print(f" Train/Test holdout fraction: {Config.TEST_FRACTION:.0%}  "
          f"EMA decay: {Config.EMA_DECAY}")
    print("=" * 80)

    device, n_gpus = configure_device()

    # ---------- Phase 1: dataset + DNS validation + FIXED test holdout -------
    print("\n[Phase 1] Dataset generation, DNS validation, test holdout.")
    payload = generate_dataset(Config, device)
    validate_dns(Config, payload, device)
    full_ds = TurbulenceDataset(payload["fields"], payload["re_labels"])

    train_indices, test_indices = prepare_splits(full_ds, Config)
    train_pool = Subset(full_ds, train_indices)
    test_set   = Subset(full_ds, test_indices)
    print(f"[Splits] train_pool={len(train_pool)}  test={len(test_set)}  "
          f"(test seed=0, fixed across all variants and seeds; no leakage)")

    # Cap eval samples by test set size
    Config.N_EVAL_SAMPLES = min(Config.N_EVAL_SAMPLES, len(test_set))

    # ---------- Phase 2: train and evaluate ---------------------------------
    summary_records, per_sample_all, per_regime_all, compute_records = [], [], [], []
    all_histories, last_samples_per_variant, last_true_samples = {}, {}, None
    samples_by_variant_regime = {v: {} for v in args.variants}
    true_by_regime = {}
    paired_samples_per_variant = {}                       # for vorticity figure (shared noise)
    paired_true_sample = None

    print("\n[Phase 2] Training and evaluation on FIXED test holdout.")
    for variant in args.variants:
        for seed in Config.SEEDS:
            use_phys = variant in PHYSICS_VARIANTS
            model, sched, history, compute = train_one_model(
                variant, seed, Config, train_pool, device, n_gpus, use_physics=use_phys
            )
            all_histories[(variant, seed)] = history
            agg, per_samp, per_reg, gen, true, re_eval = evaluate(
                model, sched, test_set, Config, device, variant, seed
            )
            summary_records.append({"variant": variant, "seed": seed, **agg})
            per_sample_all.extend(per_samp)
            per_regime_all.extend(per_reg)
            compute_records.append({"variant": variant, "seed": seed, **compute,
                                    "n_epochs": Config.EPOCHS})
            last_samples_per_variant[variant] = gen[:64]
            last_true_samples = true[:64]
            for r in range(len(Config.NU_LIST)):
                mask = (re_eval == r)
                if mask.sum() > 0:
                    samples_by_variant_regime[variant][r] = gen[mask]
                    if r not in true_by_regime:
                        true_by_regime[r] = true[mask]

            # Paired-noise sample for the vorticity figure (shared initial latent
            # across all variants -> fair side-by-side comparison)
            if seed == Config.SEEDS[0]:
                paired = sample_dpmpp_2m(model, sched, 1, 0, Config.GRID, device,
                                         n_steps=Config.N_SAMPLE_STEPS,
                                         noise_seed=20260521)
                paired_samples_per_variant[variant] = paired[0]      # (1, H, W)
                if paired_true_sample is None:
                    # First DNS sample from the test set, regime 0
                    re_test = re_eval
                    r0_idx = (re_test == 0).nonzero(as_tuple=False).flatten()
                    if len(r0_idx) > 0:
                        paired_true_sample = true[r0_idx[0]]

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ---------- Phase 3: persist results ------------------------------------
    print("\n[Phase 3] Persisting results and seed-level statistics.")
    summary_df    = pd.DataFrame(summary_records)
    per_sample_df = pd.DataFrame(per_sample_all)
    per_regime_df = pd.DataFrame(per_regime_all)
    compute_df    = pd.DataFrame(compute_records)
    summary_df   .to_csv(Config.CSV_DIR / "summary_metrics.csv",        index=False)
    per_sample_df.to_csv(Config.CSV_DIR / "per_sample_diagnostics.csv", index=False)
    per_regime_df.to_csv(Config.CSV_DIR / "per_regime_metrics.csv",     index=False)
    compute_df   .to_csv(Config.CSV_DIR / "compute_profile.csv",        index=False)

    agg_ci_df = aggregate_with_bootstrap_ci(summary_df, ALL_HEADLINE, Config)
    agg_ci_df.to_csv(Config.CSV_DIR / "headline_table_with_ci.csv", index=False)
    print("\n[Headline] Per-variant means with 95% bootstrap CI over seeds:")
    print(agg_ci_df.round(4).to_string(index=False))

    print("\n[Stats] Seed-level paired tests vs WRSD:")
    for baseline in ["vanilla", "se", "fno", "wrsd_arch_only", "wrsd_loss_only"]:
        if "wrsd" in args.variants and baseline in summary_df["variant"].unique():
            tdf = seed_level_paired_tests(summary_df, "wrsd", baseline,
                                          HEADLINE_PHYSICS, Config)
            tdf.to_csv(Config.CSV_DIR / f"seed_level_tests_wrsd_vs_{baseline}.csv",
                       index=False)
            if len(tdf):
                print(f"\n  WRSD vs {baseline}:")
                print(tdf[["metric", "n_seeds", "target_mean", "baseline_mean",
                           "diff_mean", "diff_ci_lo", "diff_ci_hi", "wilcoxon_p",
                           "cohen_d"]].round(4).to_string(index=False))

    # ---------- Phase 4: visualization --------------------------------------
    print("\n[Phase 4] Visualization.")
    plot_training_curves(all_histories, Config)
    if last_true_samples is not None and last_samples_per_variant:
        plot_dealiased_spectra(last_samples_per_variant, last_true_samples, Config)
        plot_structure_functions(last_samples_per_variant, last_true_samples, Config)
        if true_by_regime:
            plot_flux_panels(samples_by_variant_regime, true_by_regime, Config)
    if paired_true_sample is not None and paired_samples_per_variant:
        plot_vorticity_panel(paired_samples_per_variant, paired_true_sample, Config)
    plot_headline_bars_with_ci(agg_ci_df, Config)
    plot_pareto(summary_df, compute_df, agg_ci_df, Config)
    plot_uq_panel(summary_df, Config)

    print("\n[Phase 5] Aggregate report (seed means):")
    print(summary_df.groupby("variant").mean(numeric_only=True).round(4).to_string())
    print(f"\n[Done] All artifacts under {Config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
