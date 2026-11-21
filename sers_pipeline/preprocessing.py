"""Module 2 — The Deterministic DSP Pipeline ("the washing machine").

Strict sequential order matters here. Each step has a specific job and the
order is not interchangeable:

    1. Despike     — kills cosmic-ray pixels. Must run FIRST: a single 100×
                     spike will destroy any baseline fit or smoothing window
                     that touches it.
    2. Baseline    — removes the smooth fluorescence offset. Must run before
                     normalization, otherwise normalization scale is dominated
                     by the baseline rather than the signal.
    3. Smooth      — Savitzky-Golay; reduces high-frequency noise while
                     preserving peak shape. Runs after baseline because SG
                     is a polynomial fit and a strong baseline trend can pull
                     the polynomial off-shape near the spectrum edges.
    4. Normalize   — SNV or vector norm; makes spectra comparable across
                     samples (focus, substrate, laser power variability).
                     Always last because everything before changes the scale.

Each step is a class with `.fit_transform(X)` returning a new (n, m) matrix.
The Pipeline class chains them. Stateless ops re-do the same computation per
sample, so fit==transform for everything here except the (optional) PCA in
classifiers.py.

Math notes:
    * ALS (Eilers, 2005, Anal. Chem.): minimize ||y - z||²_W + λ ||D²z||²
      where W is a diagonal weight matrix that down-weights points BELOW the
      current baseline estimate (asymmetry p, typically 0.001-0.1) and λ
      controls smoothness (1e4 to 1e7).
    * Savitzky-Golay: local polynomial fit. Window length ~ peak FWHM in
      pixels; polynomial order 2-3 keeps peak heights intact.
    * SNV: per-spectrum (x - μ) / σ. Removes additive offset AND scales
      multiplicatively. Equivalent to z-scoring each spectrum independently.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve


# --------------------------------------------------------------------------- #
# Abstract base class — every step has the same interface                      #
# --------------------------------------------------------------------------- #

class SpectralTransform(ABC):
    """All preprocessing steps implement this interface so they're swappable
    inside Pipeline and easy to mock in tests."""

    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Apply to a 2D batch. Input shape (n_samples, n_features)."""

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        return self(x[np.newaxis, :])[0]


# --------------------------------------------------------------------------- #
# Step 1: Despiking (Whitaker-Hayes)                                           #
# --------------------------------------------------------------------------- #

@dataclass
class WhitakerHayesDespiker(SpectralTransform):
    """Cosmic-ray removal via modified Z-score on the differenced signal.

    Reference: Whitaker & Hayes, Chemom. Intell. Lab. Syst. 179 (2018).

    Detection: |modified_z(diff(y))| > threshold marks a spike.
    Repair:    replace flagged points with the mean of nearby unflagged points.

    The 0.6745 constant comes from the relationship between MAD and σ for
    Gaussian data: σ ≈ MAD / 0.6745.
    """
    threshold: float = 7.0
    repair_window: int = 5

    def __call__(self, X: np.ndarray) -> np.ndarray:
        out = X.copy()
        for i in range(X.shape[0]):
            out[i] = self._despike_one(X[i])
        return out

    def _despike_one(self, y: np.ndarray) -> np.ndarray:
        diff_y = np.diff(y)
        median = np.median(diff_y)
        mad = np.median(np.abs(diff_y - median)) + 1e-12
        mod_z = 0.6745 * (diff_y - median) / mad
        # A real spike causes a large diff *both entering and leaving* it.
        # Mark indices where the absolute z-score exceeds threshold.
        spike_mask = np.abs(mod_z) > self.threshold
        # diff has length n-1; align to original by marking both endpoints of
        # each flagged diff
        n = len(y)
        flagged = np.zeros(n, dtype=bool)
        idx = np.where(spike_mask)[0]
        flagged[idx] = True
        flagged[idx + 1] = True

        y_out = y.copy()
        w = self.repair_window
        for i in np.where(flagged)[0]:
            lo, hi = max(0, i - w), min(n, i + w + 1)
            neighbor_idx = [j for j in range(lo, hi) if j != i and not flagged[j]]
            if neighbor_idx:
                y_out[i] = np.mean(y[neighbor_idx])
        return y_out


# --------------------------------------------------------------------------- #
# Step 2: Baseline correction (Asymmetric Least Squares — Eilers 2005)         #
# --------------------------------------------------------------------------- #

@dataclass
class ALSBaselineCorrector(SpectralTransform):
    """Asymmetric Least Squares baseline removal.

    Parameters
    ----------
    lam : float
        Smoothness penalty. Larger → stiffer baseline. Typical 1e4 to 1e7.
        For 1400-point spectra, 1e5 to 1e6 is a sensible starting range.
    p : float
        Asymmetry. Points above current baseline get weight p, points below
        get weight 1-p. Typical 0.001 to 0.05. Lower p → baseline hugs the
        valleys more aggressively (better when peaks are sharp and positive).
    niter : int
        Reweighting iterations. 10 usually converges.

    Tune lam and p by visual inspection on a handful of representative spectra
    BEFORE committing them as pipeline constants — they are the most
    sensitive knobs in the whole DSP chain.
    """
    lam: float = 1e5
    p: float = 0.01
    niter: int = 10

    def __call__(self, X: np.ndarray) -> np.ndarray:
        out = np.empty_like(X)
        for i in range(X.shape[0]):
            baseline = self._estimate_baseline(X[i])
            out[i] = X[i] - baseline
        return out

    def _estimate_baseline(self, y: np.ndarray) -> np.ndarray:
        L = len(y)
        # Second-difference operator — penalizes curvature of baseline
        D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(L - 2, L))
        DtD = D.T @ D
        w = np.ones(L)
        z = y  # initial guess
        for _ in range(self.niter):
            W = sparse.diags(w, 0, shape=(L, L))
            Z = W + self.lam * DtD
            z = spsolve(Z.tocsc(), w * y)
            w = self.p * (y > z) + (1.0 - self.p) * (y < z)
        return z


# --------------------------------------------------------------------------- #
# Step 3: Smoothing (Savitzky-Golay)                                           #
# --------------------------------------------------------------------------- #

@dataclass
class SavitzkyGolaySmoother(SpectralTransform):
    """Local-polynomial smoothing.

    Window must be odd. Common defaults for Raman:
        window 7-15, polyorder 2-3
    Larger window → smoother but blunter peaks. If your peak FWHM in pixels
    is W, keep window length ≲ W to avoid measurably reducing peak height.

    Optional deriv > 0 returns the n-th derivative, which is useful for
    additional baseline suppression downstream (peaks become more localized).
    Leave deriv=0 for the standard chain.
    """
    window_length: int = 9
    polyorder: int = 3
    deriv: int = 0

    def __post_init__(self):
        if self.window_length % 2 == 0:
            raise ValueError("Savitzky-Golay window must be odd")
        if self.polyorder >= self.window_length:
            raise ValueError("polyorder must be < window_length")

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return savgol_filter(
            X,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            axis=-1,
        )


# --------------------------------------------------------------------------- #
# Step 4: Normalization                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class SNVNormalizer(SpectralTransform):
    """Standard Normal Variate: per-spectrum (x - mean) / std.

    Removes additive offsets that survived baseline correction AND scales
    spectra to unit variance. After SNV, a spectrum is essentially a z-scored
    sequence — magnitudes are comparable across samples regardless of laser
    focus, integration time, or substrate enhancement variability.
    """
    epsilon: float = 1e-10  # guard against zero-variance flatlines

    def __call__(self, X: np.ndarray) -> np.ndarray:
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True) + self.epsilon
        return (X - mean) / std


@dataclass
class VectorNormalizer(SpectralTransform):
    """L2 (vector) normalization. Alternative to SNV when you want to
    preserve relative peak ratios more strictly."""
    epsilon: float = 1e-10

    def __call__(self, X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True) + self.epsilon
        return X / norms


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #

class Pipeline:
    """Chains preprocessing steps in order. Stateless — fit == transform.

    Example
    -------
    >>> pipe = Pipeline.default()
    >>> X_clean = pipe(X_raw)
    """

    def __init__(self, steps: list[tuple[str, SpectralTransform]]):
        self.steps = steps

    @classmethod
    def default(cls) -> "Pipeline":
        return cls([
            ("despike", WhitakerHayesDespiker()),
            ("baseline", ALSBaselineCorrector()),
            ("smooth", SavitzkyGolaySmoother()),
            ("normalize", SNVNormalizer()),
        ])

    @classmethod
    def from_spec(cls, spec_path: str) -> "Pipeline":
        """Build pipeline from a biomarker spec YAML's preprocessing section."""
        import yaml
        from pathlib import Path
        with open(Path(spec_path), encoding="utf-8") as f:
            data = yaml.safe_load(f)
        pre = data.get("preprocessing", {})
        return cls.from_config(pre)

    @classmethod
    def from_config(cls, config: dict) -> "Pipeline":
        """Build pipeline from a preprocessing config dict."""
        steps: list[tuple[str, SpectralTransform]] = []

        despike_cfg = config.get("despike", {})
        if despike_cfg.get("enabled", True):
            steps.append(("despike", WhitakerHayesDespiker(
                threshold=despike_cfg.get("threshold", 7.0),
            )))

        baseline_cfg = config.get("baseline", {})
        method = baseline_cfg.get("method", "als")
        lam = baseline_cfg.get("lambda", 1e5)
        if method in ("als", "arpls"):
            steps.append(("baseline", ALSBaselineCorrector(lam=lam)))

        smooth_cfg = config.get("smoothing", {})
        steps.append(("smooth", SavitzkyGolaySmoother(
            window_length=smooth_cfg.get("window", 9),
            polyorder=smooth_cfg.get("polyorder", 3),
        )))

        norm_cfg = config.get("normalization", {})
        norm_method = norm_cfg.get("method", "snv")
        if norm_method == "vector":
            steps.append(("normalize", VectorNormalizer()))
        else:
            steps.append(("normalize", SNVNormalizer()))

        return cls(steps)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        for name, step in self.steps:
            X = step(X)
        return X

    def transform_with_intermediates(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Useful for debugging and for the visual QA in scripts/demo.py."""
        out = {"raw": X}
        current = X
        for name, step in self.steps:
            current = step(current)
            out[name] = current.copy()
        return out
