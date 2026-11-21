"""Synthetic SERS spectra generator.

Why this module exists:
    No hardware yet → we need data that *behaves* like real SERS so the DSP
    chain and classifiers can be developed, profiled, and unit-tested.

What gets simulated (and why each matters):
    1. Lorentzian / pseudo-Voigt peaks at class-specific wavenumbers
       — molecular vibrational modes; this is the *signal* the classifier must learn.
    2. Polynomial + broad-Gaussian fluorescence background
       — dominant low-frequency artifact in real SERS, especially from biofluids.
       The baseline correction step in preprocessing.py is justified by this.
    3. Heteroscedastic Gaussian noise
       — detector shot noise + electronic noise; Savitzky-Golay targets this.
    4. Cosmic-ray spikes (single- or two-pixel)
       — random ionizing events hitting the CCD; despiking targets these.
    5. Random multiplicative scaling per spectrum
       — instrument focus / substrate variability; SNV normalization targets this.

This is NOT a physics-grade simulator. Real SERS has substrate hot-spot effects,
enhancement-factor variability, and complex Raman cross-sections we are NOT
modeling. Treat synthetic data as a *unit-test fixture for the software*, not
as a substitute for real spectra.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Class definitions: each "molecular fingerprint" is a list of peak centers    #
# with relative intensities. Tune these to whatever analytes you care about.   #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PeakSpec:
    """A single Lorentzian peak."""
    center_cm1: float       # wavenumber position
    rel_intensity: float    # 0..1, scaled by overall spectrum intensity
    fwhm_cm1: float = 12.0  # full-width-half-max; real Raman peaks ~10-30 cm^-1


@dataclass(frozen=True)
class ClassFingerprint:
    """A class label plus its characteristic peaks."""
    label: str
    peaks: Sequence[PeakSpec]


# NAFLD/MASH classes based on published SERS serum peak positions.
# Source: "Fully connected neural network-based serum SERS accurately
# identifies NASH" (Hepatology International, 2022) — 261 biopsy-proven
# patients, silver nanoparticles, 785 nm excitation.
#
# Healthy and NAFL share many serum peaks but differ in relative intensity.
# NASH shows distinctly elevated peaks at 497, 533, 593, 638, 813, 887,
# 1071, 1133, 1204, 1366, 1500 cm⁻¹ (lipid peroxidation, oxidative stress
# metabolites, altered protein conformations).
# NAFL-dominant peaks at 726, 744, 959, 1095, 1270, 1330, 1395, 1443,
# 1580, 1653 cm⁻¹ (normal lipid/protein bands at higher relative intensity
# before oxidative damage shifts the profile).
DEFAULT_CLASSES: list[ClassFingerprint] = [
    ClassFingerprint(
        label="healthy",
        peaks=[
            PeakSpec(726, 0.50, 15.0),   # adenine/hypoxanthine
            PeakSpec(1003, 0.80, 12.0),  # phenylalanine (universal serum)
            PeakSpec(1330, 0.45, 14.0),  # CH₃CH₂ wagging
            PeakSpec(1443, 0.60, 16.0),  # CH₂ deformation (lipids)
            PeakSpec(1653, 0.55, 18.0),  # amide I (proteins)
        ],
    ),
    ClassFingerprint(
        label="nafl",
        peaks=[
            PeakSpec(726, 0.65, 15.0),   # adenine — elevated in steatosis
            PeakSpec(744, 0.40, 12.0),   # thymine
            PeakSpec(959, 0.35, 14.0),   # carotenoid / C-C stretch
            PeakSpec(1003, 0.78, 12.0),  # phenylalanine (shared)
            PeakSpec(1095, 0.45, 14.0),  # PO₂ symmetric stretch (lipids)
            PeakSpec(1270, 0.50, 16.0),  # amide III
            PeakSpec(1330, 0.55, 14.0),  # CH₃CH₂ wagging — elevated
            PeakSpec(1395, 0.60, 15.0),  # COO⁻ stretch (fatty acids)
            PeakSpec(1443, 0.70, 16.0),  # CH₂ deformation — fat accumulation
            PeakSpec(1580, 0.45, 14.0),  # C=C stretch (nucleic acids)
            PeakSpec(1653, 0.65, 18.0),  # amide I — elevated
        ],
    ),
    ClassFingerprint(
        label="nash",
        peaks=[
            PeakSpec(497, 0.40, 14.0),   # S-S disulfide (oxidative stress)
            PeakSpec(533, 0.35, 12.0),   # S-S stretch
            PeakSpec(593, 0.30, 12.0),   # C-S stretch
            PeakSpec(638, 0.45, 14.0),   # C-S stretch (tyrosine)
            PeakSpec(766, 0.40, 12.0),   # tryptophan
            PeakSpec(813, 0.35, 12.0),   # C-O-C stretch (phospholipids)
            PeakSpec(887, 0.30, 14.0),   # C-C-N stretch
            PeakSpec(1003, 0.75, 12.0),  # phenylalanine (shared)
            PeakSpec(1071, 0.50, 14.0),  # C-O stretch (lipid peroxidation)
            PeakSpec(1133, 0.45, 14.0),  # C-N stretch (proteins)
            PeakSpec(1204, 0.40, 14.0),  # amide III / Tyr/Phe
            PeakSpec(1366, 0.55, 15.0),  # tryptophan / CH₃ symmetric
            PeakSpec(1443, 0.50, 16.0),  # CH₂ deformation — reduced vs NAFL
            PeakSpec(1500, 0.45, 14.0),  # C=C stretch (oxidized lipids)
        ],
    ),
    ClassFingerprint(
        label="advanced_fibrosis",
        peaks=[
            PeakSpec(497, 0.50, 14.0),   # S-S disulfide — elevated further
            PeakSpec(533, 0.45, 12.0),   # S-S stretch — more oxidative damage
            PeakSpec(638, 0.55, 14.0),   # C-S stretch — elevated
            PeakSpec(813, 0.45, 12.0),   # phospholipid reorganization
            PeakSpec(1003, 0.65, 12.0),  # phenylalanine — reduced (liver damage)
            PeakSpec(1071, 0.60, 14.0),  # C-O stretch — collagen crosslinks
            PeakSpec(1133, 0.55, 14.0),  # C-N stretch — elevated
            PeakSpec(1204, 0.50, 14.0),  # amide III — collagen signature
            PeakSpec(1270, 0.55, 16.0),  # amide III — collagen
            PeakSpec(1366, 0.65, 15.0),  # CH₃ — elevated in fibrotic tissue
            PeakSpec(1443, 0.40, 16.0),  # CH₂ — decreased (lipid depletion)
            PeakSpec(1500, 0.55, 14.0),  # oxidized lipids — elevated
            PeakSpec(1653, 0.45, 18.0),  # amide I — altered protein structure
        ],
    ),
]


# --------------------------------------------------------------------------- #
# Building blocks                                                              #
# --------------------------------------------------------------------------- #

def _lorentzian(x: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    """Unit-height Lorentzian. Real Raman lines are usually Lorentzian-ish;
    use pseudo-Voigt if you want to mix with Gaussian later."""
    gamma = fwhm / 2.0
    return gamma**2 / ((x - center) ** 2 + gamma**2)


def _fluorescence_background(
    wavenumbers: np.ndarray,
    rng: np.random.Generator,
    magnitude: float = 1.0,
    biofluid_mode: bool = False,
) -> np.ndarray:
    """Smooth low-frequency offset that simulates fluorescence.

    Combines a low-order polynomial with a broad Gaussian bump at a random
    position. When `biofluid_mode=True`, adds an exponential-decay component
    that approximates the high-fluorescence backgrounds typical of serum,
    saliva, and other biofluids — these are *much* harder to remove and are
    the realistic stress test for your ALS / airPLS parameters.
    """
    x = (wavenumbers - wavenumbers.min()) / (wavenumbers.max() - wavenumbers.min())
    # Random low-order polynomial (degree 3)
    coefs = rng.normal(0, 1, size=4)
    poly = sum(c * x**i for i, c in enumerate(coefs))
    # Broad Gaussian bump
    bump_center = rng.uniform(0.2, 0.8)
    bump_width = rng.uniform(0.15, 0.4)
    bump = np.exp(-0.5 * ((x - bump_center) / bump_width) ** 2)
    bump_amp = rng.uniform(0.5, 2.0)
    bg = poly + bump_amp * bump
    if biofluid_mode:
        # Exponential-decay arm: high at low wavenumbers, decays toward high.
        # This is the dominant feature in serum/plasma fluorescence.
        decay_rate = rng.uniform(2.0, 5.0)
        decay_amp = rng.uniform(2.0, 5.0)
        bg = bg + decay_amp * np.exp(-decay_rate * x)
    # Shift to be non-negative and scale
    bg = bg - bg.min()
    return magnitude * bg


def _inject_cosmic_rays(
    spectrum: np.ndarray,
    rng: np.random.Generator,
    rate_per_spectrum: float = 1.5,
    spike_magnitude_range: tuple[float, float] = (5.0, 25.0),
) -> np.ndarray:
    """Insert a Poisson-distributed number of single-pixel spikes."""
    out = spectrum.copy()
    n_spikes = rng.poisson(rate_per_spectrum)
    if n_spikes == 0:
        return out
    positions = rng.integers(0, len(spectrum), size=n_spikes)
    magnitudes = rng.uniform(*spike_magnitude_range, size=n_spikes)
    # Spike scales relative to the local signal so it stays distinguishable
    local_scale = np.median(np.abs(spectrum)) + 1e-6
    out[positions] += magnitudes * local_scale
    return out


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def make_wavenumber_grid(
    start: float = 400.0,
    stop: float = 1800.0,
    n_points: int = 1401,
) -> np.ndarray:
    """Standard 1 cm^-1 spacing grid covering the Raman fingerprint region."""
    return np.linspace(start, stop, n_points)


def synthesize_spectrum(
    wavenumbers: np.ndarray,
    fingerprint: ClassFingerprint,
    rng: np.random.Generator,
    signal_intensity: float = 100.0,
    fluorescence_magnitude: float = 300.0,
    noise_sigma: float = 2.0,
    cosmic_ray_rate: float = 1.5,
    # ── SERS-specific realism knobs (defaults give "easy" synthetic) ──
    hotspot_lognormal_sigma: float = 0.0,   # 0 = uniform; 0.3-0.7 = realistic SERS
    position_jitter_cm1: float = 0.0,       # 0 = no drift; 1-2 cm^-1 = realistic
    fwhm_jitter_frac: float = 0.0,          # 0 = fixed widths; 0.1-0.2 = realistic
    biofluid_fluorescence: bool = False,    # True for serum/plasma-style backgrounds
) -> np.ndarray:
    """Generate one noisy SERS-like spectrum for the given class.

    The realism knobs are off by default (so unit tests stay stable). The
    `make_realistic_dataset` convenience function below turns them all on.

    What each knob models:
        hotspot_lognormal_sigma — peak intensities in real SERS vary wildly
            across substrate hot-spots. Each peak gets an independent
            multiplicative factor drawn from lognormal(0, sigma). At sigma=0.5
            you'll see peak intensities ranging over roughly a factor of 5
            within a single spectrum.
        position_jitter_cm1 — calibration drift between measurements. Each
            peak's center moves by a Gaussian random amount of this std-dev.
        fwhm_jitter_frac — fractional variation in peak width (instrument
            focus, temperature). At 0.15, FWHMs vary by ~±15%.
        biofluid_fluorescence — adds exponential-decay arm to baseline.
    """
    # Per-peak Lorentzians with optional jitter
    clean = np.zeros_like(wavenumbers)
    intensity_jitter = rng.uniform(0.8, 1.2)  # whole-spectrum scaling (focus)
    for peak in fingerprint.peaks:
        # Hot-spot enhancement variability — lognormal so it's heavy-tailed
        if hotspot_lognormal_sigma > 0:
            hotspot_factor = float(rng.lognormal(0.0, hotspot_lognormal_sigma))
        else:
            hotspot_factor = float(rng.uniform(0.85, 1.15))
        # Position drift
        center = peak.center_cm1
        if position_jitter_cm1 > 0:
            center = center + float(rng.normal(0.0, position_jitter_cm1))
        # FWHM jitter
        fwhm = peak.fwhm_cm1
        if fwhm_jitter_frac > 0:
            fwhm = fwhm * float(1.0 + rng.normal(0.0, fwhm_jitter_frac))
            fwhm = max(fwhm, 2.0)  # don't let it collapse
        clean += peak.rel_intensity * hotspot_factor * _lorentzian(wavenumbers, center, fwhm)
    clean *= signal_intensity * intensity_jitter

    # Fluorescence baseline
    baseline = _fluorescence_background(
        wavenumbers, rng, fluorescence_magnitude, biofluid_mode=biofluid_fluorescence,
    )

    # Heteroscedastic Gaussian noise (Poisson approximation for high counts)
    composite = clean + baseline
    noise = rng.normal(0, noise_sigma * np.sqrt(np.abs(composite) + 1.0))

    # Cosmic rays
    spectrum = _inject_cosmic_rays(composite + noise, rng, cosmic_ray_rate)

    return spectrum


def synthesize_dataset(
    n_per_class: int = 100,
    classes: Sequence[ClassFingerprint] = DEFAULT_CLASSES,
    wavenumbers: np.ndarray | None = None,
    seed: int = 0,
    **synth_kwargs,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate a labeled dataset.

    Returns
    -------
    X : (n_samples, n_features) float array of spectra
    y : (n_samples,) int array of class indices
    label_names : list[str] mapping class index → label string
    """
    if wavenumbers is None:
        wavenumbers = make_wavenumber_grid()
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for class_idx, fingerprint in enumerate(classes):
        for _ in range(n_per_class):
            X_list.append(synthesize_spectrum(wavenumbers, fingerprint, rng, **synth_kwargs))
            y_list.append(class_idx)
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int64)
    return X, y, [c.label for c in classes]


# Convenience presets — call these instead of remembering all the realism knobs.

def synthesize_realistic_dataset(
    n_per_class: int = 100,
    classes: Sequence[ClassFingerprint] = DEFAULT_CLASSES,
    wavenumbers: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Realistic biofluid-SERS preset: hot-spot variability ON, position
    jitter ON, FWHM jitter ON, biofluid-style fluorescence ON.

    This is the dataset your classifiers should ultimately have to handle.
    Expect accuracy to drop several points vs the easy preset — that drop
    is informative; it tells you which DSP / classifier knobs need tuning.
    """
    return synthesize_dataset(
        n_per_class=n_per_class,
        classes=classes,
        wavenumbers=wavenumbers,
        seed=seed,
        hotspot_lognormal_sigma=0.45,
        position_jitter_cm1=1.5,
        fwhm_jitter_frac=0.15,
        biofluid_fluorescence=True,
        fluorescence_magnitude=600.0,    # ~2× the easy preset
        cosmic_ray_rate=2.5,
    )
