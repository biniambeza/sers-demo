"""Module 1 — Data ingestion & wavenumber-grid alignment.

Real-world raw SERS files come from many vendors (Renishaw WiRE, Horiba LabSpec,
WITec, B&W Tek, in-house LabVIEW dumps...) and each emits a slightly different
text format. Common patterns:

    * Multi-line header (metadata, acquisition params, comments) before data
    * Two-column text: wavenumber, intensity — sometimes whitespace-separated,
      sometimes comma, sometimes tab
    * Wavenumber axis is NOT identical across files — sampling step varies
      between 0.5 and 2 cm^-1, start/end wavenumbers differ

This module:
    (a) sniffs the header / detects the first numeric line robustly,
    (b) parses the two-column data,
    (c) interpolates onto a *single canonical grid* that downstream code expects.

Interpolation rationale:
    A classifier trained on a grid of 1401 points (400..1800, 1 cm^-1) cannot
    accept a 700-point input. Resampling with cubic interpolation preserves
    peak shape better than nearest/linear for well-sampled Raman peaks, but
    can introduce ringing near sharp features — linear is the safer default
    when in doubt. Make this configurable.
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.interpolate import interp1d


# A line is considered numeric data if it contains exactly two numeric tokens
# (wavenumber, intensity). Tokens may be separated by comma/tab/whitespace.
_NUM_RE = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")
_SEP_RE = re.compile(r"[,\t\s]+")


def _is_data_line(line: str) -> bool:
    parts = _SEP_RE.split(line.strip())
    if len(parts) != 2:
        return False
    return all(_NUM_RE.match(p) for p in parts)


def parse_spectrum_file(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Parse a raw SERS text file.

    Strategy: scan line-by-line; once we hit a numeric two-column line, all
    *preceding* lines are header. We do NOT assume the header has a known
    delimiter — many instruments emit free-form text.

    Returns
    -------
    wavenumbers : (n,) float array (NOT yet interpolated)
    intensities : (n,) float array
    header_lines : list[str] (preserved for audit)
    """
    path = Path(path)
    header_lines: list[str] = []
    data_lines: list[str] = []
    data_started = False

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not data_started:
                if _is_data_line(line):
                    data_started = True
                    data_lines.append(line)
                else:
                    header_lines.append(line.rstrip("\n"))
            else:
                if line.strip() == "":
                    continue
                if _is_data_line(line):
                    data_lines.append(line)
                else:
                    # Trailing footer; stop reading
                    break

    if not data_lines:
        raise ValueError(f"No numeric two-column data found in {path}")

    # Sniff delimiter from the first data line
    first = data_lines[0]
    if "," in first:
        delim = ","
    elif "\t" in first:
        delim = "\t"
    elif ";" in first:
        delim = ";"
    else:
        delim = None  # whitespace
    arr = np.loadtxt(io.StringIO("".join(data_lines)), delimiter=delim)
    if arr.ndim == 1:
        # Single line — pathological but handle gracefully
        arr = arr.reshape(1, -1)

    wavenumbers = arr[:, 0].astype(float)
    intensities = arr[:, 1].astype(float)

    # Some vendors emit decreasing wavenumber order; canonicalize to ascending
    if wavenumbers[0] > wavenumbers[-1]:
        wavenumbers = wavenumbers[::-1]
        intensities = intensities[::-1]

    return wavenumbers, intensities, header_lines


def interpolate_to_grid(
    src_wavenumbers: np.ndarray,
    src_intensities: np.ndarray,
    target_grid: np.ndarray,
    kind: str = "linear",
    extrapolation_fill: float | None = 0.0,
) -> np.ndarray:
    """Resample a spectrum onto the canonical target grid.

    Parameters
    ----------
    kind : "linear" | "cubic" | "quadratic"
        "linear" is the safe default. "cubic" preserves peak shape but can
        ring near sharp edges and amplify noise.
    extrapolation_fill : float | None
        Value to use outside src range. None → raise on out-of-range.

    Out-of-range target points should be rare (you should configure your
    target grid to lie inside what your instrument actually produces) but
    we handle it explicitly so silent garbage values can't slip through.
    """
    if extrapolation_fill is None:
        if target_grid.min() < src_wavenumbers.min() or target_grid.max() > src_wavenumbers.max():
            raise ValueError(
                f"target grid [{target_grid.min()}, {target_grid.max()}] "
                f"is outside source range [{src_wavenumbers.min()}, {src_wavenumbers.max()}]"
            )
        f = interp1d(src_wavenumbers, src_intensities, kind=kind, assume_sorted=True)
    else:
        f = interp1d(
            src_wavenumbers,
            src_intensities,
            kind=kind,
            assume_sorted=True,
            bounds_error=False,
            fill_value=extrapolation_fill,
        )
    return f(target_grid).astype(float)


def load_batch(
    paths: Iterable[str | Path],
    target_grid: np.ndarray,
    kind: str = "linear",
) -> tuple[np.ndarray, list[dict]]:
    """Load many files and stack them into a (n_files, n_grid_points) matrix.

    Returns the matrix plus a list of per-file metadata dicts (for the audit log).
    """
    rows: list[np.ndarray] = []
    metadata: list[dict] = []
    for p in paths:
        wn, intens, header = parse_spectrum_file(p)
        row = interpolate_to_grid(wn, intens, target_grid, kind=kind)
        rows.append(row)
        metadata.append({
            "path": str(p),
            "n_source_points": len(wn),
            "source_range": (float(wn.min()), float(wn.max())),
            "header_line_count": len(header),
        })
    return np.vstack(rows), metadata
