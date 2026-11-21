"""Minimal reader for Thermo Galactic SPC files (the "new" LSB format, version 0x4B).

Covers the common single-spectrum case: one subfile, evenly spaced x values
(ffirst..flast), y stored either as IEEE float32 or as scaled int32.
Multi-file and XY-array variants raise NotImplementedError rather than being
read wrongly.

Header layout (512 bytes, little-endian):
    0   ftflgs  uint8   file type flags
    1   fversn  uint8   0x4B = new LSB format
    2   fexper  uint8
    3   fexp    int8    y exponent; 0x80 (-128) means y is float32
    4   fnpts   int32   number of points
    8   ffirst  float64 first x value
    16  flast   float64 last x value
    24  fnsub   int32   number of subfiles
Subfile header (32 bytes) follows; byte 1 is subexp, which overrides fexp.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

_TXYXYS = 0x40   # each subfile has its own x array
_TXVALS = 0x80   # a single x array precedes the subfiles
_TMULTI = 0x04   # multiple y arrays


def read_spc(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) for a single-spectrum SPC file."""
    raw = Path(path).read_bytes()
    if len(raw) < 512 + 32:
        raise ValueError(f"{path}: too short for an SPC file")
    ftflgs, fversn, _fexper, fexp = struct.unpack_from("<BBBb", raw, 0)
    if fversn != 0x4B:
        raise NotImplementedError(f"{path}: SPC version 0x{fversn:02X} not supported (need 0x4B)")
    if ftflgs & (_TXYXYS | _TXVALS | _TMULTI):
        raise NotImplementedError(f"{path}: XY or multi-file SPC not supported (flags 0x{ftflgs:02X})")
    (fnpts,) = struct.unpack_from("<i", raw, 4)
    ffirst, flast = struct.unpack_from("<dd", raw, 8)
    if fnpts <= 1:
        raise ValueError(f"{path}: invalid point count {fnpts}")

    sub = 512
    (subexp,) = struct.unpack_from("<b", raw, sub + 1)
    exp = subexp if subexp != 0 else fexp
    start = sub + 32
    end = start + 4 * fnpts
    if end > len(raw):
        raise ValueError(f"{path}: file ends before {fnpts} points")

    if exp == -128:                         # 0x80: IEEE float32
        y = np.frombuffer(raw, dtype="<f4", count=fnpts, offset=start).astype(float)
    else:                                   # scaled int32: y = int * 2^exp / 2^32
        ints = np.frombuffer(raw, dtype="<i4", count=fnpts, offset=start).astype(float)
        y = ints * (2.0 ** exp) / (2.0 ** 32)
    x = np.linspace(ffirst, flast, fnpts)
    return x, y
