"""Vendor-specific file parsers.

Real instruments emit text files in vendor-specific formats. The base parser
in ingestion.py handles the common case (multi-line header + two columns of
numbers) but vendors add quirks:

    * Renishaw WiRE — exports as tab-separated, header is "#"-prefixed,
      sometimes includes acquisition metadata in key=value form.
    * Horiba LabSpec — exports as semicolon-separated by default in EU
      locales (decimal comma!), header block delimited by "#Acq.".
    * Ocean Insight (formerly Ocean Optics) — simple two-column .txt with
      ">>>>>Begin Spectral Data<<<<<" sentinel line.
    * Generic CSV — what your competitor's data dump probably looks like.

Adapter pattern:
    Each vendor implements `VendorParser` and exposes `.detect(path)` for
    auto-detection plus `.parse(path)` for extraction. A `VendorRegistry`
    tries detectors in order; the first match wins. Fall back to the
    generic two-column parser from ingestion.py if nothing matches.

This module deliberately uses *only stdlib + numpy* — vendor file parsing
should never need exotic dependencies, and you'll want to read it during
an audit.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .ingestion import parse_spectrum_file as _generic_parse


@dataclass
class ParseResult:
    wavenumbers: np.ndarray
    intensities: np.ndarray
    metadata: dict
    vendor: str
    source_path: str


class VendorParser(ABC):
    name: str = "unknown"

    @abstractmethod
    def detect(self, path: Path, head_bytes: bytes) -> bool:
        """Return True if this parser thinks it can handle the file.

        `head_bytes` is the first ~4KB of the file; use it to look for vendor
        signatures without re-reading the file in every detector.
        """

    @abstractmethod
    def parse(self, path: Path) -> ParseResult:
        ...


# --------------------------------------------------------------------------- #
# Renishaw WiRE text export                                                    #
# --------------------------------------------------------------------------- #

class RenishawWiREParser(VendorParser):
    """Renishaw inVia / WiRE text exports.

    Typical structure:
        # Renishaw inVia Reflex
        # Operator: ...
        # Laser: 785nm
        # Acquisition: 5s x 3
        # Date: 2025-...
        500.0	123.4
        501.0	145.6
        ...
    """
    name = "renishaw_wire"

    _signature = re.compile(rb"#\s*Renishaw|#\s*WiRE", re.IGNORECASE)
    _meta_line = re.compile(r"^#\s*([A-Za-z][A-Za-z0-9 _.-]*)\s*[:=]\s*(.+?)\s*$")

    def detect(self, path: Path, head_bytes: bytes) -> bool:
        return bool(self._signature.search(head_bytes))

    def parse(self, path: Path) -> ParseResult:
        metadata: dict = {}
        data_lines: list[str] = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("#"):
                    m = self._meta_line.match(line)
                    if m:
                        metadata[m.group(1).strip().lower()] = m.group(2).strip()
                    continue
                if line.strip():
                    data_lines.append(line)
        if not data_lines:
            raise ValueError(f"{path}: no data lines after header")
        arr = np.loadtxt(data_lines, delimiter=None)
        wn, intens = arr[:, 0], arr[:, 1]
        if wn[0] > wn[-1]:
            wn, intens = wn[::-1], intens[::-1]
        return ParseResult(
            wavenumbers=wn, intensities=intens,
            metadata=metadata, vendor=self.name, source_path=str(path),
        )


# --------------------------------------------------------------------------- #
# Horiba LabSpec text export                                                   #
# --------------------------------------------------------------------------- #

class HoribaLabSpecParser(VendorParser):
    """Horiba LabSpec text exports.

    Often semicolon-separated with EU decimal comma. Header block starts
    with "#Acq." or "#Title=".
    """
    name = "horiba_labspec"

    _signature = re.compile(rb"#Acq\.|#Title=|LabSpec", re.IGNORECASE)

    def detect(self, path: Path, head_bytes: bytes) -> bool:
        return bool(self._signature.search(head_bytes))

    def parse(self, path: Path) -> ParseResult:
        metadata: dict = {}
        data_lines: list[str] = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    # "#Key=Value" or "#Key: Value"
                    body = stripped.lstrip("#").strip()
                    for sep in ("=", ":"):
                        if sep in body:
                            k, v = body.split(sep, 1)
                            metadata[k.strip().lower()] = v.strip()
                            break
                    continue
                data_lines.append(stripped)
        if not data_lines:
            raise ValueError(f"{path}: no data lines after header")

        # Decide delimiter: prefer semicolon if present, fall back to whitespace
        delim = ";" if ";" in data_lines[0] else None
        # Replace EU decimal comma with period
        normalized = [ln.replace(",", ".") for ln in data_lines] if delim == ";" else data_lines
        arr = np.loadtxt(normalized, delimiter=delim)
        wn, intens = arr[:, 0], arr[:, 1]
        if wn[0] > wn[-1]:
            wn, intens = wn[::-1], intens[::-1]
        return ParseResult(
            wavenumbers=wn, intensities=intens,
            metadata=metadata, vendor=self.name, source_path=str(path),
        )


# --------------------------------------------------------------------------- #
# Ocean Insight / Ocean Optics                                                 #
# --------------------------------------------------------------------------- #

class OceanInsightParser(VendorParser):
    """Ocean Insight / Ocean Optics ASCII exports.

    Uses a sentinel line to mark the start of data:
        >>>>>Begin Spectral Data<<<<<
    """
    name = "ocean_insight"

    _signature_begin = re.compile(rb"Begin Spectral Data", re.IGNORECASE)

    def detect(self, path: Path, head_bytes: bytes) -> bool:
        return bool(self._signature_begin.search(head_bytes))

    def parse(self, path: Path) -> ParseResult:
        metadata: dict = {}
        data_lines: list[str] = []
        started = False
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not started:
                    if "Begin Spectral Data" in line:
                        started = True
                        continue
                    if ":" in line:
                        k, v = line.split(":", 1)
                        metadata[k.strip().lower()] = v.strip()
                    continue
                if "End Spectral Data" in line:
                    break
                if line.strip():
                    data_lines.append(line)
        if not data_lines:
            raise ValueError(f"{path}: 'Begin Spectral Data' present but no data follows")
        arr = np.loadtxt(data_lines, delimiter=None)
        wn, intens = arr[:, 0], arr[:, 1]
        if wn[0] > wn[-1]:
            wn, intens = wn[::-1], intens[::-1]
        return ParseResult(
            wavenumbers=wn, intensities=intens,
            metadata=metadata, vendor=self.name, source_path=str(path),
        )


# --------------------------------------------------------------------------- #
# Generic fallback                                                             #
# --------------------------------------------------------------------------- #

class GenericTwoColumnParser(VendorParser):
    """Fallback: trust the heuristic two-column parser in ingestion.py."""
    name = "generic"

    def detect(self, path: Path, head_bytes: bytes) -> bool:
        return True  # always matches; must be LAST in the registry

    def parse(self, path: Path) -> ParseResult:
        wn, intens, header = _generic_parse(path)
        return ParseResult(
            wavenumbers=wn, intensities=intens,
            metadata={"raw_header_lines": header},
            vendor=self.name, source_path=str(path),
        )


# --------------------------------------------------------------------------- #
# Registry / dispatcher                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class VendorRegistry:
    parsers: list[VendorParser] = field(default_factory=list)

    @classmethod
    def default(cls) -> "VendorRegistry":
        return cls(parsers=[
            RenishawWiREParser(),
            HoribaLabSpecParser(),
            OceanInsightParser(),
            GenericTwoColumnParser(),   # MUST be last — its detect() returns True always
        ])

    def parse(self, path: str | Path, head_size: int = 4096) -> ParseResult:
        path = Path(path)
        with path.open("rb") as f:
            head = f.read(head_size)
        for parser in self.parsers:
            if parser.detect(path, head):
                return parser.parse(path)
        raise RuntimeError(
            f"No parser matched {path}. This should be impossible because "
            "GenericTwoColumnParser.detect always returns True — check the registry order."
        )
