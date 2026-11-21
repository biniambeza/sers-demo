"""Module 5 — Compliance scaffolding.

Important framing for your team:
    The patterns below are *necessary but nowhere near sufficient* for
    CLIA/FDA CSA compliance of a clinical LDT. They put the codebase into
    a defensible shape (input validation, model integrity, tamper-resistant
    audit logging) so that the *artifacts you produce during R&D* are usable
    later when you build out a full QMS. They do NOT constitute a validation
    package — that needs analytical/clinical studies, risk analysis (ISO
    14971), software lifecycle docs (IEC 62304), and design controls.

What this module provides:
    1. SpectrumValidator — rejects malformed input arrays before they
       reach the DSP / model. Flatlines, NaN/Inf, clipped/saturated arrays,
       wrong shape, out-of-physical-range values.
    2. ModelRegistry — SHA-256 hash of model files at load time; refuses
       to use a model whose hash doesn't match the registered value.
       Prevents silent model swap or corruption.
    3. AuditLogger — append-only JSON-lines log with hash chaining.
       Each entry includes the SHA-256 of the previous entry; you can detect
       tampering by re-walking the chain. The log file is opened in append-
       only mode and `fsync`'d after each write.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Input validation                                                             #
# --------------------------------------------------------------------------- #

class InvalidSpectrumError(ValueError):
    """Raised when a spectrum fails defensive validation."""


@dataclass
class SpectrumValidator:
    """Rejects input arrays that cannot be safely classified.

    Each check returns a short error code (machine-parsable) plus a
    human-readable message. We raise on the first failure rather than
    collecting all failures — fail fast is safer in a clinical context.
    """
    expected_length: int                       # must match training grid
    expected_dtype: type = np.floating
    min_intensity: float = -1e6                # paranoid lower bound
    max_intensity: float = 1e9                 # detector saturation guard
    min_variance: float = 1e-6                 # flatline detector
    max_clipped_fraction: float = 0.01         # fraction of pixels at max allowed

    def validate(self, x: np.ndarray) -> None:
        if not isinstance(x, np.ndarray):
            raise InvalidSpectrumError("E_TYPE: input is not a numpy array")
        if x.ndim != 1:
            raise InvalidSpectrumError(f"E_NDIM: expected 1D, got {x.ndim}D")
        if x.shape[0] != self.expected_length:
            raise InvalidSpectrumError(
                f"E_LENGTH: expected {self.expected_length} points, got {x.shape[0]}"
            )
        if not np.issubdtype(x.dtype, self.expected_dtype):
            raise InvalidSpectrumError(f"E_DTYPE: expected float, got {x.dtype}")
        if not np.isfinite(x).all():
            n_nan = int(np.isnan(x).sum())
            n_inf = int(np.isinf(x).sum())
            raise InvalidSpectrumError(f"E_NONFINITE: nan={n_nan} inf={n_inf}")
        if x.min() < self.min_intensity or x.max() > self.max_intensity:
            raise InvalidSpectrumError(
                f"E_RANGE: values in [{x.min()}, {x.max()}] outside "
                f"allowed [{self.min_intensity}, {self.max_intensity}]"
            )
        if x.var() < self.min_variance:
            raise InvalidSpectrumError(
                f"E_FLATLINE: variance {x.var():.3e} below {self.min_variance}"
            )
        # Clipped/saturated detection: many pixels at exactly the max value
        max_val = x.max()
        clipped_fraction = float(np.mean(x >= max_val - 1e-9))
        if clipped_fraction > self.max_clipped_fraction:
            raise InvalidSpectrumError(
                f"E_CLIPPED: {clipped_fraction:.3f} of pixels at saturation "
                f"(threshold {self.max_clipped_fraction})"
            )

    def validate_batch(self, X: np.ndarray) -> None:
        if X.ndim != 2:
            raise InvalidSpectrumError(f"E_NDIM_BATCH: expected 2D, got {X.ndim}D")
        for i in range(X.shape[0]):
            try:
                self.validate(X[i])
            except InvalidSpectrumError as e:
                raise InvalidSpectrumError(f"row {i}: {e}") from None


# --------------------------------------------------------------------------- #
# Model registry: SHA-256 integrity check                                      #
# --------------------------------------------------------------------------- #

def sha256_file(path: str | Path, chunk: int = 1 << 16) -> str:
    """SHA-256 of a file, computed in chunks so large model files stream."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


@dataclass
class ModelEntry:
    name: str
    path: str
    sha256: str
    trained_at: str
    n_classes: int
    classes: list[str]
    notes: str = ""


class ModelRegistry:
    """Trivial JSON-backed registry. Tracks expected hash per model.

    Workflow:
        registry.register(name, path, ...)  — after training, before deployment
        model = registry.load(name)         — at runtime; raises if hash mismatches

    In production this lives alongside your CI artifacts. The JSON file itself
    should be version-controlled and ideally signed.

    Optional HMAC mode (`hmac_key` kwarg or `SERS_REGISTRY_HMAC_KEY` env var):
        Each entry carries a `hmac_sha256` field computed over the entry's
        canonical JSON. A tampered entry — including swapping the path to
        point at a different model file with a different real hash — fails
        verification. The key should be stored outside the repo (env var,
        secret manager) and rotated when compromised.
    """

    ENV_KEY = "SERS_REGISTRY_HMAC_KEY"

    def __init__(self, registry_path: str | Path, hmac_key: Optional[bytes] = None):
        self.path = Path(registry_path)
        self._lock = threading.Lock()
        if hmac_key is None:
            env_val = os.environ.get(self.ENV_KEY)
            hmac_key = env_val.encode("utf-8") if env_val else None
        self._hmac_key = hmac_key
        if not self.path.exists():
            self.path.write_text(json.dumps({"entries": {}}, indent=2))

    def _read(self) -> dict:
        return json.loads(self.path.read_text())

    def _write(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True))

    def _sign(self, entry_dict: dict) -> Optional[str]:
        if self._hmac_key is None:
            return None
        canonical = json.dumps(entry_dict, sort_keys=True, separators=(",", ":"))
        import hmac
        return hmac.new(self._hmac_key, canonical.encode("utf-8"), hashlib.sha256).hexdigest()

    def register(self, entry: ModelEntry) -> None:
        with self._lock:
            data = self._read()
            entry_dict = asdict(entry)
            sig = self._sign(entry_dict)
            if sig is not None:
                entry_dict["hmac_sha256"] = sig
            data["entries"][entry.name] = entry_dict
            self._write(data)

    def verify(self, name: str) -> ModelEntry:
        data = self._read()
        if name not in data["entries"]:
            raise KeyError(f"Model '{name}' not in registry")
        raw = dict(data["entries"][name])
        signature = raw.pop("hmac_sha256", None)
        # If a key is configured OR the entry was signed, demand a valid signature
        if self._hmac_key is not None or signature is not None:
            if self._hmac_key is None:
                raise RuntimeError(
                    f"Entry '{name}' is signed but no HMAC key configured "
                    f"(set ${self.ENV_KEY})"
                )
            if signature is None:
                raise RuntimeError(f"Entry '{name}' missing HMAC signature")
            expected_sig = self._sign(raw)
            import hmac as _hmac
            if not _hmac.compare_digest(expected_sig or "", signature):
                raise RuntimeError(f"Entry '{name}' HMAC mismatch — registry tampered")
        entry = ModelEntry(**raw)
        actual = sha256_file(entry.path)
        if actual != entry.sha256:
            raise RuntimeError(
                f"MODEL INTEGRITY FAILURE for '{name}': "
                f"expected {entry.sha256[:16]}..., got {actual[:16]}..."
            )
        return entry


# --------------------------------------------------------------------------- #
# Audit logger: append-only, hash-chained JSON lines                           #
# --------------------------------------------------------------------------- #

class AuditLogger:
    """Append-only audit trail.

    Each line is a JSON object containing:
        ts_iso      — ISO-8601 UTC timestamp
        event       — short event name
        payload     — arbitrary JSON-serializable dict
        prev_sha256 — SHA-256 of the previous line (or 64 zeros if first)
        sha256      — SHA-256 of this line MINUS the sha256 field itself

    Tampering check: walk the file top-to-bottom, recompute each line's hash,
    and verify that each `prev_sha256` matches. Any silent edit will break
    the chain at the edited line.

    fsync after every write so logs survive crashes. This is intentionally
    slow for safety; batch your logging at the caller if throughput matters.
    """

    GENESIS_HASH = "0" * 64

    def __init__(self, log_path: str | Path):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()

    def _last_hash(self) -> str:
        if self.path.stat().st_size == 0:
            return self.GENESIS_HASH
        # Read final line
        with self.path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            # Walk backwards to find the last newline
            buf_size = min(4096, size)
            f.seek(size - buf_size)
            tail = f.read(buf_size).decode("utf-8", errors="replace")
        last_line = tail.strip().splitlines()[-1]
        return json.loads(last_line)["sha256"]

    def log(self, event: str, payload: dict[str, Any]) -> str:
        with self._lock:
            prev = self._last_hash()
            entry: dict[str, Any] = {
                "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": event,
                "payload": payload,
                "host": platform.node(),
                "pid": os.getpid(),
                "prev_sha256": prev,
            }
            # Compute self-hash over canonical JSON (sorted keys, no trailing space)
            canonical = json.dumps(entry, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            entry["sha256"] = digest
            line = json.dumps(entry, sort_keys=True) + "\n"
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            return digest

    def verify_chain(self) -> tuple[bool, int]:
        """Walk the chain. Returns (is_intact, line_number_of_first_break)."""
        prev = self.GENESIS_HASH
        line_no = 0
        with self.path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                entry = json.loads(raw)
                stored_self = entry.pop("sha256")
                stored_prev = entry["prev_sha256"]
                canonical = json.dumps(entry, sort_keys=True, separators=(",", ":"))
                computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                if stored_prev != prev or stored_self != computed:
                    return False, line_no
                prev = stored_self
        return True, line_no

    def merkle_root(self) -> str:
        """Compute a Merkle root over all log entries' self-hashes.

        Why bother when we already have hash-chaining? Two reasons:

            1. Snapshot publication: you can periodically commit the
               root hash to an *external* immutable store (git tag, an
               internal blockchain, a notary service, a signed receipt
               from your QA system). Anyone who has the root can later
               verify that the log was untouched at that point in time.
            2. Efficient range proofs: with a Merkle tree you can prove
               a single entry was present without disclosing the whole
               log — useful when an auditor wants to verify one specific
               sample's audit trail without seeing other patients' data.

        Implementation is a standard binary Merkle tree with SHA-256.
        Empty log → returns the genesis hash.
        """
        leaves: list[bytes] = []
        with self.path.open("r", encoding="utf-8") as f:
            for raw in f:
                entry = json.loads(raw)
                leaves.append(bytes.fromhex(entry["sha256"]))
        if not leaves:
            return self.GENESIS_HASH
        # Duplicate the last leaf if odd count, then iteratively hash pairs
        level = leaves
        while len(level) > 1:
            if len(level) % 2 == 1:
                level = level + [level[-1]]
            next_level = []
            for i in range(0, len(level), 2):
                h = hashlib.sha256(level[i] + level[i + 1]).digest()
                next_level.append(h)
            level = next_level
        return level[0].hex()
