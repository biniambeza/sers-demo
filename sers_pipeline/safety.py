"""Module 4 — Clinical safety gate ("grey-zone guardrails").

A weak probabilistic prediction MUST NOT propagate into a clinical decision.
The gate intercepts low-margin outputs and replaces them with an explicit
"INCONCLUSIVE — re-run sample" disposition, which the downstream LIS/UI
treats as a non-result requiring action.

What "weak" means here:
    Binary case: the natural margin is the top-class probability itself,
    so grey zone = (low_threshold, high_threshold), e.g. (0.40, 0.60).

    Multiclass case: the top-class probability can be high (>0.6) while the
    runner-up is also high (>0.4). In that case the model is "confidently
    confused." We expose THREE checks and trigger inconclusive on any:

        1. Top probability in [grey_low, grey_high]
        2. Margin between top-1 and top-2 < min_margin
        3. Predictive entropy > max_entropy  (information-theoretic
           uncertainty; complements 1 and 2 for >2 classes)

    Tune thresholds via ROC/Net-Reclassification analysis on a held-out set
    once you have real samples. Reasonable defaults below.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class Disposition(str, Enum):
    CONFIDENT = "confident"
    INCONCLUSIVE = "inconclusive"


@dataclass
class GatedPrediction:
    sample_id: str
    disposition: Disposition
    predicted_class: Optional[int]      # None if inconclusive
    predicted_label: Optional[str]
    top_prob: float
    runner_up_prob: float
    margin: float
    entropy: float
    reasons: list[str]                  # human-readable explanations for the audit log


@dataclass
class GreyZoneGate:
    grey_low: float = 0.40
    grey_high: float = 0.60
    min_margin: float = 0.15            # min gap between top-1 and top-2
    max_entropy: float | None = None    # set e.g. log(n_classes)*0.7 for 3-way

    def evaluate(
        self,
        probs: np.ndarray,
        class_labels: list[str],
        sample_ids: list[str] | None = None,
    ) -> list[GatedPrediction]:
        """Gate a batch of predict_proba outputs.

        Parameters
        ----------
        probs : (n_samples, n_classes) array of calibrated probabilities
        class_labels : human-readable names indexed by class id
        sample_ids : optional list of per-sample identifiers for the audit log
        """
        if sample_ids is None:
            sample_ids = [f"sample_{i:06d}" for i in range(len(probs))]
        if len(sample_ids) != len(probs):
            raise ValueError("sample_ids length must equal number of probability rows")

        results: list[GatedPrediction] = []
        for sid, p in zip(sample_ids, probs):
            results.append(self._evaluate_one(sid, p, class_labels))
        return results

    def _evaluate_one(
        self,
        sample_id: str,
        p: np.ndarray,
        class_labels: list[str],
    ) -> GatedPrediction:
        # Defensive: probabilities should sum to ~1
        p = np.asarray(p, dtype=float)
        if not np.isfinite(p).all() or p.min() < 0:
            return GatedPrediction(
                sample_id=sample_id,
                disposition=Disposition.INCONCLUSIVE,
                predicted_class=None,
                predicted_label=None,
                top_prob=float("nan"),
                runner_up_prob=float("nan"),
                margin=float("nan"),
                entropy=float("nan"),
                reasons=["malformed_probability_vector"],
            )

        order = np.argsort(p)[::-1]
        top_idx = int(order[0])
        top_p = float(p[top_idx])
        runner_p = float(p[order[1]]) if len(order) > 1 else 0.0
        margin = top_p - runner_p
        # Predictive entropy (nats)
        p_safe = p + 1e-12
        entropy = float(-(p_safe * np.log(p_safe)).sum())

        reasons: list[str] = []

        if self.grey_low <= top_p <= self.grey_high:
            reasons.append(f"top_prob_{top_p:.3f}_in_grey_zone[{self.grey_low},{self.grey_high}]")

        if margin < self.min_margin:
            reasons.append(f"margin_{margin:.3f}_below_min_{self.min_margin}")

        if self.max_entropy is not None and entropy > self.max_entropy:
            reasons.append(f"entropy_{entropy:.3f}_above_max_{self.max_entropy}")

        if reasons:
            return GatedPrediction(
                sample_id=sample_id,
                disposition=Disposition.INCONCLUSIVE,
                predicted_class=None,
                predicted_label=None,
                top_prob=top_p,
                runner_up_prob=runner_p,
                margin=margin,
                entropy=entropy,
                reasons=reasons,
            )

        return GatedPrediction(
            sample_id=sample_id,
            disposition=Disposition.CONFIDENT,
            predicted_class=top_idx,
            predicted_label=class_labels[top_idx],
            top_prob=top_p,
            runner_up_prob=runner_p,
            margin=margin,
            entropy=entropy,
            reasons=[],
        )


# ---------------------------------------------------------------------------
# Sensitivity-first triage threshold
# ---------------------------------------------------------------------------
#
# Why this exists: on the HCC serum data, held-out-day testing showed that the
# GreyZoneGate above does not catch missed cancers. The misses are confident
# wrong answers (low probability of disease), not borderline ones, so no
# grey-zone setting flags them. For a triage test the safer design is to move
# the decision threshold itself: choose it so that a target share of known
# positives is referred, and accept more false referrals in exchange.
#
# The threshold MUST be fitted on out-of-fold scores from development data
# (scores the model produced for samples it was not trained on), and then
# evaluated on separate data. Fitting it on in-sample scores makes the model
# look far more sensitive than it is.
#
# A CLEAR result never rules the disease out. It means "this test found no
# reason to refer"; clinical judgement and guidelines still apply.


class TriageResult(str, Enum):
    REFER = "refer"                # send for confirmatory testing
    CLEAR = "clear"                # no reason to refer from this test
    INCONCLUSIVE = "inconclusive"  # the measurement cannot be assessed


@dataclass
class TriageDecision:
    sample_id: str
    result: TriageResult
    score: float                   # model probability of the positive class
    threshold: float
    reasons: list[str]


@dataclass
class TriageThreshold:
    """Decide REFER / CLEAR with a threshold chosen for a target sensitivity.

    Usage:
        t = TriageThreshold(target_sensitivity=0.95)
        t.fit(oof_scores, y_dev)          # out-of-fold scores on development data
        decisions = t.decide(scores_new, sample_ids=ids, qc_passed=qc)
    """

    target_sensitivity: float = 0.95
    threshold: Optional[float] = None
    n_positives_fitted: int = 0
    fitted_sensitivity: float = float("nan")
    fitted_specificity: float = float("nan")

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "TriageThreshold":
        """Pick the highest threshold that still refers >= target of positives.

        scores : probability of the positive class, one per sample (out-of-fold)
        y      : true labels, 1 = disease, 0 = no disease
        """
        scores = np.asarray(scores, dtype=float)
        y = np.asarray(y).astype(int)
        if scores.shape != y.shape:
            raise ValueError("scores and y must have the same length")
        if not 0.0 < self.target_sensitivity <= 1.0:
            raise ValueError("target_sensitivity must be in (0, 1]")
        if not np.isfinite(scores).all():
            raise ValueError("scores must be finite")
        pos = np.sort(scores[y == 1])
        neg = scores[y == 0]
        if len(pos) == 0:
            raise ValueError("cannot fit a sensitivity threshold without positive samples")

        # Allow at most floor((1 - target) * n_pos) positives below the threshold.
        k = int(np.floor((1.0 - self.target_sensitivity) * len(pos) + 1e-9))
        self.threshold = float(pos[k])
        self.n_positives_fitted = int(len(pos))
        self.fitted_sensitivity = float((pos >= self.threshold).mean())
        self.fitted_specificity = (
            float((neg < self.threshold).mean()) if len(neg) else float("nan")
        )
        return self

    def decide(
        self,
        scores: np.ndarray,
        sample_ids: list[str] | None = None,
        qc_passed: np.ndarray | None = None,
    ) -> list[TriageDecision]:
        """Apply the fitted threshold.

        qc_passed : optional boolean per sample; False -> INCONCLUSIVE
                    (e.g. spectrum failed validation, run calibration out of limits).
        """
        if self.threshold is None:
            raise RuntimeError("TriageThreshold.fit() must be called before decide()")
        scores = np.asarray(scores, dtype=float)
        if sample_ids is None:
            sample_ids = [f"sample_{i:06d}" for i in range(len(scores))]
        if len(sample_ids) != len(scores):
            raise ValueError("sample_ids length must equal number of scores")
        if qc_passed is None:
            qc_passed = np.ones(len(scores), dtype=bool)
        qc_passed = np.asarray(qc_passed, dtype=bool)
        if len(qc_passed) != len(scores):
            raise ValueError("qc_passed length must equal number of scores")

        out: list[TriageDecision] = []
        for sid, s, ok in zip(sample_ids, scores, qc_passed):
            if not np.isfinite(s) or s < 0 or s > 1:
                out.append(TriageDecision(sid, TriageResult.INCONCLUSIVE, float(s),
                                          self.threshold, ["malformed_score"]))
            elif not ok:
                out.append(TriageDecision(sid, TriageResult.INCONCLUSIVE, float(s),
                                          self.threshold, ["quality_control_failed"]))
            elif s >= self.threshold:
                out.append(TriageDecision(sid, TriageResult.REFER, float(s),
                                          self.threshold,
                                          [f"score_{s:.3f}_at_or_above_{self.threshold:.3f}"]))
            else:
                out.append(TriageDecision(sid, TriageResult.CLEAR, float(s),
                                          self.threshold,
                                          [f"score_{s:.3f}_below_{self.threshold:.3f}"]))
        return out
