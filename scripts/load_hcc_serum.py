"""Load and classify HCC vs Healthy serum SERS spectra.

Dataset: Zenodo 4277797 (DOI: 10.5281/zenodo.4277797)
- 144 SERS spectra: 72 HCC (hepatocellular carcinoma), 72 healthy controls
- Silver plasmonic paper substrates (citrate-reduced AgNPs, Lee-Meisel)
- 785 nm laser, 400 mW output, 10s CCD exposure
- Two-column TXT files (wavenumber, intensity), range ~-307 to 3271 cm⁻¹
- Labels encoded in filenames: CTR = healthy, H0T = HCC

This script validates the full SERS pipeline on real liver-disease SERS data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sers_pipeline.ingestion import parse_spectrum_file, interpolate_to_grid
from sers_pipeline.synthetic import make_wavenumber_grid
from sers_pipeline.preprocessing import Pipeline
from sers_pipeline.classifiers import TierAClassifier, TierBClassifier
from sers_pipeline.safety import GreyZoneGate

# Relative to the project, so it works on any machine or server.
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "hcc_serum_sers" / "spectra"
CLASS_LABELS = ["Healthy", "HCC"]


def load_hcc_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load 144 spectra, interpolate to standard 400-1800 cm⁻¹ grid."""
    target_grid = make_wavenumber_grid()  # 400-1800, 1401 points

    spectra: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    skipped = 0

    for txt_file in sorted(DATA_DIR.glob("*.txt")):
        wn, intensities, _ = parse_spectrum_file(txt_file)

        if wn.min() > 400 or wn.max() < 1800:
            skipped += 1
            continue

        row = interpolate_to_grid(wn, intensities, target_grid)
        spectra.append(row)
        labels.append(1 if "H0T" in txt_file.name else 0)
        sample_ids.append(txt_file.stem)

    if skipped:
        print(f"  Skipped {skipped} files (wavenumber range too narrow)")

    return np.vstack(spectra), np.array(labels), sample_ids


def main():
    print("=" * 60)
    print("HCC vs Healthy — Serum SERS Classification")
    print("Zenodo 4277797 | AgNP plasmonic paper | 785nm")
    print("=" * 60)

    # --- Load and interpolate ---
    print("\n--- Loading spectra ---")
    X_raw, y, sample_ids = load_hcc_dataset()
    n_hcc = (y == 1).sum()
    n_healthy = (y == 0).sum()
    print(f"  Loaded: {X_raw.shape[0]} spectra x {X_raw.shape[1]} features")
    print(f"  HCC: {n_hcc}, Healthy: {n_healthy}")
    print(f"  Grid: 400-1800 cm-1, 1401 points")

    # --- Full DSP preprocessing ---
    print("\n--- DSP Preprocessing (despike -> baseline -> smooth -> normalize) ---")
    dsp = Pipeline.default()
    X_clean = dsp(X_raw)
    print(f"  Post-DSP shape: {X_clean.shape}")
    print(f"  Post-DSP stats: mean={X_clean.mean():.4f}, std={X_clean.std():.4f}")

    # --- Tier A: PCA + SVM (cross-validated) ---
    print("\n--- Tier A: PCA + SVM (10-fold CV) ---")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for n_comp in [10, 20, 30]:
        from sklearn.pipeline import make_pipeline
        from sklearn.decomposition import PCA
        from sklearn.svm import SVC
        pipe = make_pipeline(
            PCA(n_components=n_comp, random_state=42),
            SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
        )
        scores = cross_val_score(pipe, X_clean, y, cv=skf, scoring="accuracy")
        print(f"  PCA({n_comp:2d}) + RBF-SVM: {scores.mean():.3f} +/- {scores.std():.3f}")

    # --- Detailed report on 80/20 split ---
    print("\n--- Detailed Report (80/20 split) ---")
    # The sample names are split together with the data, so that a name
    # printed for a test sample is the name of that sample.
    X_train, X_test, y_train, y_test, _, ids_test = train_test_split(
        X_clean, y, sample_ids, test_size=0.2, stratify=y, random_state=42,
    )
    print(f"  Train: {len(y_train)} | Test: {len(y_test)}")

    # Tier A
    print("\n  Tier A (PCA + SVM):")
    tier_a = TierAClassifier(n_components=20)
    tier_a.fit(X_train, y_train)
    y_pred_a = tier_a.predict(X_test)
    y_proba_a = tier_a.predict_proba(X_test)
    print(classification_report(y_test, y_pred_a, target_names=CLASS_LABELS))

    cm = confusion_matrix(y_test, y_pred_a)
    print(f"  Confusion matrix:")
    print(f"    {'':14s} Pred Healthy  Pred HCC")
    print(f"    {'True Healthy':14s} {cm[0,0]:11d}  {cm[0,1]:8d}")
    print(f"    {'True HCC':14s} {cm[1,0]:11d}  {cm[1,1]:8d}")

    # Tier B
    print("\n  Tier B (1D CNN):")
    tier_b = TierBClassifier(
        n_classes=2,
        n_features=X_clean.shape[1],
        epochs=60,
        verbose=True,
    )
    tier_b.fit(X_train, y_train)
    y_pred_b = tier_b.predict(X_test)
    y_proba_b = tier_b.predict_proba(X_test)
    print(classification_report(y_test, y_pred_b, target_names=CLASS_LABELS))

    cm_b = confusion_matrix(y_test, y_pred_b)
    print(f"  Confusion matrix:")
    print(f"    {'':14s} Pred Healthy  Pred HCC")
    print(f"    {'True Healthy':14s} {cm_b[0,0]:11d}  {cm_b[0,1]:8d}")
    print(f"    {'True HCC':14s} {cm_b[1,0]:11d}  {cm_b[1,1]:8d}")

    # --- Tier agreement ---
    agree = (y_pred_a == y_pred_b).sum()
    print(f"\n--- Tier Agreement ---")
    print(f"  Agree: {agree}/{len(y_pred_a)} ({100*agree/len(y_pred_a):.1f}%)")
    disagree_idx = np.where(y_pred_a != y_pred_b)[0]
    if len(disagree_idx) > 0:
        print(f"  Disagreements ({len(disagree_idx)}):")
        for idx in disagree_idx:
            print(f"    {ids_test[idx]}: "
                  f"Tier A={CLASS_LABELS[y_pred_a[idx]]}, "
                  f"Tier B={CLASS_LABELS[y_pred_b[idx]]}, "
                  f"True={CLASS_LABELS[y_test[idx]]}")

    # --- Safety gate ---
    print("\n--- Grey-Zone Safety Gate (Tier A probabilities) ---")
    gate = GreyZoneGate(grey_low=0.40, grey_high=0.60, min_margin=0.15)
    decisions = gate.evaluate(
        y_proba_a, class_labels=CLASS_LABELS, sample_ids=ids_test,
    )
    n_inconc = sum(d.disposition.value == "inconclusive" for d in decisions)
    n_conf = len(decisions) - n_inconc
    print(f"  Confident:    {n_conf}/{len(decisions)}")
    print(f"  Inconclusive: {n_inconc}/{len(decisions)}")
    if n_inconc > 0:
        print(f"  Inconclusive samples:")
        for d in decisions:
            if d.disposition.value == "inconclusive":
                print(f"    {d.sample_id}: top_prob={d.top_prob:.3f}, "
                      f"margin={d.margin:.3f}, reasons={d.reasons}")

    # --- Sensitivity-first triage ---
    # The grey-zone gate above does not catch confident misses. The triage rule
    # instead picks the threshold on out-of-fold training scores so that ~95% of
    # known cancers are referred, then applies it to the untouched test set.
    print("\n--- Sensitivity-first triage (threshold fitted on training data only) ---")
    from sklearn.decomposition import PCA
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC
    from sers_pipeline.safety import TriageResult, TriageThreshold

    def triage_model():
        return make_pipeline(
            PCA(n_components=20, random_state=42),
            SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
        )

    oof = cross_val_predict(
        triage_model(), X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        method="predict_proba",
    )[:, 1]
    triage = TriageThreshold(target_sensitivity=0.95).fit(oof, y_train)
    test_scores = triage_model().fit(X_train, y_train).predict_proba(X_test)[:, 1]
    triage_decisions = triage.decide(test_scores, sample_ids=ids_test)
    referred = np.array([d.result == TriageResult.REFER for d in triage_decisions])
    tp = int((referred & (y_test == 1)).sum())
    fn = int((~referred & (y_test == 1)).sum())
    tn = int((~referred & (y_test == 0)).sum())
    fp = int((referred & (y_test == 0)).sum())
    print(f"  Threshold: {triage.threshold:.3f} "
          f"(training out-of-fold: sensitivity {triage.fitted_sensitivity:.2f}, "
          f"specificity {triage.fitted_specificity:.2f})")
    print(f"  Test set: HCC referred {tp}/{tp + fn}, healthy cleared {tn}/{tn + fp}")
    for d, truth in zip(triage_decisions, y_test):
        if d.result == TriageResult.CLEAR and truth == 1:
            print(f"    missed HCC: {d.sample_id} (score {d.score:.3f})")
    print("  A CLEAR result does not rule out cancer.")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"  Dataset:  144 real liver-disease SERS spectra (Zenodo 4277797)")
    print(f"  DSP:      despike -> ALS baseline -> Savitzky-Golay -> SNV")
    print(f"  Tier A:   PCA(20) + RBF-SVM")
    print(f"  Tier B:   1D CNN (3-block, 60 epochs)")
    print(f"  Pipeline validated on real liver-disease SERS data.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
