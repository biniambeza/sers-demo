"""SERS liver cancer (HCC) triage: research demo on real public serum spectra.

Data: Zenodo 4277797, 144 serum SERS spectra (72 hepatocellular carcinoma,
72 healthy controls), portable 785 nm spectrometer, silver plasmonic paper,
measured on 5 days with 3 substrate batches.

Model (biomarkers/hcc_v1/spec.yaml): despike -> ALS baseline -> Savitzky-Golay
-> SNV, PCA(20) + RBF-SVM, sensitivity-first TriageThreshold (95%) giving
REFER / CLEAR / INCONCLUSIVE.

Honesty rule: a spectrum from the public dataset is always scored by the model
trained on the OTHER four measurement days (leave-one-day-out), never by a
model that saw it. Only genuinely new uploads use the model fitted on all data.

Run: streamlit run liver_app.py
"""
from __future__ import annotations

import html
import math
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("SERS_REGISTRY_HMAC_KEY", "demo-key-do-not-use-in-production")

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from reports.generator import (
    LabInfo, PatientInfo, ReportData, ReportGenerator, ReportStatus, ReportType, ResultEntry,
)
from sers_pipeline.ingestion import interpolate_to_grid, parse_spectrum_file
from sers_pipeline.preprocessing import Pipeline
from sers_pipeline.safety import TriageResult, TriageThreshold
from sers_pipeline.synthetic import make_wavenumber_grid

st.set_page_config(page_title="HCC Triage — Serum SERS", page_icon="🩺", layout="wide",
                   initial_sidebar_state="expanded")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "hcc_serum_sers" / "spectra"
TARGET_SENSITIVITY = 0.95
CLASS_NAMES = {0: "Healthy control", 1: "HCC (liver cancer)"}
_ID_RE = re.compile(r"^(?P<day>\d{8})_batch-(?P<batch>[A-Z])_(?P<cls>CTR|H0T)_(?P<code>\d+)$")

# From biomarkers/hcc_v1/spec.yaml: data-derived on these 144 spectra (2026-11-17),
# not yet assigned to molecules.
PEAKS_HIGHER_IN_HCC = [598, 778, 1057, 1156, 1272, 1506]
PEAKS_LOWER_IN_HCC = [479, 1440, 1564, 1594]

DISCLAIMER = (
    "Research demonstration. The spectra are real, public data from one laboratory, "
    "one week, one instrument, all male participants, and compare liver cancer with "
    "healthy people only. **Not a medical device and not for diagnosis.**"
)

REFLEX = {
    TriageResult.REFER: ("🔴 REFER", "result-refer",
                         "Refer for liver ultrasound and serum AFP."),
    TriageResult.CLEAR: ("🟢 CLEAR", "result-clear",
                         "No reason to refer from this test. CLEAR does not rule out HCC; "
                         "surveillance guidelines and clinical judgement still apply."),
    TriageResult.INCONCLUSIVE: ("🟠 INCONCLUSIVE", "result-inconclusive",
                                "Repeat the measurement. If still inconclusive, refer."),
}

st.markdown("""
<style>
  .page-title { font-size: 2rem; font-weight: 700; margin-bottom: 0; }
  .page-sub { font-size: 1rem; opacity: 0.7; margin-top: 0; }
  .result-refer { color: #e74c3c; font-weight: 700; font-size: 1.6rem; }
  .result-clear { color: #27ae60; font-weight: 700; font-size: 1.6rem; }
  .result-inconclusive { color: #f39c12; font-weight: 700; font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)

LAB_INFO = LabInfo(
    name="HCC Triage — Research Demo (not a clinical laboratory)",
    address="Demonstration only", clia_number="N/A (demo)",
    director_name="N/A", director_credentials="",
)


def header(title: str, subtitle: str) -> None:
    st.markdown(f'<p class="page-title">{title}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="page-sub">{subtitle}</p>', unsafe_allow_html=True)
    st.divider()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #

def make_model():
    return make_pipeline(
        PCA(n_components=20, random_state=42),
        SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
    )


def fit_triage(X: np.ndarray, y: np.ndarray):
    """Fit the classifier and a referral threshold chosen on out-of-fold scores."""
    oof = cross_val_predict(make_model(), X, y, method="predict_proba",
                            cv=StratifiedKFold(5, shuffle=True, random_state=0))[:, 1]
    triage = TriageThreshold(target_sensitivity=TARGET_SENSITIVITY).fit(oof, y)
    return make_model().fit(X, y), triage


@st.cache_resource(show_spinner="Loading 144 real serum spectra and fitting the triage model…")
def load_bundle():
    grid = make_wavenumber_grid()  # 400–1800 cm⁻¹, 1401 points
    rows, labels, meta = [], [], []
    for f in sorted(DATA_DIR.glob("*.txt")):
        m = _ID_RE.match(f.stem)
        if not m:
            continue
        wn, inten, _ = parse_spectrum_file(f)
        if wn.min() > grid[0] or wn.max() < grid[-1]:
            continue
        rows.append(interpolate_to_grid(wn, inten, grid))
        labels.append(1 if m["cls"] == "H0T" else 0)
        meta.append({"id": f.stem, "file": f.name, "day": m["day"], "batch": m["batch"],
                     "code": m["code"]})
    if not rows:
        raise FileNotFoundError(f"No spectra found in {DATA_DIR}")
    X_raw, y = np.vstack(rows), np.array(labels)
    dsp = Pipeline.default()
    X = dsp(X_raw)
    days = np.array([m["day"] for m in meta])
    day_list = sorted(set(days))

    # Leave-one-day-out: one model per held-out day, threshold fitted on the other days.
    lodo_models: dict[str, tuple] = {}
    score, thr = np.zeros(len(y)), np.zeros(len(y))
    for d in day_list:
        te, tr = days == d, days != d
        model, triage = fit_triage(X[tr], y[tr])
        lodo_models[d] = (model, triage)
        score[te] = model.predict_proba(X[te])[:, 1]
        thr[te] = triage.threshold
    referred = score >= thr
    pred = (score >= 0.5).astype(int)
    tp, fn = int((referred & (y == 1)).sum()), int((~referred & (y == 1)).sum())
    tn, fp = int((~referred & (y == 0)).sum()), int((referred & (y == 0)).sum())
    per_day = []
    for d in day_list:
        m = days == d
        per_day.append({"Day": f"{d[:4]}-{d[4:6]}-{d[6:]}", "Spectra": int(m.sum()),
                        "HCC": int((y[m] == 1).sum()), "Healthy": int((y[m] == 0).sum()),
                        "Accuracy": round(float((pred[m] == y[m]).mean()), 3),
                        "Referral threshold": round(float(thr[m][0]), 3)})
    final_model, final_triage = fit_triage(X, y)
    return {
        "grid": grid, "dsp": dsp, "X_raw": X_raw, "X": X, "y": y, "meta": meta, "days": days,
        "lodo": lodo_models, "final": (final_model, final_triage),
        "eval": {"score": score, "threshold": thr, "referred": referred, "pred": pred,
                 "accuracy": float((pred == y).mean()), "auroc": float(roc_auc_score(y, score)),
                 "tp": tp, "fn": fn, "tn": tn, "fp": fp, "per_day": per_day},
    }


def parse_upload(uploaded, grid: np.ndarray) -> np.ndarray:
    """Parse an uploaded spectrum file and resample it onto the model grid."""
    suffix = Path(uploaded.name).suffix or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = Path(tmp.name)
    try:
        wn, inten, _ = parse_spectrum_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    if wn.min() > grid[0] or wn.max() < grid[-1]:
        raise ValueError(
            f"The spectrum covers {wn.min():.0f}–{wn.max():.0f} cm⁻¹ but the model needs "
            f"{grid[0]:.0f}–{grid[-1]:.0f} cm⁻¹."
        )
    return interpolate_to_grid(wn, inten, grid)


MIN_SERUM_CORRELATION = 0.30  # all 144 dataset spectra ≥ 0.49; pure noise ≈ 0; non-serum ≈ 0.15


def qc_checks(spectrum_clean: np.ndarray, reference: np.ndarray) -> list[tuple[str, bool, str]]:
    """Checks that an input can be scored at all. They do not judge the result."""
    finite = bool(np.isfinite(spectrum_clean).all())
    rng_ = float(spectrum_clean.max() - spectrum_clean.min()) if finite else 0.0
    zeros = float(np.mean(np.abs(spectrum_clean) < 1e-6)) if finite else 1.0
    corr = float(np.corrcoef(spectrum_clean, reference)[0, 1]) if finite and rng_ > 0 else 0.0
    return [
        ("Finite values", finite, "all finite" if finite else "NaN/inf present"),
        ("Not flat", rng_ > 0.01, f"range {rng_:.2f}"),
        ("Zero fraction", zeros < 0.5, f"{zeros:.1%}"),
        ("Serum SERS profile", corr >= MIN_SERUM_CORRELATION,
         f"r = {corr:.2f} vs dataset mean (need ≥ {MIN_SERUM_CORRELATION})"),
    ]


def spectrum_figure(grid, traces, yaxis: str, peaks: bool = False) -> go.Figure:
    fig = go.Figure()
    for y_vals, name, color, width in traces:
        fig.add_trace(go.Scatter(x=grid, y=y_vals, mode="lines", name=name,
                                 line=dict(color=color, width=width)))
    if peaks:
        for p in PEAKS_HIGHER_IN_HCC:
            fig.add_vline(x=p, line_dash="dot", line_color="#e74c3c", opacity=0.45)
        for p in PEAKS_LOWER_IN_HCC:
            fig.add_vline(x=p, line_dash="dot", line_color="#3498db", opacity=0.45)
    fig.update_layout(xaxis_title="Raman shift (cm⁻¹)", yaxis_title=yaxis, height=340,
                      margin=dict(t=20, b=40), legend=dict(orientation="h", y=1.1))
    return fig


# --------------------------------------------------------------------------- #
# Pages                                                                        #
# --------------------------------------------------------------------------- #

def page_screening():
    header("Liver Cancer (HCC) Triage",
           "Score a serum SERS spectrum: REFER for ultrasound + AFP, or CLEAR")
    b = load_bundle()
    grid, dsp, meta, y = b["grid"], b["dsp"], b["meta"], b["y"]

    col_src, col_opt = st.columns([1, 2])
    with col_src:
        source = st.radio("Spectrum source", ["Real dataset sample", "Upload spectrum"])
    truth = None
    with col_opt:
        if source == "Real dataset sample":
            wanted = st.selectbox("True status of the sample", [1, 0],
                                  format_func=CLASS_NAMES.get)
            pool = [i for i in range(len(y)) if y[i] == wanted]
            i = st.selectbox(
                "Sample", pool,
                format_func=lambda k: f"{meta[k]['id']}   (day {meta[k]['day']}, batch {meta[k]['batch']})",
            )
            spectrum_raw, sample_id, truth = b["X_raw"][i], meta[i]["id"], int(y[i])
            model, triage = b["lodo"][meta[i]["day"]]
            model_note = (f"Scored by the model trained on the other four measurement days "
                          f"(day {meta[i]['day']} held out), so this sample was never seen in training.")
        else:
            uploaded = st.file_uploader(
                "Two-column text file (Raman shift, intensity) covering 400–1800 cm⁻¹; "
                "B&W Tek / Renishaw exports work as-is",
                type=["txt", "csv", "tsv", "dat"],
            )
            if uploaded is None:
                st.info("Upload a spectrum file to begin. Dataset files are in "
                        "`data/hcc_serum_sers/spectra/` (CTR = healthy, H0T = HCC).")
                return
            try:
                spectrum_raw = parse_upload(uploaded, grid)
            except Exception as exc:  # noqa: BLE001 — surface any parse failure
                st.error(f"Could not use this file: {exc}")
                return
            sample_id = Path(uploaded.name).stem
            known = {m["id"]: k for k, m in enumerate(meta)}
            if sample_id in known:
                k = known[sample_id]
                truth = int(y[k])
                model, triage = b["lodo"][meta[k]["day"]]
                model_note = (f"This file is part of the public dataset, so it is scored by the "
                              f"model that never saw its measurement day ({meta[k]['day']}).")
            else:
                model, triage = b["final"]
                model_note = "New spectrum: scored by the model fitted on all 144 dataset spectra."

    st.divider()
    st.subheader("Step 1 — Raw spectrum (resampled to 400–1800 cm⁻¹)")
    c1, c2 = st.columns([3, 1])
    with c1:
        st.plotly_chart(spectrum_figure(grid, [(spectrum_raw, "Raw", "#e74c3c", 1.5)],
                                        "Intensity (a.u.)"), width="stretch")
    with c2:
        st.metric("Sample", sample_id)
        st.metric("Points", len(spectrum_raw))
        st.metric("Max intensity", f"{spectrum_raw.max():.0f}")

    st.subheader("Step 2 — Preprocessing (despike → ALS baseline → Savitzky-Golay → SNV)")
    spectrum_clean = dsp(spectrum_raw.reshape(1, -1))[0]
    c1, c2 = st.columns([3, 1])
    with c1:
        st.plotly_chart(spectrum_figure(
            grid, [(spectrum_clean, "Processed", "#2ecc71", 2)], "SNV intensity", peaks=True),
            width="stretch")
        st.caption("Dotted lines: wavenumbers that differ most between HCC and healthy serum in "
                   "this dataset (red: higher in HCC, blue: lower in HCC).")
    with c2:
        for step_name, _ in dsp.steps:
            st.success(f"✅ {step_name}")

    st.subheader("Step 3 — Quality control")
    checks = qc_checks(spectrum_clean, b["X"].mean(0))
    qc_pass = all(ok for _, ok, _ in checks)
    for col, (label, ok, shown) in zip(st.columns(len(checks)), checks):
        with col:
            st.markdown(f"{'✅' if ok else '❌'} **{label}**")
            st.caption(shown)
    if not qc_pass:
        st.error("Spectrum failed QC: the result is INCONCLUSIVE. The 'serum SERS profile' check "
                 "fails for inputs that do not look like serum on this substrate, including "
                 "spectra from other sample types or synthetic generators.")

    st.subheader("Step 4 — Triage decision")
    score = float(model.predict_proba(spectrum_clean.reshape(1, -1))[0, 1])
    decision = triage.decide(np.array([score]), sample_ids=[sample_id],
                             qc_passed=np.array([qc_pass]))[0]
    label, css, action = REFLEX[decision.result]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f'<p class="{css}">{label}</p>', unsafe_allow_html=True)
        st.markdown(f"**Action:** {action}")
        st.caption(model_note)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[score], y=["HCC score"], orientation="h",
                             marker_color="#e74c3c" if score >= triage.threshold else "#27ae60",
                             text=[f"{score:.3f}"], textposition="outside"))
        fig.add_vline(x=triage.threshold, line_dash="dash", line_color="#333",
                      annotation_text=f"referral threshold {triage.threshold:.3f}",
                      annotation_position="top")
        fig.update_layout(height=160, xaxis_range=[0, 1], margin=dict(t=30, b=30, l=10, r=10),
                          xaxis_title="Model probability of HCC", showlegend=False)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"The threshold is set so that about {TARGET_SENSITIVITY:.0%} of known cancers "
                   f"in the training days are referred (fitted on out-of-fold scores: sensitivity "
                   f"{triage.fitted_sensitivity:.2f}, specificity {triage.fitted_specificity:.2f}).")

    if truth is not None:
        correct = (decision.result == TriageResult.REFER) == (truth == 1)
        if decision.result == TriageResult.INCONCLUSIVE:
            st.info(f"True status: **{CLASS_NAMES[truth]}**.")
        elif correct:
            st.success(f"True status: **{CLASS_NAMES[truth]}** — triage was right.")
        else:
            st.error(f"True status: **{CLASS_NAMES[truth]}** — triage was wrong. "
                     "Misses like this are why CLEAR never rules out cancer.")

    st.session_state["last_result"] = {
        "sample_id": sample_id, "result": decision.result.value, "score": score,
        "threshold": float(triage.threshold), "qc_pass": qc_pass, "action": action,
        "model_note": model_note,
    }
    st.caption("This result is available on the **Report** page.")


def page_report():
    header("Triage Report", "Printable summary of the most recent analysis")
    res = st.session_state.get("last_result")
    if res is None:
        st.info("Run an analysis on the **Screening** page first.")
        return
    c1, c2 = st.columns(2)
    with c1:
        patient = st.text_input("Patient / sample name", "Demo Patient")
        dob = st.text_input("Date of birth", "1970-01-01")
        sex = st.selectbox("Sex", ["Male", "Female", "Other"])
    with c2:
        mrn = st.text_input("Sample ID", res["sample_id"])
        provider = st.text_input("Requesting clinician", "—")
        specimen = st.selectbox("Specimen", ["Serum"])
    if not st.button("Generate report", type="primary", width="stretch"):
        return

    # ReportGenerator interpolates fields into HTML unescaped; escape user input.
    patient, dob, mrn, provider = (html.escape(s) for s in (patient, dob, mrn, provider))
    result = res["result"].upper()
    results = [
        ResultEntry("HCC triage (serum SERS)", result, "", "CLEAR",
                    "A" if result == "REFER" else "", "PCA(20)+SVM, sensitivity-first threshold"),
        ResultEntry("Model probability of HCC", f"{res['score']:.3f}", "probability",
                    f"< {res['threshold']:.3f}", "", "Tier A classifier"),
        ResultEntry("Spectrum QC", "pass" if res["qc_pass"] else "FAIL", "", "pass", "", ""),
    ]
    now = datetime.now().strftime("%Y-%m-%dT%H:%M")
    report = ReportGenerator(LAB_INFO).generate(ReportData(
        patient=PatientInfo(name=patient, dob=dob, mrn=mrn, sex=sex, ordering_provider=provider),
        report_type=ReportType.QUALITATIVE, status=ReportStatus.FINAL, results=results,
        collection_datetime=now, result_datetime=now, specimen_type=specimen,
        ordering_provider=provider,
        method_description=(
            "Surface-enhanced Raman spectroscopy of serum on silver plasmonic paper, 785 nm. "
            "Spectra are despiked, baseline-corrected (ALS), smoothed (Savitzky-Golay) and "
            "normalized (SNV), then scored by PCA(20) + RBF-SVM. The referral threshold is "
            f"chosen for {TARGET_SENSITIVITY:.0%} sensitivity on out-of-fold development scores."
        ),
        method_limitations=(
            "RESEARCH DEMO ONLY. Developed on 144 public spectra from one laboratory, one week, "
            "one instrument, all male, HCC versus healthy controls only (no hepatitis B or "
            "cirrhosis group). A CLEAR result does not rule out HCC. Not for diagnostic use."
        ),
        comments=f"Recommended action: {res['action']} {res['model_note']}",
    ))
    st.iframe(report.html, height=650)
    safe_id = "".join(ch for ch in res["sample_id"] if ch.isalnum() or ch in "-_") or "report"
    st.download_button("Download report (HTML)", report.html,
                       file_name=f"hcc_triage_{safe_id}.html", mime="text/html")


def page_performance():
    header("Model Performance", "Leave-one-day-out evaluation on the 144 real spectra")
    b = load_bundle()
    e, y = b["eval"], b["y"]
    n_pos, n_neg = e["tp"] + e["fn"], e["tn"] + e["fp"]
    sens_lo, sens_hi = wilson(e["tp"], n_pos)
    spec_lo, spec_hi = wilson(e["tn"], n_neg)

    c = st.columns(4)
    c[0].metric("Accuracy", f"{e['accuracy']:.3f}")
    c[1].metric("AUROC", f"{e['auroc']:.3f}")
    c[2].metric("HCC referred", f"{e['tp']}/{n_pos}", f"{sens_lo:.2f}–{sens_hi:.2f} (95% CI)",
                delta_color="off")
    c[3].metric("Healthy cleared", f"{e['tn']}/{n_neg}", f"{spec_lo:.2f}–{spec_hi:.2f} (95% CI)",
                delta_color="off")
    st.caption(
        "Each measurement day is held out in turn; the model and its referral threshold are "
        "fitted on the other four days only. No per-day correction is applied, because a clinic "
        "cannot apply one without a daily reference serum. Accuracy uses a 0.5 cut-off; "
        f"referral uses the {TARGET_SENSITIVITY:.0%}-sensitivity threshold."
    )
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Triage outcome by true status")
        z = [[e["tn"], e["fp"]], [e["fn"], e["tp"]]]
        fig = go.Figure(go.Heatmap(z=z, x=["CLEAR", "REFER"],
                                   y=[CLASS_NAMES[0], CLASS_NAMES[1]], colorscale="Blues",
                                   text=z, texttemplate="%{text}", showscale=False))
        fig.update_layout(height=340, xaxis_title="Triage result", yaxis_title="True status",
                          margin=dict(t=20, b=40))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Held-out HCC scores")
        fig = go.Figure()
        for cls, color in ((0, "#27ae60"), (1, "#e74c3c")):
            fig.add_trace(go.Histogram(x=e["score"][y == cls], nbinsx=20, name=CLASS_NAMES[cls],
                                       marker_color=color, opacity=0.7))
        fig.update_layout(barmode="overlay", height=340, xaxis_title="Model probability of HCC",
                          yaxis_title="Spectra", margin=dict(t=20, b=40),
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Per measurement day")
    st.dataframe(e["per_day"], hide_index=True, width="stretch")

    st.subheader("What the model looks at")
    X = b["X"]
    st.plotly_chart(spectrum_figure(
        b["grid"],
        [(X[y == 0].mean(0), f"Mean {CLASS_NAMES[0]} (n={int((y == 0).sum())})", "#27ae60", 2),
         (X[y == 1].mean(0), f"Mean {CLASS_NAMES[1]} (n={int((y == 1).sum())})", "#e74c3c", 2)],
        "SNV intensity", peaks=True), width="stretch")
    st.caption(
        f"Higher in HCC: {', '.join(map(str, PEAKS_HIGHER_IN_HCC))} cm⁻¹. "
        f"Lower in HCC: {', '.join(map(str, PEAKS_LOWER_IN_HCC))} cm⁻¹. "
        "Data-derived (Welch t-test per wavenumber); molecular assignment still pending. "
        "The source paper tentatively points to uric acid, hypoxanthine, ergothioneine and "
        "glutathione, and notes some differences may reflect liver function rather than cancer."
    )


def page_about():
    header("About", "What this test is for, where the data comes from, and its limits")
    st.warning(DISCLAIMER)
    st.markdown(f"""
**Intended use (target).** Triage for hepatocellular carcinoma (HCC) in clinics without
imaging or hepatology services. Adults at risk (chronic hepatitis B or C, cirrhosis, or
symptoms) give a drop of serum; a **REFER** result sends them for ultrasound and AFP.
**CLEAR** does not rule out HCC. Not a diagnosis and not a replacement for surveillance imaging.

**Data.** Zenodo record 4277797 (DOI 10.5281/zenodo.4277797): 144 serum SERS spectra,
72 HCC and 72 healthy controls, B&W Tek i-Raman Plus portable spectrometer at 785 nm,
citrate-reduced silver nanoparticles on paper, 5 measurement days, 3 substrate batches.
The source paper reports about 81% accuracy with PCA-LDA.

**Model.** Despike → ALS baseline → Savitzky-Golay → SNV, then PCA(20) + RBF-SVM.
The referral threshold is chosen so that {TARGET_SENSITIVITY:.0%} of known cancers in the
development data are referred, accepting more false referrals in exchange. The threshold is
always fitted on out-of-fold scores; a spectrum from the dataset is only ever scored by a
model that never saw its measurement day.

**Known limits**
- One laboratory, one week, one instrument.
- All male; 13-year median age gap between groups (HCC 69, healthy 56).
- HCC compared with healthy people only: no hepatitis B or cirrhosis-without-cancer group,
  which is the comparison a clinic needs.
- The measurement day can be recognised from the spectrum, so day-to-day drift is real.

**Next steps in the project**
- Patient cohort with four groups: HCC, hepatitis B without cancer, cirrhosis without cancer,
  healthy; men and women of similar ages.
- Daily pooled control serum as a reference to correct drift.
- Repeat on a second portable instrument; assign the discriminative peaks to molecules.
""")


PAGES = {
    "🩺 Screening": page_screening,
    "📄 Report": page_report,
    "🎯 Model performance": page_performance,
    "ℹ️ About": page_about,
}

st.sidebar.title("HCC Triage")
st.sidebar.caption("Liver cancer · serum SERS · real public data · research demo")
st.sidebar.divider()
choice = st.sidebar.radio("Navigation", list(PAGES))
st.sidebar.divider()
_e = load_bundle()["eval"]
st.sidebar.caption(f"Leave-one-day-out: accuracy {_e['accuracy']:.3f} · AUROC {_e['auroc']:.3f}")
st.sidebar.caption(f"HCC referred {_e['tp']}/{_e['tp'] + _e['fn']} · "
                   f"healthy cleared {_e['tn']}/{_e['tn'] + _e['fp']}")
st.sidebar.caption("⚠️ Not for clinical use.")

PAGES[choice]()
