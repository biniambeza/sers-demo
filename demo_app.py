"""SERS Liver Disease Screening: research demo on synthetic serum spectra.

Classifies a serum SERS spectrum as healthy, NAFL (simple steatosis), NASH, or
advanced fibrosis using two independent classifiers and a grey-zone safety gate.

Run: streamlit run liver_app.py
"""
from __future__ import annotations

import html
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("SERS_REGISTRY_HMAC_KEY", "demo-key-do-not-use-in-production")

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from reports.generator import (
    LabInfo, PatientInfo, ReportData, ReportGenerator, ReportStatus, ReportType, ResultEntry,
)
from sers_pipeline.classifiers import TierAClassifier, TierBClassifier
from sers_pipeline.preprocessing import Pipeline
from sers_pipeline.safety import GreyZoneGate
from sers_pipeline.synthetic import DEFAULT_CLASSES, make_wavenumber_grid, synthesize_realistic_dataset

st.set_page_config(
    page_title="SERS Liver Disease Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# label -> (display name, short description, abnormal?)
CONDITIONS: dict[str, tuple[str, str, bool]] = {
    "healthy": ("Healthy", "No serum signature of fatty liver disease.", False),
    "nafl": ("NAFL (simple steatosis)",
             "Fat accumulation in the liver without significant inflammation.", True),
    "nash": ("NASH / MASH",
             "Fatty liver with inflammation and liver-cell injury; can progress to fibrosis.", True),
    "advanced_fibrosis": ("Advanced fibrosis (F3–F4)",
                          "Extensive scarring of the liver, up to and including cirrhosis.", True),
}

DISCLAIMER = (
    "Research demonstration on **synthetic** spectra. Not a medical device, not validated "
    "on patients, and not for diagnosis or treatment decisions."
)

CSS = """
<style>
  .page-title { font-size: 2rem; font-weight: 700; margin-bottom: 0; }
  .page-sub { font-size: 1rem; opacity: 0.7; margin-top: 0; }
  .result-abnormal { color: #e74c3c; font-weight: 700; font-size: 1.3rem; }
  .result-normal { color: #27ae60; font-weight: 700; font-size: 1.3rem; }
  .result-inconclusive { color: #f39c12; font-weight: 700; font-size: 1.3rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

LAB_INFO = LabInfo(
    name="SERS Liver Screening — Research Demo (not a clinical laboratory)",
    address="Demonstration only",
    clia_number="N/A (demo)",
    director_name="N/A",
    director_credentials="",
)

GATE = GreyZoneGate(grey_low=0.40, grey_high=0.60, min_margin=0.15)


def name(label: str) -> str:
    return CONDITIONS.get(label, (label, "", False))[0]


def is_abnormal(label: str) -> bool:
    return CONDITIONS.get(label, ("", "", True))[2]


def header(title: str, subtitle: str) -> None:
    st.markdown(f'<p class="page-title">{title}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="page-sub">{subtitle}</p>', unsafe_allow_html=True)
    st.divider()


@st.cache_resource(show_spinner="Training liver disease models on synthetic spectra (about a minute, first load only)…")
def load_models():
    wavenumbers = make_wavenumber_grid()
    X, y, labels = synthesize_realistic_dataset(
        n_per_class=300, classes=DEFAULT_CLASSES, wavenumbers=wavenumbers, seed=42,
    )
    dsp = Pipeline.default()
    X_clean = dsp(X)

    idx = np.random.default_rng(0).permutation(len(X))
    split = int(0.8 * len(idx))
    train, test = idx[:split], idx[split:]

    tier_a = TierAClassifier(n_components=25, C=4.0, kernel="rbf", random_state=0)
    tier_a.fit(X_clean[train], y[train])
    tier_b = TierBClassifier(
        n_classes=len(labels), n_features=X_clean.shape[1],
        lr=1e-3, batch_size=32, epochs=40, device="cpu", random_state=0, verbose=False,
    )
    tier_b.fit(X_clean[train], y[train])

    acc_a = float((tier_a.predict_proba(X_clean[test]).argmax(1) == y[test]).mean())
    acc_b = float((tier_b.predict_proba(X_clean[test]).argmax(1) == y[test]).mean())
    return {
        "wavenumbers": wavenumbers, "dsp": dsp, "tier_a": tier_a, "tier_b": tier_b,
        "labels": labels, "acc_a": acc_a, "acc_b": acc_b,
        "X_raw": X, "X_clean": X_clean, "y": y, "test_idx": test,
    }


def result_html(label: str, inconclusive: bool = False) -> str:
    if inconclusive:
        return '<p class="result-inconclusive">INCONCLUSIVE</p>'
    css = "result-abnormal" if is_abnormal(label) else "result-normal"
    return f'<p class="{css}">{name(label).upper()}</p>'


# --------------------------------------------------------------------------- #
# Pages                                                                        #
# --------------------------------------------------------------------------- #

def page_screening():
    header("Liver Disease Screening",
           "Classify a serum SERS spectrum: healthy, NAFL, NASH, or advanced fibrosis")
    models = load_models()
    wavenumbers, dsp, labels = models["wavenumbers"], models["dsp"], models["labels"]

    col_src, col_opt = st.columns([2, 1])
    with col_src:
        source = st.radio("Spectrum source", ["Synthetic sample", "Upload spectrum"], horizontal=True)
    with col_opt:
        if source == "Synthetic sample":
            sample_label = st.selectbox("Simulated patient condition", labels, format_func=name)
            pool = np.where(models["y"] == labels.index(sample_label))[0]
            sample_num = st.slider("Sample #", 0, len(pool) - 1, 0)
            spectrum_raw = models["X_raw"][pool[sample_num]]
            sample_id = f"SYN-{sample_label}-{sample_num:03d}"
        else:
            uploaded = st.file_uploader(
                f"Spectrum file: {len(wavenumbers)} points over "
                f"{wavenumbers[0]:.0f}–{wavenumbers[-1]:.0f} cm⁻¹. Renishaw/WiRE .txt "
                "(wavenumber + intensity columns), .csv, or .npy",
                type=["txt", "csv", "npy"],
            )
            if uploaded is None:
                st.info("Upload a spectrum file to begin.")
                return
            try:
                if uploaded.name.lower().endswith(".npy"):
                    data = np.load(uploaded, allow_pickle=False)
                else:
                    text = uploaded.getvalue().decode("utf-8", errors="replace").replace(",", " ")
                    data = np.loadtxt(text.splitlines(), comments="#")
                data = np.asarray(data, dtype=float)
                # Two columns = (wavenumber, intensity); otherwise a flat list of intensities
                spectrum_raw = data[:, 1] if data.ndim == 2 and data.shape[1] == 2 else data.ravel()
            except Exception as exc:  # noqa: BLE001 — show any parse failure to the user
                st.error(f"Could not read the file: {exc}")
                return
            if spectrum_raw.size != len(wavenumbers):
                st.error(f"Expected {len(wavenumbers)} intensity values, got {spectrum_raw.size}.")
                return
            sample_id = Path(uploaded.name).stem

    st.divider()

    st.subheader("Step 1 — Raw spectrum")
    c1, c2 = st.columns([3, 1])
    with c1:
        fig = go.Figure(go.Scatter(x=wavenumbers, y=spectrum_raw, mode="lines",
                                   line=dict(color="#e74c3c", width=1.5)))
        fig.update_layout(xaxis_title="Raman shift (cm⁻¹)", yaxis_title="Intensity (a.u.)",
                          height=330, margin=dict(t=20, b=40))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.metric("Points", len(spectrum_raw))
        st.metric("Range (cm⁻¹)", f"{wavenumbers[0]:.0f}–{wavenumbers[-1]:.0f}")
        st.metric("Max intensity", f"{spectrum_raw.max():.0f}")

    st.subheader("Step 2 — Preprocessing (despike → baseline → smooth → normalize)")
    spectrum_clean = dsp(spectrum_raw.reshape(1, -1))[0]
    c1, c2 = st.columns([3, 1])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wavenumbers, y=spectrum_raw / max(spectrum_raw.max(), 1),
                                 mode="lines", name="Raw (scaled)", line=dict(color="#95a5a6", width=1)))
        fig.add_trace(go.Scatter(x=wavenumbers, y=spectrum_clean, mode="lines",
                                 name="Processed", line=dict(color="#2ecc71", width=2)))
        fig.update_layout(xaxis_title="Raman shift (cm⁻¹)", yaxis_title="Normalized intensity",
                          height=330, margin=dict(t=20, b=40))
        st.plotly_chart(fig, width="stretch")
    with c2:
        for step_name, _ in dsp.steps:
            st.success(f"✅ {step_name}")

    st.subheader("Step 3 — Quality control")
    snr = float(np.mean(np.abs(spectrum_clean)) / max(np.std(spectrum_clean), 1e-10))
    rng_ = float(spectrum_clean.max() - spectrum_clean.min())
    zeros = float(np.mean(np.abs(spectrum_clean) < 1e-6))
    checks = [
        ("Signal-to-noise", snr > 1.0, f"{snr:.2f}"),
        ("Intensity range", rng_ > 0.01, f"{rng_:.4f}"),
        ("Zero fraction", zeros < 0.5, f"{zeros:.1%}"),
    ]
    qc_pass = all(ok for _, ok, _ in checks)
    for col, (label, ok, shown) in zip(st.columns(len(checks)), checks):
        with col:
            st.markdown(f"{'✅' if ok else '❌'} **{label}**")
            st.caption(shown)
    if not qc_pass:
        st.error("⚠️ Spectrum failed QC. The result below may be unreliable.")

    st.subheader("Step 4 — Liver disease classification")
    probs_a = models["tier_a"].predict_proba(spectrum_clean.reshape(1, -1))[0]
    probs_b = models["tier_b"].predict_proba(spectrum_clean.reshape(1, -1))[0]
    decision = GATE.evaluate(probs_a.reshape(1, -1), class_labels=labels)[0]
    inconclusive = decision.disposition.value == "inconclusive"
    pred_a, pred_b = labels[int(probs_a.argmax())], labels[int(probs_b.argmax())]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Tier A — PCA + SVM (explainable)**")
        st.markdown(result_html(pred_a, inconclusive), unsafe_allow_html=True)
        st.progress(int(probs_a.max() * 100))
        st.caption(f"Confidence {probs_a.max():.1%}")
    with c2:
        st.markdown("**Tier B — 1D CNN**")
        st.markdown(result_html(pred_b), unsafe_allow_html=True)
        st.progress(int(probs_b.max() * 100))
        st.caption(f"Confidence {probs_b.max():.1%}")
    with c3:
        st.markdown("**Safety gate**")
        if inconclusive:
            st.warning("⚠️ Inconclusive: confirmatory testing recommended (e.g. FIB-4, elastography).")
            for reason in decision.reasons:
                st.caption(f"• {reason}")
        elif pred_a != pred_b:
            st.warning("⚠️ Classifiers disagree: flagged for review.")
        else:
            st.success(f"✅ Confident: {name(pred_a)}")
        st.caption(f"Margin {decision.margin:.3f}")

    if not inconclusive:
        st.info(f"**{name(pred_a)}**: {CONDITIONS.get(pred_a, ('', '', False))[1]}")

    with st.expander("Probability for each condition"):
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Tier A", x=[name(l) for l in labels], y=probs_a, marker_color="#667eea"))
        fig.add_trace(go.Bar(name="Tier B", x=[name(l) for l in labels], y=probs_b, marker_color="#e67e22"))
        fig.update_layout(barmode="group", height=320, yaxis_title="Probability",
                          yaxis_range=[0, 1], margin=dict(t=20, b=40))
        st.plotly_chart(fig, width="stretch")

    st.session_state["last_result"] = {
        "sample_id": sample_id, "labels": labels,
        "probs_a": probs_a.tolist(), "probs_b": probs_b.tolist(),
        "pred_a": pred_a, "pred_b": pred_b,
        "inconclusive": inconclusive, "qc_pass": qc_pass,
    }
    st.caption("This result is available on the **Report** page.")


def page_report():
    header("Screening Report", "Printable summary of the most recent analysis")
    res = st.session_state.get("last_result")
    if res is None:
        st.info("Run an analysis on the **Screening** page first.")
        return

    c1, c2 = st.columns(2)
    with c1:
        patient = st.text_input("Patient / sample name", "Demo Patient")
        dob = st.text_input("Date of birth", "1970-01-01")
        sex = st.selectbox("Sex", ["Female", "Male", "Other"])
    with c2:
        mrn = st.text_input("Sample ID", res["sample_id"])
        provider = st.text_input("Requesting clinician", "—")
        specimen = st.selectbox("Specimen", ["Serum", "Plasma"])

    if not st.button("Generate report", type="primary", width="stretch"):
        return

    if res["inconclusive"]:
        value, flag = "Inconclusive — confirmatory testing recommended", ""
    else:
        value = name(res["pred_a"])
        flag = "A" if is_abnormal(res["pred_a"]) else ""
    results = [ResultEntry("Liver disease classification (SERS)", value, "", "Healthy", flag,
                           "Serum SERS, PCA+SVM with grey-zone gate")]
    for label, pa in zip(res["labels"], res["probs_a"]):
        results.append(ResultEntry(f"P({name(label)})", f"{pa:.2f}", "probability", "", "",
                                   "Tier A classifier"))

    # ReportGenerator interpolates fields into HTML unescaped; escape everything user-supplied.
    patient, dob, mrn, provider = (html.escape(s) for s in (patient, dob, mrn, provider))

    now = datetime.now().strftime("%Y-%m-%dT%H:%M")
    report = ReportGenerator(LAB_INFO).generate(ReportData(
        patient=PatientInfo(name=patient, dob=dob, mrn=mrn, sex=sex, ordering_provider=provider),
        report_type=ReportType.QUALITATIVE,
        status=ReportStatus.FINAL,
        results=results,
        collection_datetime=now,
        result_datetime=now,
        specimen_type=specimen,
        ordering_provider=provider,
        method_description=(
            "Surface-enhanced Raman spectroscopy (SERS) of serum. Spectra are despiked, "
            "baseline-corrected, smoothed and normalized, then classified by PCA+SVM (Tier A) "
            "and a 1D CNN (Tier B), with a grey-zone gate that withholds low-confidence calls."
        ),
        method_limitations=(
            "RESEARCH DEMO ONLY. Models are trained on synthetic spectra and have not been "
            "validated on patient samples. Not for diagnostic use."
        ),
        comments=f"Tier B (CNN) result: {name(res['pred_b'])}. QC: {'pass' if res['qc_pass'] else 'FAIL'}.",
    ))
    st.iframe(report.html, height=650)
    safe_id = "".join(ch for ch in res["sample_id"] if ch.isalnum() or ch in "-_") or "report"
    st.download_button("Download report (HTML)", report.html,
                       file_name=f"liver_report_{safe_id}.html", mime="text/html")


def page_performance():
    header("Model Performance", "Held-out accuracy on synthetic liver disease spectra")
    m = load_models()
    labels, y_test = m["labels"], m["y"][m["test_idx"]]
    X_test = m["X_clean"][m["test_idx"]]
    probs_a, probs_b = m["tier_a"].predict_proba(X_test), m["tier_b"].predict_proba(X_test)
    preds_a, preds_b = probs_a.argmax(1), probs_b.argmax(1)
    names = [name(l) for l in labels]

    decisions = GATE.evaluate(probs_a, class_labels=labels)
    inconclusive_rate = np.mean([d.disposition.value == "inconclusive" for d in decisions])
    c = st.columns(4)
    c[0].metric("Tier A accuracy", f"{(preds_a == y_test).mean():.1%}")
    c[1].metric("Tier B accuracy", f"{(preds_b == y_test).mean():.1%}")
    c[2].metric("Tier agreement", f"{(preds_a == preds_b).mean():.1%}")
    c[3].metric("Inconclusive rate", f"{inconclusive_rate:.1%}")
    st.caption(f"{len(y_test)} held-out synthetic spectra. Real-patient performance will be lower.")
    st.divider()

    n = len(labels)
    for col, preds, title, scale in zip(st.columns(2), (preds_a, preds_b),
                                        ("Tier A — confusion matrix", "Tier B — confusion matrix"),
                                        ("Blues", "Oranges")):
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(y_test, preds):
            cm[t][p] += 1
        with col:
            st.subheader(title)
            fig = go.Figure(go.Heatmap(z=cm, x=names, y=names, colorscale=scale,
                                       text=cm, texttemplate="%{text}"))
            fig.update_layout(height=400, xaxis_title="Predicted", yaxis_title="True",
                              margin=dict(t=20, b=40))
            st.plotly_chart(fig, width="stretch")

    st.subheader("Prediction confidence")
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Tier A", "Tier B"])
    fig.add_trace(go.Histogram(x=probs_a.max(1), nbinsx=30, marker_color="#667eea"), row=1, col=1)
    fig.add_trace(go.Histogram(x=probs_b.max(1), nbinsx=30, marker_color="#e67e22"), row=1, col=2)
    fig.update_layout(height=330, showlegend=False, margin=dict(t=40, b=40))
    fig.update_xaxes(title_text="Max probability")
    st.plotly_chart(fig, width="stretch")


def page_about():
    header("About", "What this demo does, and what it does not")
    st.warning(DISCLAIMER)
    st.markdown(
        """
**Goal.** Screen for non-alcoholic fatty liver disease (NAFLD / MASLD) from a drop of serum,
using surface-enhanced Raman spectroscopy (SERS), as a low-cost triage step before biopsy
or elastography.

**Conditions classified**
"""
    )
    for label, (display, desc, _) in CONDITIONS.items():
        st.markdown(f"- **{display}**: {desc}")
    st.markdown(
        """
**How it works**
1. A serum SERS spectrum (400–1800 cm⁻¹) is cleaned: spikes removed, baseline subtracted,
   smoothed, normalized.
2. Two independent classifiers score it. **Tier A** (PCA + SVM) is explainable;
   **Tier B** (1D CNN) catches subtler patterns.
3. A **grey-zone safety gate** withholds a call when Tier A is not confident,
   and disagreement between tiers is flagged.

**Data.** The spectra are *synthetic*, generated from published serum SERS peak positions
for NAFL, NASH and fibrosis (Hepatology International, 2022: 261 biopsy-proven patients,
silver nanoparticles, 785 nm). High accuracy here shows the pipeline works end to end;
it does not show clinical accuracy.
"""
    )


# --------------------------------------------------------------------------- #
# Navigation                                                                   #
# --------------------------------------------------------------------------- #

PAGES = {
    "🩺 Screening": page_screening,
    "📄 Report": page_report,
    "🎯 Model performance": page_performance,
    "ℹ️ About": page_about,
}

st.sidebar.title("Liver Disease Screening")
st.sidebar.caption("Serum SERS · research demo")
st.sidebar.divider()
choice = st.sidebar.radio("Navigation", list(PAGES))
st.sidebar.divider()
_m = load_models()
st.sidebar.caption(f"Held-out accuracy (synthetic): Tier A {_m['acc_a']:.1%} · Tier B {_m['acc_b']:.1%}")
st.sidebar.caption("⚠️ Not for clinical use.")

PAGES[choice]()
