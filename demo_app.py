"""SERS Clinical Diagnostics Platform — Live Demo
Run: streamlit run demo_app.py
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("SERS_REGISTRY_HMAC_KEY", "demo-key-do-not-use-in-production")

import numpy as np
import streamlit as st

from sers_pipeline.synthetic import synthesize_realistic_dataset, make_wavenumber_grid, DEFAULT_CLASSES
from sers_pipeline.preprocessing import Pipeline
from sers_pipeline.classifiers import TierAClassifier, TierBClassifier
from sers_pipeline.safety import GreyZoneGate
from reports.generator import (
    ReportGenerator, ReportData, PatientInfo, LabInfo,
    ResultEntry, ReportType, ReportStatus,
)

st.set_page_config(
    page_title="SERS Clinical Diagnostics Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .positive-result { color: #e74c3c; font-weight: bold; font-size: 1.3rem; }
    .negative-result { color: #27ae60; font-weight: bold; font-size: 1.3rem; }
    .inconclusive-result { color: #f39c12; font-weight: bold; font-size: 1.3rem; }
    .qc-pass { color: #27ae60; }
    .qc-fail { color: #e74c3c; }
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

LAB_INFO = LabInfo(
    name="HealSend SERS Diagnostics Laboratory",
    address="Clinical Laboratory, Suite 200",
    clia_number="99D2345678",
    director_name="Laboratory Director",
    director_credentials="MD, PhD, FCAP",
    phone="(555) 123-4567",
)


@st.cache_resource
def load_models():
    wavenumbers = make_wavenumber_grid()
    X, y, labels = synthesize_realistic_dataset(
        n_per_class=300, classes=DEFAULT_CLASSES, wavenumbers=wavenumbers, seed=42,
    )
    dsp = Pipeline.default()
    X_clean = dsp(X)

    rng = np.random.default_rng(0)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    X_train, y_train = X_clean[idx[:split]], y[idx[:split]]
    X_test, y_test = X_clean[idx[split:]], y[idx[split:]]

    tier_a = TierAClassifier(n_components=25, C=4.0, kernel="rbf", random_state=0)
    tier_a.fit(X_train, y_train)

    tier_b = TierBClassifier(
        n_classes=len(labels), n_features=X_train.shape[1],
        lr=1e-3, batch_size=32, epochs=40, device="cpu", random_state=0, verbose=False,
    )
    tier_b.fit(X_train, y_train)

    acc_a = float((tier_a.predict_proba(X_test).argmax(1) == y_test).mean())
    acc_b = float((tier_b.predict_proba(X_test).argmax(1) == y_test).mean())

    return {
        "wavenumbers": wavenumbers,
        "dsp": dsp,
        "tier_a": tier_a,
        "tier_b": tier_b,
        "labels": labels,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "X_raw": X,
        "X_clean": X_clean,
        "y": y,
        "test_idx": idx[split:],
    }


def page_pipeline():
    st.markdown('<p class="main-header">Spectrum Analysis Pipeline</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a Raman spectrum or use a synthetic sample for analysis</p>', unsafe_allow_html=True)
    st.divider()

    models = load_models()
    wavenumbers = models["wavenumbers"]
    dsp = models["dsp"]
    labels = models["labels"]

    col_upload, col_options = st.columns([2, 1])
    with col_upload:
        upload_method = st.radio("Spectrum source", ["Synthetic sample", "Upload CSV"], horizontal=True)
    with col_options:
        if upload_method == "Synthetic sample":
            sample_class = st.selectbox("Select condition", labels)
            class_idx = labels.index(sample_class)
            mask = models["y"] == class_idx
            available_indices = np.where(mask)[0]
            sample_num = st.slider("Sample #", 0, len(available_indices) - 1, 0)
            spectrum_raw = models["X_raw"][available_indices[sample_num]]
        else:
            uploaded = st.file_uploader("Upload spectrum CSV (one value per line)", type=["csv", "npy"])
            if uploaded is None:
                st.info("Upload a spectrum file to begin analysis.")
                return
            if uploaded.name.endswith(".npy"):
                spectrum_raw = np.load(uploaded)
            else:
                spectrum_raw = np.loadtxt(uploaded, delimiter=",")
            if spectrum_raw.ndim > 1:
                spectrum_raw = spectrum_raw.flatten()

    st.divider()

    # Step 1: Raw spectrum
    st.subheader("Step 1 — Raw Spectrum")
    col1, col2 = st.columns([3, 1])
    with col1:
        import plotly.graph_objects as go
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(x=wavenumbers, y=spectrum_raw, mode='lines',
                                     name='Raw', line=dict(color='#e74c3c', width=1.5)))
        fig_raw.update_layout(
            xaxis_title="Wavenumber (cm⁻¹)", yaxis_title="Intensity (a.u.)",
            template="plotly_white", height=350, margin=dict(t=30, b=40),
            xaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_raw, width="stretch")
    with col2:
        st.metric("Points", len(spectrum_raw))
        st.metric("Range (cm⁻¹)", f"{wavenumbers[0]:.0f}–{wavenumbers[-1]:.0f}")
        st.metric("Max intensity", f"{spectrum_raw.max():.1f}")
        snr = float(np.mean(spectrum_raw) / max(np.std(spectrum_raw), 1e-10))
        st.metric("SNR estimate", f"{snr:.1f}")

    # Step 2: Preprocessing
    st.subheader("Step 2 — Preprocessing (Despike → Baseline → Smooth → Normalize)")
    spectrum_clean = dsp(spectrum_raw.reshape(1, -1))[0]

    col1, col2 = st.columns([3, 1])
    with col1:
        fig_clean = go.Figure()
        fig_clean.add_trace(go.Scatter(x=wavenumbers, y=spectrum_raw / max(spectrum_raw.max(), 1),
                                       mode='lines', name='Raw (scaled)', line=dict(color='#bdc3c7', width=1)))
        fig_clean.add_trace(go.Scatter(x=wavenumbers, y=spectrum_clean, mode='lines',
                                       name='Processed', line=dict(color='#2ecc71', width=2)))
        fig_clean.update_layout(
            xaxis_title="Wavenumber (cm⁻¹)", yaxis_title="Normalized intensity",
            template="plotly_white", height=350, margin=dict(t=30, b=40),
            xaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_clean, width="stretch")
    with col2:
        steps = [name for name, _ in dsp.steps]
        for i, step in enumerate(steps):
            st.success(f"✅ {step}")

    # Step 3: QC checks
    st.subheader("Step 3 — Quality Control Checks")
    qc_cols = st.columns(4)
    clean_snr = float(np.mean(np.abs(spectrum_clean)) / max(np.std(spectrum_clean), 1e-10))
    intensity_range = float(spectrum_clean.max() - spectrum_clean.min())
    zero_fraction = float(np.sum(np.abs(spectrum_clean) < 1e-6) / len(spectrum_clean))

    checks = [
        ("Signal-to-Noise", clean_snr, clean_snr > 1.0, f"{clean_snr:.2f}"),
        ("Intensity Range", intensity_range, intensity_range > 0.01, f"{intensity_range:.4f}"),
        ("Zero Fraction", zero_fraction, zero_fraction < 0.5, f"{zero_fraction:.1%}"),
        ("Spectrum Length", len(spectrum_clean), len(spectrum_clean) == len(wavenumbers), f"{len(spectrum_clean)} pts"),
    ]
    all_qc_pass = True
    for i, (name, val, passed, display) in enumerate(checks):
        with qc_cols[i]:
            if passed:
                st.markdown(f'<p class="qc-pass">✅ {name}</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="qc-fail">❌ {name}</p>', unsafe_allow_html=True)
                all_qc_pass = False
            st.caption(display)

    if not all_qc_pass:
        st.error("⚠️ Spectrum failed QC — results may be unreliable.")

    # Step 4: Prediction
    st.subheader("Step 4 — Diagnostic Prediction")
    tier_a = models["tier_a"]
    tier_b = models["tier_b"]
    gate = GreyZoneGate(grey_low=0.40, grey_high=0.60, min_margin=0.15)

    probs_a = tier_a.predict_proba(spectrum_clean.reshape(1, -1))
    probs_b = tier_b.predict_proba(spectrum_clean.reshape(1, -1))

    decisions = gate.evaluate(probs_a, class_labels=labels)
    decision = decisions[0]

    pred_cols = st.columns(3)
    with pred_cols[0]:
        st.markdown("**Tier A (PCA + SVC)**")
        pred_class_a = labels[probs_a[0].argmax()]
        conf_a = probs_a[0].max() * 100
        if decision.disposition.value == "inconclusive":
            st.markdown(f'<p class="inconclusive-result">INCONCLUSIVE</p>', unsafe_allow_html=True)
        else:
            color_class = "positive-result" if "nash" in pred_class_a.lower() or "cancer" in pred_class_a.lower() else "negative-result"
            st.markdown(f'<p class="{color_class}">{pred_class_a.upper()}</p>', unsafe_allow_html=True)
        st.progress(int(conf_a))
        st.caption(f"Confidence: {conf_a:.1f}%")

    with pred_cols[1]:
        st.markdown("**Tier B (1D CNN)**")
        pred_class_b = labels[probs_b[0].argmax()]
        conf_b = probs_b[0].max() * 100
        color_class_b = "positive-result" if "nash" in pred_class_b.lower() or "cancer" in pred_class_b.lower() else "negative-result"
        st.markdown(f'<p class="{color_class_b}">{pred_class_b.upper()}</p>', unsafe_allow_html=True)
        st.progress(int(conf_b))
        st.caption(f"Confidence: {conf_b:.1f}%")

    with pred_cols[2]:
        st.markdown("**Safety Gate**")
        if decision.disposition.value == "confident":
            st.success(f"✅ CONFIDENT — {decision.predicted_label}")
        else:
            st.warning(f"⚠️ INCONCLUSIVE — Reflex testing recommended")
            for reason in decision.reasons:
                st.caption(f"• {reason}")
        st.caption(f"Margin: {decision.margin:.3f}")

    # Probability breakdown
    with st.expander("Detailed probability breakdown"):
        prob_cols = st.columns(2)
        with prob_cols[0]:
            st.markdown("**Tier A probabilities**")
            fig_probs_a = go.Figure(go.Bar(
                x=[labels[i] for i in range(len(labels))],
                y=probs_a[0],
                marker_color=['#e74c3c' if i == probs_a[0].argmax() else '#bdc3c7' for i in range(len(labels))],
            ))
            fig_probs_a.update_layout(height=300, template="plotly_white", margin=dict(t=20, b=40),
                                      yaxis_title="Probability")
            st.plotly_chart(fig_probs_a, width="stretch")
        with prob_cols[1]:
            st.markdown("**Tier B probabilities**")
            fig_probs_b = go.Figure(go.Bar(
                x=[labels[i] for i in range(len(labels))],
                y=probs_b[0],
                marker_color=['#3498db' if i == probs_b[0].argmax() else '#bdc3c7' for i in range(len(labels))],
            ))
            fig_probs_b.update_layout(height=300, template="plotly_white", margin=dict(t=20, b=40),
                                      yaxis_title="Probability")
            st.plotly_chart(fig_probs_b, width="stretch")


def page_report():
    st.markdown('<p class="main-header">Clinical Report Generator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Generate CLIA-compliant patient result reports</p>', unsafe_allow_html=True)
    st.divider()

    models = load_models()
    labels = models["labels"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Patient Information")
        patient_name = st.text_input("Patient name", "Jane Doe")
        patient_dob = st.text_input("Date of birth", "1985-03-15")
        patient_mrn = st.text_input("MRN", "MRN-2026-001234")
        patient_sex = st.selectbox("Sex", ["Female", "Male", "Other"])
        ordering_provider = st.text_input("Ordering provider", "Dr. Smith")

    with col2:
        st.subheader("Test Configuration")
        test_type = st.selectbox("Report type", ["Qualitative (NASH screening)", "Cancer screening", "Multi-analyte panel"])
        specimen_type = st.selectbox("Specimen type", ["Serum", "Plasma", "Whole Blood", "Urine"])
        collection_date = st.text_input("Collection date/time", "2026-05-22T09:30:00")
        result_date = st.text_input("Result date/time", "2026-05-22T14:15:00")

    if st.button("Generate Report", type="primary", width="stretch"):
        if test_type.startswith("Qualitative"):
            report_type = ReportType.QUALITATIVE
            results = [ResultEntry(
                test_name="NASH SERS Screening",
                value="Positive",
                units="",
                reference_range="Negative",
                flag="",
                method="SERS-based serum biomarker analysis",
            )]
        elif test_type.startswith("Cancer"):
            report_type = ReportType.CANCER_SCREENING
            results = [ResultEntry(
                test_name="Multi-Cancer SERS Screen",
                value="No cancer signal detected",
                units="",
                reference_range="No cancer signal detected",
                flag="",
                method="SERS-based multi-cancer serum screening",
            )]
        else:
            report_type = ReportType.MULTI_ANALYTE
            results = [
                ResultEntry("NASH Score", "0.82", "index", "< 0.50", "H", "SERS biomarker panel"),
                ResultEntry("Inflammation Marker", "12.5", "ng/mL", "< 10.0", "H", "SERS quantitative"),
                ResultEntry("Fibrosis Score", "0.35", "index", "< 0.60", "", "SERS biomarker panel"),
            ]

        gen = ReportGenerator(LAB_INFO)
        report = gen.generate(ReportData(
            patient=PatientInfo(
                name=patient_name, dob=patient_dob, mrn=patient_mrn,
                sex=patient_sex, ordering_provider=ordering_provider,
            ),
            report_type=report_type,
            status=ReportStatus.FINAL,
            results=results,
            collection_datetime=collection_date,
            result_datetime=result_date,
            specimen_type=specimen_type,
            ordering_provider=ordering_provider,
            method_description="Surface-Enhanced Raman Spectroscopy (SERS) analysis of serum biomarkers using nano-gold substrate platform. PCA+SVC (Tier A) and 1D-CNN (Tier B) dual-classifier system with grey-zone safety gate.",
            method_limitations="This is a laboratory developed test (LDT). Results should be interpreted in conjunction with clinical findings.",
        ))

        st.divider()
        clia_col, report_col = st.columns([1, 3])
        with clia_col:
            st.subheader("CLIA Compliance")
            if report.is_clia_compliant:
                st.success("✅ CLIA Compliant")
            else:
                st.error("❌ Missing required fields")
            st.caption(f"Report ID: {report.report_id}")
            st.caption(f"Fields present: {len(report.clia_fields_present)}/14")
            for f in report.clia_fields_present:
                st.caption(f"  ✅ {f}")

        with report_col:
            st.subheader("Generated Report")
            st.components.v1.html(report.html, height=600, scrolling=True)


def page_dashboard():
    st.markdown('<p class="main-header">Operational Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Lab throughput, quality metrics, and capacity planning</p>', unsafe_allow_html=True)
    st.divider()

    models = load_models()
    import plotly.graph_objects as go

    # Generate synthetic operational data
    np.random.seed(42)
    days = 30
    dates = [f"2026-05-{d+1:02d}" for d in range(days)]
    daily_volume = np.random.poisson(85, days)
    daily_tat_median = np.random.normal(4.2, 0.5, days)
    daily_rejection = np.random.binomial(1, 0.03, (days, 85)).sum(axis=1)

    # KPIs
    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        st.metric("Monthly Volume", f"{daily_volume.sum():,}", f"+{int(daily_volume.sum() * 0.12)} vs last month")
    with kpi_cols[1]:
        st.metric("Avg TAT", f"{daily_tat_median.mean():.1f} hrs", "-0.3 hrs")
    with kpi_cols[2]:
        rejection_rate = daily_rejection.sum() / daily_volume.sum() * 100
        st.metric("Rejection Rate", f"{rejection_rate:.1f}%", "-0.2%")
    with kpi_cols[3]:
        st.metric("QC Pass Rate", "98.7%", "+0.3%")
    with kpi_cols[4]:
        st.metric("Tier A Accuracy", f"{models['acc_a']*100:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Daily Sample Volume")
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=dates, y=daily_volume, marker_color='#667eea'))
        fig_vol.add_trace(go.Scatter(x=dates, y=[np.mean(daily_volume)] * days,
                                     mode='lines', name='Average',
                                     line=dict(color='#e74c3c', dash='dash')))
        fig_vol.update_layout(height=350, template="plotly_white", margin=dict(t=30, b=40),
                              showlegend=False, yaxis_title="Samples")
        st.plotly_chart(fig_vol, width="stretch")

    with col2:
        st.subheader("Turnaround Time Trend")
        fig_tat = go.Figure()
        fig_tat.add_trace(go.Scatter(x=dates, y=daily_tat_median, mode='lines+markers',
                                     name='Median TAT', line=dict(color='#27ae60', width=2)))
        fig_tat.add_hline(y=6.0, line_dash="dash", line_color="red",
                         annotation_text="Target: 6h")
        fig_tat.update_layout(height=350, template="plotly_white", margin=dict(t=30, b=40),
                              yaxis_title="Hours")
        st.plotly_chart(fig_tat, width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Tests by Biomarker")
        biomarker_volumes = {"NASH": 820, "Multi-Cancer": 560, "Diabetes": 490, "Inflammation": 380, "PSA Panel": 290}
        fig_bm = go.Figure(go.Pie(
            labels=list(biomarker_volumes.keys()),
            values=list(biomarker_volumes.values()),
            hole=0.4,
            marker_colors=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b'],
        ))
        fig_bm.update_layout(height=350, template="plotly_white", margin=dict(t=30, b=40))
        st.plotly_chart(fig_bm, width="stretch")

    with col4:
        st.subheader("Instrument Utilization")
        instruments = ["SERS-Alpha", "SERS-Beta", "SERS-Gamma"]
        utilization = [0.78, 0.65, 0.42]
        fig_util = go.Figure(go.Bar(
            x=instruments, y=[u * 100 for u in utilization],
            marker_color=['#e74c3c' if u > 0.8 else '#f39c12' if u > 0.6 else '#27ae60' for u in utilization],
        ))
        fig_util.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80% threshold")
        fig_util.update_layout(height=350, template="plotly_white", margin=dict(t=30, b=40),
                              yaxis_title="Utilization %", yaxis_range=[0, 100])
        st.plotly_chart(fig_util, width="stretch")

    # Quality breakdown
    st.subheader("Specimen Rejection Breakdown")
    rejection_reasons = {"Hemolysis": 38, "Lipemia": 15, "Insufficient Volume": 12, "Clotted": 8, "Labeling Error": 5, "Other": 3}
    fig_rej = go.Figure(go.Bar(
        x=list(rejection_reasons.values()),
        y=list(rejection_reasons.keys()),
        orientation='h',
        marker_color='#e74c3c',
    ))
    fig_rej.update_layout(height=300, template="plotly_white", margin=dict(t=20, b=40),
                          xaxis_title="Count (past 30 days)")
    st.plotly_chart(fig_rej, width="stretch")


def page_model_performance():
    st.markdown('<p class="main-header">Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Classifier accuracy and evaluation metrics</p>', unsafe_allow_html=True)
    st.divider()

    models = load_models()
    labels = models["labels"]
    tier_a = models["tier_a"]
    tier_b = models["tier_b"]
    X_clean = models["X_clean"]
    y = models["y"]
    test_idx = models["test_idx"]

    X_test = X_clean[test_idx]
    y_test = y[test_idx]

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    probs_a = tier_a.predict_proba(X_test)
    probs_b = tier_b.predict_proba(X_test)
    preds_a = probs_a.argmax(1)
    preds_b = probs_b.argmax(1)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tier A Accuracy", f"{(preds_a == y_test).mean() * 100:.1f}%")
    with col2:
        st.metric("Tier B Accuracy", f"{(preds_b == y_test).mean() * 100:.1f}%")
    with col3:
        agreement = (preds_a == preds_b).mean() * 100
        st.metric("Tier Agreement", f"{agreement:.1f}%")
    with col4:
        gate = GreyZoneGate(grey_low=0.40, grey_high=0.60, min_margin=0.15)
        decisions = gate.evaluate(probs_a, class_labels=labels)
        inconclusive_rate = sum(1 for d in decisions if d.disposition.value == "inconclusive") / len(decisions) * 100
        st.metric("Inconclusive Rate", f"{inconclusive_rate:.1f}%")

    st.divider()

    # Confusion matrix
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tier A — Confusion Matrix")
        n_classes = len(labels)
        cm_a = np.zeros((n_classes, n_classes), dtype=int)
        for true, pred in zip(y_test, preds_a):
            cm_a[true][pred] += 1
        fig_cm_a = go.Figure(go.Heatmap(
            z=cm_a, x=labels, y=labels,
            colorscale='Blues', text=cm_a, texttemplate="%{text}",
        ))
        fig_cm_a.update_layout(
            height=400, template="plotly_white",
            xaxis_title="Predicted", yaxis_title="True",
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_cm_a, width="stretch")

    with col2:
        st.subheader("Tier B — Confusion Matrix")
        cm_b = np.zeros((n_classes, n_classes), dtype=int)
        for true, pred in zip(y_test, preds_b):
            cm_b[true][pred] += 1
        fig_cm_b = go.Figure(go.Heatmap(
            z=cm_b, x=labels, y=labels,
            colorscale='Purples', text=cm_b, texttemplate="%{text}",
        ))
        fig_cm_b.update_layout(
            height=400, template="plotly_white",
            xaxis_title="Predicted", yaxis_title="True",
            margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_cm_b, width="stretch")

    # Per-class accuracy
    st.subheader("Per-Class Accuracy")
    per_class_a = []
    per_class_b = []
    for i, label in enumerate(labels):
        mask = y_test == i
        if mask.sum() > 0:
            per_class_a.append((preds_a[mask] == i).mean() * 100)
            per_class_b.append((preds_b[mask] == i).mean() * 100)
        else:
            per_class_a.append(0)
            per_class_b.append(0)

    fig_pc = go.Figure()
    fig_pc.add_trace(go.Bar(name="Tier A", x=labels, y=per_class_a, marker_color='#667eea'))
    fig_pc.add_trace(go.Bar(name="Tier B", x=labels, y=per_class_b, marker_color='#764ba2'))
    fig_pc.update_layout(
        barmode='group', height=400, template="plotly_white",
        yaxis_title="Accuracy %", margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_pc, width="stretch")

    # Confidence distributions
    st.subheader("Prediction Confidence Distribution")
    fig_conf = make_subplots(rows=1, cols=2, subplot_titles=["Tier A", "Tier B"])
    fig_conf.add_trace(go.Histogram(x=probs_a.max(axis=1), nbinsx=30, marker_color='#667eea', name="Tier A"), row=1, col=1)
    fig_conf.add_trace(go.Histogram(x=probs_b.max(axis=1), nbinsx=30, marker_color='#764ba2', name="Tier B"), row=1, col=2)
    fig_conf.update_layout(height=350, template="plotly_white", showlegend=False, margin=dict(t=40, b=40))
    fig_conf.update_xaxes(title_text="Max probability", row=1, col=1)
    fig_conf.update_xaxes(title_text="Max probability", row=1, col=2)
    st.plotly_chart(fig_conf, width="stretch")


# --- Sidebar navigation ---
st.sidebar.image("https://img.icons8.com/fluency/96/microscope.png", width=60)
st.sidebar.title("SERS Platform")
st.sidebar.caption("Clinical Diagnostics v0.2")
st.sidebar.divider()

page = st.sidebar.radio("Navigation", [
    "🔬 Spectrum Analysis",
    "📋 Clinical Report",
    "📊 Operations Dashboard",
    "🎯 Model Performance",
])

st.sidebar.divider()
st.sidebar.caption("HealSend SERS Diagnostics")
st.sidebar.caption(f"Tier A: {load_models()['acc_a']*100:.1f}% | Tier B: {load_models()['acc_b']*100:.1f}%")

if page == "🔬 Spectrum Analysis":
    page_pipeline()
elif page == "📋 Clinical Report":
    page_report()
elif page == "📊 Operations Dashboard":
    page_dashboard()
elif page == "🎯 Model Performance":
    page_model_performance()
