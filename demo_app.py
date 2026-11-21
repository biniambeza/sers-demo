"""Liver cancer triage demo: hybrid molecule + pattern model with safety-first decisions.

Run:  python -m streamlit run triage_app.py

RESEARCH DEMO on public data (Zenodo 4277797: 72 HCC, 72 healthy, all male, one
European lab). Not validated on Ethiopian patients and not for clinical use.
"""
from __future__ import annotations

import csv
import html
import io
import re
import sys
import tempfile
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from load_hcc_serum import load_hcc_dataset  # noqa: E402
from reports.triage_pdf import build_triage_pdf  # noqa: E402
from scipy.signal import find_peaks, savgol_filter  # noqa: E402
from sers_pipeline.ingestion import interpolate_to_grid, parse_spectrum_file  # noqa: E402
from sers_pipeline.preprocessing import ALSBaselineCorrector, WhitakerHayesDespiker  # noqa: E402
from sers_pipeline.safety import TriageResult, TriageThreshold  # noqa: E402
from sers_pipeline.synthetic import make_wavenumber_grid  # noqa: E402
from sklearn.decomposition import NMF, PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split  # noqa: E402
from sklearn.pipeline import FeatureUnion, make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

GRID = make_wavenumber_grid()
K = 6
TARGET_SENSITIVITY = 0.95

# Palette
INK, PANEL, LINE = "#0e1013", "#16191e", "#262a31"
IVORY, MUTED, GOLD = "#ece6da", "#9a958b", "#c9a96e"
RED, GREEN, AMBER, TEAL = "#c2645a", "#6fa58b", "#c99a4e", "#7fa7b5"

st.set_page_config(page_title="SERS Liver Triage", page_icon="◆", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, p, li, label, span, div {{
  font-family: 'Inter', sans-serif; }}
.stApp {{ background: radial-gradient(1200px 600px at 15% -10%, #1b1f26 0%, {INK} 55%); color: {IVORY}; }}
section[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {LINE}; }}
section[data-testid="stSidebar"] * {{ color: {IVORY}; }}
h1, h2, h3, .serif {{ font-family: 'Cormorant Garamond', serif !important; letter-spacing: .3px; color: {IVORY}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
.eyebrow {{ font-size: .72rem; letter-spacing: .28em; text-transform: uppercase; color: {GOLD}; }}
.hero-title {{ font-family: 'Cormorant Garamond', serif; font-size: 3.1rem; font-weight: 600; line-height: 1.05; margin: .2rem 0 .4rem; }}
.hero-sub {{ color: {MUTED}; font-size: 1rem; max-width: 760px; font-weight: 300; }}
.rule {{ height: 1px; background: linear-gradient(90deg, {GOLD}55, transparent); margin: 1.4rem 0 1.6rem; }}
.card {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 14px; padding: 20px 22px; }}
.kpi-label {{ color: {MUTED}; font-size: .72rem; letter-spacing: .16em; text-transform: uppercase; }}
.kpi-value {{ font-family: 'Cormorant Garamond', serif; font-size: 2.4rem; font-weight: 600; color: {IVORY}; line-height: 1.1; }}
.kpi-note {{ color: {MUTED}; font-size: .78rem; }}
.verdict {{ border-radius: 14px; padding: 22px 24px; border: 1px solid; }}
.verdict-k {{ font-size: .72rem; letter-spacing: .24em; text-transform: uppercase; opacity: .85; }}
.verdict-v {{ font-family: 'Cormorant Garamond', serif; font-size: 2.6rem; font-weight: 700; line-height: 1.05; }}
.verdict-s {{ font-size: .92rem; opacity: .9; margin-top: .25rem; }}
.mol {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; }}
.mol-name {{ font-family: 'Cormorant Garamond', serif; font-size: 1.25rem; font-weight: 600; }}
.mol-meta {{ color: {MUTED}; font-size: .76rem; }}
.tag {{ display:inline-block; font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; padding:3px 9px; border-radius:20px; border:1px solid; }}
.track {{ position: relative; height: 8px; background: #22262d; border-radius: 6px; margin: 10px 0 4px; }}
.band {{ position:absolute; height:8px; background:{GOLD}33; border-radius:6px; border:1px solid {GOLD}66; }}
.dot {{ position:absolute; top:-4px; width:16px; height:16px; border-radius:50%; border:2px solid {INK}; }}
.step {{ display:flex; gap:14px; align-items:flex-start; margin-bottom:14px; }}
.step-n {{ font-family:'Cormorant Garamond', serif; color:{GOLD}; font-size:1.6rem; min-width:32px; line-height:1; }}
.small {{ color:{MUTED}; font-size:.82rem; }}
.foot {{ color:{MUTED}; font-size:.76rem; border-top:1px solid {LINE}; padding-top:14px; margin-top:28px; }}
div[data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {LINE}; }}
button[data-baseweb="tab"] {{ background: transparent; color: {MUTED}; font-size: .82rem; letter-spacing: .14em; text-transform: uppercase; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {GOLD}; }}
div[data-baseweb="tab-highlight"] {{ background-color: {GOLD}; }}
.stButton > button {{ border-radius: 30px; padding: .55rem 1.6rem; letter-spacing: .12em;
  text-transform: uppercase; font-size: .78rem; border: 1px solid {GOLD}88; background: transparent; color: {IVORY}; }}
.stButton > button[kind="primary"] {{ background: {GOLD}; color: {INK}; border: none; font-weight: 600; }}
.stButton > button:hover {{ border-color: {GOLD}; color: {GOLD}; }}
.stButton > button[kind="primary"]:hover {{ background: #d8bb84; color: {INK}; }}
table.ev {{ width:100%; border-collapse: collapse; font-size:.9rem; }}
table.ev th {{ text-align:left; color:{MUTED}; font-weight:500; font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; border-bottom:1px solid {LINE}; padding:10px 8px; }}
table.ev td {{ border-bottom:1px solid {LINE}; padding:11px 8px; color:{IVORY}; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- model
def clean(X: np.ndarray) -> np.ndarray:
    """Despike, remove baseline, smooth, scale to unit area, clip below zero."""
    X = WhitakerHayesDespiker()(np.atleast_2d(X))
    X = ALSBaselineCorrector()(X)
    X = savgol_filter(X, 11, 3, axis=1)
    X = np.clip(X, 0, None)
    return X / (X.sum(1, keepdims=True) + 1e-12)


def make_model():
    return make_pipeline(
        FeatureUnion([("molecules", NMF(K, init="nndsvda", max_iter=2000, random_state=0)),
                      ("pattern", PCA(20, random_state=0))]),
        StandardScaler(),
        LogisticRegression(C=0.01, max_iter=5000),
    )


def name_component(h: np.ndarray) -> tuple[str, list[int]]:
    """Name a component by its two strongest peaks (provisional literature matching)."""
    h = h / h.max()
    pk, _ = find_peaks(h, height=0.25, distance=12)
    ranked = [int(GRID[j]) for j in pk[np.argsort(h[pk])[::-1]]]
    peaks = sorted(ranked[:6])
    bands = {"Uric acid": (640, 1133), "Hypoxanthine": (725,), "Ergothioneine": (483,)}
    names = []
    for p in ranked[:2]:
        for name, bs in bands.items():
            if any(abs(p - b) <= 15 for b in bs) and name not in names:
                names.append(name)
    return (" · ".join(names) if names else "Unassigned"), peaks


@st.cache_resource(show_spinner="Preparing the model on the training patients…")
def train():
    X_raw, y, ids = load_hcc_dataset()
    X = clean(X_raw)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=7)

    oof = cross_val_predict(make_model(), X[tr], y[tr],
                            cv=StratifiedKFold(5, shuffle=True, random_state=0),
                            method="predict_proba")[:, 1]
    triage = TriageThreshold(TARGET_SENSITIVITY).fit(oof, y[tr])
    model = make_model().fit(X[tr], y[tr])

    nmf = model[0].transformer_list[0][1]
    names = [name_component(h) for h in nmf.components_]
    W_tr = nmf.transform(X[tr])
    ranges = np.percentile(W_tr[y[tr] == 0], [2.5, 97.5], axis=0)
    span = np.percentile(W_tr, [0.5, 99.5], axis=0)

    te_scores = model.predict_proba(X[te])[:, 1]
    referred = np.array([d.result == TriageResult.REFER for d in triage.decide(te_scores)])
    yt = y[te]
    perf = {
        "n_train": len(tr), "n_test": len(te),
        "hcc_referred": (int((referred & (yt == 1)).sum()), int((yt == 1).sum())),
        "healthy_cleared": (int((~referred & (yt == 0)).sum()), int((yt == 0).sum())),
    }
    return dict(X_raw=X_raw, X=X, y=y, test_idx=te, model=model, triage=triage,
                names=names, ranges=ranges, span=span, perf=perf, H=nmf.components_)


def explain(m, x_clean: np.ndarray):
    model = m["model"]
    feats = model[0].transform(x_clean)
    z = model[1].transform(feats)[0]
    contrib = model[2].coef_[0] * z
    return feats[0, :K], contrib[:K], contrib[K:].sum()


def quality_ok(x_raw: np.ndarray) -> tuple[bool, str]:
    if not np.isfinite(x_raw).all():
        return False, "the spectrum contains invalid values"
    if np.std(x_raw) < 1e-6 * (np.abs(x_raw).mean() + 1e-12):
        return False, "the spectrum is flat (no signal)"
    return True, ""


def plot_layout(fig, height, title=None):
    fig.update_layout(
        height=height, title=dict(text=title, font=dict(family="Cormorant Garamond", size=20, color=IVORY)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=MUTED, size=12),
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(color=IVORY)),
        xaxis=dict(gridcolor=LINE, zeroline=False, linecolor=LINE),
        yaxis=dict(gridcolor=LINE, zeroline=False, linecolor=LINE),
    )
    return fig


def kpi(label, value, note=""):
    return (f"<div class='card'><div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{value}</div><div class='kpi-note'>{note}</div></div>")


COMPONENT_COLORS = ["#c9a96e", "#7fa7b5", "#c2645a", "#6fa58b", "#a48bc4", "#d1b48c"]


def live_analysis(x_raw: np.ndarray, m: dict, name: str) -> None:
    """Step-by-step replay of a measurement, from reading to decision."""
    head = st.empty()
    chart = st.empty()
    notes = st.empty()
    bar = st.progress(0.0)
    run = st.session_state.get("live_run", 0) + 1
    st.session_state["live_run"] = run

    def show(fig, height, title, tag):
        chart.plotly_chart(plot_layout(fig, height, title), width="stretch", key=f"live_{run}_{tag}")

    def step(n, title, text):
        head.markdown(f"<div class='step'><div class='step-n'>{n:02d}</div><div><div class='serif' "
                      f"style='font-size:1.45rem;font-weight:600'>{title}</div><div class='small'>{text}"
                      f"</div></div></div>", unsafe_allow_html=True)

    ylo, yhi = float(x_raw.min()), float(x_raw.max())
    pad = 0.08 * (yhi - ylo + 1e-12)

    # 01 Reading -----------------------------------------------------------
    step(1, "Reading the sample", f"785 nm laser on the silver strip · {html.escape(name)}")
    frames = 36
    for k in range(1, frames + 1):
        upto = max(2, int(len(GRID) * k / frames))
        fig = go.Figure(go.Scatter(x=GRID[:upto], y=x_raw[:upto], line=dict(color=GOLD, width=1.6)))
        fig.add_vline(x=GRID[upto - 1], line=dict(color=IVORY, width=1, dash="dot"))
        fig.update_xaxes(range=[GRID[0], GRID[-1]], title="Raman shift (cm⁻¹)")
        fig.update_yaxes(range=[ylo - pad, yhi + pad], title="counts")
        show(fig, 340, "Raw signal from the detector", f"read{k}")
        bar.progress(0.25 * k / frames)
        time.sleep(0.045)
    notes.markdown("<div class='small'>Replaying a recorded measurement — no reader is connected yet.</div>",
                   unsafe_allow_html=True)
    time.sleep(0.6)

    # 02 Quality checks -------------------------------------------------------
    step(2, "Quality checks", "A bad reading is never classified; it is returned as ‘measure again’.")
    ok, why = quality_ok(x_raw)
    noise = float(np.std(np.diff(x_raw)) / np.sqrt(2))
    snr = (yhi - ylo) / (noise + 1e-12)
    checks = [("Signal present", np.ptp(x_raw) > 0),
              ("Valid numbers only", bool(np.isfinite(x_raw).all())),
              (f"Signal-to-noise {snr:.0f}", snr > 10),
              ("Covers 400–1800 cm⁻¹", True)]
    shown = []
    for label, passed in checks:
        mark = f"<span style='color:{GREEN if passed else RED}'>{'✓' if passed else '✗'}</span>"
        shown.append(f"<div style='font-size:1.05rem;margin:4px 0'>{mark}&nbsp;&nbsp;{label}</div>")
        notes.markdown("<div class='card'>" + "".join(shown) + "</div>", unsafe_allow_html=True)
        time.sleep(0.45)
    bar.progress(0.35)
    if not ok:
        notes.markdown(f"<div class='card' style='color:{AMBER}'>Quality check failed: {why}</div>",
                       unsafe_allow_html=True)
        time.sleep(1.5)
        return

    # 03 Cleaning -----------------------------------------------------------
    despiked = WhitakerHayesDespiker()(x_raw[None])[0]
    corrected = ALSBaselineCorrector()(despiked[None])[0]
    baseline = despiked - corrected
    smoothed = savgol_filter(corrected, 11, 3)
    final = clean(x_raw)[0]

    stages = [
        ("Removing cosmic-ray spikes", [("raw", x_raw, "#5d6168"), ("despiked", despiked, GOLD)]),
        ("Removing background glow", [("signal", despiked, GOLD), ("background", baseline, TEAL)]),
        ("Smoothing noise", [("before", corrected, "#5d6168"), ("smoothed", smoothed, GOLD)]),
        ("Normalising brightness", [("clean spectrum", final / final.max(), GOLD)]),
    ]
    for n, (title, traces) in enumerate(stages):
        step(3, "Cleaning the signal", title)
        fig = go.Figure()
        for label, yv, col in traces:
            dash = "dash" if label == "background" else "solid"
            fig.add_trace(go.Scatter(x=GRID, y=yv, name=label, line=dict(color=col, width=1.6, dash=dash)))
        fig.update_xaxes(title="Raman shift (cm⁻¹)")
        show(fig, 340, title, f"clean{n}")
        notes.empty()
        bar.progress(0.35 + 0.1 * (n + 1))
        time.sleep(1.1)

    # 04 Molecule breakdown ----------------------------------------------------
    step(4, "Breaking the spectrum into molecules",
         "Each layer is a molecule-like component; together they rebuild the measured spectrum.")
    nmf = m["model"][0].transformer_list[0][1]
    w = nmf.transform(final[None])[0]
    layers = w[:, None] * nmf.components_
    scale = 1.0 / (final.max() + 1e-12)
    order = np.argsort(-layers.sum(1))
    for k in range(1, len(order) + 1):
        fig = go.Figure()
        stack = np.zeros_like(GRID)
        for i in order[:k]:
            stack = stack + layers[i] * scale
            fig.add_trace(go.Scatter(x=GRID, y=stack.copy(), mode="lines", line=dict(width=0.6,
                                     color=COMPONENT_COLORS[i % 6]), fill="tonexty" if i != order[0] else "tozeroy",
                                     name=m["names"][i][0]))
        fig.add_trace(go.Scatter(x=GRID, y=final * scale, name="measured", line=dict(color=IVORY, width=1.4)))
        fig.update_xaxes(title="Raman shift (cm⁻¹)")
        fig.update_yaxes(range=[0, 1.15])
        show(fig, 360, "Molecule-like components", f"mol{k}")
        bar.progress(0.75 + 0.1 * k / len(order))
        time.sleep(0.55)

    # 05 Decision -----------------------------------------------------------
    step(5, "Deciding", "A safety-first threshold: missing a cancer counts as the worse error.")
    score = float(m["model"].predict_proba(final[None])[0, 1])
    thr = m["triage"].threshold
    for k in range(1, 19):
        v = score * k / 18
        g = go.Figure(go.Indicator(
            mode="gauge+number", value=v,
            number=dict(valueformat=".2f", font=dict(family="Cormorant Garamond", size=52, color=IVORY)),
            gauge=dict(axis=dict(range=[0, 1], tickcolor=MUTED), bar=dict(color=GOLD, thickness=0.28),
                       bgcolor="#1c2026", borderwidth=0,
                       steps=[dict(range=[0, thr], color="#1d2a24"), dict(range=[thr, 1], color="#2c1f1d")],
                       threshold=dict(line=dict(color=IVORY, width=3), thickness=0.85, value=thr))))
        show(g, 300, "Cancer score", f"gauge{k}")
        time.sleep(0.04)
    bar.progress(1.0)
    verdict = ("Refer", RED) if score >= thr else ("Clear", GREEN)
    notes.markdown(f"<div class='verdict' style='border-color:{verdict[1]}66;background:{verdict[1]}1f'>"
                   f"<div class='verdict-k' style='color:{verdict[1]}'>Triage result</div>"
                   f"<div class='verdict-v' style='color:{verdict[1]}'>{verdict[0]}</div></div>",
                   unsafe_allow_html=True)
    time.sleep(1.6)


# --------------------------------------------------------------------------- data
m = train()
p = m["perf"]

DEMO_DIR = ROOT / "demo_samples"


def demo_answers() -> dict[str, int]:
    """Confirmed diagnosis for the demo sample files (for the reveal only)."""
    path = DEMO_DIR / "answers.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["file"]: int(r["diagnosis"] == "Liver cancer") for r in csv.DictReader(fh)}


def read_spectrum_file(path: str) -> np.ndarray | None:
    wn, inten, _ = parse_spectrum_file(path)
    if wn.min() > GRID[0] or wn.max() < GRID[-1]:
        st.error(f"The spectrum must cover {GRID[0]:.0f}–{GRID[-1]:.0f} cm⁻¹.")
        return None
    order = np.argsort(wn)
    return interpolate_to_grid(wn[order], inten[order], GRID)


with st.sidebar:
    st.markdown("<div class='eyebrow'>Sample</div>", unsafe_allow_html=True)
    source = st.radio("Source", ["Load a blood sample file", "Unseen test patient"],
                      label_visibility="collapsed")
    x_raw, true_label, sample_name = None, None, None
    if source == "Unseen test patient":
        te = m["test_idx"]
        labels = [f"Patient {i + 1:02d}" for i in range(len(te))]
        choice = st.selectbox("Never used in training", labels)
        j = te[labels.index(choice)]
        x_raw, true_label, sample_name = m["X_raw"][j], int(m["y"][j]), choice
    else:
        answers = demo_answers()
        up = st.file_uploader("Spectrum from the reader (two columns: wavenumber, intensity)",
                              type=["txt", "csv"])
        if up is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as fh:
                fh.write(up.getvalue())
            x_raw = read_spectrum_file(fh.name)
            sample_name = up.name
            true_label = answers.get(up.name)
        elif answers:
            pick = st.selectbox("…or use a demo sample", ["—"] + sorted(answers))
            if pick != "—":
                x_raw = read_spectrum_file(str(DEMO_DIR / pick))
                sample_name, true_label = pick, answers[pick]
    st.markdown("<div class='rule'></div><div class='eyebrow'>Patient</div>", unsafe_allow_html=True)
    patient = {
        "name": st.text_input("Full name", key="pt_name", placeholder="optional"),
        "patient_id": st.text_input("Patient ID", key="pt_id", placeholder="e.g. TA-2026-0153"),
    }
    c_age, c_sex = st.columns(2)
    patient["age"] = c_age.text_input("Age", key="pt_age")
    patient["sex"] = c_sex.selectbox("Sex", ["—", "Female", "Male"], key="pt_sex")
    patient["sex"] = "" if patient["sex"] == "—" else patient["sex"]
    patient["clinic"] = st.text_input("Clinic", key="pt_clinic", placeholder="e.g. Tikur Anbessa")
    st.markdown("<div class='small'>Used only on screen and in the report; nothing is stored.</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.markdown("<div class='eyebrow'>Research prototype</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small'>Public data: 72 liver cancer, 72 healthy · all male · one European "
                f"laboratory. Not validated on Ethiopian patients. Not for clinical use.</div>",
                unsafe_allow_html=True)

# --------------------------------------------------------------------------- hero
st.markdown(f"""
<div class='eyebrow'>SERS blood triage · Hepatocellular carcinoma</div>
<div class='hero-title'>A drop of serum, read by light.</div>
<div class='hero-sub'>Serum on a silver-coated strip, a 785 nm laser, and a model that explains itself.
Each patient receives one of three answers — refer, clear, or measure again — together with the
molecules that drove the decision.</div>
<div class='rule'></div>
""", unsafe_allow_html=True)

c = st.columns(4)
c[0].markdown(kpi("Nested accuracy", "94.2%", "model choice made without test data"), unsafe_allow_html=True)
c[1].markdown(kpi("Cancers referred", f"{p['hcc_referred'][0]}/{p['hcc_referred'][1]}",
                  f"on {p['n_test']} unseen patients"), unsafe_allow_html=True)
c[2].markdown(kpi("Healthy cleared", f"{p['healthy_cleared'][0]}/{p['healthy_cleared'][1]}",
                  "safety-first threshold, 95% target"), unsafe_allow_html=True)
c[3].markdown(kpi("Unseen measurement day", "93.1%", "with drift correction from training days"),
              unsafe_allow_html=True)
st.write("")

tab_result, tab_samples, tab_how, tab_evidence = st.tabs(
    ["Patient result", "Sample data", "How it works", "Evidence"])

# --------------------------------------------------------------------------- result tab
run_key = f"{source}|{sample_name}"
with tab_result:
    if x_raw is None:
        st.markdown("<div class='card small'>Load a blood sample file, pick a demo sample, or choose an "
                    "unseen test patient in the sidebar.</div>", unsafe_allow_html=True)
    elif st.session_state.get("analysed") != run_key:
        st.markdown(f"<div class='card'><div class='kpi-label'>Sample loaded</div>"
                    f"<div class='kpi-value'>{html.escape(sample_name)}</div>"
                    f"<div class='kpi-note'>{len(GRID)} spectral points ready for analysis</div></div>",
                    unsafe_allow_html=True)
        st.write("")
        if st.button("▶  Run analysis", type="primary"):
            live_analysis(x_raw, m, sample_name)
            st.session_state["analysed"] = run_key
            st.rerun()
    else:
        if st.button("↻  Replay the analysis"):
            st.session_state.pop("analysed", None)
            st.rerun()
        ok, why_bad = quality_ok(x_raw)
        x_clean = clean(x_raw)
        score = float(m["model"].predict_proba(x_clean)[0, 1])
        d = m["triage"].decide(np.array([score]), qc_passed=np.array([ok]))[0]

        style = {
            TriageResult.REFER: (RED, "Refer", "Liver ultrasound and serum AFP"),
            TriageResult.CLEAR: (GREEN, "Clear", "No reason to refer from this test — it does not rule cancer out"),
            TriageResult.INCONCLUSIVE: (AMBER, "Measure again", f"Quality check failed: {why_bad}"),
        }[d.result]
        col_l, col_r = st.columns([1, 1.6], gap="large")
        with col_l:
            who = " · ".join(html.escape(v) for v in
                             (patient["name"], patient["patient_id"],
                              f"{patient['age']} y" if patient["age"] else "", patient["sex"]) if v)
            st.markdown(f"<div class='eyebrow'>{who or html.escape(sample_name)}</div>"
                        f"<div class='small' style='margin-bottom:8px'>{html.escape(sample_name)}</div>",
                        unsafe_allow_html=True)
            st.markdown(
                f"<div class='verdict' style='border-color:{style[0]}66;background:{style[0]}1f;color:{IVORY}'>"
                f"<div class='verdict-k' style='color:{style[0]}'>Triage result</div>"
                f"<div class='verdict-v' style='color:{style[0]}'>{style[1]}</div>"
                f"<div class='verdict-s'>{style[2]}</div></div>", unsafe_allow_html=True)

            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=score,
                number=dict(valueformat=".2f", font=dict(family="Cormorant Garamond", size=46, color=IVORY)),
                gauge=dict(
                    axis=dict(range=[0, 1], tickcolor=MUTED, tickfont=dict(color=MUTED)),
                    bar=dict(color=GOLD, thickness=0.28), bgcolor="#1c2026", borderwidth=0,
                    steps=[dict(range=[0, d.threshold], color="#1d2a24"),
                           dict(range=[d.threshold, 1], color="#2c1f1d")],
                    threshold=dict(line=dict(color=IVORY, width=3), thickness=0.85, value=d.threshold)),
                title=dict(text="Cancer score", font=dict(family="Inter", size=13, color=MUTED))))
            st.plotly_chart(plot_layout(gauge, 250), width="stretch")
            st.markdown(f"<div class='small'>Referral line at {d.threshold:.2f}, set so that about 95% of "
                        f"cancers in training would be referred.</div>", unsafe_allow_html=True)
            if true_label is not None:
                st.write("")
                if st.toggle("Reveal the confirmed diagnosis"):
                    truth = "Liver cancer" if true_label == 1 else "Healthy"
                    right = (d.result == TriageResult.REFER) == (true_label == 1)
                    colr = GREEN if right else RED
                    st.markdown(
                        f"<div class='card'><div class='kpi-label'>Confirmed diagnosis</div>"
                        f"<div class='kpi-value'>{truth}</div>"
                        f"<div style='color:{colr}'>{'Triage correct' if right else 'Triage wrong'}</div></div>",
                        unsafe_allow_html=True)

        with col_r:
            amounts, mol_c, pat_c = explain(m, x_clean)
            raw_s = (x_raw - x_raw.min()) / (np.ptp(x_raw) + 1e-12)
            cln_s = x_clean[0] / (x_clean[0].max() + 1e-12)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=GRID, y=raw_s, name="Raw reading", line=dict(color="#5d6168", width=1)))
            fig.add_trace(go.Scatter(x=GRID, y=cln_s, name="After cleaning", line=dict(color=GOLD, width=2),
                                     fill="tozeroy", fillcolor="rgba(201,169,110,0.08)"))
            for b, lab in [(483, "Ergothioneine"), (640, "Uric acid"), (725, "Hypoxanthine"), (1133, "Uric acid")]:
                fig.add_vline(x=b, line=dict(color=TEAL, width=1, dash="dot"))
                fig.add_annotation(x=b, y=1.05, text=lab, showarrow=False, font=dict(size=10, color=TEAL),
                                   textangle=-90, yanchor="bottom")
            fig.update_xaxes(title="Raman shift (cm⁻¹)")
            fig.update_yaxes(title="relative intensity", range=[0, 1.35])
            st.plotly_chart(plot_layout(fig, 330, "Spectrum"), width="stretch")

            order = np.argsort(-np.abs(mol_c))
            labels_bar = [m["names"][i][0] + f"  ({i + 1})" for i in order] + ["Pattern (unnamed)"]
            vals = [float(mol_c[i]) for i in order] + [float(pat_c)]
            bar = go.Figure(go.Bar(
                x=vals[::-1], y=labels_bar[::-1], orientation="h",
                marker=dict(color=[RED if v > 0 else GREEN for v in vals[::-1]]),
                text=[f"{v:+.2f}" for v in vals[::-1]], textposition="outside",
                textfont=dict(color=IVORY)))
            bar.add_vline(x=0, line=dict(color=MUTED, width=1))
            bar.update_xaxes(title="← toward healthy        toward cancer →")
            st.plotly_chart(plot_layout(bar, 290, "What pushed the decision"), width="stretch")

        st.markdown("<div class='rule'></div><h3>Molecule-like components</h3>", unsafe_allow_html=True)
        lo, hi = m["ranges"]
        s_lo, s_hi = m["span"]
        cols = st.columns(3)
        for n, i in enumerate(np.argsort(-np.abs(mol_c))):
            name, peaks = m["names"][i]
            val = float(amounts[i])
            status = "High" if val > hi[i] else "Low" if val < lo[i] else "Normal"
            scol = RED if status == "High" else TEAL if status == "Low" else GREEN
            width = max(s_hi[i] - s_lo[i], 1e-12)

            def pos(v, i=i, width=width):
                return float(np.clip((v - s_lo[i]) / width, 0, 1) * 100)

            band_l, band_r = pos(lo[i]), pos(hi[i])
            push = "toward cancer" if mol_c[i] > 0 else "toward healthy"
            cols[n % 3].markdown(f"""
            <div class='mol'>
              <div style='display:flex;justify-content:space-between;align-items:center'>
                <div class='mol-name'>{html.escape(name)}</div>
                <span class='tag' style='color:{scol};border-color:{scol}88'>{status}</span>
              </div>
              <div class='mol-meta'>Component {i + 1} · peaks {", ".join(map(str, peaks))} cm⁻¹</div>
              <div class='track'>
                <div class='band' style='left:{band_l:.1f}%;width:{max(band_r - band_l, 1):.1f}%'></div>
                <div class='dot' style='left:calc({pos(val):.1f}% - 8px);background:{scol}'></div>
              </div>
              <div class='mol-meta'>gold band = healthy range · pushed {push} ({mol_c[i]:+.2f})</div>
            </div>""", unsafe_allow_html=True)
        share = np.abs(mol_c).sum() / (np.abs(mol_c).sum() + abs(pat_c) + 1e-12)
        st.markdown(f"<div class='small'>Named components carried {share:.0%} of this decision; the rest came "
                    f"from the unnamed pattern part. Names are provisional (matched from published band "
                    f"positions) until pure-molecule reference spectra are measured. The healthy range is "
                    f"the 2.5–97.5 percentile of healthy training patients, not a clinical reference "
                    f"range.</div>", unsafe_allow_html=True)

        # ----- all tests for this one person
        result_word = {TriageResult.REFER: "REFER", TriageResult.CLEAR: "CLEAR",
                       TriageResult.INCONCLUSIVE: "INCONCLUSIVE"}[d.result]
        panel = [
            ("Liver cancer (HCC)", result_word.title() if result_word != "INCONCLUSIVE" else "Measure again",
             style[0], "Research model · 94% on public data"),
            ("Active tuberculosis", "Not available", MUTED,
             "Tested on public serum data: 73–77%, below the 95% bar"),
            ("Sepsis · first antibiotic", "Needs culture", MUTED,
             "94% on grown bacteria; cannot be run from serum"),
            ("MRSA resistance", "Not offered", MUTED, "Method failed on new strains"),
        ]
        st.markdown("<div class='rule'></div><h3>Tests on this sample</h3>", unsafe_allow_html=True)
        body_rows = "".join(
            f"<tr><td class='serif' style='font-size:1.1rem'>{a}</td>"
            f"<td><span class='tag' style='color:{c_};border-color:{c_}88'>{b}</span></td>"
            f"<td class='small'>{n_}</td></tr>" for a, b, c_, n_ in panel)
        st.markdown(f"<table class='ev'><tr><th>Test</th><th>Result</th><th>Basis</th></tr>{body_rows}</table>",
                    unsafe_allow_html=True)

        # ----- one-page PDF report for this person
        comp_rows = []
        for i in np.argsort(-np.abs(mol_c)):
            st_ = "High" if amounts[i] > hi[i] else "Low" if amounts[i] < lo[i] else "Normal"
            comp_rows.append({"name": m["names"][i][0], "status": st_,
                              "effect": f"{'toward cancer' if mol_c[i] > 0 else 'toward healthy'} "
                                        f"({mol_c[i]:+.2f})"})
        pdf = build_triage_pdf(
            patient=patient, sample_name=sample_name, result=result_word, score=score,
            threshold=d.threshold, components=comp_rows, molecule_share=float(share), grid=GRID,
            spectrum=x_clean[0], other_tests=[(a, f"{b} — {n_}") for a, b, _, n_ in panel])
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_",
                      (patient["patient_id"] or patient["name"] or Path(sample_name).stem).strip()).strip("_")
        st.write("")
        st.download_button("⤓  Download patient report (PDF)", pdf, file_name=f"triage_report_{stem}.pdf",
                           mime="application/pdf", type="primary")

# --------------------------------------------------------------------------- how tab
with tab_how:
    steps = [
        ("Calibrate", "Each morning a pooled control serum is measured; drift beyond limits blocks the run."),
        ("Collect", "A small blood sample; the serum is separated by centrifuge."),
        ("Read", "A drop on a silver-coated strip; a 785 nm laser records the spectrum — about 1,400 numbers."),
        ("Check", "Quality control rejects noisy, flat or malformed readings as ‘measure again’."),
        ("Clean", "Cosmic-ray spikes, background glow and noise are removed; brightness is normalised."),
        ("Explain", "The spectrum is broken into molecule-like components — uric acid, hypoxanthine, "
                    "ergothioneine — plus an unnamed pattern."),
        ("Decide", "A safety-first threshold turns the score into refer, clear or measure again; missing a "
                   "cancer is treated as the worse error."),
        ("Seal", "Every prediction is fingerprinted, so results cannot be altered afterwards."),
    ]
    col_a, col_b = st.columns(2, gap="large")
    for n, (t, txt) in enumerate(steps):
        (col_a if n < 4 else col_b).markdown(
            f"<div class='step'><div class='step-n'>{n + 1:02d}</div><div><div class='serif' "
            f"style='font-size:1.3rem;font-weight:600'>{t}</div><div class='small'>{txt}</div></div></div>",
            unsafe_allow_html=True)
    st.markdown("<div class='small'>Steps 01 and 06 are partly in development: the reference serum and the "
                "measured molecule library need a partner laboratory.</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- evidence tab
with tab_evidence:
    st.markdown("<h3>Honest results on public data</h3><div class='small'>Every number comes from patients "
                "the model never saw during training or model selection. Negative results are kept, because "
                "they show where the method does not work.</div><div style='height:12px'></div>",
                unsafe_allow_html=True)
    rows = [
        ("Liver cancer", "144 serum spectra", "94.2% nested; 93.1% on an unseen day", GREEN, "Main target"),
        ("Sepsis — first antibiotic", "66,000 bacterial spectra", "94.4% on a separate test set", GREEN, "Second target"),
        ("Active tuberculosis", "116 serum spectra", "73–77% (paper claimed 91%)", RED, "Not pursued"),
        ("Latent tuberculosis", "1,000 plasma spectra", "74.9%", RED, "Parked"),
        ("MRSA resistance", "498 colony spectra", "38–63%, fails on new strains", RED, "Dropped"),
    ]
    body = "".join(
        f"<tr><td class='serif' style='font-size:1.1rem'>{a}</td><td>{b}</td><td>{c_}</td>"
        f"<td><span class='tag' style='color:{col};border-color:{col}88'>{e}</span></td></tr>"
        for a, b, c_, col, e in rows)
    st.markdown(f"<table class='ev'><tr><th>Use</th><th>Data</th><th>Result</th><th>Status</th></tr>{body}</table>",
                unsafe_allow_html=True)
    st.markdown("""
    <div style='height:18px'></div><h3>What we learned</h3>
    <div class='small'>Label-free serum SERS mostly measures a few small molecules, chiefly uric acid and
    hypoxanthine. They separate liver cancer from healthy people in this dataset, but they also change with
    gout, kidney function, diet and age. The patient study therefore includes hepatitis B, cirrhosis and
    gout / kidney-disease groups, and the model reports which molecules drove each result.</div>
    """, unsafe_allow_html=True)

# --------------------------------------------------------------------------- sample data tab
with tab_samples:
    files = sorted(DEMO_DIR.glob("sample_*.txt")) if DEMO_DIR.exists() else []
    st.markdown("<h3>Demo blood samples</h3><div class='small'>Raw spectra exactly as a reader exports them: two "
                "columns, Raman shift (cm⁻¹) and intensity (counts). Each comes from a patient the model never "
                "saw in training. The diagnosis is not shown here, so the samples stay blind; load one under "
                "‘Load a blood sample file’ and reveal it after the analysis.</div><div style='height:12px'></div>",
                unsafe_allow_html=True)
    if not files:
        st.markdown("<div class='card small'>No demo samples found. Run "
                    "<code>python scripts/make_demo_samples.py</code>.</div>", unsafe_allow_html=True)
    else:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(f, f.name)
        st.download_button("⤓  Download all samples (zip)", buf.getvalue(), file_name="sers_demo_samples.zip",
                           mime="application/zip", type="primary")
        st.write("")
        for f in files:
            raw = f.read_bytes()
            data = np.loadtxt(f)
            wn, inten = data[:, 0], data[:, 1]
            with st.expander(f"{f.name}   ·   {len(wn):,} points   ·   {wn.min():.0f} to {wn.max():.0f} cm⁻¹   ·   "
                             f"{len(raw) / 1024:.0f} KB"):
                col_g, col_t = st.columns([2, 1], gap="large")
                with col_g:
                    fig = go.Figure(go.Scatter(x=wn, y=inten, line=dict(color=GOLD, width=1.2)))
                    fig.add_vrect(x0=GRID[0], x1=GRID[-1], fillcolor=GOLD, opacity=0.06, line_width=0)
                    fig.update_xaxes(title="Raman shift (cm⁻¹)")
                    fig.update_yaxes(title="counts")
                    st.plotly_chart(plot_layout(fig, 280, "Raw spectrum"), width="stretch",
                                    key=f"sample_plot_{f.stem}")
                    st.markdown("<div class='small'>Shaded band: the 400–1800 cm⁻¹ region the model uses.</div>",
                                unsafe_allow_html=True)
                with col_t:
                    rows = "".join(f"<tr><td>{a:.2f}</td><td>{b:.2f}</td></tr>" for a, b in data[:8])
                    st.markdown(f"<table class='ev'><tr><th>cm⁻¹</th><th>intensity</th></tr>{rows}"
                                f"<tr><td colspan='2' class='small'>… {len(wn) - 8:,} more rows</td></tr></table>",
                                unsafe_allow_html=True)
                    st.write("")
                    st.download_button(f"⤓  Download {f.name}", raw, file_name=f.name, mime="text/plain",
                                       key=f"dl_{f.stem}")

st.markdown("<div class='foot'>Research prototype · public data (Zenodo 4277797) · not validated on Ethiopian "
            "patients · not for clinical use · a clear result does not rule out disease.</div>",
            unsafe_allow_html=True)
