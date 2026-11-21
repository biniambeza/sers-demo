"""One-page PDF report for a single patient's SERS triage result.

Pure function: build_triage_pdf(...) -> bytes. Used by triage_app.py; testable
without Streamlit. The report states on every page that it is a research
prototype and not for clinical use, and that a CLEAR result does not rule
out disease. Suggested next steps are placeholders until approved by the
clinical partner.
"""
from __future__ import annotations

import hashlib
import io
from datetime import datetime
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

GOLD = colors.HexColor("#a8884d")
INK = colors.HexColor("#1b1d22")
MUTED = colors.HexColor("#6b675f")
LINE = colors.HexColor("#d9d3c7")
RESULT_COLOR = {"REFER": colors.HexColor("#b4473c"), "CLEAR": colors.HexColor("#3f7a5e"),
                "INCONCLUSIVE": colors.HexColor("#b07a2a")}

NEXT_STEPS = {
    "REFER": "Arrange a liver ultrasound and serum AFP, and clinical review.",
    "CLEAR": "No referral is indicated by this test. This does not rule out liver cancer; "
             "follow clinical judgement and local guidelines.",
    "INCONCLUSIVE": "Repeat the measurement with a new sample. If it is inconclusive again, refer.",
}


def report_id(spectrum: np.ndarray, when: datetime) -> str:
    h = hashlib.sha256(np.asarray(spectrum, dtype=np.float64).tobytes() + when.isoformat().encode())
    return h.hexdigest()[:12].upper()


def _spectrum_png(grid: np.ndarray, spectrum: np.ndarray) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(7.2, 2.0), dpi=160)
    y = spectrum / (spectrum.max() + 1e-12)
    ax.plot(grid, y, color="#a8884d", lw=1.1)
    ax.fill_between(grid, y, color="#a8884d", alpha=0.10)
    for b, lab in [(483, "Ergothioneine"), (640, "Uric acid"), (725, "Hypoxanthine"), (1133, "Uric acid")]:
        ax.axvline(b, color="#5f8a99", lw=0.6, ls=":")
        ax.text(b, 1.02, lab, rotation=90, fontsize=5.5, color="#5f8a99", ha="center", va="bottom")
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, 1.45)
    ax.set_xlabel("Raman shift (cm$^{-1}$)", fontsize=7)
    ax.set_yticks([])
    ax.tick_params(labelsize=6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def build_triage_pdf(
    patient: dict,
    sample_name: str,
    result: str,
    score: float,
    threshold: float,
    components: Sequence[dict],
    molecule_share: float,
    grid: np.ndarray,
    spectrum: np.ndarray,
    other_tests: Sequence[tuple[str, str]],
    when: datetime | None = None,
) -> bytes:
    """Build the PDF.

    patient      keys: name, patient_id, age, sex, clinic (any may be empty)
    result       "REFER" | "CLEAR" | "INCONCLUSIVE"
    components   dicts with keys: name, status, effect
    other_tests  (test name, status text) for the panel
    """
    when = when or datetime.now()
    rid = report_id(spectrum, when)
    ss = getSampleStyleSheet()
    h_title = ParagraphStyle("t", parent=ss["Title"], fontName="Times-Bold", fontSize=20, textColor=INK,
                             alignment=TA_LEFT, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Times-Bold", fontSize=12.5, textColor=INK,
                        spaceBefore=8, spaceAfter=4)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=8.6, leading=11.5,
                          textColor=INK)
    small = ParagraphStyle("s", parent=body, fontSize=7.2, leading=9.2, textColor=MUTED)
    eyebrow = ParagraphStyle("e", parent=small, textColor=GOLD, fontName="Helvetica-Bold")

    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=13 * mm, bottomMargin=13 * mm,
                            title=f"SERS triage report {rid}", author="SERS research prototype")
    story = []

    banner = Table([[Paragraph("<b>RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.</b> Trained on public data "
                               "(72 liver cancer, 72 healthy; one European laboratory). Not validated on "
                               "Ethiopian patients.", ParagraphStyle("bn", parent=small, textColor=colors.white))]],
                   colWidths=[178 * mm])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), INK), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [banner, Spacer(1, 8),
              Paragraph("SERS BLOOD TRIAGE · HEPATOCELLULAR CARCINOMA", eyebrow),
              Paragraph("Patient triage report", h_title),
              Paragraph(f"Report {rid} · {when:%d %B %Y, %H:%M}", small), Spacer(1, 8)]

    def val(k):
        v = str(patient.get(k) or "").strip()
        return v if v else "—"

    pt = Table([["Patient", val("name"), "Patient ID", val("patient_id")],
                ["Age", val("age"), "Sex", val("sex")],
                ["Clinic", val("clinic"), "Sample file", sample_name]],
               colWidths=[24 * mm, 65 * mm, 24 * mm, 65 * mm])
    pt.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 8.6),
                            ("TEXTCOLOR", (0, 0), (0, -1), MUTED), ("TEXTCOLOR", (2, 0), (2, -1), MUTED),
                            ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
                            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story += [pt, Spacer(1, 10)]

    col = RESULT_COLOR.get(result, INK)
    label = {"REFER": "Refer", "CLEAR": "Clear", "INCONCLUSIVE": "Measure again"}.get(result, result)
    v_small = ParagraphStyle("vs", parent=body, fontSize=7.5, leading=10, textColor=colors.white)
    v_big = ParagraphStyle("vb", parent=body, fontName="Times-Bold", fontSize=22, leading=25,
                           textColor=colors.white)
    verdict = Table([[[Paragraph("LIVER CANCER TRIAGE", v_small), Paragraph(label, v_big)],
                      Paragraph(f"<font color='#ffffff'>Cancer score <b>{score:.2f}</b> · referral line "
                                f"{threshold:.2f}<br/>{NEXT_STEPS.get(result, '')}</font>", body)]],
                    colWidths=[55 * mm, 123 * mm])
    verdict.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), col), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 9),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 9)]))
    story += [verdict, Spacer(1, 4),
              Paragraph("The referral line is set so that about 95% of cancers in the training data are "
                        "referred. A CLEAR result does not rule out cancer.", small)]

    story.append(Paragraph("Tests on this sample", h2))
    panel = [["Test", "Status"]] + [[a, b] for a, b in other_tests]
    tp = Table(panel, colWidths=[60 * mm, 118 * mm])
    tp.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 8.2),
                            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.5),
                            ("TEXTCOLOR", (0, 0), (-1, 0), MUTED), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
                            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story.append(tp)

    story.append(Paragraph("Why: molecule-like components", h2))
    comp = [["Component", "Versus healthy range", "Effect on decision"]] + \
           [[c["name"], c["status"], c["effect"]] for c in components]
    tc = Table(comp, colWidths=[78 * mm, 45 * mm, 55 * mm])
    tc.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 8.2),
                            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.5),
                            ("TEXTCOLOR", (0, 0), (-1, 0), MUTED), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
                            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story += [tc, Spacer(1, 3),
              Paragraph(f"Named components carried {molecule_share:.0%} of this decision. Names are provisional "
                        "(matched to published band positions) until pure-molecule reference spectra are "
                        "measured. The healthy range is from healthy training patients, not a clinical "
                        "reference range.", small)]

    story += [Paragraph("Spectrum (after cleaning)", h2), Image(_spectrum_png(grid, spectrum), width=178 * mm,
                                                                 height=49 * mm)]

    story += [Spacer(1, 10),
              Paragraph("Suggested next steps are placeholders pending approval by the clinical partner.", small),
              Spacer(1, 14)]
    sign = Table([["Reviewed by clinician", "", "Date", ""]], colWidths=[38 * mm, 70 * mm, 14 * mm, 56 * mm])
    sign.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 8.2), ("TEXTCOLOR", (0, 0), (-1, -1), MUTED),
                              ("LINEBELOW", (1, 0), (1, 0), 0.5, INK), ("LINEBELOW", (3, 0), (3, 0), 0.5, INK)]))
    story += [sign, Spacer(1, 8),
              Paragraph(f"Model: hybrid NMF components + PCA pattern, logistic regression; sensitivity-first "
                        f"threshold. Report fingerprint {rid}.", small)]
    doc.build(story)
    return out.getvalue()
