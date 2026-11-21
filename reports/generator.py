"""Result report generation — CLIA-compliant patient result reports."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ReportType(Enum):
    QUANTITATIVE = "quantitative"
    QUALITATIVE = "qualitative"
    MULTI_ANALYTE = "multi_analyte"
    CANCER_SCREENING = "cancer_screening"
    TREND = "trend"


class ReportStatus(Enum):
    PRELIMINARY = "Preliminary"
    FINAL = "Final"
    CORRECTED = "Corrected"
    AMENDED = "Amended"


@dataclass
class PatientInfo:
    name: str
    dob: str
    mrn: str
    sex: str = ""
    ordering_provider: str = ""


@dataclass
class LabInfo:
    name: str
    address: str
    clia_number: str
    director_name: str
    director_credentials: str
    phone: str = ""


@dataclass
class ResultEntry:
    test_name: str
    value: str
    units: str
    reference_range: str
    flag: str = ""  # H, L, C (critical)
    method: str = ""
    reference_population: str = ""


@dataclass
class ReportData:
    patient: PatientInfo
    report_type: ReportType
    status: ReportStatus
    results: list[ResultEntry]
    collection_datetime: str
    result_datetime: str
    report_id: str = ""
    method_description: str = ""
    method_limitations: str = ""
    specimen_type: str = ""
    ordering_provider: str = ""
    comments: str = ""


@dataclass
class GeneratedReport:
    report_id: str
    report_type: ReportType
    html: str
    generated_at: datetime = field(default_factory=datetime.now)
    clia_fields_present: list[str] = field(default_factory=list)

    @property
    def is_clia_compliant(self) -> bool:
        required = {
            "patient_name", "patient_dob", "patient_mrn",
            "test_name", "result_value", "reference_range",
            "lab_name", "lab_address", "clia_number",
            "director_name", "collection_datetime", "result_datetime",
            "report_status", "method_description",
        }
        return required.issubset(set(self.clia_fields_present))


CLIA_REQUIRED_FIELDS = {
    "patient_name", "patient_dob", "patient_mrn",
    "test_name", "result_value", "reference_range",
    "lab_name", "lab_address", "clia_number",
    "director_name", "collection_datetime", "result_datetime",
    "report_status", "method_description",
}


class ReportGenerator:
    """Generate CLIA-compliant patient result reports."""

    def __init__(self, lab_info: LabInfo):
        self.lab = lab_info
        self._counter = 0

    def generate(self, data: ReportData) -> GeneratedReport:
        self._counter += 1
        report_id = data.report_id or f"RPT-{self._counter:06d}"

        present_fields = self._check_clia_fields(data)
        html = self._render(data, report_id)

        return GeneratedReport(
            report_id=report_id,
            report_type=data.report_type,
            html=html,
            clia_fields_present=present_fields,
        )

    def _check_clia_fields(self, data: ReportData) -> list[str]:
        present = []
        if data.patient.name:
            present.append("patient_name")
        if data.patient.dob:
            present.append("patient_dob")
        if data.patient.mrn:
            present.append("patient_mrn")
        if data.results:
            present.append("test_name")
            present.append("result_value")
            if any(r.reference_range for r in data.results):
                present.append("reference_range")
        if self.lab.name:
            present.append("lab_name")
        if self.lab.address:
            present.append("lab_address")
        if self.lab.clia_number:
            present.append("clia_number")
        if self.lab.director_name:
            present.append("director_name")
        if data.collection_datetime:
            present.append("collection_datetime")
        if data.result_datetime:
            present.append("result_datetime")
        if data.status:
            present.append("report_status")
        if data.method_description:
            present.append("method_description")
        return present

    def _render(self, data: ReportData, report_id: str) -> str:
        lines = [
            "<html><body>",
            f"<h1>Laboratory Report — {data.status.value}</h1>",
            f"<p>Report ID: {report_id}</p>",
            "<h2>Patient Information</h2>",
            f"<p>Name: {data.patient.name}</p>",
            f"<p>DOB: {data.patient.dob}</p>",
            f"<p>MRN: {data.patient.mrn}</p>",
            f"<p>Specimen: {data.specimen_type}</p>",
            f"<p>Collected: {data.collection_datetime}</p>",
            f"<p>Reported: {data.result_datetime}</p>",
        ]

        if data.report_type == ReportType.QUANTITATIVE:
            lines.extend(self._render_quantitative(data))
        elif data.report_type == ReportType.QUALITATIVE:
            lines.extend(self._render_qualitative(data))
        elif data.report_type == ReportType.MULTI_ANALYTE:
            lines.extend(self._render_multi_analyte(data))
        elif data.report_type == ReportType.CANCER_SCREENING:
            lines.extend(self._render_cancer_screening(data))
        elif data.report_type == ReportType.TREND:
            lines.extend(self._render_trend(data))

        lines.extend([
            "<h2>Method</h2>",
            f"<p>{data.method_description}</p>",
        ])
        if data.method_limitations:
            lines.append(f"<p><em>Limitations: {data.method_limitations}</em></p>")
        if data.comments:
            lines.append(f"<p>Comments: {data.comments}</p>")

        lines.extend([
            "<hr>",
            f"<p>{self.lab.name}</p>",
            f"<p>{self.lab.address}</p>",
            f"<p>CLIA: {self.lab.clia_number}</p>",
            f"<p>Director: {self.lab.director_name}, {self.lab.director_credentials}</p>",
            "</body></html>",
        ])
        return "\n".join(lines)

    def _render_quantitative(self, data: ReportData) -> list[str]:
        lines = ["<h2>Results</h2>", "<table><tr><th>Test</th><th>Value</th><th>Units</th><th>Range</th><th>Flag</th></tr>"]
        for r in data.results:
            flag = f"<b>{r.flag}</b>" if r.flag else ""
            lines.append(
                f"<tr><td>{r.test_name}</td><td>{r.value}</td>"
                f"<td>{r.units}</td><td>{r.reference_range}</td><td>{flag}</td></tr>"
            )
        lines.append("</table>")
        return lines

    def _render_qualitative(self, data: ReportData) -> list[str]:
        lines = ["<h2>Results</h2>"]
        for r in data.results:
            lines.append(f"<p><b>{r.test_name}:</b> {r.value}</p>")
            if r.reference_range:
                lines.append(f"<p>Expected: {r.reference_range}</p>")
        return lines

    def _render_multi_analyte(self, data: ReportData) -> list[str]:
        lines = ["<h2>Panel Results</h2>", "<table><tr><th>Analyte</th><th>Value</th><th>Units</th><th>Range</th><th>Flag</th></tr>"]
        for r in data.results:
            flag = f"<b>{r.flag}</b>" if r.flag else ""
            lines.append(
                f"<tr><td>{r.test_name}</td><td>{r.value}</td>"
                f"<td>{r.units}</td><td>{r.reference_range}</td><td>{flag}</td></tr>"
            )
        lines.append("</table>")
        return lines

    def _render_cancer_screening(self, data: ReportData) -> list[str]:
        lines = ["<h2>Cancer Screening Results</h2>"]
        for r in data.results:
            lines.append(f"<p><b>{r.test_name}:</b> {r.value} {r.units}</p>")
            if r.flag:
                lines.append(f"<p class='flag'>Flag: {r.flag}</p>")
        lines.append("<p><em>Follow-up recommendations should be discussed with ordering physician.</em></p>")
        return lines

    def _render_trend(self, data: ReportData) -> list[str]:
        lines = ["<h2>Longitudinal Results</h2>", "<table><tr><th>Date</th><th>Test</th><th>Value</th><th>Units</th><th>Flag</th></tr>"]
        for r in data.results:
            flag = f"<b>{r.flag}</b>" if r.flag else ""
            lines.append(
                f"<tr><td>{r.method}</td><td>{r.test_name}</td><td>{r.value}</td>"
                f"<td>{r.units}</td><td>{flag}</td></tr>"
            )
        lines.append("</table>")
        return lines
