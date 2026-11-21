"""Per-biomarker template selection and lab branding."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reports.generator import ReportType


@dataclass
class LabBranding:
    logo_path: str = ""
    primary_color: str = "#003366"
    secondary_color: str = "#666666"
    footer_text: str = ""
    header_text: str = ""


@dataclass
class BiomarkerTemplate:
    biomarker_id: str
    report_type: ReportType
    method_description: str
    method_limitations: str = ""
    reference_population: str = ""
    follow_up_text: str = ""


class TemplateRegistry:
    """Registry for per-biomarker report template customizations."""

    def __init__(self, branding: LabBranding | None = None):
        self.branding = branding or LabBranding()
        self._templates: dict[str, BiomarkerTemplate] = {}

    def register(self, template: BiomarkerTemplate) -> None:
        self._templates[template.biomarker_id] = template

    def get(self, biomarker_id: str) -> BiomarkerTemplate | None:
        return self._templates.get(biomarker_id)

    def get_report_type(self, biomarker_id: str) -> ReportType:
        tmpl = self._templates.get(biomarker_id)
        if tmpl:
            return tmpl.report_type
        return ReportType.QUANTITATIVE

    def list_biomarkers(self) -> list[str]:
        return list(self._templates.keys())

    @property
    def count(self) -> int:
        return len(self._templates)
