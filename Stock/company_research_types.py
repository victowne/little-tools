"""Shared immutable types for company-specific research candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from Stock.company_profiles import CompanyProfileLookupResult
from Stock.valuation import MultiStageDCFAssumptions


ConfidenceLevel = Literal["High", "Medium", "Low"]


@dataclass(frozen=True)
class ResearchRange:
    """Evidence-supported range around one research assumption."""

    assumption_id: str
    low: float
    central: float
    high: float
    rationale: str
    evidence_references: tuple[str, ...]


@dataclass(frozen=True)
class RevenueEvidenceRow:
    """One comparable historical or forward Revenue observation."""

    label: str
    period: str | None
    revenue: float | None
    growth: float | None
    source: str
    source_date: str | None
    retrieved_at: str | None
    analyst_count: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class ConfidenceAssessment:
    """Research confidence assigned to one major assumption category."""

    category: str
    confidence: ConfidenceLevel
    rationale: str


class CompanyResearchResult(Protocol):
    """Common result contract returned by every researched company builder."""

    lookup: CompanyProfileLookupResult
    revenue_evidence: tuple[RevenueEvidenceRow, ...]
    growth_ranges: tuple[ResearchRange, ...]
    current_assumptions: MultiStageDCFAssumptions
    period_reconciliation: tuple[str, ...]
    warnings: tuple[str, ...]

