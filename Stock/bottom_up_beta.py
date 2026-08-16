"""Pure bottom-up beta evidence and DCF translation.

This module deliberately does not fetch market data, import Streamlit, select a
production beta, or create a Research WACC.  Callers supply observed peer data
and the existing WACC/DCF inputs explicitly.
"""

from dataclasses import dataclass, replace
import math
import statistics

from Stock.beta_audit import (
    BetaWACCContext,
    implied_beta_from_target_wacc,
    wacc_from_beta,
)
from Stock.multistage_integration import RealCompanyDCFInputs, run_multistage_dcf
from Stock.valuation import MultiStageDCFAssumptions


PEER_DISPERSION_THRESHOLD = 0.50
MEAN_MEDIAN_DIFFERENCE_THRESHOLD = 0.15
HISTORICAL_DIFFERENCE_THRESHOLD = 0.40
LEAVE_ONE_OUT_THRESHOLD = 0.25
MINIMUM_VALID_PEERS = 3


@dataclass(frozen=True)
class PeerDefinition:
    ticker: str
    issuer: str
    inclusion_rationale: str


@dataclass(frozen=True)
class PeerGroupDefinition:
    issuer: str
    name: str
    peers: tuple[PeerDefinition, ...]
    exclusions: tuple[tuple[str, str], ...]
    damodaran_industries: tuple[tuple[str, str], ...]


NVDA_PEER_GROUP = PeerGroupDefinition(
    issuer="NVIDIA Corporation",
    name="Accelerated computing and fabless semiconductor peers",
    peers=(
        PeerDefinition("AMD", "Advanced Micro Devices", "Direct GPU and accelerated-computing competitor."),
        PeerDefinition("AVGO", "Broadcom", "AI connectivity and custom-accelerator semiconductor exposure."),
        PeerDefinition("QCOM", "Qualcomm", "Large fabless semiconductor designer with platform economics."),
        PeerDefinition("MRVL", "Marvell Technology", "Data-infrastructure and custom silicon exposure."),
    ),
    exclusions=(
        ("INTC", "Integrated manufacturing and restructuring make capital structure less comparable."),
        ("TXN", "Analog and embedded exposure is less relevant to accelerated computing."),
        ("ADI", "Predominantly analog semiconductor economics are less comparable."),
    ),
    damodaran_industries=(
        ("Semiconductor", "Primary broad semiconductor reference; mixes end markets."),
        ("Semiconductor Equip", "Adjacent AI-infrastructure supply chain, not a direct operating peer set."),
        ("Computers/Peripherals", "Computing label is plausible but includes different hardware economics."),
    ),
)


ALPHABET_PEER_GROUP = PeerGroupDefinition(
    issuer="Alphabet Inc.",
    name="Large digital platform, advertising, and cloud peers",
    peers=(
        PeerDefinition("META", "Meta Platforms", "Closest scaled digital-advertising and consumer-platform peer."),
        PeerDefinition("MSFT", "Microsoft", "Scaled cloud and software platform with AI infrastructure exposure."),
        PeerDefinition("AMZN", "Amazon", "Scaled cloud, digital platform, and advertising exposure."),
    ),
    exclusions=(
        ("AAPL", "Hardware-led economics are not sufficiently comparable to Alphabet's core businesses."),
        ("NFLX", "Subscription entertainment has limited advertising and cloud comparability."),
    ),
    damodaran_industries=(
        ("Advertising", "Captures the core advertising engine but not cloud or platform breadth."),
        ("Software (Internet)", "Captures internet-platform exposure but is a broad, high-dispersion group."),
        ("Software (System & Application)", "Captures software/cloud exposure but not advertising economics."),
    ),
)


def peer_group_for_target(ticker: str) -> PeerGroupDefinition | None:
    normalized = ticker.strip().upper()
    if normalized == "NVDA":
        return NVDA_PEER_GROUP
    if normalized in {"GOOG", "GOOGL"}:
        return ALPHABET_PEER_GROUP
    return None


@dataclass(frozen=True)
class PeerBetaInput:
    ticker: str
    issuer: str
    inclusion_rationale: str
    levered_beta: float | None
    adjusted_beta: float | None
    beta_method: str
    market_cap: float | None
    gross_debt: float | None
    tax_rate: float | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PeerBetaObservation:
    ticker: str
    issuer: str
    inclusion_rationale: str
    levered_beta: float | None
    adjusted_beta: float | None
    beta_method: str
    market_cap: float | None
    gross_debt: float | None
    tax_rate: float | None
    debt_to_equity: float | None
    unlevered_beta: float | None
    adjusted_unlevered_beta: float | None
    valid: bool
    reason: str | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class BetaDistribution:
    minimum: float | None
    maximum: float | None
    median: float | None
    mean: float | None
    standard_deviation: float | None
    count: int


@dataclass(frozen=True)
class LeaveOneOutRange:
    median_minimum: float | None
    median_maximum: float | None
    mean_minimum: float | None
    mean_maximum: float | None

    @property
    def maximum_span(self) -> float | None:
        spans = []
        if self.median_minimum is not None and self.median_maximum is not None:
            spans.append(self.median_maximum - self.median_minimum)
        if self.mean_minimum is not None and self.mean_maximum is not None:
            spans.append(self.mean_maximum - self.mean_minimum)
        return max(spans) if spans else None


@dataclass(frozen=True)
class IndustryBetaReference:
    industry: str
    number_of_firms: int | None
    levered_beta: float | None
    unlevered_beta: float | None
    debt_to_equity: float | None
    source_date: str | None
    mapping_note: str


@dataclass(frozen=True)
class BottomUpBetaResult:
    target_ticker: str
    issuer: str
    peer_group_name: str
    peer_observations: tuple[PeerBetaObservation, ...]
    exclusion_rationales: tuple[tuple[str, str], ...]
    valid_peer_count: int
    raw_unlevered_distribution: BetaDistribution
    adjusted_unlevered_distribution: BetaDistribution
    peer_unlevered_beta_median: float | None
    peer_unlevered_beta_mean: float | None
    target_debt_to_equity: float | None
    target_tax_rate: float | None
    relevered_beta_median: float | None
    relevered_beta_mean: float | None
    adjusted_relevered_beta_median: float | None
    adjusted_relevered_beta_mean: float | None
    raw_leave_one_out: LeaveOneOutRange
    adjusted_leave_one_out: LeaveOneOutRange
    industry_references: tuple[IndustryBetaReference, ...]
    warnings: tuple[str, ...]
    classification: str


@dataclass(frozen=True)
class BetaEvidencePoint:
    evidence_method: str
    beta: float | None
    beta_description: str
    formula_based_wacc: float
    intrinsic_value_per_share: float | None
    reason: str | None


@dataclass(frozen=True)
class BottomUpBetaEvidenceComparison:
    target_ticker: str
    issuer: str
    points: tuple[BetaEvidencePoint, ...]


def _finite(value: float | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def unlever_beta(
    levered_beta: float,
    gross_debt: float,
    equity: float,
    tax_rate: float,
) -> float:
    """Unlever beta using gross debt and current equity market value."""
    values = tuple(map(_finite, (levered_beta, gross_debt, equity, tax_rate)))
    if any(value is None for value in values):
        raise ValueError("unlevering_inputs_must_be_finite")
    beta, debt, market_equity, tax = values
    if market_equity <= 0:
        raise ValueError("equity_must_be_positive")
    if debt < 0:
        raise ValueError("gross_debt_must_be_non_negative")
    if not 0 <= tax <= 1:
        raise ValueError("tax_rate_must_be_between_zero_and_one")
    return beta / (1 + (1 - tax) * debt / market_equity)


def relever_beta(
    unlevered_beta: float,
    gross_debt: float,
    equity: float,
    tax_rate: float,
) -> float:
    """Relever a peer beta reference to the target's gross D/E."""
    values = tuple(map(_finite, (unlevered_beta, gross_debt, equity, tax_rate)))
    if any(value is None for value in values):
        raise ValueError("relevering_inputs_must_be_finite")
    beta, debt, market_equity, tax = values
    if market_equity <= 0:
        raise ValueError("equity_must_be_positive")
    if debt < 0:
        raise ValueError("gross_debt_must_be_non_negative")
    if not 0 <= tax <= 1:
        raise ValueError("tax_rate_must_be_between_zero_and_one")
    return beta * (1 + (1 - tax) * debt / market_equity)


def build_peer_observation(peer: PeerBetaInput) -> PeerBetaObservation:
    beta = _finite(peer.levered_beta)
    adjusted = _finite(peer.adjusted_beta)
    equity = _finite(peer.market_cap)
    debt = _finite(peer.gross_debt)
    tax = _finite(peer.tax_rate)
    reason = None
    if beta is None or adjusted is None:
        reason = "invalid_or_insufficient_beta"
    elif equity is None or equity <= 0:
        reason = "invalid_market_cap"
    elif debt is None or debt < 0:
        reason = "invalid_gross_debt"
    elif tax is None or not 0 <= tax <= 1:
        reason = "invalid_tax_rate"
    if reason is not None:
        return PeerBetaObservation(
            peer.ticker, peer.issuer, peer.inclusion_rationale, beta, adjusted,
            peer.beta_method, equity, debt, tax, None, None, None, False,
            reason, tuple(dict.fromkeys((*peer.warnings, reason))),
        )
    debt_to_equity = debt / equity
    return PeerBetaObservation(
        peer.ticker, peer.issuer, peer.inclusion_rationale, beta, adjusted,
        peer.beta_method, equity, debt, tax, debt_to_equity,
        unlever_beta(beta, debt, equity, tax),
        unlever_beta(adjusted, debt, equity, tax), True, None,
        tuple(dict.fromkeys(peer.warnings)),
    )


def _distribution(values) -> BetaDistribution:
    valid = tuple(value for value in map(_finite, values) if value is not None)
    if not valid:
        return BetaDistribution(None, None, None, None, None, 0)
    return BetaDistribution(
        minimum=min(valid), maximum=max(valid), median=statistics.median(valid),
        mean=statistics.mean(valid),
        standard_deviation=statistics.pstdev(valid) if len(valid) > 1 else 0.0,
        count=len(valid),
    )


def _leave_one_out(values, debt: float, equity: float, tax: float) -> LeaveOneOutRange:
    values = tuple(values)
    median_values = []
    mean_values = []
    if len(values) >= 2:
        for index in range(len(values)):
            remaining = values[:index] + values[index + 1:]
            median_values.append(relever_beta(statistics.median(remaining), debt, equity, tax))
            mean_values.append(relever_beta(statistics.mean(remaining), debt, equity, tax))
    return LeaveOneOutRange(
        min(median_values) if median_values else None,
        max(median_values) if median_values else None,
        min(mean_values) if mean_values else None,
        max(mean_values) if mean_values else None,
    )


def build_bottom_up_beta_result(
    *,
    target_ticker: str,
    issuer: str,
    peer_group_name: str,
    peer_inputs: tuple[PeerBetaInput, ...],
    target_market_cap: float | None,
    target_gross_debt: float | None,
    target_tax_rate: float | None,
    historical_raw_beta: float | None = None,
    exclusion_rationales: tuple[tuple[str, str], ...] = (),
    industry_references: tuple[IndustryBetaReference, ...] = (),
    industry_mapping_ambiguous: bool = False,
) -> BottomUpBetaResult:
    """Build transparent peer distributions without choosing a preferred beta."""
    normalized = target_ticker.strip().upper()
    issuer_key = "ALPHABET_INC" if normalized in {"GOOG", "GOOGL"} else issuer
    observations = tuple(build_peer_observation(peer) for peer in peer_inputs)
    valid = tuple(observation for observation in observations if observation.valid)
    raw_values = tuple(observation.unlevered_beta for observation in valid)
    adjusted_values = tuple(observation.adjusted_unlevered_beta for observation in valid)
    raw_distribution = _distribution(raw_values)
    adjusted_distribution = _distribution(adjusted_values)

    equity = _finite(target_market_cap)
    debt = _finite(target_gross_debt)
    tax = _finite(target_tax_rate)
    target_inputs_valid = (
        equity is not None and equity > 0 and debt is not None and debt >= 0
        and tax is not None and 0 <= tax <= 1
    )
    target_de = debt / equity if target_inputs_valid else None

    def target_beta(value):
        return relever_beta(value, debt, equity, tax) if value is not None and target_inputs_valid else None

    raw_relevered_median = target_beta(raw_distribution.median)
    raw_relevered_mean = target_beta(raw_distribution.mean)
    adjusted_relevered_median = target_beta(adjusted_distribution.median)
    adjusted_relevered_mean = target_beta(adjusted_distribution.mean)
    raw_loo = (
        _leave_one_out(raw_values, debt, equity, tax)
        if target_inputs_valid else LeaveOneOutRange(None, None, None, None)
    )
    adjusted_loo = (
        _leave_one_out(adjusted_values, debt, equity, tax)
        if target_inputs_valid else LeaveOneOutRange(None, None, None, None)
    )

    warnings = []
    if len(valid) < MINIMUM_VALID_PEERS:
        warnings.append("insufficient_valid_peers")
    if not target_inputs_valid:
        warnings.append("invalid_target_capital_structure_or_tax")
    if (
        raw_distribution.minimum is not None
        and raw_distribution.maximum - raw_distribution.minimum >= PEER_DISPERSION_THRESHOLD
    ):
        warnings.append("peer_beta_dispersion_high")
    if (
        raw_distribution.mean is not None and raw_distribution.median is not None
        and abs(raw_distribution.mean - raw_distribution.median)
        >= MEAN_MEDIAN_DIFFERENCE_THRESHOLD
    ):
        warnings.append("peer_mean_median_materially_different")
    if raw_loo.maximum_span is not None and raw_loo.maximum_span >= LEAVE_ONE_OUT_THRESHOLD:
        warnings.append("peer_result_sensitive_to_single_company")
    historical = _finite(historical_raw_beta)
    if (
        historical is not None and raw_relevered_median is not None
        and abs(raw_relevered_median - historical) >= HISTORICAL_DIFFERENCE_THRESHOLD
    ):
        warnings.append("bottom_up_beta_far_from_historical_beta")
    if industry_mapping_ambiguous:
        warnings.append("target_industry_mapping_ambiguous")
    if any(not observation.valid for observation in observations):
        warnings.append("invalid_peers_excluded")
    warnings = list(dict.fromkeys(warnings))

    if (
        len(valid) < MINIMUM_VALID_PEERS
        or "peer_result_sensitive_to_single_company" in warnings
        or (
            raw_distribution.minimum is not None
            and raw_distribution.maximum - raw_distribution.minimum >= 0.75
        )
    ):
        classification = "highly_peer_sensitive"
    elif any(
        warning in {
            "peer_beta_dispersion_high",
            "peer_mean_median_materially_different",
            "invalid_peers_excluded",
        }
        for warning in warnings
    ):
        classification = "moderately_peer_sensitive"
    else:
        classification = "reasonably_stable_peer_reference"

    return BottomUpBetaResult(
        target_ticker=normalized, issuer=issuer_key, peer_group_name=peer_group_name,
        peer_observations=observations, exclusion_rationales=exclusion_rationales,
        valid_peer_count=len(valid), raw_unlevered_distribution=raw_distribution,
        adjusted_unlevered_distribution=adjusted_distribution,
        peer_unlevered_beta_median=raw_distribution.median,
        peer_unlevered_beta_mean=raw_distribution.mean,
        target_debt_to_equity=target_de, target_tax_rate=tax,
        relevered_beta_median=raw_relevered_median,
        relevered_beta_mean=raw_relevered_mean,
        adjusted_relevered_beta_median=adjusted_relevered_median,
        adjusted_relevered_beta_mean=adjusted_relevered_mean,
        raw_leave_one_out=raw_loo, adjusted_leave_one_out=adjusted_loo,
        industry_references=industry_references, warnings=tuple(warnings),
        classification=classification,
    )


def build_beta_evidence_comparison(
    *,
    inputs: RealCompanyDCFInputs,
    base_assumptions: MultiStageDCFAssumptions,
    wacc_context: BetaWACCContext,
    historical_raw_beta: float,
    historical_adjusted_beta: float,
    bottom_up_result: BottomUpBetaResult,
) -> BottomUpBetaEvidenceComparison:
    """Translate beta evidence through existing WACC and full DCF orchestration."""
    candidates = (
        (
            "Provisional DCF Default",
            implied_beta_from_target_wacc(base_assumptions.wacc, wacc_context),
            "Beta implied by provisional DCF default",
            base_assumptions.wacc,
        ),
        ("Historical Raw", historical_raw_beta, "5Y monthly raw regression beta", None),
        ("Historical Adjusted", historical_adjusted_beta, "Blume-adjusted historical beta", None),
        ("Bottom-Up Median", bottom_up_result.relevered_beta_median, "Peer median unlevered beta relevered to target", None),
        ("Bottom-Up Mean", bottom_up_result.relevered_beta_mean, "Peer mean unlevered beta relevered to target", None),
    )
    points = []
    for label, beta, description, fixed_wacc in candidates:
        numeric_beta = _finite(beta)
        if numeric_beta is None:
            points.append(BetaEvidencePoint(label, None, description, math.nan, None, "beta_unavailable"))
            continue
        formula_wacc = fixed_wacc if fixed_wacc is not None else wacc_from_beta(numeric_beta, wacc_context)
        try:
            run = run_multistage_dcf(inputs, replace(base_assumptions, wacc=formula_wacc))
            value = run.per_share_value.intrinsic_value_per_share if run.per_share_value else None
            reason = run.per_share_unavailable_reason
        except (TypeError, ValueError) as exc:
            value, reason = None, str(exc)
        points.append(BetaEvidencePoint(label, numeric_beta, description, formula_wacc, value, reason))
    return BottomUpBetaEvidenceComparison(
        target_ticker=bottom_up_result.target_ticker,
        issuer=bottom_up_result.issuer,
        points=tuple(points),
    )
