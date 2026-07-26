"""Business_Module - turn a forecast into an actionable recommendation (Requirement 7).

The whole pipeline exists to produce *business* value: a broad model comparison
and an honest evaluation are only useful if the selected forecast is translated
into something an operations team can act on. This module does that translation.

Three pure functions, mirroring Requirement 7:

* :func:`positioning_recommendation` (R7.1) - read a :class:`~src.models.base.Forecast`
  and derive *where* and *when* to position drivers, expressed at the project's
  Time_Grain and Geographic_Grain (from :class:`~src.config.ScopeConfig`). The
  recommendation ranks predicted demand and highlights the peak region/period plus
  the top region per period.
* :func:`quantify_impact` (R7.2, R7.3) - express the expected benefit in business
  terms (reduced rider wait time / reduced driver idle time) and, crucially, expose
  *both* the assumptions and the calculation so the number is defensible and
  reproducible (design Correctness Property 12).
* :func:`india_generalization` (R7.4) - a plain-language narrative mapping the
  NYC-TLC-based approach to Ola / Uber / Rapido operations in India.

Design references:
- Components and Interfaces -> Business_Module (`src/business.py`)
- Data Models -> Recommendation, ImpactStatement
- Correctness Properties -> Property 12 (Impact calculation reproducibility)
- Requirements 7.1, 7.2, 7.3, 7.4

Note on reproducibility (Property 12): :func:`quantify_impact` is a *pure*
function of ``recommendation.predicted_demand`` and the explicit ``assumptions``
mapping. It uses no randomness, wall-clock time, or hidden state, so recomputing
the documented formula from the same inputs always yields the same benefit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from src.config import ScopeConfig
    from src.models.base import Forecast


# --- Documented default assumptions for the impact calculation (R7.3) ---------
#
# These are illustrative, clearly-labelled planning assumptions - not measured
# facts - so every quantified benefit can be traced back to an explicit number
# the reader can challenge or replace. They are surfaced in the returned
# :class:`ImpactStatement.assumptions` mapping alongside the formula.

#: Rider wait (minutes) in an under-served peak before proactive positioning.
DEFAULT_BASELINE_WAIT_MINUTES = 6.0

#: Fraction of that wait removed by pre-positioning drivers into the predicted
#: hotspot (0..1). 0.30 => a 30% reduction in the peak wait.
DEFAULT_WAIT_REDUCTION_PCT = 0.30

#: Idle minutes per driver per peak period before proactive positioning.
DEFAULT_BASELINE_IDLE_MINUTES = 12.0

#: Fraction of driver idle time removed by positioning drivers where demand is
#: predicted rather than letting them wait in low-demand zones (0..1).
DEFAULT_IDLE_REDUCTION_PCT = 0.25

#: Trips one driver can serve within a single period at the Time_Grain. Used to
#: convert predicted demand (trip count) into a driver count.
DEFAULT_TRIPS_PER_DRIVER = 8.0

#: The full documented default assumption set used when a caller passes none.
DEFAULT_IMPACT_ASSUMPTIONS: dict[str, float] = {
    "baseline_wait_minutes": DEFAULT_BASELINE_WAIT_MINUTES,
    "wait_reduction_pct": DEFAULT_WAIT_REDUCTION_PCT,
    "baseline_idle_minutes": DEFAULT_BASELINE_IDLE_MINUTES,
    "idle_reduction_pct": DEFAULT_IDLE_REDUCTION_PCT,
    "trips_per_driver": DEFAULT_TRIPS_PER_DRIVER,
}

#: Human-readable statement of the exact formula :func:`quantify_impact` applies.
#: Kept as a constant so the returned :class:`ImpactStatement.formula` is the same
#: text every time and a reader can recompute the benefit by hand (Property 12).
IMPACT_FORMULA = (
    "drivers_positioned = predicted_demand / trips_per_driver; "
    "rider_wait_minutes_saved = predicted_demand * baseline_wait_minutes * wait_reduction_pct; "
    "driver_idle_minutes_saved = drivers_positioned * baseline_idle_minutes * idle_reduction_pct; "
    "total_minutes_saved = rider_wait_minutes_saved + driver_idle_minutes_saved"
)


@dataclass
class Placement:
    """A single "position drivers here, then" instruction at the defined grain.

    One :class:`Placement` per period in the forecast horizon: the region with the
    highest predicted demand in that period (at the Geographic_Grain) is where
    drivers should be concentrated for that period (at the Time_Grain).

    Attributes:
        period: The forecast period (at Time_Grain) this placement covers.
        region: The highest predicted-demand region (at Geographic_Grain), or
            ``None`` when the forecast is univariate and carries no region.
        predicted_demand: Predicted demand for that region/period.
    """

    period: Any
    region: Optional[str]
    predicted_demand: float


@dataclass
class ImpactStatement:
    """A quantified, fully-documented business benefit (Requirements 7.2, 7.3).

    Expresses the recommendation's expected impact in business terms and carries
    everything needed to reproduce the number: the ``assumptions`` used and the
    ``formula`` applied (design Property 12). Because :func:`quantify_impact`
    derives ``rider_wait_minutes_saved``/``driver_idle_minutes_saved`` purely from
    ``predicted_demand`` and ``assumptions``, recomputing the formula from the same
    ``assumptions`` yields the same ``total_minutes_saved``.

    Attributes:
        predicted_demand: The demand the benefit was computed from (trip count).
        drivers_positioned: Drivers implied by ``predicted_demand`` given the
            ``trips_per_driver`` assumption.
        rider_wait_minutes_saved: Total rider wait minutes saved (R7.2).
        driver_idle_minutes_saved: Total driver idle minutes saved (R7.2).
        total_minutes_saved: Sum of the two benefit components.
        assumptions: The explicit assumption mapping the benefit was derived from
            (a copy, so the statement is self-contained and auditable - R7.3).
        formula: Human-readable statement of the calculation applied (R7.3).
        narrative: Plain-language summary of the benefit in business terms (R7.2).
    """

    predicted_demand: float
    drivers_positioned: float
    rider_wait_minutes_saved: float
    driver_idle_minutes_saved: float
    total_minutes_saved: float
    assumptions: dict[str, float]
    formula: str
    narrative: str


@dataclass
class Recommendation:
    """A driver-positioning recommendation derived from a forecast (Requirement 7.1).

    Produced by :func:`positioning_recommendation`. The core fields identify the
    single peak opportunity (the region/period with the highest predicted demand);
    ``placements`` lists the recommended top region per period across the whole
    horizon, and the ``time_grain``/``geographic_grain`` fields record the scope at
    which the recommendation is expressed (R7.1). ``impact`` is filled in by
    :func:`quantify_impact` (left ``None`` until then).

    Attributes:
        region: Peak predicted-demand region (Geographic_Grain), or ``None`` for a
            univariate forecast that carries no region.
        period: The period (Time_Grain) of the peak predicted demand.
        predicted_demand: The peak predicted demand value.
        action: Plain-language positioning instruction.
        impact: The quantified :class:`ImpactStatement`, or ``None`` until
            :func:`quantify_impact` is applied.
        placements: Top region per period across the horizon (may be empty when the
            forecast is empty).
        time_grain: Time_Grain the recommendation is expressed at (from scope).
        geographic_grain: Geographic_Grain the recommendation is expressed at.
    """

    region: Optional[str]
    period: Any
    predicted_demand: float
    action: str
    impact: Optional[ImpactStatement] = None
    placements: list[Placement] = field(default_factory=list)
    time_grain: Optional[str] = None
    geographic_grain: Optional[str] = None


# --- Forecast parsing helpers -------------------------------------------------


def _values_list(values: Any) -> list[float]:
    """Coerce a forecast's ``values`` (list/ndarray/Series) into a list of floats."""
    # pandas Series / numpy array both expose ``tolist``; fall back to iteration.
    tolist = getattr(values, "tolist", None)
    if callable(tolist):
        raw = tolist()
    else:
        raw = list(values)
    return [float(v) for v in raw]


def _rows_from_forecast(forecast: "Forecast") -> list[tuple[Any, Optional[str], float]]:
    """Flatten a :class:`Forecast` into ``(period, region, predicted_demand)`` rows.

    Handles the two index shapes the modeling layer produces:

    * a ``(period, region)`` MultiIndex (or list of 2-tuples) for multivariate /
      per-region forecasts - each entry keeps its region; and
    * a plain ``DatetimeIndex`` / list of scalar periods for univariate forecasts -
      region is unknown, so it is ``None``.

    When ``forecast.index`` is ``None`` the values are labelled by positional
    integer period so the recommendation is still well-defined.
    """
    values = _values_list(forecast.values)
    index = forecast.index

    # Prefer a pandas Series' own index if values already carry one and no
    # explicit index was supplied.
    if index is None:
        # Only borrow a pandas-style index from the values themselves. Guard
        # against plain lists/tuples, whose ``.index`` attribute is a *method*
        # (callable), not an index-like sequence.
        series_index = getattr(forecast.values, "index", None)
        if series_index is not None and not callable(series_index):
            index = series_index

    if index is None:
        return [(pos, None, val) for pos, val in enumerate(values)]

    # Normalize the index into a list of labels aligned with values.
    tolist = getattr(index, "tolist", None)
    labels = tolist() if callable(tolist) else list(index)

    rows: list[tuple[Any, Optional[str], float]] = []
    for label, val in zip(labels, values):
        if isinstance(label, tuple) and len(label) == 2:
            period, region = label
            rows.append((period, None if region is None else str(region), val))
        else:
            rows.append((label, None, val))
    return rows


def positioning_recommendation(
    forecast: "Forecast", scope: "ScopeConfig"
) -> Recommendation:
    """Derive a driver-positioning recommendation from a forecast (Requirement 7.1).

    Reads the selected model's :class:`~src.models.base.Forecast` and turns it into
    an operational instruction expressed at the project's Time_Grain and
    Geographic_Grain (from ``scope``): where predicted demand is highest, position
    more drivers there for that period.

    Concretely it flattens the forecast into ``(period, region, predicted_demand)``
    rows (see :func:`_rows_from_forecast`), then:

    * builds one :class:`Placement` per period - the highest predicted-demand region
      in that period (the region to concentrate drivers in for that period); and
    * identifies the single peak opportunity across the whole horizon (the
      region/period with the largest predicted demand) as the headline
      recommendation.

    Ties are broken deterministically (first by descending demand, then by the
    period/region ordering encountered) so the same forecast always yields the same
    recommendation.

    Args:
        forecast: The selected model's forecast over the holdout / future horizon.
        scope: The project :class:`~src.config.ScopeConfig`, supplying the
            Time_Grain and Geographic_Grain the recommendation is expressed at.

    Returns:
        A :class:`Recommendation` with the peak region/period, a per-period
        placement list, and a plain-language ``action``. ``impact`` is ``None``
        until :func:`quantify_impact` is applied. An empty forecast yields a
        recommendation with ``predicted_demand`` 0 and no placements.
    """
    time_grain = getattr(scope, "time_grain", None)
    geographic_grain = getattr(scope, "geographic_grain", None)
    rows = _rows_from_forecast(forecast)

    if not rows:
        return Recommendation(
            region=None,
            period=None,
            predicted_demand=0.0,
            action=(
                "No forecast values available; cannot derive a positioning "
                "recommendation."
            ),
            placements=[],
            time_grain=time_grain,
            geographic_grain=geographic_grain,
        )

    # One placement per period: the top predicted-demand region within that period.
    # Iterate in encounter order so ties keep the first-seen region deterministically.
    best_per_period: dict[Any, Placement] = {}
    period_order: list[Any] = []
    for period, region, demand in rows:
        current = best_per_period.get(period)
        if current is None:
            best_per_period[period] = Placement(period, region, demand)
            period_order.append(period)
        elif demand > current.predicted_demand:
            best_per_period[period] = Placement(period, region, demand)

    placements = [best_per_period[p] for p in period_order]

    # Headline: the single highest predicted-demand placement across all periods.
    peak = max(placements, key=lambda p: p.predicted_demand)

    where = peak.region if peak.region is not None else "the highest-demand area"
    grain_note = geographic_grain if geographic_grain else "region"
    action = (
        f"Position more drivers in {where} ({grain_note}) for period {peak.period}: "
        f"predicted demand is {peak.predicted_demand:.0f} trips, the peak across the "
        f"forecast horizon. Rebalance idle drivers toward the top predicted-demand "
        f"{grain_note} in each period."
    )

    return Recommendation(
        region=peak.region,
        period=peak.period,
        predicted_demand=float(peak.predicted_demand),
        action=action,
        impact=None,
        placements=placements,
        time_grain=time_grain,
        geographic_grain=geographic_grain,
    )


def quantify_impact(
    recommendation: Recommendation,
    assumptions: Optional[Mapping[str, float]] = None,
) -> ImpactStatement:
    """Quantify the recommendation's benefit, showing assumptions and formula (R7.2, R7.3).

    Expresses the expected impact in business terms - reduced rider wait time and
    reduced driver idle time - and returns *both* the explicit ``assumptions`` and
    the ``formula`` used, so the number is defensible and reproducible.

    The calculation is a pure function of ``recommendation.predicted_demand`` and
    the ``assumptions`` mapping (no randomness, clock, or hidden state), applying
    :data:`IMPACT_FORMULA`::

        drivers_positioned        = predicted_demand / trips_per_driver
        rider_wait_minutes_saved  = predicted_demand * baseline_wait_minutes * wait_reduction_pct
        driver_idle_minutes_saved = drivers_positioned * baseline_idle_minutes * idle_reduction_pct
        total_minutes_saved       = rider_wait_minutes_saved + driver_idle_minutes_saved

    Because of this purity, recomputing the formula from the same ``assumptions``
    always yields the same ``total_minutes_saved`` (design Correctness Property 12).

    Args:
        recommendation: The :class:`Recommendation` whose ``predicted_demand`` the
            benefit is computed from.
        assumptions: Explicit planning assumptions. Missing keys fall back to
            :data:`DEFAULT_IMPACT_ASSUMPTIONS`; the returned statement records the
            *effective* assumptions actually used. ``trips_per_driver`` must be
            positive.

    Returns:
        An :class:`ImpactStatement` carrying the computed benefit components, the
        effective assumptions (a copy), the formula text, and a business-terms
        narrative.

    Raises:
        ValueError: If the effective ``trips_per_driver`` is not positive (a driver
            must be able to serve at least a fraction of a trip per period).
    """
    # Effective assumptions: caller-supplied values override the documented
    # defaults; the copy makes the returned statement self-contained and auditable.
    effective: dict[str, float] = dict(DEFAULT_IMPACT_ASSUMPTIONS)
    if assumptions:
        effective.update({k: float(v) for k, v in assumptions.items()})

    trips_per_driver = effective["trips_per_driver"]
    if trips_per_driver <= 0:
        raise ValueError(
            "Assumption 'trips_per_driver' must be positive to convert predicted "
            f"demand into a driver count; got {trips_per_driver}."
        )

    predicted_demand = float(recommendation.predicted_demand)

    # The documented formula (IMPACT_FORMULA) - kept in lock-step with that string.
    drivers_positioned = predicted_demand / trips_per_driver
    rider_wait_minutes_saved = (
        predicted_demand
        * effective["baseline_wait_minutes"]
        * effective["wait_reduction_pct"]
    )
    driver_idle_minutes_saved = (
        drivers_positioned
        * effective["baseline_idle_minutes"]
        * effective["idle_reduction_pct"]
    )
    total_minutes_saved = rider_wait_minutes_saved + driver_idle_minutes_saved

    narrative = (
        f"Positioning about {drivers_positioned:.0f} drivers ahead of a predicted "
        f"{predicted_demand:.0f} trips is estimated to save "
        f"{rider_wait_minutes_saved:.0f} rider wait-minutes and "
        f"{driver_idle_minutes_saved:.0f} driver idle-minutes "
        f"({total_minutes_saved:.0f} minutes total) for this period, under the "
        "stated assumptions. Shorter waits improve rider experience and conversion; "
        "less idle time raises driver earnings and utilization."
    )

    return ImpactStatement(
        predicted_demand=predicted_demand,
        drivers_positioned=drivers_positioned,
        rider_wait_minutes_saved=rider_wait_minutes_saved,
        driver_idle_minutes_saved=driver_idle_minutes_saved,
        total_minutes_saved=total_minutes_saved,
        assumptions=effective,
        formula=IMPACT_FORMULA,
        narrative=narrative,
    )


def india_generalization() -> str:
    """Describe how the NYC approach generalizes to India operations (Requirement 7.4).

    Returns a plain-language narrative mapping this project's NYC-TLC-based
    forecasting-and-positioning approach onto Ola, Uber, and Rapido operations in
    India, noting what transfers directly and what must be adapted (two-wheeler /
    auto-rickshaw supply, denser and more heterogeneous cities, event/monsoon
    seasonality, and data-source differences).

    Returns:
        A multi-paragraph explanatory string suitable for the notebook and
        dashboard business sections.
    """
    return (
        "How this generalizes to India (Ola, Uber, Rapido)\n"
        "\n"
        "The method here is platform-agnostic: aggregate historical trips into a "
        "demand series at a chosen time and geographic grain, compare a broad set of "
        "forecasting models, pick the most accurate, and translate its predictions "
        "into a driver-positioning plan. None of that is specific to New York, so the "
        "same pipeline applies to Ola, Uber, and Rapido in Indian cities - swap the "
        "NYC TLC trip records for each platform's own booking logs and the "
        "borough-level geography for Indian city zones (for example ward or "
        "pin-code clusters, or hex grids used for dispatch).\n"
        "\n"
        "What transfers directly: predicting where and when demand will peak, then "
        "pre-positioning supply to cut rider wait time and driver idle time. In India "
        "this maps to Ola and Uber cabs and autos, and to Rapido's bike taxis and "
        "autos - the objective (match supply to predicted demand) is identical.\n"
        "\n"
        "What must be adapted: Indian demand is dominated by two-wheelers and "
        "auto-rickshaws rather than four-wheeler cabs, so 'demand' should be modelled "
        "per vehicle type. Cities are denser and more heterogeneous, so a finer "
        "geographic grain and intraday (hourly) forecasting usually matter more than "
        "at NYC borough/daily grain. Seasonality drivers differ - monsoon, festivals "
        "(Diwali, Holi), cricket matches, and local events cause sharp, city-specific "
        "spikes that the calendar/exogenous features must capture. Finally, data comes "
        "from each platform's internal systems rather than a public TLC feed, but the "
        "validate-against-ground-truth discipline and the model-comparison-then-"
        "positioning workflow stay exactly the same."
    )
