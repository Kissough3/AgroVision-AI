"""Pure irrigation calculations using caller-supplied, referenced inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from math import isfinite
from typing import Literal

from app.core.soil_intelligence import CROP_PROFILES


@dataclass(frozen=True)
class ApplicabilityRange:
    variable: str
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class CropCoefficient:
    crop_key: str
    value: float
    growth_stage: str
    reference_name: str
    table_identifier: str
    applicability_conditions: str
    applicability_ranges: tuple[ApplicabilityRange, ...] = ()


@dataclass(frozen=True)
class ReferencedCropParameter:
    crop_key: str
    parameter: Literal["root_depth", "depletion_fraction_p"]
    value: float
    unit: str
    growth_stage: str
    reference_name: str
    table_identifier: str


@dataclass(frozen=True)
class GrowthStageLength:
    crop_key: str
    growth_stage: str
    duration_days: int
    stage_order: int
    reference_name: str
    table_identifier: str


@dataclass(frozen=True)
class SoilWaterLayer:
    depth_top_cm: float
    depth_bottom_cm: float
    awc_mm: float


@dataclass(frozen=True)
class RootZoneWater:
    total_available_water_mm: float
    readily_available_water_mm: float


@dataclass(frozen=True)
class EffectiveRainMethod:
    method_id: str
    name: str
    reference: str
    reference_locator: str
    growth_stage: str
    verified: bool


DASTANE_FAO25_CANDIDATE = "dastane_fao25_candidate"
EFFECTIVE_RAIN_METHODS: dict[str, EffectiveRainMethod] = {
    DASTANE_FAO25_CANDIDATE: EffectiveRainMethod(
        method_id=DASTANE_FAO25_CANDIDATE,
        name="Dastane effective-rainfall method candidate",
        reference=(
            "Dastane, N.G. (1978), Effective rainfall in irrigated agriculture, "
            "FAO Irrigation and Drainage Paper 25."
        ),
        reference_locator="Specific method/table requires primary-source verification",
        growth_stage="All stages, applicability unverified",
        verified=False,
    ),
}


@dataclass(frozen=True)
class DailyWeatherInput:
    day: date
    et0_mm: float | None
    forecast_rain_mm: float | None
    crop_coefficient: CropCoefficient | None
    climate_values: Mapping[str, float] | None = None


@dataclass(frozen=True)
class IrrigationRecommendation:
    state: Literal["ready", "needs_more_data", "review_required"]
    window_start: date | None
    window_end: date | None
    etc_mm: float | None
    forecast_rain_mm: float | None
    effective_rain_mm: float | None
    net_requirement_mm: float | None
    field_area_ha: float | None
    quantity_m3: float | None
    rain_netted: bool
    explanation: tuple[str, ...]
    warnings: tuple[str, ...]
    effective_rain_method: EffectiveRainMethod | None = None


def _has_text(value: str | None) -> bool:
    return bool(value and value.strip())


def _valid_nonnegative(value: float | None) -> bool:
    return value is not None and isfinite(value) and value >= 0


def mm_to_m3(depth_mm: float, field_area_ha: float) -> float:
    """Convert a water depth over hectares to cubic metres."""
    if not _valid_nonnegative(depth_mm):
        raise ValueError("depth_mm must be finite and non-negative")
    if not isfinite(field_area_ha) or field_area_ha <= 0:
        raise ValueError("field_area_ha must be finite and positive")
    return depth_mm * field_area_ha * 10


def resolve_growth_stage(
    manual_growth_stage: str | None,
    days_since_planting: int | None,
    stage_lengths: Sequence[GrowthStageLength],
    crop_key: str,
) -> str | None:
    """Prefer the manually entered stage; otherwise use referenced durations."""
    if manual_growth_stage and manual_growth_stage.strip():
        return manual_growth_stage.strip()
    if days_since_planting is None or days_since_planting < 0:
        return None

    rows = sorted(stage_lengths, key=lambda row: row.stage_order)
    if not rows or any(
        row.crop_key != crop_key
        or not _has_text(row.growth_stage)
        or not _has_text(row.reference_name)
        or not _has_text(row.table_identifier)
        or row.duration_days <= 0
        or row.stage_order < 0
        for row in rows
    ):
        return None
    if len({row.stage_order for row in rows}) != len(rows):
        return None

    elapsed_days = 0
    for row in rows:
        elapsed_days += row.duration_days
        if days_since_planting < elapsed_days:
            return row.growth_stage
    return rows[-1].growth_stage


def calculate_root_zone_water(
    layers: Sequence[SoilWaterLayer],
    root_depth: ReferencedCropParameter,
    depletion_fraction: ReferencedCropParameter,
) -> RootZoneWater:
    """Integrate modeled layer AWC to referenced rooting depth; do not infer moisture."""
    for parameter, expected_name in (
        (root_depth, "root_depth"),
        (depletion_fraction, "depletion_fraction_p"),
    ):
        if (
            parameter.parameter != expected_name
            or not isfinite(parameter.value)
            or parameter.value <= 0
            or not _has_text(parameter.crop_key)
            or not _has_text(parameter.growth_stage)
            or not _has_text(parameter.reference_name)
            or not _has_text(parameter.table_identifier)
        ):
            raise ValueError(f"Invalid or unreferenced {expected_name}")
    if (
        root_depth.crop_key != depletion_fraction.crop_key
        or root_depth.growth_stage != depletion_fraction.growth_stage
    ):
        raise ValueError(
            "Root depth and p must apply to the same crop and growth stage"
        )
    if root_depth.unit != "cm" or depletion_fraction.unit != "fraction":
        raise ValueError("Root depth must use cm and p must be a fraction")
    if not 0 < depletion_fraction.value < 1:
        raise ValueError("depletion_fraction_p must be between 0 and 1")
    if not layers:
        raise ValueError("At least one soil water layer is required")

    total_available_water = 0.0
    for layer in layers:
        if (
            not isfinite(layer.depth_top_cm)
            or not isfinite(layer.depth_bottom_cm)
            or not _valid_nonnegative(layer.awc_mm)
            or layer.depth_top_cm < 0
            or layer.depth_bottom_cm <= layer.depth_top_cm
        ):
            raise ValueError("Invalid soil water layer")
        overlap_cm = max(
            0.0,
            min(layer.depth_bottom_cm, root_depth.value) - layer.depth_top_cm,
        )
        total_available_water += layer.awc_mm * (
            overlap_cm / (layer.depth_bottom_cm - layer.depth_top_cm)
        )

    return RootZoneWater(
        total_available_water_mm=total_available_water,
        readily_available_water_mm=total_available_water * depletion_fraction.value,
    )


def _empty_result(
    state: Literal["needs_more_data", "review_required"],
    *,
    window_start: date | None,
    window_end: date | None,
    forecast_rain_mm: float | None,
    explanation: str,
    warnings: Sequence[str] = (),
) -> IrrigationRecommendation:
    return IrrigationRecommendation(
        state=state,
        window_start=window_start,
        window_end=window_end,
        etc_mm=None,
        forecast_rain_mm=forecast_rain_mm,
        effective_rain_mm=None,
        net_requirement_mm=None,
        field_area_ha=None,
        quantity_m3=None,
        rain_netted=False,
        explanation=(explanation,),
        warnings=tuple(warnings),
    )


def calculate_irrigation_recommendation(
    crop_type: str | None,
    growth_stage: str | None,
    daily_inputs: Sequence[DailyWeatherInput],
    field_area_ha: float | None,
    selected_effective_rain_method_id: str | None = None,
    effective_rain_mm: float | None = None,
) -> IrrigationRecommendation:
    """Calculate ETc and optionally net a caller-supplied effective-rain estimate.

    This function deliberately does not implement an effective-rain formula. The
    owner-selected method and its effective-rain result must be supplied by a
    separately verified source implementation.
    """
    dates = [item.day for item in daily_inputs]
    window_start = min(dates) if dates else None
    window_end = max(dates) if dates else None
    rain_complete = bool(daily_inputs) and all(
        _valid_nonnegative(item.forecast_rain_mm) for item in daily_inputs
    )
    forecast_rain = (
        sum(item.forecast_rain_mm for item in daily_inputs) if rain_complete else None
    )

    crop_key = crop_type.strip().lower() if crop_type else ""
    if crop_key not in CROP_PROFILES:
        return _empty_result(
            "needs_more_data",
            window_start=window_start,
            window_end=window_end,
            forecast_rain_mm=forecast_rain,
            explanation="Crop type does not match a supported soil-intelligence crop key.",
        )
    if not growth_stage or not growth_stage.strip():
        return _empty_result(
            "needs_more_data",
            window_start=window_start,
            window_end=window_end,
            forecast_rain_mm=forecast_rain,
            explanation="A manually entered or referenced growth stage is required.",
        )
    if not daily_inputs:
        return _empty_result(
            "needs_more_data",
            window_start=None,
            window_end=None,
            forecast_rain_mm=None,
            explanation="Daily ET0 and forecast rainfall are required for the stated window.",
        )

    etc_mm = 0.0
    warnings: list[str] = []
    stage_key = growth_stage.strip().lower()
    for item in daily_inputs:
        coefficient = item.crop_coefficient
        if not _valid_nonnegative(item.et0_mm) or not _valid_nonnegative(
            item.forecast_rain_mm
        ):
            return _empty_result(
                "needs_more_data",
                window_start=window_start,
                window_end=window_end,
                forecast_rain_mm=forecast_rain,
                explanation="A daily ET0 or forecast rainfall value is missing or invalid.",
                warnings=warnings,
            )
        if coefficient is None:
            return _empty_result(
                "needs_more_data",
                window_start=window_start,
                window_end=window_end,
                forecast_rain_mm=forecast_rain,
                explanation="A referenced crop coefficient is required for every day.",
                warnings=warnings,
            )
        reference_fields = (
            coefficient.reference_name,
            coefficient.table_identifier,
            coefficient.growth_stage,
            coefficient.applicability_conditions,
        )
        if (
            not all(_has_text(value) for value in reference_fields)
            or not isfinite(coefficient.value)
            or coefficient.value < 0
        ):
            return _empty_result(
                "needs_more_data",
                window_start=window_start,
                window_end=window_end,
                forecast_rain_mm=forecast_rain,
                explanation=(
                    "Kc is rejected unless its value, source, table, growth stage, "
                    "and applicability conditions are present."
                ),
                warnings=warnings,
            )
        if (
            coefficient.crop_key.strip().lower() != crop_key
            or coefficient.growth_stage.strip().lower() != stage_key
        ):
            return _empty_result(
                "needs_more_data",
                window_start=window_start,
                window_end=window_end,
                forecast_rain_mm=forecast_rain,
                explanation="Kc crop and stage must match the field crop and selected stage.",
                warnings=warnings,
            )

        if not coefficient.applicability_ranges:
            warnings.append(
                "Kc applicability conditions cannot be checked against climate data."
            )
        for condition in coefficient.applicability_ranges:
            observed = (item.climate_values or {}).get(condition.variable)
            if observed is None or not isfinite(observed):
                warnings.append(
                    f"Kc applicability condition {condition.variable} was not checked."
                )
                continue
            if (condition.minimum is not None and observed < condition.minimum) or (
                condition.maximum is not None and observed > condition.maximum
            ):
                warnings.append(
                    f"Climate value for {condition.variable} is outside the Kc applicability range."
                )
        etc_mm += item.et0_mm * coefficient.value

    if forecast_rain is None:
        warnings.append("Forecast rainfall is incomplete for this window.")

    selected_method = None
    if selected_effective_rain_method_id:
        selected_method = EFFECTIVE_RAIN_METHODS.get(selected_effective_rain_method_id)
        if selected_method is None:
            return IrrigationRecommendation(
                state="needs_more_data",
                window_start=window_start,
                window_end=window_end,
                etc_mm=etc_mm,
                forecast_rain_mm=forecast_rain,
                effective_rain_mm=None,
                net_requirement_mm=None,
                field_area_ha=field_area_ha,
                quantity_m3=None,
                rain_netted=False,
                explanation=("The selected effective-rain method is not registered.",),
                warnings=tuple(warnings),
            )

    if selected_method is None:
        warnings.append(
            "No owner-selected effective-rain method; forecast rain was not netted."
        )
        return IrrigationRecommendation(
            state="needs_more_data",
            window_start=window_start,
            window_end=window_end,
            etc_mm=etc_mm,
            forecast_rain_mm=forecast_rain,
            effective_rain_mm=None,
            net_requirement_mm=None,
            field_area_ha=field_area_ha,
            quantity_m3=None,
            rain_netted=False,
            explanation=(
                "ETc and forecast rainfall are reported separately; rain was not netted.",
            ),
            warnings=tuple(warnings),
        )

    if not _valid_nonnegative(effective_rain_mm):
        warnings.append(
            "An effective-rain estimate from the selected method is required."
        )
        return IrrigationRecommendation(
            state="needs_more_data",
            window_start=window_start,
            window_end=window_end,
            etc_mm=etc_mm,
            forecast_rain_mm=forecast_rain,
            effective_rain_mm=None,
            net_requirement_mm=None,
            field_area_ha=field_area_ha,
            quantity_m3=None,
            rain_netted=False,
            explanation=(
                "No net requirement was calculated because effective rain is unavailable.",
            ),
            warnings=tuple(warnings),
            effective_rain_method=selected_method,
        )
    if forecast_rain is None:
        warnings.append(
            "Forecast rainfall is incomplete; effective rain cannot be netted."
        )
        return IrrigationRecommendation(
            state="needs_more_data",
            window_start=window_start,
            window_end=window_end,
            etc_mm=etc_mm,
            forecast_rain_mm=None,
            effective_rain_mm=None,
            net_requirement_mm=None,
            field_area_ha=field_area_ha,
            quantity_m3=None,
            rain_netted=False,
            explanation=(
                "No net requirement was calculated because the forecast is incomplete.",
            ),
            warnings=tuple(warnings),
            effective_rain_method=selected_method,
        )
    if effective_rain_mm > forecast_rain:
        warnings.append(
            "Effective rain cannot exceed total forecast rainfall for the window."
        )
        return IrrigationRecommendation(
            state="needs_more_data",
            window_start=window_start,
            window_end=window_end,
            etc_mm=etc_mm,
            forecast_rain_mm=forecast_rain,
            effective_rain_mm=None,
            net_requirement_mm=None,
            field_area_ha=field_area_ha,
            quantity_m3=None,
            rain_netted=False,
            explanation=(
                "No net requirement was calculated because effective rain is invalid.",
            ),
            warnings=tuple(warnings),
            effective_rain_method=selected_method,
        )

    raw_net_mm = etc_mm - effective_rain_mm
    net_mm = max(0.0, raw_net_mm)
    explanation = ["Net requirement is ETc minus the supplied effective-rain estimate."]
    if raw_net_mm < 0:
        explanation.append("Negative net demand was clamped to 0 mm.")
    if not selected_method.verified:
        warnings.append(
            "The selected effective-rain method is not verified; review is required."
        )

    quantity_m3 = None
    if field_area_ha is not None:
        if isfinite(field_area_ha) and field_area_ha > 0:
            quantity_m3 = mm_to_m3(net_mm, field_area_ha)
        else:
            warnings.append("Field area is invalid; volume could not be calculated.")
    else:
        warnings.append("Field area is missing; volume could not be calculated.")

    climate_review = any(
        "outside the Kc applicability range" in warning for warning in warnings
    )
    state: Literal["ready", "needs_more_data", "review_required"] = "ready"
    if field_area_ha is None or not isfinite(field_area_ha) or field_area_ha <= 0:
        state = "needs_more_data"
    elif not selected_method.verified or climate_review:
        state = "review_required"
    return IrrigationRecommendation(
        state=state,
        window_start=window_start,
        window_end=window_end,
        etc_mm=etc_mm,
        forecast_rain_mm=forecast_rain,
        effective_rain_mm=effective_rain_mm,
        net_requirement_mm=net_mm,
        field_area_ha=field_area_ha,
        quantity_m3=quantity_m3,
        rain_netted=True,
        explanation=tuple(explanation),
        warnings=tuple(warnings),
        effective_rain_method=selected_method,
    )
