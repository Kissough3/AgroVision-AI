"""Pure fertilizer calculations that require matched lab and calibration evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from math import isfinite
from typing import Literal

from app.core.soil_intelligence import CROP_PROFILES

NUTRIENTS = ("N", "P", "K")
_P_TO_P2O5 = 2.291
_K_TO_K2O = 1.205
_MAX_TEXT_LENGTH = 200


@dataclass(frozen=True)
class LabNutrientResult:
    nutrient: Literal["N", "P", "K"]
    value: float | None
    unit: str | None
    analysis_method: str | None
    basis: str | None
    sample_date: date | None
    depth_top_cm: float | None
    depth_bottom_cm: float | None
    laboratory_source: str | None

    def __post_init__(self) -> None:
        if self.nutrient not in NUTRIENTS:
            raise ValueError("nutrient must be N, P, or K")
        if self.value is not None and (not isfinite(self.value) or self.value < 0):
            raise ValueError("lab result values must be finite and non-negative")
        if self.depth_top_cm is not None and self.depth_top_cm < 0:
            raise ValueError("sampling depth must be non-negative")
        if (
            self.depth_top_cm is not None
            and self.depth_bottom_cm is not None
            and self.depth_bottom_cm <= self.depth_top_cm
        ):
            raise ValueError("sampling bottom depth must exceed top depth")
        for value in (
            self.unit,
            self.analysis_method,
            self.basis,
            self.laboratory_source,
        ):
            if value is not None and len(value) > _MAX_TEXT_LENGTH:
                raise ValueError("lab result text fields are too long")


@dataclass(frozen=True)
class ProductAnalysis:
    n_percent: float
    p2o5_percent: float
    k2o_percent: float

    def __post_init__(self) -> None:
        values = (self.n_percent, self.p2o5_percent, self.k2o_percent)
        if any(not isfinite(value) or value < 0 or value > 100 for value in values):
            raise ValueError("product percentages must be finite and between 0 and 100")
        if sum(values) > 100:
            raise ValueError("N + P2O5 + K2O product percentages cannot exceed 100")
        if not any(value > 0 for value in values):
            raise ValueError(
                "at least one product nutrient percentage must be positive"
            )

    def percentage_for(self, nutrient: str) -> float:
        return {
            "N": self.n_percent,
            "P": self.p2o5_percent,
            "K": self.k2o_percent,
        }[nutrient]


@dataclass(frozen=True)
class FertilizerCalibrationBand:
    crop_key: str
    region: str
    nutrient: Literal["N", "P", "K"]
    growth_stage: str
    analysis_method: str
    result_unit: str
    basis: str
    minimum_test_value: float
    maximum_test_value: float | None
    recommended_nutrient_kg_ha: float
    reference_name: str
    table_identifier: str
    user_entered: bool = False
    verified: bool = False

    def __post_init__(self) -> None:
        if (
            not isfinite(self.minimum_test_value)
            or self.minimum_test_value < 0
            or not isfinite(self.recommended_nutrient_kg_ha)
            or self.recommended_nutrient_kg_ha < 0
        ):
            raise ValueError("calibration values must be finite and non-negative")
        if self.maximum_test_value is not None and (
            not isfinite(self.maximum_test_value)
            or self.maximum_test_value <= self.minimum_test_value
        ):
            raise ValueError("maximum test value must exceed the minimum")
        for value in (
            self.crop_key,
            self.region,
            self.growth_stage,
            self.analysis_method,
            self.result_unit,
            self.basis,
            self.reference_name,
            self.table_identifier,
        ):
            if len(value) > _MAX_TEXT_LENGTH:
                raise ValueError("calibration text fields are too long")


@dataclass(frozen=True)
class FertilizerRecommendation:
    state: Literal["ready", "needs_more_data", "review_required"]
    nutrient_requirements_kg_ha: Mapping[str, float] = field(default_factory=dict)
    product_equivalent_kg_ha_by_nutrient: Mapping[str, float] = field(
        default_factory=dict
    )
    missing_information: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def nutrient_to_product_kg_ha(
    nutrient_requirement_kg_ha: float,
    product_nutrient_percentage: float,
) -> float:
    """Convert a single nutrient requirement to equivalent product mass."""
    if not isfinite(nutrient_requirement_kg_ha) or nutrient_requirement_kg_ha < 0:
        raise ValueError("nutrient requirement must be finite and non-negative")
    if (
        not isfinite(product_nutrient_percentage)
        or product_nutrient_percentage <= 0
        or product_nutrient_percentage > 100
    ):
        raise ValueError(
            "product nutrient percentage must be greater than 0 and at most 100"
        )
    return nutrient_requirement_kg_ha / (product_nutrient_percentage / 100)


def normalize_lab_nutrient_value(result: LabNutrientResult) -> tuple[float, str] | None:
    if result.value is None or not result.basis:
        return None
    basis = result.basis.strip().upper()
    if result.nutrient == "N" and basis in {"N", "ELEMENTAL"}:
        return result.value, "N"
    if result.nutrient == "P":
        if basis in {"P", "ELEMENTAL"}:
            return result.value * _P_TO_P2O5, "P2O5"
        if basis == "P2O5":
            return result.value, "P2O5"
    if result.nutrient == "K":
        if basis in {"K", "ELEMENTAL"}:
            return result.value * _K_TO_K2O, "K2O"
        if basis == "K2O":
            return result.value, "K2O"
    return None


def _lab_result_is_complete(result: LabNutrientResult) -> bool:
    return all(
        (
            result.value is not None,
            bool(result.unit and result.unit.strip()),
            bool(result.analysis_method and result.analysis_method.strip()),
            bool(result.basis and result.basis.strip()),
            result.sample_date is not None,
            result.depth_top_cm is not None,
            result.depth_bottom_cm is not None,
            bool(result.laboratory_source and result.laboratory_source.strip()),
        )
    )


def calculate_fertilizer_recommendation(
    crop_type: str | None,
    region: str | None,
    growth_stage: str | None,
    lab_results: Sequence[LabNutrientResult],
    calibration_bands: Sequence[FertilizerCalibrationBand],
    product_analysis: ProductAnalysis | None = None,
) -> FertilizerRecommendation:
    """Return rates only when all NPK lab methods and matching references exist."""
    crop_key = crop_type.strip().lower() if crop_type else ""
    if crop_key not in CROP_PROFILES:
        return FertilizerRecommendation(
            state="needs_more_data",
            missing_information=(
                "A crop key matching soil_intelligence.CROP_PROFILES is required.",
            ),
        )

    missing_data: list[str] = []
    missing_calibration: list[str] = []
    if not region or not region.strip():
        missing_data.append("Region is required to select a calibration.")
    if not growth_stage or not growth_stage.strip():
        missing_data.append("Growth stage is required.")

    results_by_nutrient: dict[str, LabNutrientResult] = {}
    for result in lab_results:
        if result.nutrient in results_by_nutrient:
            missing_data.append(
                f"Duplicate {result.nutrient} lab results are ambiguous."
            )
        results_by_nutrient[result.nutrient] = result

    matching_bands: dict[str, FertilizerCalibrationBand] = {}
    if not lab_results:
        missing_data.append("N, P, and K laboratory results are required.")

    for nutrient in NUTRIENTS:
        result = results_by_nutrient.get(nutrient)
        if result is None:
            missing_data.append(f"A {nutrient} laboratory result is missing.")
            continue
        if not _lab_result_is_complete(result):
            missing_data.append(
                f"The {nutrient} result needs a value, unit, analysis method, basis, "
                "sample date, sampling depth, and laboratory source."
            )
            continue
        normalized = normalize_lab_nutrient_value(result)
        if normalized is None:
            missing_data.append(
                f"The reported basis for {nutrient} cannot be converted safely."
            )
            continue
        normalized_value, normalized_basis = normalized

        candidates = [
            band
            for band in calibration_bands
            if band.crop_key == crop_key
            and band.region.strip().lower() == (region or "").strip().lower()
            and band.nutrient == nutrient
            and band.growth_stage.strip().lower()
            == (growth_stage or "").strip().lower()
            and band.analysis_method.strip().lower()
            == result.analysis_method.strip().lower()
            and band.result_unit.strip().lower() == result.unit.strip().lower()
            and band.basis.upper() == normalized_basis
            and normalized_value >= band.minimum_test_value
            and (
                band.maximum_test_value is None
                or normalized_value < band.maximum_test_value
            )
            and bool(band.reference_name.strip())
            and bool(band.table_identifier.strip())
        ]
        if not candidates:
            missing_calibration.append(
                f"No referenced calibration band matches {nutrient}, crop, region, "
                "stage, method, unit, and soil-test value."
            )
        else:
            matching_bands[nutrient] = candidates[0]

    if missing_data or missing_calibration:
        state: Literal["needs_more_data", "review_required"] = (
            "needs_more_data" if missing_data else "review_required"
        )
        return FertilizerRecommendation(
            state=state,
            missing_information=tuple(missing_data + missing_calibration),
        )

    nutrient_rates = {
        nutrient: band.recommended_nutrient_kg_ha
        for nutrient, band in matching_bands.items()
    }
    product_equivalents: dict[str, float] = {}
    warnings: list[str] = []
    if product_analysis is not None:
        for nutrient, required_rate in nutrient_rates.items():
            percentage = product_analysis.percentage_for(nutrient)
            if percentage > 0:
                product_equivalents[nutrient] = nutrient_to_product_kg_ha(
                    required_rate, percentage
                )
        if len(product_equivalents) > 1:
            warnings.append(
                "Per-nutrient product equivalents are alternatives and must not be summed "
                "as a blended-product rate."
            )

    review_required = any(
        band.user_entered or not band.verified for band in matching_bands.values()
    )
    if review_required:
        warnings.append(
            "User-entered or unverified calibration requires agronomic review."
        )
    return FertilizerRecommendation(
        state="review_required" if review_required else "ready",
        nutrient_requirements_kg_ha=nutrient_rates,
        product_equivalent_kg_ha_by_nutrient=product_equivalents,
        warnings=tuple(warnings),
    )
