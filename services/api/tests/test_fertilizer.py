from datetime import date

import pytest

from app.core.fertilizer import (
    FertilizerCalibrationBand,
    LabNutrientResult,
    ProductAnalysis,
    calculate_fertilizer_recommendation,
    normalize_lab_nutrient_value,
    nutrient_to_product_kg_ha,
)

SYNTHETIC_RESULTS = [
    LabNutrientResult(
        "N",
        20,
        "mg/kg",
        "Synthetic N method",
        "N",
        date(2026, 5, 1),
        0,
        15,
        "Synthetic lab",
    ),
    LabNutrientResult(
        "P",
        10,
        "mg/kg",
        "Synthetic P method",
        "P",
        date(2026, 5, 1),
        0,
        15,
        "Synthetic lab",
    ),
    LabNutrientResult(
        "K",
        100,
        "mg/kg",
        "Synthetic K method",
        "K",
        date(2026, 5, 1),
        0,
        15,
        "Synthetic lab",
    ),
]

SYNTHETIC_CALIBRATIONS = [
    FertilizerCalibrationBand(
        "wheat",
        "synthetic-region",
        "N",
        "vegetative",
        "Synthetic N method",
        "mg/kg",
        "N",
        0,
        100,
        46,
        "Synthetic source, not agronomic guidance",
        "Synthetic N table",
        verified=True,
    ),
    FertilizerCalibrationBand(
        "wheat",
        "synthetic-region",
        "P",
        "vegetative",
        "Synthetic P method",
        "mg/kg",
        "P2O5",
        20,
        30,
        30,
        "Synthetic source, not agronomic guidance",
        "Synthetic P table",
        verified=True,
    ),
    FertilizerCalibrationBand(
        "wheat",
        "synthetic-region",
        "K",
        "vegetative",
        "Synthetic K method",
        "mg/kg",
        "K2O",
        120,
        130,
        40,
        "Synthetic source, not agronomic guidance",
        "Synthetic K table",
        verified=True,
    ),
]


def test_elemental_phosphorus_and_potassium_convert_to_oxide_basis() -> None:
    recommendation = calculate_fertilizer_recommendation(
        "wheat",
        "synthetic-region",
        "vegetative",
        SYNTHETIC_RESULTS,
        SYNTHETIC_CALIBRATIONS,
    )

    assert recommendation.state == "ready"
    assert recommendation.nutrient_requirements_kg_ha == {"N": 46, "P": 30, "K": 40}


def test_oxide_conversions_are_explicit() -> None:
    phosphorus = normalize_lab_nutrient_value(SYNTHETIC_RESULTS[1])
    potassium = normalize_lab_nutrient_value(SYNTHETIC_RESULTS[2])

    assert phosphorus == pytest.approx((22.91, "P2O5"))
    assert potassium == pytest.approx((120.5, "K2O"))


def test_nutrient_to_product_conversion_uses_percent_analysis() -> None:
    assert nutrient_to_product_kg_ha(46, 46) == pytest.approx(100)


@pytest.mark.parametrize(
    "analysis",
    [
        (101, 0, 0),
        (50, 30, 25),
        (-1, 0, 1),
        (float("nan"), 0, 1),
        (0, 0, 0),
    ],
)
def test_invalid_product_percentages_are_rejected(
    analysis: tuple[float, float, float],
) -> None:
    with pytest.raises(ValueError):
        ProductAnalysis(*analysis)


def test_product_equivalent_rates_are_reported_per_nutrient_not_summed() -> None:
    recommendation = calculate_fertilizer_recommendation(
        "wheat",
        "synthetic-region",
        "vegetative",
        SYNTHETIC_RESULTS,
        SYNTHETIC_CALIBRATIONS,
        ProductAnalysis(46, 18, 12),
    )

    assert recommendation.product_equivalent_kg_ha_by_nutrient["N"] == pytest.approx(
        100
    )
    assert recommendation.product_equivalent_kg_ha_by_nutrient["P"] == pytest.approx(
        30 / 0.18
    )
    assert recommendation.product_equivalent_kg_ha_by_nutrient["K"] == pytest.approx(
        40 / 0.12
    )
    assert any("must not be summed" in warning for warning in recommendation.warnings)


def test_missing_laboratory_method_returns_no_rate() -> None:
    results = [
        LabNutrientResult(
            item.nutrient,
            item.value,
            item.unit,
            None if item.nutrient == "P" else item.analysis_method,
            item.basis,
            item.sample_date,
            item.depth_top_cm,
            item.depth_bottom_cm,
            item.laboratory_source,
        )
        for item in SYNTHETIC_RESULTS
    ]
    recommendation = calculate_fertilizer_recommendation(
        "wheat", "synthetic-region", "vegetative", results, SYNTHETIC_CALIBRATIONS
    )

    assert recommendation.state == "needs_more_data"
    assert recommendation.nutrient_requirements_kg_ha == {}
    assert recommendation.product_equivalent_kg_ha_by_nutrient == {}


def test_missing_laboratory_result_returns_no_rate() -> None:
    recommendation = calculate_fertilizer_recommendation(
        "wheat",
        "synthetic-region",
        "vegetative",
        SYNTHETIC_RESULTS[:2],
        SYNTHETIC_CALIBRATIONS,
    )

    assert recommendation.state == "needs_more_data"
    assert recommendation.nutrient_requirements_kg_ha == {}


def test_missing_crop_region_calibration_returns_no_rate() -> None:
    recommendation = calculate_fertilizer_recommendation(
        "wheat",
        "synthetic-region",
        "vegetative",
        SYNTHETIC_RESULTS,
        SYNTHETIC_CALIBRATIONS[:2],
    )

    assert recommendation.state == "review_required"
    assert recommendation.nutrient_requirements_kg_ha == {}


def test_user_entered_calibration_always_requires_review() -> None:
    calibrations = [
        FertilizerCalibrationBand(
            band.crop_key,
            band.region,
            band.nutrient,
            band.growth_stage,
            band.analysis_method,
            band.result_unit,
            band.basis,
            band.minimum_test_value,
            band.maximum_test_value,
            band.recommended_nutrient_kg_ha,
            band.reference_name,
            band.table_identifier,
            user_entered=True,
            verified=True,
        )
        for band in SYNTHETIC_CALIBRATIONS
    ]
    recommendation = calculate_fertilizer_recommendation(
        "wheat", "synthetic-region", "vegetative", SYNTHETIC_RESULTS, calibrations
    )

    assert recommendation.state == "review_required"
    assert recommendation.nutrient_requirements_kg_ha


def test_unmatched_free_text_crop_needs_more_data() -> None:
    recommendation = calculate_fertilizer_recommendation(
        "winter durum wheat variety",
        "synthetic-region",
        "vegetative",
        SYNTHETIC_RESULTS,
        SYNTHETIC_CALIBRATIONS,
    )

    assert recommendation.state == "needs_more_data"
    assert recommendation.nutrient_requirements_kg_ha == {}


def test_negative_lab_result_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LabNutrientResult(
            "N",
            -0.1,
            "mg/kg",
            "Synthetic method",
            "N",
            date(2026, 5, 1),
            0,
            15,
            "Synthetic lab",
        )
