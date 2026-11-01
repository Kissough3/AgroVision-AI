from datetime import date

import pytest

from app.core.irrigation import (
    DASTANE_FAO25_CANDIDATE,
    EFFECTIVE_RAIN_METHODS,
    ApplicabilityRange,
    CropCoefficient,
    DailyWeatherInput,
    GrowthStageLength,
    ReferencedCropParameter,
    SoilWaterLayer,
    calculate_irrigation_recommendation,
    calculate_root_zone_water,
    mm_to_m3,
    resolve_growth_stage,
)

SYNTHETIC_KC = CropCoefficient(
    crop_key="wheat",
    value=0.8,
    growth_stage="vegetative",
    reference_name="Synthetic test input, not an agronomic source",
    table_identifier="Synthetic table for unit tests",
    applicability_conditions="Synthetic temperature interval for unit tests",
    applicability_ranges=(ApplicabilityRange("temperature_c", 10, 25),),
)


def synthetic_day(
    *,
    et0: float | None = 5.0,
    rain: float | None = 2.0,
    kc: CropCoefficient | None = SYNTHETIC_KC,
    temperature: float = 20.0,
) -> DailyWeatherInput:
    return DailyWeatherInput(
        day=date(2026, 6, 1),
        et0_mm=et0,
        forecast_rain_mm=rain,
        crop_coefficient=kc,
        climate_values={"temperature_c": temperature},
    )


def test_et0_and_forecast_rain_are_separate_without_selected_method() -> None:
    result = calculate_irrigation_recommendation(
        "wheat", "vegetative", [synthetic_day()], 1.0
    )

    assert result.etc_mm == pytest.approx(4.0)
    assert result.forecast_rain_mm == pytest.approx(2.0)
    assert result.net_requirement_mm is None
    assert result.rain_netted is False
    assert result.state == "needs_more_data"
    assert any("rain was not netted" in warning for warning in result.warnings)


def test_selected_unverified_method_nets_supplied_rain_and_clamps_negative() -> None:
    result = calculate_irrigation_recommendation(
        "wheat",
        "vegetative",
        [synthetic_day(rain=6.0)],
        1.0,
        selected_effective_rain_method_id=DASTANE_FAO25_CANDIDATE,
        effective_rain_mm=5.0,
    )

    assert result.net_requirement_mm == 0
    assert result.quantity_m3 == 0
    assert result.rain_netted is True
    assert result.state == "review_required"
    assert any("clamped to 0 mm" in text for text in result.explanation)
    assert result.effective_rain_method is not None
    assert result.effective_rain_method.verified is False


def test_effective_rain_candidate_has_published_reference_but_no_formula() -> None:
    method = EFFECTIVE_RAIN_METHODS[DASTANE_FAO25_CANDIDATE]

    assert "Dastane" in method.reference
    assert "FAO Irrigation and Drainage Paper 25" in method.reference
    assert method.verified is False
    assert "requires primary-source verification" in method.reference_locator


def test_kc_without_reference_or_applicability_is_rejected() -> None:
    missing_reference = CropCoefficient(
        crop_key="wheat",
        value=0.8,
        growth_stage="vegetative",
        reference_name="",
        table_identifier="",
        applicability_conditions="",
    )
    result = calculate_irrigation_recommendation(
        "wheat", "vegetative", [synthetic_day(kc=missing_reference)], 1.0
    )

    assert result.state == "needs_more_data"
    assert result.etc_mm is None
    assert "Kc is rejected" in result.explanation[0]


def test_kc_climate_outside_applicability_warns_and_requires_review() -> None:
    result = calculate_irrigation_recommendation(
        "wheat",
        "vegetative",
        [synthetic_day(temperature=30)],
        1.0,
        selected_effective_rain_method_id=DASTANE_FAO25_CANDIDATE,
        effective_rain_mm=0.5,
    )

    assert result.state == "review_required"
    assert any(
        "outside the Kc applicability range" in warning for warning in result.warnings
    )


def test_effective_rain_cannot_exceed_forecast_rain() -> None:
    result = calculate_irrigation_recommendation(
        "wheat",
        "vegetative",
        [synthetic_day(rain=2.0)],
        1.0,
        selected_effective_rain_method_id=DASTANE_FAO25_CANDIDATE,
        effective_rain_mm=3.0,
    )

    assert result.state == "needs_more_data"
    assert result.net_requirement_mm is None
    assert result.rain_netted is False


def test_incomplete_forecast_cannot_be_netted() -> None:
    result = calculate_irrigation_recommendation(
        "wheat",
        "vegetative",
        [synthetic_day(rain=None)],
        1.0,
        selected_effective_rain_method_id=DASTANE_FAO25_CANDIDATE,
        effective_rain_mm=0.0,
    )

    assert result.state == "needs_more_data"
    assert result.net_requirement_mm is None
    assert result.rain_netted is False


def test_unmatched_free_text_crop_needs_more_data() -> None:
    result = calculate_irrigation_recommendation(
        "winter durum wheat variety", "vegetative", [synthetic_day()], 1.0
    )

    assert result.state == "needs_more_data"
    assert result.etc_mm is None


def test_missing_et0_prevents_etc_calculation() -> None:
    result = calculate_irrigation_recommendation(
        "wheat", "vegetative", [synthetic_day(et0=None)], 1.0
    )

    assert result.state == "needs_more_data"
    assert result.etc_mm is None


def test_growth_stage_uses_manual_value_or_referenced_lengths() -> None:
    lengths = [
        GrowthStageLength("wheat", "initial", 20, 0, "Synthetic", "Synthetic table"),
        GrowthStageLength("wheat", "vegetative", 40, 1, "Synthetic", "Synthetic table"),
    ]

    assert resolve_growth_stage("manual stage", 45, lengths, "wheat") == "manual stage"
    assert resolve_growth_stage(None, 25, lengths, "wheat") == "vegetative"
    assert resolve_growth_stage(None, 100, lengths, "wheat") == "vegetative"
    assert resolve_growth_stage(None, 25, [], "wheat") is None


def test_root_zone_water_uses_referenced_root_depth_and_p() -> None:
    root_depth = ReferencedCropParameter(
        "wheat", "root_depth", 45, "cm", "vegetative", "Synthetic", "Synthetic table"
    )
    depletion_fraction = ReferencedCropParameter(
        "wheat",
        "depletion_fraction_p",
        0.5,
        "fraction",
        "vegetative",
        "Synthetic",
        "Synthetic table",
    )
    water = calculate_root_zone_water(
        [SoilWaterLayer(0, 30, 45), SoilWaterLayer(30, 60, 60)],
        root_depth,
        depletion_fraction,
    )

    assert water.total_available_water_mm == pytest.approx(75)
    assert water.readily_available_water_mm == pytest.approx(37.5)


def test_root_depth_without_reference_is_rejected() -> None:
    root_depth = ReferencedCropParameter(
        "wheat", "root_depth", 45, "cm", "vegetative", "", ""
    )
    depletion_fraction = ReferencedCropParameter(
        "wheat",
        "depletion_fraction_p",
        0.5,
        "fraction",
        "vegetative",
        "Synthetic",
        "Synthetic table",
    )

    with pytest.raises(ValueError, match="unreferenced root_depth"):
        calculate_root_zone_water(
            [SoilWaterLayer(0, 60, 105)], root_depth, depletion_fraction
        )


def test_one_mm_over_one_hectare_is_ten_cubic_metres() -> None:
    assert mm_to_m3(1, 1) == 10


def test_recommendation_uses_one_field_area_for_volume() -> None:
    result = calculate_irrigation_recommendation(
        "wheat",
        "vegetative",
        [synthetic_day()],
        2.0,
        selected_effective_rain_method_id=DASTANE_FAO25_CANDIDATE,
        effective_rain_mm=1.0,
    )

    assert result.net_requirement_mm == pytest.approx(3.0)
    assert result.quantity_m3 == pytest.approx(60.0)


def test_missing_field_area_prevents_volume_recommendation() -> None:
    result = calculate_irrigation_recommendation(
        "wheat",
        "vegetative",
        [synthetic_day()],
        None,
        selected_effective_rain_method_id=DASTANE_FAO25_CANDIDATE,
        effective_rain_mm=1.0,
    )

    assert result.state == "needs_more_data"
    assert result.net_requirement_mm == pytest.approx(3.0)
    assert result.quantity_m3 is None


def test_invalid_area_is_rejected() -> None:
    with pytest.raises(ValueError, match="field_area_ha"):
        mm_to_m3(1, 0)
