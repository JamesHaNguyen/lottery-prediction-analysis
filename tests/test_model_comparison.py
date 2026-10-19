import pandas as pd
import pytest

from src import model_comparison as comparison


def test_load_common_dates_uses_longest_lookback(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        comparison,
        "FREQUENCY_DIR",
        tmp_path,
    )
    monkeypatch.setattr(
        comparison,
        "LOOKBACK_WINDOWS",
        [10, 25, 100],
    )

    pd.DataFrame({
        "target_draw_date": [
            "2026-01-10",
            "2026-01-13",
        ]
    }).to_csv(
        tmp_path / "powerball_lookback_100.csv",
        index=False,
    )

    dates = comparison.load_common_evaluation_dates(
        "powerball"
    )

    assert dates == [
        "2026-01-10",
        "2026-01-13",
    ]


def test_frequency_predictions_are_filtered_to_common_dates():
    predictions = pd.DataFrame({
        "target_draw_date": [
            "2026-01-01",
            "2026-01-04",
            "2026-01-07",
        ],
        "predicted_special": [1, 2, 3],
    })

    common_dates = [
        "2026-01-04",
        "2026-01-07",
    ]

    filtered = comparison.filter_frequency_to_common_dates(
        predictions,
        common_dates,
        lookback=10,
    )

    assert filtered["target_draw_date"].tolist() == common_dates


def test_frequency_predictions_fail_if_common_date_is_missing():
    predictions = pd.DataFrame({
        "target_draw_date": ["2026-01-04"],
        "predicted_special": [1],
    })

    with pytest.raises(ValueError, match="missing 1 common"):
        comparison.filter_frequency_to_common_dates(
            predictions,
            ["2026-01-04", "2026-01-07"],
            lookback=50,
        )


def test_random_predictions_are_filtered_to_common_dates():
    predictions = pd.DataFrame({
        "target_draw_date": [
            "2026-01-01",
            "2026-01-01",
            "2026-01-04",
            "2026-01-04",
        ],
        "simulation": [1, 2, 1, 2],
        "predicted_white_1": [1, 2, 3, 4],
        "predicted_white_2": [10, 11, 12, 13],
        "predicted_white_3": [20, 21, 22, 23],
        "predicted_white_4": [30, 31, 32, 33],
        "predicted_white_5": [40, 41, 42, 43],
        "predicted_special": [5, 6, 7, 8],
    })

    filtered = comparison.filter_random_to_common_dates(
        predictions,
        ["2026-01-04"],
    )

    assert filtered["target_draw_date"].nunique() == 1
    assert set(filtered["target_draw_date"]) == {"2026-01-04"}
    assert len(filtered) == 2


def test_random_predictions_fail_when_date_is_missing():
    predictions = pd.DataFrame({
        "target_draw_date": ["2026-01-04"],
        "simulation": [1],
        "predicted_white_1": [1],
        "predicted_white_2": [10],
        "predicted_white_3": [20],
        "predicted_white_4": [30],
        "predicted_white_5": [40],
        "predicted_special": [5],
    })

    with pytest.raises(ValueError, match="missing 1 common"):
        comparison.filter_random_to_common_dates(
            predictions,
            ["2026-01-04", "2026-01-07"],
        )


def test_score_predictions_calculates_matches():
    predictions = pd.DataFrame({
        "target_draw_date": ["2026-01-04"],
        "predicted_white_1": [1],
        "predicted_white_2": [2],
        "predicted_white_3": [8],
        "predicted_white_4": [9],
        "predicted_white_5": [10],
        "predicted_special": [7],
    })

    actual_results = pd.DataFrame({
        "target_draw_date": ["2026-01-04"],
        "actual_white_1": [1],
        "actual_white_2": [2],
        "actual_white_3": [3],
        "actual_white_4": [4],
        "actual_white_5": [5],
        "actual_special": [7],
    })

    scored = comparison.score_predictions(
        predictions,
        actual_results,
    )

    assert scored.loc[0, "white_matches"] == 2
    assert scored.loc[0, "has_white_match"] == 1
    assert scored.loc[0, "special_match"] == 1