import random

import pandas as pd
import pytest

from src import frequency_model as model


WHITE_COLUMNS = [
    "white_1",
    "white_2",
    "white_3",
    "white_4",
    "white_5",
]


def make_sample_draws():
    """Create artificial drawings for testing."""

    numbers = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 6],
        [1, 2, 3, 7, 8],
        [2, 4, 6, 8, 10],
        [1, 5, 7, 9, 10],
    ]

    rows = []

    for index, white_numbers in enumerate(numbers):
        row = {
            "draw_date": (
                pd.Timestamp("2026-01-01")
                + pd.Timedelta(days=3 * index)
            ),
            "special": index + 1,
        }

        for column, number in zip(
            WHITE_COLUMNS,
            white_numbers,
        ):
            row[column] = number

        rows.append(row)

    return pd.DataFrame(rows)


def test_rank_numbers_by_frequency():
    history = make_sample_draws().iloc[:2]

    ranking = model.rank_numbers_by_frequency(
        history,
        WHITE_COLUMNS,
        10,
        random.Random(42),
    )

    # These four numbers each appear twice.
    assert set(ranking[:4]) == {1, 2, 3, 4}

    # Every legal number appears exactly once.
    assert len(ranking) == 10
    assert set(ranking) == set(range(1, 11))


def test_generate_frequency_ticket_obeys_number_ranges():
    history = make_sample_draws().iloc[:3]

    white, special = model.generate_frequency_ticket(
        history,
        "special",
        10,
        5,
        random.Random(42),
    )

    assert len(white) == 5
    assert len(set(white)) == 5
    assert white == sorted(white)
    assert all(1 <= number <= 10 for number in white)
    assert 1 <= special <= 5


def test_frequency_ticket_is_reproducible():
    history = make_sample_draws().iloc[:3]

    first = model.generate_frequency_ticket(
        history, "special", 10, 5, random.Random(42)
    )

    second = model.generate_frequency_ticket(
        history, "special", 10, 5, random.Random(42)
    )

    assert first == second


def test_predictions_use_only_prior_draws():
    df = make_sample_draws()

    results = model.generate_predictions_for_window(
        draws=df,
        special_column="special",
        white_pool_max=10,
        special_pool_max=5,
        lookback_draws=2,
    )

    # Five draws minus two initial history draws.
    assert len(results) == 3

    first = results.iloc[0]

    assert first["history_start_date"] == "2026-01-01"
    assert first["history_end_date"] == "2026-01-04"
    assert first["target_draw_date"] == "2026-01-07"
    assert first["lookback_draws"] == 2


def test_target_result_does_not_change_its_own_prediction():
    df = make_sample_draws()

    original = model.generate_predictions_for_window(
        df, "special", 10, 5, lookback_draws=2
    )

    modified_df = df.copy()

    # Change only the actual numbers of the first target drawing.
    modified_df.loc[2, WHITE_COLUMNS] = [2, 5, 7, 8, 9]
    modified_df.loc[2, "special"] = 5

    modified = model.generate_predictions_for_window(
        modified_df, "special", 10, 5, lookback_draws=2
    )

    prediction_columns = [
        *(f"predicted_white_{i}" for i in range(1, 6)),
        "predicted_special",
    ]

    assert (
        original.loc[0, prediction_columns].tolist()
        == modified.loc[0, prediction_columns].tolist()
    )


def test_invalid_lookback_is_rejected():
    df = make_sample_draws()

    with pytest.raises(ValueError, match="at least 1"):
        model.generate_predictions_for_window(
            df, "special", 10, 5, lookback_draws=0
        )


def test_insufficient_history_is_rejected():
    df = make_sample_draws()

    with pytest.raises(ValueError, match="Not enough drawings"):
        model.generate_predictions_for_window(
            df, "special", 10, 5, lookback_draws=5
        )


def test_duplicate_drawing_dates_are_rejected():
    df = make_sample_draws()
    df.loc[1, "draw_date"] = df.loc[0, "draw_date"]

    with pytest.raises(ValueError, match="unique"):
        model.generate_predictions_for_window(
            df, "special", 10, 5, lookback_draws=2
        )


def test_each_lookback_gets_a_separate_output_file(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(model, "PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr(model, "LOOKBACK_WINDOWS", [2, 3])

    model.generate_game_predictions(
        game_name="Powerball",
        draws=make_sample_draws(),
        special_column="special",
        white_pool_max=10,
        special_pool_max=5,
    )

    file_2 = tmp_path / "powerball_lookback_2.csv"
    file_3 = tmp_path / "powerball_lookback_3.csv"

    assert file_2.exists()
    assert file_3.exists()

    assert len(pd.read_csv(file_2)) == 3
    assert len(pd.read_csv(file_3)) == 2