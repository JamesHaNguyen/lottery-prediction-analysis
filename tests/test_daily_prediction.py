from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.daily_prediction import next_drawing_date
from src.frequency_model import generate_next_draw_prediction
from src.random_baseline import generate_next_random_prediction


CENTRAL_TIME = ZoneInfo("America/Chicago")

WHITE_COLUMNS = [
    "white_1",
    "white_2",
    "white_3",
    "white_4",
    "white_5",
]


def make_sample_draws():
    """Create a small artificial dataset for testing."""

    white_numbers = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 6],
        [1, 2, 3, 7, 8],
        [2, 4, 6, 8, 10],
        [1, 5, 7, 9, 10],
    ]

    rows = []

    for index, numbers in enumerate(white_numbers):
        row = {
            "draw_date": (
                pd.Timestamp("2026-01-01")
                + pd.Timedelta(days=3 * index)
            ),
            "special": index + 1,
        }

        for column, number in zip(WHITE_COLUMNS, numbers):
            row[column] = number

        rows.append(row)

    return pd.DataFrame(rows)


def test_frequency_next_prediction_uses_latest_history_window():
    draws = make_sample_draws()

    prediction = generate_next_draw_prediction(
        draws=draws,
        special_column="special",
        white_pool_max=10,
        special_pool_max=5,
        lookback_draws=2,
        target_draw_date="2026-01-20",
    )

    assert prediction["target_draw_date"] == "2026-01-20"
    assert prediction["lookback_draws"] == 2
    assert prediction["history_start_date"] == "2026-01-10"
    assert prediction["history_end_date"] == "2026-01-13"

    whites = [
        prediction[f"predicted_white_{i}"]
        for i in range(1, 6)
    ]

    assert len(set(whites)) == 5
    assert whites == sorted(whites)
    assert all(1 <= number <= 10 for number in whites)
    assert 1 <= prediction["predicted_special"] <= 5


def test_frequency_next_prediction_ignores_draws_outside_window():
    draws = make_sample_draws()

    original = generate_next_draw_prediction(
        draws, "special", 10, 5, 2, "2026-01-20"
    )

    modified_draws = draws.copy()
    modified_draws.loc[0, WHITE_COLUMNS] = [6, 7, 8, 9, 10]
    modified_draws.loc[0, "special"] = 5

    modified = generate_next_draw_prediction(
        modified_draws, "special", 10, 5, 2, "2026-01-20"
    )

    # The altered drawing is outside the last two draws used.
    assert original == modified


def test_frequency_next_prediction_rejects_insufficient_history():
    draws = make_sample_draws().iloc[:2]

    try:
        generate_next_draw_prediction(
            draws, "special", 10, 5, 3, "2026-01-20"
        )
    except ValueError as error:
        assert "Not enough historical drawings" in str(error)
    else:
        raise AssertionError("Expected insufficient history to fail.")


def test_random_next_prediction_is_reproducible():
    draws = make_sample_draws()

    first = generate_next_random_prediction(
        "Powerball", draws, 69, 26, "2026-01-20"
    )
    second = generate_next_random_prediction(
        "Powerball", draws, 69, 26, "2026-01-20"
    )

    assert first == second


def test_random_next_prediction_obeys_number_ranges():
    prediction = generate_next_random_prediction(
        "Powerball",
        make_sample_draws(),
        69,
        26,
        "2026-01-20",
    )

    whites = [
        prediction[f"predicted_white_{i}"]
        for i in range(1, 6)
    ]

    assert len(set(whites)) == 5
    assert whites == sorted(whites)
    assert all(1 <= number <= 69 for number in whites)
    assert 1 <= prediction["predicted_special"] <= 26


def test_powerball_uses_tonight_when_before_drawing_time():
    now = datetime(
        2026, 10, 19, 16, 0,
        tzinfo=CENTRAL_TIME,
    )

    assert next_drawing_date("powerball", now) == "2026-10-19"


def test_powerball_skips_tonight_after_drawing_time():
    now = datetime(
        2026, 10, 19, 22, 30,
        tzinfo=CENTRAL_TIME,
    )

    assert next_drawing_date("powerball", now) == "2026-10-21"


def test_mega_millions_next_drawing_is_tuesday():
    now = datetime(
        2026, 10, 19, 16, 0,
        tzinfo=CENTRAL_TIME,
    )

    assert next_drawing_date("mega_millions", now) == "2026-10-20"


def test_mega_millions_skips_tuesday_after_drawing_time():
    now = datetime(
        2026, 10, 20, 22, 15,
        tzinfo=CENTRAL_TIME,
    )

    assert next_drawing_date("mega_millions", now) == "2026-10-23"