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
    """Create a small, artificial dataset for testing."""

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
            "draw_date": pd.Timestamp("2026-01-01")
            + pd.Timedelta(days=3 * index),
            "special": index + 1,
        }

        for column, number in zip(
            WHITE_COLUMNS, white_numbers
        ):
            row[column] = number

        rows.append(row)

    return pd.DataFrame(rows)


def test_rank_numbers_by_frequency():
    history = make_sample_draws().iloc[:2]

    ranking = model.rank_numbers_by_frequency(
        history=history,
        columns=WHITE_COLUMNS,
        pool_max=10,
        rng=random.Random(42),
    )

    # The numbers 1, 2, 3 and 4 each appeared twice.
    assert set(ranking[:4]) == {1, 2, 3, 4}

    # Every legal number appears exactly once in the ranking.
    assert len(ranking) == 10
    assert set(ranking) == set(range(1, 11))


def test_generate_frequency_ticket_obeys_number_ranges():
    history = make_sample_draws().iloc[:3]

    white_numbers, special_number = (
        model.generate_frequency_ticket(
            history=history,
            special_column="special",
            white_pool_max=10,
            special_pool_max=5,
            rng=random.Random(42),
        )
    )

    assert len(white_numbers) == 5
    assert len(set(white_numbers)) == 5
    assert white_numbers == sorted(white_numbers)

    assert all(1 <= number <= 10 for number in white_numbers)
    assert 1 <= special_number <= 5


def test_frequency_ticket_is_reproducible():
    history = make_sample_draws().iloc[:3]

    first = model.generate_frequency_ticket(
        history,
        "special",
        10,
        5,
        random.Random(42),
    )

    second = model.generate_frequency_ticket(
        history,
        "special",
        10,
        5,
        random.Random(42),
    )

    assert first == second


def test_score_ticket_counts_matches_correctly():
    actual = pd.Series({
        "white_1": 1,
        "white_2": 2,
        "white_3": 3,
        "white_4": 4,
        "white_5": 5,
        "special": 2,
    })

    score = model.score_ticket(
        predicted_white=[1, 2, 8, 9, 10],
        predicted_special=2,
        actual_row=actual,
        special_column="special",
    )

    assert score == {
        "white_matches": 2,
        "has_white_match": 1,
        "special_match": 1,
    }


def test_backtest_uses_only_prior_draws():
    df = make_sample_draws()

    results = model.run_backtest(
        df=df,
        special_column="special",
        white_pool_max=10,
        special_pool_max=5,
        lookback_draws=2,
    )

    # Five draws minus two initial history draws = three predictions.
    assert len(results) == 3

    # The first prediction uses only draws 1 and 2.
    assert results.iloc[0]["history_start_date"] == (
        df.iloc[0]["draw_date"].date().isoformat()
    )
    assert results.iloc[0]["history_end_date"] == (
        df.iloc[1]["draw_date"].date().isoformat()
    )

    assert results.iloc[0]["target_draw_date"] == (
        df.iloc[2]["draw_date"].date().isoformat()
    )


def test_target_result_does_not_change_its_own_prediction():
    df = make_sample_draws()

    original = model.run_backtest(
        df,
        "special",
        10,
        5,
        lookback_draws=2,
    )

    modified_df = df.copy()

    # Change only the first target drawing's actual result.
    # The prediction for that drawing must remain unchanged.
    modified_df.loc[
        modified_df.index[2],
        WHITE_COLUMNS
    ] = [2, 5, 7, 8, 9]

    modified_df.loc[modified_df.index[2], "special"] = 5

    modified = model.run_backtest(
        modified_df,
        "special",
        10,
        5,
        lookback_draws=2,
    )

    prediction_columns = [
        *(f"frequency_white_{i}" for i in range(1, 6)),
        "frequency_special",
    ]

    assert (
        original.loc[0, prediction_columns].tolist()
        == modified.loc[0, prediction_columns].tolist()
    )


def test_random_simulations_return_expected_metrics():
    df = make_sample_draws()

    results = model.run_random_simulations(
        df=df,
        special_column="special",
        white_pool_max=10,
        special_pool_max=5,
        lookback_draws=2,
        simulations=4,
    )

    assert len(results) == 4

    expected_columns = {
        "simulation",
        "average_white_matches",
        "tickets_with_white_match_pct",
        "special_ball_match_rate_pct",
    }

    assert expected_columns.issubset(results.columns)

    assert results["average_white_matches"].between(0, 5).all()
    assert results["tickets_with_white_match_pct"].between(
        0, 100
    ).all()
    assert results["special_ball_match_rate_pct"].between(
        0, 100
    ).all()


def test_random_simulations_are_reproducible():
    df = make_sample_draws()

    first = model.run_random_simulations(
        df, "special", 10, 5,
        lookback_draws=2,
        simulations=4,
    )

    second = model.run_random_simulations(
        df, "special", 10, 5,
        lookback_draws=2,
        simulations=4,
    )

    pd.testing.assert_frame_equal(first, second)


def test_backtest_rejects_invalid_lookback():
    df = make_sample_draws()

    with pytest.raises(ValueError, match="at least 1"):
        model.run_backtest(
            df,
            "special",
            10,
            5,
            lookback_draws=0,
        )


def test_random_simulations_reject_zero_runs():
    df = make_sample_draws()

    with pytest.raises(ValueError, match="at least 1"):
        model.run_random_simulations(
            df,
            "special",
            10,
            5,
            lookback_draws=2,
            simulations=0,
        )