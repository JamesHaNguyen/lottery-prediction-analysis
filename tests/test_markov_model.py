
import pandas as pd
import pytest

from src.markov_model import (
    generate_markov_ticket,
    generate_next_draw_prediction,
    generate_predictions_for_backtest,
)


def make_sample_draws(count=12):
    records = []

    for index in range(count):
        white_start = (index * 5) % 60 + 1
        white_balls = [
            ((white_start + offset - 1) % 69) + 1
            for offset in range(5)
        ]

        records.append({
            "draw_date": pd.Timestamp("2026-01-01")
            + pd.Timedelta(days=index * 3),
            "white_1": white_balls[0],
            "white_2": white_balls[1],
            "white_3": white_balls[2],
            "white_4": white_balls[3],
            "white_5": white_balls[4],
            "powerball": index % 26 + 1,
        })

    return pd.DataFrame(records)


def test_markov_ticket_has_valid_numbers():
    draws = make_sample_draws()

    ticket = generate_markov_ticket(
        draws, "powerball", 69, 26
    )

    white_balls = [
        ticket[f"predicted_white_{index}"]
        for index in range(1, 6)
    ]

    assert len(set(white_balls)) == 5
    assert white_balls == sorted(white_balls)
    assert all(1 <= number <= 69 for number in white_balls)
    assert 1 <= ticket["predicted_special"] <= 26


def test_markov_ticket_is_deterministic():
    draws = make_sample_draws()

    first = generate_markov_ticket(draws, "powerball", 69, 26)
    second = generate_markov_ticket(draws, "powerball", 69, 26)

    assert first == second


def test_backtest_predictions_use_only_past_draws():
    draws = make_sample_draws(count=8)

    predictions = generate_predictions_for_backtest(
        draws,
        "powerball",
        69,
        26,
        min_history_draws=3,
    )

    assert len(predictions) == 8 - 3
    assert predictions.iloc[0]["history_end_date"] == "2026-01-07"
    assert predictions.iloc[0]["target_draw_date"] == "2026-01-10"

    for _, row in predictions.iterrows():
        assert row["history_end_date"] < row["target_draw_date"]


def test_backtest_requires_enough_drawings():
    draws = make_sample_draws(count=3)

    with pytest.raises(ValueError, match="Not enough drawings"):
        generate_predictions_for_backtest(
            draws,
            "powerball",
            69,
            26,
            min_history_draws=3,
        )


def test_backtest_requires_at_least_two_history_draws():
    draws = make_sample_draws()

    with pytest.raises(ValueError, match="at least 2"):
        generate_predictions_for_backtest(
            draws,
            "powerball",
            69,
            26,
            min_history_draws=1,
        )


def test_next_draw_prediction_has_history_metadata():
    draws = make_sample_draws(count=8)

    prediction = generate_next_draw_prediction(
        draws,
        "powerball",
        69,
        26,
        "2026-01-25",
    )

    assert prediction["target_draw_date"] == "2026-01-25"
    assert prediction["history_draws"] == 8
    assert prediction["history_end_date"] == "2026-01-22"


def test_next_draw_rejects_date_not_after_history():
    draws = make_sample_draws()

    with pytest.raises(ValueError, match="after the latest"):
        generate_next_draw_prediction(
            draws,
            "powerball",
            69,
            26,
            "2026-01-01",
        )


def test_markov_ticket_rejects_invalid_alpha():
    draws = make_sample_draws()

    with pytest.raises(ValueError, match="alpha"):
        generate_markov_ticket(
            draws,
            "powerball",
            69,
            26,
            alpha=0,
        )


def test_markov_ticket_rejects_duplicate_white_balls():
    draws = make_sample_draws()
    draws.loc[0, "white_2"] = draws.loc[0, "white_1"]

    with pytest.raises(ValueError, match="5 unique white balls"):
        generate_markov_ticket(draws, "powerball", 69, 26)
