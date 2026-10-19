import pandas as pd
import pytest

from src import random_baseline as baseline


@pytest.mark.parametrize(
    "white_max,special_max",
    [(69, 26), (70, 24)],
)
def test_generate_ticket_obeys_number_ranges(
    white_max,
    special_max,
):
    import random

    white, special = baseline.generate_ticket(
        random.Random(42),
        white_max,
        special_max,
    )

    assert len(white) == 5
    assert len(set(white)) == 5
    assert white == sorted(white)
    assert all(1 <= number <= white_max for number in white)
    assert 1 <= special <= special_max


def test_generate_ticket_is_reproducible():
    import random

    first = baseline.generate_ticket(random.Random(42), 69, 26)
    second = baseline.generate_ticket(random.Random(42), 69, 26)

    assert first == second


@pytest.mark.parametrize(
    "white_max,special_max",
    [(4, 26), (69, 0)],
)
def test_generate_ticket_rejects_invalid_pool_sizes(
    white_max,
    special_max,
):
    import random

    with pytest.raises(ValueError):
        baseline.generate_ticket(
            random.Random(42),
            white_max,
            special_max,
        )


def test_generate_random_predictions_saves_compressed_file(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        baseline,
        "PREDICTIONS_DIR",
        tmp_path / "predictions",
    )

    draws = pd.DataFrame({
        "draw_date": pd.to_datetime([
            "2026-01-01",
            "2026-01-04",
        ])
    })

    output_path = baseline.generate_random_predictions(
        game_name="Powerball",
        draws=draws,
        white_pool_max=69,
        special_pool_max=26,
        simulations=3,
    )

    assert output_path.exists()
    assert output_path.name == "powerball_random.csv.gz"

    results = pd.read_csv(output_path, compression="gzip")

    # Two target dates times three simulations.
    assert len(results) == 6
    assert results["target_draw_date"].nunique() == 2
    assert results["simulation"].nunique() == 3

    # Each date/simulation combination should occur once.
    assert not results.duplicated(
        ["target_draw_date", "simulation"]
    ).any()

    # This file stores predictions, not actual results or scores.
    assert "actual_special" not in results.columns
    assert "white_matches" not in results.columns


def test_random_predictions_are_reproducible(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        baseline,
        "PREDICTIONS_DIR",
        tmp_path / "predictions",
    )

    draws = pd.DataFrame({
        "draw_date": pd.to_datetime([
            "2026-01-01",
            "2026-01-04",
        ])
    })

    output_path = baseline.generate_random_predictions(
        "Powerball", draws, 69, 26, simulations=3
    )

    first = pd.read_csv(output_path, compression="gzip")

    baseline.generate_random_predictions(
        "Powerball", draws, 69, 26, simulations=3
    )

    second = pd.read_csv(output_path, compression="gzip")

    pd.testing.assert_frame_equal(first, second)


def test_random_predictions_reject_zero_simulations():
    draws = pd.DataFrame({
        "draw_date": pd.to_datetime(["2026-01-01"])
    })

    with pytest.raises(ValueError, match="at least 1"):
        baseline.generate_random_predictions(
            "Powerball", draws, 69, 26, simulations=0
        )


def test_random_predictions_reject_empty_data():
    draws = pd.DataFrame({"draw_date": pd.to_datetime([])})

    with pytest.raises(ValueError, match="cannot be empty"):
        baseline.generate_random_predictions(
            "Powerball", draws, 69, 26, simulations=1
        )