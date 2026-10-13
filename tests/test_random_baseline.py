import random

import pandas as pd
import pytest

from src import random_baseline as baseline


def make_powerball_row(draw_date):
    """Create a sample Powerball record for testing."""

    return {
        "draw_date": draw_date,
        "white_1": 1,
        "white_2": 12,
        "white_3": 23,
        "white_4": 36,
        "white_5": 54,
        "powerball": 9,
    }


def make_mega_millions_row(draw_date):
    """Create a sample Mega Millions record for testing."""

    return {
        "draw_date": draw_date,
        "white_1": 2,
        "white_2": 15,
        "white_3": 27,
        "white_4": 48,
        "white_5": 68,
        "mega_ball": 12,
    }


# --------------------------------------------------
# 1. Test random ticket generation
# --------------------------------------------------

@pytest.mark.parametrize(
    "white_max, special_max",
    [(69, 26), (70, 24)],
)
def test_generate_ticket_obeys_number_ranges(
    white_max,
    special_max,
):
    rng = random.Random(42)

    white_numbers, special_number = baseline.generate_ticket(
        rng,
        white_max,
        special_max,
    )

    # A ticket must contain exactly five distinct white balls.
    assert len(white_numbers) == 5
    assert len(set(white_numbers)) == 5

    # White balls must be sorted and within the legal range.
    assert white_numbers == sorted(white_numbers)
    assert all(1 <= n <= white_max for n in white_numbers)

    # The special ball must be within its legal range.
    assert 1 <= special_number <= special_max


def test_generate_ticket_is_reproducible():
    first_ticket = baseline.generate_ticket(
        random.Random(42),
        69,
        26,
    )

    second_ticket = baseline.generate_ticket(
        random.Random(42),
        69,
        26,
    )

    assert first_ticket == second_ticket


@pytest.mark.parametrize(
    "white_max, special_max",
    [(4, 26), (69, 0)],
)
def test_generate_ticket_rejects_invalid_pool_sizes(
    white_max,
    special_max,
):
    with pytest.raises(ValueError):
        baseline.generate_ticket(
            random.Random(42),
            white_max,
            special_max,
        )


# --------------------------------------------------
# 2. Test theoretical probability calculations
# --------------------------------------------------

@pytest.mark.parametrize(
    "pool_size, expected_probability",
    [
        (69, 0.3215728806827024),
        (70, 0.31753462402009947),
    ],
)
def test_theoretical_white_match_probability(
    pool_size,
    expected_probability,
):
    actual_probability = (
        baseline.theoretical_any_white_match_probability(
            pool_size
        )
    )

    assert actual_probability == pytest.approx(
        expected_probability,
    )


# --------------------------------------------------
# 3. Test loading and filtering CSV data
# --------------------------------------------------

def test_load_draws_filters_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "DATA_DIR", tmp_path)

    rows = [
        make_powerball_row("2015-10-06"),
        make_powerball_row("2015-10-07"),
        make_powerball_row("2026-10-05"),
    ]

    pd.DataFrame(rows).to_csv(
        tmp_path / "powerball.csv",
        index=False,
    )

    df = baseline.load_draws(
        "powerball.csv",
        "powerball",
        "2015-10-07",
    )

    # The pre-rule-change drawing must be excluded.
    assert len(df) == 2
    assert df["draw_date"].min() == pd.Timestamp("2015-10-07")

    # Number columns should be usable as integers.
    assert df["white_1"].dtype.kind in "iu"


def test_load_draws_fails_when_file_is_missing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(baseline, "DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        baseline.load_draws(
            "missing.csv",
            "powerball",
            "2015-10-07",
        )


def test_load_draws_fails_when_column_is_missing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(baseline, "DATA_DIR", tmp_path)

    row = make_powerball_row("2026-10-05")
    del row["powerball"]

    pd.DataFrame([row]).to_csv(
        tmp_path / "powerball.csv",
        index=False,
    )

    with pytest.raises(ValueError, match="missing columns"):
        baseline.load_draws(
            "powerball.csv",
            "powerball",
            "2015-10-07",
        )


def test_load_draws_fails_on_invalid_date(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(baseline, "DATA_DIR", tmp_path)

    pd.DataFrame([
        make_powerball_row("not-a-date")
    ]).to_csv(
        tmp_path / "powerball.csv",
        index=False,
    )

    with pytest.raises(ValueError, match="invalid drawing dates"):
        baseline.load_draws(
            "powerball.csv",
            "powerball",
            "2015-10-07",
        )


# --------------------------------------------------
# 4. Test Mega Millions current-rule filtering
# --------------------------------------------------

def test_mega_millions_uses_current_rule_period(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(baseline, "DATA_DIR", tmp_path)

    rows = [
        make_mega_millions_row("2025-04-04"),
        make_mega_millions_row("2025-04-08"),
        make_mega_millions_row("2026-10-06"),
    ]

    pd.DataFrame(rows).to_csv(
        tmp_path / "mega_millions.csv",
        index=False,
    )

    df = baseline.prepare_mega_millions()

    # Only drawings from the current rule period are included.
    assert len(df) == 2
    assert (df["draw_date"] >= pd.Timestamp("2025-04-08")).all()

    # Every special ball must use the current 1–24 pool.
    assert (df["special_pool_max"] == 24).all()


# --------------------------------------------------
# 5. Test simulation output
# --------------------------------------------------

def test_run_simulations_returns_and_saves_results(
    tmp_path,
    monkeypatch,
):
    # Make the test small and fast.
    monkeypatch.setattr(baseline, "SIMULATIONS", 3)
    monkeypatch.setattr(baseline, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(
        baseline,
        "REPORTS_DIR",
        tmp_path / "reports" / "random_baseline",
    )

    df = pd.DataFrame([
        {
            "draw_date": pd.Timestamp("2025-04-08"),
            "white_1": 1,
            "white_2": 2,
            "white_3": 3,
            "white_4": 4,
            "white_5": 5,
            "mega_ball": 6,
            "special_pool_max": 24,
        },
        {
            "draw_date": pd.Timestamp("2025-04-11"),
            "white_1": 10,
            "white_2": 20,
            "white_3": 30,
            "white_4": 40,
            "white_5": 50,
            "mega_ball": 12,
            "special_pool_max": 24,
        },
    ])

    results = baseline.run_simulations(
        "Mega Millions",
        df,
        "mega_ball",
        70,
    )

    # We asked for three simulation runs.
    assert len(results) == 3

    # The results should contain the expected metrics.
    assert "average_white_matches" in results.columns
    assert "tickets_with_white_match_pct" in results.columns
    assert "special_ball_match_pct" in results.columns

    # Metrics must be within their possible ranges.
    assert results["average_white_matches"].between(0, 5).all()
    assert results["tickets_with_white_match_pct"].between(0, 100).all()
    assert results["special_ball_match_pct"].between(0, 100).all()

    # The CSV output should have been created.
    output_file = (
        tmp_path
        / "reports"
        / "random_baseline"
        / "mega_millions_simulations.csv"
    )

    assert output_file.exists()

    saved_results = pd.read_csv(output_file)
    assert len(saved_results) == 3