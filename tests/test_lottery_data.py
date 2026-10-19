import pandas as pd
import pytest

from src import lottery_data


def make_powerball_row(draw_date):
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
    return {
        "draw_date": draw_date,
        "white_1": 2,
        "white_2": 15,
        "white_3": 27,
        "white_4": 48,
        "white_5": 68,
        "mega_ball": 12,
    }


def test_load_draws_filters_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(lottery_data, "DATA_DIR", tmp_path)

    rows = [
        make_powerball_row("2015-10-06"),
        make_powerball_row("2015-10-07"),
        make_powerball_row("2026-10-05"),
    ]

    pd.DataFrame(rows).to_csv(
        tmp_path / "powerball.csv",
        index=False,
    )

    df = lottery_data.load_draws(
        "powerball.csv",
        "powerball",
        "2015-10-07",
    )

    assert len(df) == 2
    assert df["draw_date"].min() == pd.Timestamp("2015-10-07")
    assert df["draw_date"].max() == pd.Timestamp("2026-10-05")
    assert df["white_1"].dtype.kind in "iu"


def test_load_draws_fails_when_file_is_missing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(lottery_data, "DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        lottery_data.load_draws(
            "missing.csv",
            "powerball",
            "2015-10-07",
        )


def test_load_draws_fails_when_column_is_missing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(lottery_data, "DATA_DIR", tmp_path)

    row = make_powerball_row("2026-10-05")
    del row["powerball"]

    pd.DataFrame([row]).to_csv(
        tmp_path / "powerball.csv",
        index=False,
    )

    with pytest.raises(ValueError, match="missing columns"):
        lottery_data.load_draws(
            "powerball.csv",
            "powerball",
            "2015-10-07",
        )


def test_load_draws_fails_on_invalid_date(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(lottery_data, "DATA_DIR", tmp_path)

    pd.DataFrame([
        make_powerball_row("not-a-date")
    ]).to_csv(
        tmp_path / "powerball.csv",
        index=False,
    )

    with pytest.raises(ValueError, match="invalid drawing dates"):
        lottery_data.load_draws(
            "powerball.csv",
            "powerball",
            "2015-10-07",
        )


def test_mega_millions_uses_current_rule_period(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(lottery_data, "DATA_DIR", tmp_path)

    rows = [
        make_mega_millions_row("2025-04-04"),
        make_mega_millions_row("2025-04-08"),
        make_mega_millions_row("2026-10-06"),
    ]

    pd.DataFrame(rows).to_csv(
        tmp_path / "mega_millions.csv",
        index=False,
    )

    df = lottery_data.prepare_mega_millions()

    assert len(df) == 2
    assert (df["draw_date"] >= pd.Timestamp("2025-04-08")).all()


def test_load_draws_rejects_duplicate_dates(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(lottery_data, "DATA_DIR", tmp_path)

    rows = [
        make_powerball_row("2026-10-05"),
        make_powerball_row("2026-10-05"),
    ]

    pd.DataFrame(rows).to_csv(
        tmp_path / "powerball.csv",
        index=False,
    )

    with pytest.raises(ValueError, match="duplicate drawing dates"):
        lottery_data.load_draws(
            "powerball.csv",
            "powerball",
            "2015-10-07",
        )