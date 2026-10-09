import csv

import pytest

from src.data_validator import validate_date, validate_lottery_file


# The columns expected in a Powerball CSV file.
COLUMNS = [
    "draw_date",
    "white_1",
    "white_2",
    "white_3",
    "white_4",
    "white_5",
    "powerball",
]


# A sample row that should pass validation.
VALID_ROW = {
    "draw_date": "2026-10-05",
    "white_1": "16",
    "white_2": "23",
    "white_3": "32",
    "white_4": "36",
    "white_5": "54",
    "powerball": "09",
}


def write_csv(path, columns=COLUMNS, rows=None):
    """Create a temporary CSV file for testing."""

    if rows is None:
        rows = [VALID_ROW]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    return path


def test_valid_date():
    assert validate_date("2026-10-05") is True


def test_invalid_date():
    assert validate_date("2026-02-30") is False


def test_valid_csv_passes(tmp_path):
    csv_file = write_csv(tmp_path / "powerball.csv")

    assert validate_lottery_file(
        str(csv_file), "powerball"
    ) is True


def test_missing_file_fails(tmp_path):
    missing_file = tmp_path / "missing.csv"

    assert validate_lottery_file(
        str(missing_file), "powerball"
    ) is False


def test_missing_column_fails(tmp_path):
    columns = [
        column for column in COLUMNS
        if column != "powerball"
    ]

    csv_file = write_csv(
        tmp_path / "powerball.csv",
        columns=columns,
    )

    assert validate_lottery_file(
        str(csv_file), "powerball"
    ) is False


def test_duplicate_dates_fail(tmp_path):
    rows = [VALID_ROW.copy(), VALID_ROW.copy()]

    csv_file = write_csv(
        tmp_path / "powerball.csv",
        rows=rows,
    )

    assert validate_lottery_file(
        str(csv_file), "powerball"
    ) is False


def test_empty_required_value_fails(tmp_path):
    row = VALID_ROW.copy()
    row["powerball"] = ""

    csv_file = write_csv(
        tmp_path / "powerball.csv",
        rows=[row],
    )

    assert validate_lottery_file(
        str(csv_file), "powerball"
    ) is False


def test_non_numeric_number_fails(tmp_path):
    row = VALID_ROW.copy()
    row["white_1"] = "ABC"

    csv_file = write_csv(
        tmp_path / "powerball.csv",
        rows=[row],
    )

    assert validate_lottery_file(
        str(csv_file), "powerball"
    ) is False


def test_duplicate_white_numbers_fail(tmp_path):
    row = VALID_ROW.copy()
    row["white_5"] = "36"

    csv_file = write_csv(
        tmp_path / "powerball.csv",
        rows=[row],
    )

    assert validate_lottery_file(
        str(csv_file), "powerball"
    ) is False


def test_unsorted_white_numbers_fail(tmp_path):
    row = VALID_ROW.copy()
    row["white_3"] = "42"
    row["white_4"] = "36"

    csv_file = write_csv(
        tmp_path / "powerball.csv",
        rows=[row],
    )

    assert validate_lottery_file(
        str(csv_file), "powerball"
    ) is False