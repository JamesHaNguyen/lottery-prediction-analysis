from pathlib import Path

import pandas as pd


# Project configuration
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

WHITE_COLUMNS = [
    "white_1",
    "white_2",
    "white_3",
    "white_4",
    "white_5",
]

# Shared evaluation settings
BACKTEST_LOOKBACK_DRAWS = 50


def load_draws(filename, special_column, start_date):
    """Load, validate, and filter historical lottery drawings."""

    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Lottery data file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = [
        "draw_date",
        *WHITE_COLUMNS,
        special_column,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{filename} is missing columns: {missing_columns}"
        )

    # Convert dates to datetime values.
    df["draw_date"] = pd.to_datetime(
        df["draw_date"],
        errors="coerce",
    )

    if df["draw_date"].isna().any():
        raise ValueError(
            f"{filename} contains invalid drawing dates."
        )

    if df["draw_date"].duplicated().any():
        raise ValueError(
            f"{filename} contains duplicate drawing dates."
        )

    # Convert lottery numbers into integers.
    for column in WHITE_COLUMNS + [special_column]:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        ).astype(int)

    # Keep only drawings from the requested rule period.
    df = df[
        df["draw_date"] >= pd.Timestamp(start_date)
    ].copy()

    if df.empty:
        raise ValueError(
            f"No drawings found in {filename} "
            f"on or after {start_date}."
        )

    # Ensure chronological order for backtesting.
    return (
        df.sort_values("draw_date")
        .reset_index(drop=True)
    )


def prepare_powerball():
    """Load Powerball drawings under the current number rules."""

    return load_draws(
        filename="powerball.csv",
        special_column="powerball",
        start_date="2015-10-07",
    )


def prepare_mega_millions():
    """Load Mega Millions drawings under the current number rules."""

    return load_draws(
        filename="mega_millions.csv",
        special_column="mega_ball",
        start_date="2025-04-08",
    )