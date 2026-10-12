from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Find the project directory regardless of where the script is run.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "reports" / "figures"

WHITE_COLUMNS = [
    "white_1",
    "white_2",
    "white_3",
    "white_4",
    "white_5",
]


def load_data(filename, special_ball_column):
    """Load a lottery CSV and check its required columns."""

    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find lottery data: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = [
        "draw_date",
        *WHITE_COLUMNS,
        special_ball_column,
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

    # Convert the date column to actual date values.
    df["draw_date"] = pd.to_datetime(
        df["draw_date"],
        errors="coerce",
    )

    return df


def print_summary(game_name, df, special_ball_column):
    """Print basic statistics about one lottery dataset."""

    print(f"\n{'=' * 50}")
    print(f"{game_name} — Dataset Summary")
    print("=" * 50)

    print(f"Total drawings: {len(df)}")

    valid_dates = df["draw_date"].dropna()

    if not valid_dates.empty:
        print(f"First drawing: {valid_dates.min().date()}")
        print(f"Latest drawing: {valid_dates.max().date()}")

    missing_values = df[
        ["draw_date", *WHITE_COLUMNS, special_ball_column]
    ].isna().sum().sum()

    duplicate_dates = df["draw_date"].duplicated().sum()

    print(f"Missing required values: {missing_values}")
    print(f"Duplicate drawing dates: {duplicate_dates}")


def get_number_frequencies(df, columns):
    """Count how often each number appears in the specified columns."""

    numbers = (
        df[columns]
        .apply(pd.to_numeric, errors="coerce")
        .stack()
        .dropna()
        .astype(int)
    )

    return numbers.value_counts().sort_index()


def print_top_numbers(frequencies, label):
    """Print the ten most frequently observed numbers."""

    print(f"\nTop 10 {label} by historical frequency:")

    top_numbers = frequencies.sort_values(
        ascending=False
    ).head(10)

    for number, count in top_numbers.items():
        print(f"  Number {number:>2}: {count} appearances")


def save_frequency_chart(
    frequencies,
    title,
    filename,
    x_label,
    number_max,
):
    """Save a bar chart showing the frequency of each number."""

    if frequencies.empty:
        print(f"Skipping chart because no data exists: {title}")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Include every legal number, even if it appeared zero times.
    complete_frequencies = frequencies.reindex(
        range(1, number_max + 1),
        fill_value=0,
    )

    plt.figure(figsize=(12, 5))

    plt.bar(
        complete_frequencies.index,
        complete_frequencies.values,
    )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Number of appearances")
    plt.xticks(range(1, number_max + 1, 2))
    plt.grid(axis="y", alpha=0.25)

    plt.tight_layout()

    output_path = FIGURES_DIR / filename

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(
        f"Chart saved to: "
        f"{output_path.relative_to(PROJECT_DIR)}"
    )


def analyze_game(
    game_name,
    filename,
    special_ball_column,
    white_start_date,
    special_start_date,
    white_max,
    special_max,
):
    """Analyze number frequencies within consistent rule periods."""

    df = load_data(filename, special_ball_column)

    print_summary(game_name, df, special_ball_column)

    chart_prefix = filename.removesuffix(".csv")

    # Analyze white balls using a consistent number pool.
    white_df = df[
        df["draw_date"] >= pd.Timestamp(white_start_date)
    ]

    print(f"\n{game_name} white-ball analysis")
    print(f"Starting date: {white_start_date}")
    print(f"Drawings analyzed: {len(white_df)}")

    white_frequencies = get_number_frequencies(
        white_df,
        WHITE_COLUMNS,
    )

    print_top_numbers(
        white_frequencies,
        "white-ball numbers",
    )

    save_frequency_chart(
        white_frequencies,
        f"{game_name}: White-Ball Frequency Under Current Rules",
        f"{chart_prefix}_white_current_rules.png",
        "White-ball number",
        white_max,
    )

    # Analyze the special ball separately because its rules
    # may have changed on a different date.
    special_df = df[
        df["draw_date"] >= pd.Timestamp(special_start_date)
    ]

    print(f"\n{game_name} special-ball analysis")
    print(f"Starting date: {special_start_date}")
    print(f"Drawings analyzed: {len(special_df)}")

    special_frequencies = get_number_frequencies(
        special_df,
        [special_ball_column],
    )

    print_top_numbers(
        special_frequencies,
        "special-ball numbers",
    )

    save_frequency_chart(
        special_frequencies,
        f"{game_name}: Special-Ball Frequency Under Current Rules",
        f"{chart_prefix}_special_current_rules.png",
        "Special-ball number",
        special_max,
    )


def main():
    """Analyze both lottery games."""

    print("Starting lottery data analysis...")

    analyze_game(
        game_name="Powerball",
        filename="powerball.csv",
        special_ball_column="powerball",
        white_start_date="2015-10-07",
        special_start_date="2015-10-07",
        white_max=69,
        special_max=26,
    )

    analyze_game(
        game_name="Mega Millions",
        filename="mega_millions.csv",
        special_ball_column="mega_ball",
        white_start_date="2017-10-31",
        special_start_date="2025-04-08",
        white_max=70,
        special_max=24,
    )

    print("\nData analysis complete.")


if __name__ == "__main__":
    main()