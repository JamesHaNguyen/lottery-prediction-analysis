import math
import random
from pathlib import Path

import pandas as pd


# Project locations
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
REPORTS_DIR = PROJECT_DIR / "reports" / "random_baseline"
POWERBALL_CURRENT_RULES_START = "2015-10-07"
MEGA_MILLIONS_CURRENT_RULES_START = "2025-04-08"

WHITE_COLUMNS = [
    "white_1",
    "white_2",
    "white_3",
    "white_4",
    "white_5",
]

# Experiment settings
SIMULATIONS = 500
RANDOM_SEED = 42


def load_draws(filename, special_column, start_date):
    """Load historical draws from CSV and filter by start date."""

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

    df["draw_date"] = pd.to_datetime(
        df["draw_date"],
        errors="coerce",
    )

    if df["draw_date"].isna().any():
        raise ValueError(
            f"{filename} contains invalid drawing dates."
        )

    df = df[
        df["draw_date"] >= pd.Timestamp(start_date)
    ].copy()

    if df.empty:
        raise ValueError(
            f"No drawings found in {filename} "
            f"on or after {start_date}."
        )

    for column in WHITE_COLUMNS + [special_column]:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        ).astype(int)

    return df


def prepare_powerball():
    """Prepare Powerball drawings under the current number rules."""

    df = load_draws(
        "powerball.csv",
        "powerball",
        POWERBALL_CURRENT_RULES_START,
    )

    # Current Powerball rules: five white balls from 1–69
    # and one special Powerball from 1–26.
    df["special_pool_max"] = 26

    return df


def prepare_mega_millions():
    """Prepare Mega Millions drawings under the current number rules."""

    df = load_draws(
        "mega_millions.csv",
        "mega_ball",
        MEGA_MILLIONS_CURRENT_RULES_START,
    )

    # Current Mega Millions rules: five white balls from 1–70
    # and one special Mega Ball from 1–24.
    df["special_pool_max"] = 24

    return df

def generate_ticket(rng, white_pool_max, special_pool_max):
    """Generate a valid random lottery ticket."""

    if white_pool_max < 5:
        raise ValueError("The white-ball pool must contain at least 5 numbers.")

    if special_pool_max < 1:
        raise ValueError("The special-ball pool must contain at least 1 number.")

    # Select 5 distinct white-ball numbers and sort them.
    white_numbers = sorted(
        rng.sample(range(1, white_pool_max + 1), 5)
    )

    # Select one special-ball number.
    special_number = rng.randint(1, special_pool_max)

    return white_numbers, special_number


def theoretical_any_white_match_probability(pool_size):
    """Calculate the probability of matching at least one white ball."""

    # Probability of zero matches:
    # Choose all five ticket numbers from the numbers not drawn.
    total_combinations = math.comb(pool_size, 5)
    zero_match_combinations = math.comb(pool_size - 5, 5)

    probability_of_zero_matches = (
        zero_match_combinations / total_combinations
    )

    return 1 - probability_of_zero_matches


def run_simulations(
    game_name,
    df,
    special_column,
    white_pool_max,
):
    """Simulate random tickets against historical drawings."""

    simulation_results = []

    # Keep only the columns needed for this experiment.
    simulation_data = df[
        [
            "draw_date",
            *WHITE_COLUMNS,
            special_column,
            "special_pool_max",
        ]
    ]

    number_of_draws = len(simulation_data)

    print(f"\n{'=' * 55}")
    print(f"{game_name} — Random Baseline Experiment")
    print("=" * 55)
    print(f"Historical drawings: {number_of_draws}")
    print(
    f"Evaluation period: "
    f"{simulation_data['draw_date'].min().date()} to "
    f"{simulation_data['draw_date'].max().date()}"
    )
    print(f"Simulations: {SIMULATIONS}")
    print(
        f"Total simulated tickets: "
        f"{SIMULATIONS * number_of_draws:,}"
    )

    # Repeat the experiment with reproducible random seeds.
    for simulation_number in range(SIMULATIONS):

        rng = random.Random(
            RANDOM_SEED + simulation_number
        )

        total_white_matches = 0
        tickets_with_white_match = 0
        special_ball_matches = 0

        for row in simulation_data.itertuples(
            index=False,
            name=None,
        ):
            # Row layout:
            # date, five white balls, special ball, special pool max
            actual_white = {
                int(number) for number in row[1:6]
            }

            actual_special = int(row[6])
            special_pool_max = int(row[7])

            # Generate five distinct white balls.
            predicted_white_numbers, predicted_special = generate_ticket(
            rng,
            white_pool_max,
            special_pool_max,
            )

            predicted_white = set(predicted_white_numbers)

            # Compare the prediction with the actual drawing.
            white_matches = len(
                predicted_white & actual_white
            )

            total_white_matches += white_matches

            if white_matches > 0:
                tickets_with_white_match += 1

            if predicted_special == actual_special:
                special_ball_matches += 1

        # Store the summary for this simulation run.
        simulation_results.append({
            "simulation": simulation_number + 1,
            "average_white_matches": (
                total_white_matches / number_of_draws
            ),
            "tickets_with_white_match_pct": (
                tickets_with_white_match
                / number_of_draws
                * 100
            ),
            "special_ball_match_pct": (
                special_ball_matches
                / number_of_draws
                * 100
            ),
        })

    results = pd.DataFrame(simulation_results)

    # Calculate theoretical expectations for comparison.
    expected_white_matches = 25 / white_pool_max

    expected_white_match_probability = (
        theoretical_any_white_match_probability(
            white_pool_max
        ) * 100
    )

    expected_special_match_pct = (
        1 / df["special_pool_max"]
    ).mean() * 100

    print("\nResults across the 500 simulations:")

    metrics = [
        (
            "Average white-ball matches per ticket",
            "average_white_matches",
            expected_white_matches,
            "",
        ),
        (
            "Tickets with at least one white match",
            "tickets_with_white_match_pct",
            expected_white_match_probability,
            "%",
        ),
        (
            "Special-ball match rate",
            "special_ball_match_pct",
            expected_special_match_pct,
            "%",
        ),
    ]

    for label, column, theoretical_value, suffix in metrics:
        average = results[column].mean()
        lower = results[column].quantile(0.05)
        upper = results[column].quantile(0.95)

        print(f"\n{label}:")
        print(f"  Simulated mean: {average:.4f}{suffix}")
        print(
            f"  5th–95th percentile across runs: "
            f"{lower:.4f}–{upper:.4f}{suffix}"
        )
        print(
            f"  Theoretical expectation: "
            f"{theoretical_value:.4f}{suffix}"
        )

    # Save the experiment results for future comparison.
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_filename = (
        f"{game_name.lower().replace(' ', '_')}_simulations.csv"
    )

    output_path = REPORTS_DIR / output_filename

    results.to_csv(
        output_path,
        index=False,
    )

    print(f"\nSimulation results saved to: {output_path}")
    print("Experiment complete.")

    return results


def main():
    """Run the random baseline for both lottery games."""

    print("Starting random baseline experiment...")

    powerball = prepare_powerball()

    run_simulations(
        game_name="Powerball",
        df=powerball,
        special_column="powerball",
        white_pool_max=69,
    )

    mega_millions = prepare_mega_millions()

    run_simulations(
        game_name="Mega Millions",
        df=mega_millions,
        special_column="mega_ball",
        white_pool_max=70,
    )

    print("\nAll random baseline experiments completed.")


if __name__ == "__main__":
    main()