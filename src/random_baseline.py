import csv
import gzip
import random
import pandas as pd

from src.lottery_data import (
    PROJECT_DIR,
    prepare_mega_millions,
    prepare_powerball,
)


# Configuration
PREDICTIONS_DIR = PROJECT_DIR / "reports" / "predictions" / "random"

RANDOM_SIMULATIONS = 1000
RANDOM_SEED = 2026

WHITE_POOL_MAX = {
    "powerball": 69,
    "mega_millions": 70,
}

SPECIAL_POOL_MAX = {
    "powerball": 26,
    "mega_millions": 24,
}


def generate_ticket(rng, white_pool_max, special_pool_max):
    """Generate one valid random lottery ticket."""

    if white_pool_max < 5:
        raise ValueError(
            "White-ball pool must contain at least 5 numbers."
        )

    if special_pool_max < 1:
        raise ValueError(
            "Special-ball pool must contain at least 1 number."
        )

    white_numbers = sorted(
        rng.sample(range(1, white_pool_max + 1), 5)
    )

    special_number = rng.randint(1, special_pool_max)

    return white_numbers, special_number


def generate_random_predictions(
    game_name,
    draws,
    white_pool_max,
    special_pool_max,
    simulations=RANDOM_SIMULATIONS,
):
    """Generate random tickets for every drawing date.

    Saves predictions only. Actual results and performance
    metrics are handled separately by model_comparison.py.
    """

    if simulations < 1:
        raise ValueError("simulations must be at least 1.")

    if draws.empty:
        raise ValueError("Drawing data cannot be empty.")

    game_prefix = game_name.lower().replace(" ", "_")

    output_path = (
        PREDICTIONS_DIR / f"{game_prefix}_random.csv.gz"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "target_draw_date",
        "simulation",
        "predicted_white_1",
        "predicted_white_2",
        "predicted_white_3",
        "predicted_white_4",
        "predicted_white_5",
        "predicted_special",
    ]

    ordered_draws = draws.sort_values("draw_date")

    with gzip.open(
        output_path,
        "wt",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        # Each simulation is one complete sequence of random tickets.
        for simulation_number in range(1, simulations + 1):

            rng = random.Random(
                f"{RANDOM_SEED}:{game_prefix}:{simulation_number}"
            )

            for draw_date in ordered_draws["draw_date"]:

                target_date = draw_date.strftime("%Y-%m-%d")

                white_numbers, special_number = generate_ticket(
                    rng,
                    white_pool_max,
                    special_pool_max,
                )

                row = {
                    "target_draw_date": target_date,
                    "simulation": simulation_number,
                    "predicted_special": special_number,
                }

                for index, number in enumerate(
                    white_numbers, start=1
                ):
                    row[f"predicted_white_{index}"] = number

                writer.writerow(row)

    total_tickets = len(ordered_draws) * simulations

    print(f"\n{game_name} random predictions generated")
    print(f"Drawing dates: {len(ordered_draws)}")
    print(f"Simulations per date: {simulations}")
    print(f"Total tickets: {total_tickets:,}")
    print(f"Saved to: {output_path}")

    return output_path

def generate_next_random_prediction(
    game_name,
    draws,
    white_pool_max,
    special_pool_max,
    target_draw_date,
):
    """Generate one reproducible random ticket for a future drawing."""

    if draws.empty:
        raise ValueError("Drawing data cannot be empty.")

    if "draw_date" not in draws.columns:
        raise ValueError("Drawing data must contain draw_date.")

    dates = pd.to_datetime(
        draws["draw_date"],
        errors="raise",
    )

    latest_history_date = dates.max().strftime("%Y-%m-%d")
    target_date = pd.to_datetime(
        target_draw_date,
        errors="raise",
    ).strftime("%Y-%m-%d")

    if target_date <= latest_history_date:
        raise ValueError(
            "The target drawing date must be after the latest "
            "historical drawing."
        )

    game_prefix = game_name.lower().replace(" ", "_")

    # Repeating the same game/date produces the same control ticket.
    rng = random.Random(
        f"{RANDOM_SEED}:{game_prefix}:next:{target_date}"
    )

    white_numbers, special_number = generate_ticket(
        rng=rng,
        white_pool_max=white_pool_max,
        special_pool_max=special_pool_max,
    )

    prediction = {
        "target_draw_date": target_date,
        "history_end_date": latest_history_date,
        "predicted_special": special_number,
    }

    for index, number in enumerate(white_numbers, start=1):
        prediction[f"predicted_white_{index}"] = number

    return prediction

def main():
    """Generate random prediction files for both games."""

    print("Generating random baseline predictions...")

    generate_random_predictions(
        game_name="Powerball",
        draws=prepare_powerball(),
        white_pool_max=WHITE_POOL_MAX["powerball"],
        special_pool_max=SPECIAL_POOL_MAX["powerball"],
    )

    generate_random_predictions(
        game_name="Mega Millions",
        draws=prepare_mega_millions(),
        white_pool_max=WHITE_POOL_MAX["mega_millions"],
        special_pool_max=SPECIAL_POOL_MAX["mega_millions"],
    )

    print("\nRandom prediction generation complete.")


if __name__ == "__main__":
    main()