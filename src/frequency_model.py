import random

import pandas as pd

from src.lottery_data import (
    PROJECT_DIR,
    WHITE_COLUMNS,
    prepare_mega_millions,
    prepare_powerball,
)


# Configuration
PREDICTIONS_DIR = PROJECT_DIR / "reports" / "predictions" / "frequency"

LOOKBACK_WINDOWS = [10, 25, 50, 75, 100]
FREQUENCY_SEED = 42


def rank_numbers_by_frequency(
    history,
    columns,
    pool_max,
    rng,
):
    """Rank legal numbers by their historical frequency."""

    if history.empty:
        raise ValueError("Historical data cannot be empty.")

    observed_numbers = history[columns].to_numpy().ravel()
    counts = pd.Series(observed_numbers).value_counts()

    # Include numbers that did not appear in the history window.
    ranking = list(range(1, pool_max + 1))

    # Shuffle first so tied numbers are not always favored by value.
    rng.shuffle(ranking)

    # Stable sorting keeps the randomized order for tied frequencies.
    ranking.sort(
        key=lambda number: int(counts.get(number, 0)),
        reverse=True,
    )

    return ranking


def generate_frequency_ticket(
    history,
    special_column,
    white_pool_max,
    special_pool_max,
    rng,
):
    """Select the five most frequent white balls and top special ball."""

    white_ranking = rank_numbers_by_frequency(
        history,
        WHITE_COLUMNS,
        white_pool_max,
        rng,
    )

    special_ranking = rank_numbers_by_frequency(
        history,
        [special_column],
        special_pool_max,
        rng,
    )

    predicted_white = sorted(white_ranking[:5])
    predicted_special = special_ranking[0]

    return predicted_white, predicted_special


def generate_predictions_for_window(
    draws,
    special_column,
    white_pool_max,
    special_pool_max,
    lookback_draws,
):
    """Generate predictions using only prior drawings."""

    if lookback_draws < 1:
        raise ValueError("lookback_draws must be at least 1.")

    df = (
        draws.sort_values("draw_date")
        .reset_index(drop=True)
        .copy()
    )

    if df["draw_date"].isna().any():
        raise ValueError("Drawing dates cannot contain missing values.")

    if df["draw_date"].duplicated().any():
        raise ValueError("Drawing dates must be unique.")

    if len(df) <= lookback_draws:
        raise ValueError(
            "Not enough drawings for the selected lookback window."
        )

    predictions = []

    for target_index in range(lookback_draws, len(df)):

        # The target drawing is deliberately excluded.
        history = df.iloc[
            target_index - lookback_draws:target_index
        ]

        target = df.iloc[target_index]
        target_date = target["draw_date"].strftime("%Y-%m-%d")

        # Reproducible tie-breaking for each target date/window.
        rng = random.Random(
            f"{FREQUENCY_SEED}:{lookback_draws}:{target_date}"
        )

        white_numbers, special_number = generate_frequency_ticket(
            history=history,
            special_column=special_column,
            white_pool_max=white_pool_max,
            special_pool_max=special_pool_max,
            rng=rng,
        )

        row = {
            "target_draw_date": target_date,
            "lookback_draws": lookback_draws,
            "history_start_date": (
                history.iloc[0]["draw_date"].strftime("%Y-%m-%d")
            ),
            "history_end_date": (
                history.iloc[-1]["draw_date"].strftime("%Y-%m-%d")
            ),
            "predicted_special": special_number,
        }

        for index, number in enumerate(
            white_numbers, start=1
        ):
            row[f"predicted_white_{index}"] = number

        predictions.append(row)

    return pd.DataFrame(predictions)


def generate_game_predictions(
    game_name,
    draws,
    special_column,
    white_pool_max,
    special_pool_max,
):
    """Generate and save predictions for every lookback window."""

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    game_prefix = game_name.lower().replace(" ", "_")

    for lookback in LOOKBACK_WINDOWS:

        predictions = generate_predictions_for_window(
            draws=draws,
            special_column=special_column,
            white_pool_max=white_pool_max,
            special_pool_max=special_pool_max,
            lookback_draws=lookback,
        )

        output_path = (
            PREDICTIONS_DIR
            / f"{game_prefix}_lookback_{lookback}.csv"
        )

        predictions.to_csv(output_path, index=False)

        print(
            f"{game_name} | lookback={lookback:>3} | "
            f"predictions={len(predictions):>4} | "
            f"saved to {output_path}"
        )


def main():
    """Generate frequency predictions for both lottery games."""

    print("Generating frequency-model predictions...")

    generate_game_predictions(
        game_name="Powerball",
        draws=prepare_powerball(),
        special_column="powerball",
        white_pool_max=69,
        special_pool_max=26,
    )

    generate_game_predictions(
        game_name="Mega Millions",
        draws=prepare_mega_millions(),
        special_column="mega_ball",
        white_pool_max=70,
        special_pool_max=24,
    )

    print("\nFrequency prediction generation complete.")


if __name__ == "__main__":
    main()