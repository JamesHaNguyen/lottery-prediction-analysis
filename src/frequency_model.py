import random
from pathlib import Path

import pandas as pd

from src.random_baseline import (
    WHITE_COLUMNS,
    generate_ticket,
    prepare_mega_millions,
    prepare_powerball,
)


# Project configuration
PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_DIR / "reports" / "frequency_model"

LOOKBACK_DRAWS = 50
RANDOM_SIMULATIONS = 1000

FREQUENCY_SEED = 42
RANDOM_SEED = 2026


def rank_numbers_by_frequency(
    history,
    columns,
    pool_max,
    rng,
):
    """Rank legal numbers by their frequency in historical draws.

    Numbers with equal frequencies are ordered using a seeded
    random generator, making the results reproducible.
    """

    if history.empty:
        raise ValueError("Historical data cannot be empty.")

    observed_numbers = history[columns].to_numpy().ravel()
    counts = pd.Series(observed_numbers).value_counts()

    # Include every legal number, even if it appeared zero times.
    ranked_numbers = list(range(1, pool_max + 1))

    # Randomize tie order before applying a stable sort.
    rng.shuffle(ranked_numbers)

    ranked_numbers.sort(
        key=lambda number: int(counts.get(number, 0)),
        reverse=True,
    )

    return ranked_numbers


def generate_frequency_ticket(
    history,
    special_column,
    white_pool_max,
    special_pool_max,
    rng,
):
    """Generate a ticket using the most frequent historical numbers."""

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


def score_ticket(
    predicted_white,
    predicted_special,
    actual_row,
    special_column,
):
    """Count white-ball and special-ball matches."""

    actual_white = {
        int(actual_row[column])
        for column in WHITE_COLUMNS
    }

    white_matches = len(set(predicted_white) & actual_white)

    special_match = (
        int(predicted_special) == int(actual_row[special_column])
    )

    return {
        "white_matches": white_matches,
        "has_white_match": int(white_matches > 0),
        "special_match": int(special_match),
    }


def validate_backtest_inputs(df, lookback_draws):
    """Check inputs shared by the backtest functions."""

    if lookback_draws < 1:
        raise ValueError("lookback_draws must be at least 1.")

    if len(df) <= lookback_draws:
        raise ValueError(
            "Not enough drawings for the selected lookback window."
        )

    if df["draw_date"].isna().any():
        raise ValueError("Drawing dates cannot contain missing values.")

    if df["draw_date"].duplicated().any():
        raise ValueError("Drawing dates must be unique.")


def run_backtest(
    df,
    special_column,
    white_pool_max,
    special_pool_max,
    lookback_draws=LOOKBACK_DRAWS,
):
    """Evaluate one frequency prediction per eligible drawing.

    Each prediction uses only drawings that occurred before
    its target drawing.
    """

    df = df.sort_values("draw_date").reset_index(drop=True).copy()

    validate_backtest_inputs(df, lookback_draws)

    results = []

    for target_index in range(lookback_draws, len(df)):
        history = df.iloc[
            target_index - lookback_draws:target_index
        ]

        target = df.iloc[target_index]
        target_date = target["draw_date"].date().isoformat()

        # A reproducible tie-breaking sequence for each target date.
        frequency_rng = random.Random(
            f"{FREQUENCY_SEED}:{target_date}"
        )

        frequency_white, frequency_special = generate_frequency_ticket(
            history=history,
            special_column=special_column,
            white_pool_max=white_pool_max,
            special_pool_max=special_pool_max,
            rng=frequency_rng,
        )

        frequency_score = score_ticket(
            frequency_white,
            frequency_special,
            target,
            special_column,
        )

        result = {
            "target_draw_date": target_date,
            "history_start_date": (
                history.iloc[0]["draw_date"].date().isoformat()
            ),
            "history_end_date": (
                history.iloc[-1]["draw_date"].date().isoformat()
            ),
            "lookback_draws": len(history),
        }

        for index, number in enumerate(
            frequency_white, start=1
        ):
            result[f"frequency_white_{index}"] = number

        result["frequency_special"] = frequency_special

        for index, column in enumerate(
            WHITE_COLUMNS, start=1
        ):
            result[f"actual_white_{index}"] = int(target[column])

        result["actual_special"] = int(target[special_column])

        result["frequency_white_matches"] = (
            frequency_score["white_matches"]
        )
        result["frequency_has_white_match"] = (
            frequency_score["has_white_match"]
        )
        result["frequency_special_match"] = (
            frequency_score["special_match"]
        )

        results.append(result)

    return pd.DataFrame(results)


def run_random_simulations(
    df,
    special_column,
    white_pool_max,
    special_pool_max,
    lookback_draws=LOOKBACK_DRAWS,
    simulations=RANDOM_SIMULATIONS,
):
    """Run independent random tickets over the same target dates.

    Each simulation creates one random ticket per eligible
    historical drawing. It uses no historical frequency information.
    """

    df = df.sort_values("draw_date").reset_index(drop=True).copy()

    validate_backtest_inputs(df, lookback_draws)

    if simulations < 1:
        raise ValueError("simulations must be at least 1.")

    # These are exactly the target dates evaluated by run_backtest().
    evaluation_rows = df.iloc[lookback_draws:].to_dict("records")
    number_of_draws = len(evaluation_rows)

    simulation_results = []

    for simulation_number in range(1, simulations + 1):
        # Each simulation gets its own reproducible random sequence.
        rng = random.Random(
            f"{RANDOM_SEED}:{simulation_number}"
        )

        total_white_matches = 0
        tickets_with_white_match = 0
        special_ball_matches = 0

        for target in evaluation_rows:
            predicted_white, predicted_special = generate_ticket(
                rng=rng,
                white_pool_max=white_pool_max,
                special_pool_max=special_pool_max,
            )

            score = score_ticket(
                predicted_white,
                predicted_special,
                target,
                special_column,
            )

            total_white_matches += score["white_matches"]
            tickets_with_white_match += score["has_white_match"]
            special_ball_matches += score["special_match"]

        simulation_results.append({
            "simulation": simulation_number,
            "average_white_matches": (
                total_white_matches / number_of_draws
            ),
            "tickets_with_white_match_pct": (
                tickets_with_white_match / number_of_draws * 100
            ),
            "special_ball_match_rate_pct": (
                special_ball_matches / number_of_draws * 100
            ),
        })

    return pd.DataFrame(simulation_results)


def summarize_results(results, model_name):
    """Calculate average performance for a prediction model."""

    return {
        "average_white_matches": (
            results[f"{model_name}_white_matches"].mean()
        ),
        "tickets_with_white_match_pct": (
            results[f"{model_name}_has_white_match"].mean() * 100
        ),
        "special_ball_match_rate_pct": (
            results[f"{model_name}_special_match"].mean() * 100
        ),
    }


def print_comparison(frequency_results, random_results):
    """Compare frequency performance with the random-run distribution."""

    frequency_metrics = summarize_results(
        frequency_results,
        "frequency",
    )

    metrics = [
        (
            "Average white-ball matches per ticket",
            "average_white_matches",
            "",
        ),
        (
            "Tickets with at least one white match",
            "tickets_with_white_match_pct",
            "%",
        ),
        (
            "Special-ball match rate",
            "special_ball_match_rate_pct",
            "%",
        ),
    ]

    print("\nComparison against 1,000 random simulations")
    print("-" * 55)

    for label, metric, suffix in metrics:
        frequency_value = frequency_metrics[metric]
        random_values = random_results[metric]

        random_mean = random_values.mean()
        random_5th = random_values.quantile(0.05)
        random_95th = random_values.quantile(0.95)

        # Percent of simulated runs that scored at or below
        # the frequency model on this particular metric.
        percentile = (
            (random_values <= frequency_value).mean() * 100
        )

        print(f"\n{label}:")
        print(f"  Frequency model: {frequency_value:.4f}{suffix}")
        print(f"  Random mean:     {random_mean:.4f}{suffix}")
        print(
            f"  Random 5th–95th percentile: "
            f"{random_5th:.4f}–{random_95th:.4f}{suffix}"
        )
        print(
            f"  Frequency score's percentile among random runs: "
            f"{percentile:.1f}%"
        )


def evaluate_game(
    game_name,
    df,
    special_column,
    white_pool_max,
    special_pool_max,
):
    """Backtest a game and save detailed results."""

    print(f"\n{'=' * 55}")
    print(f"{game_name} — Frequency Model Backtest")
    print("=" * 55)

    print(f"Historical drawings: {len(df)}")
    print(f"Lookback window: {LOOKBACK_DRAWS} drawings")
    print(f"Random simulations: {RANDOM_SIMULATIONS}")

    frequency_results = run_backtest(
        df=df,
        special_column=special_column,
        white_pool_max=white_pool_max,
        special_pool_max=special_pool_max,
    )

    random_results = run_random_simulations(
        df=df,
        special_column=special_column,
        white_pool_max=white_pool_max,
        special_pool_max=special_pool_max,
    )

    print(f"Evaluation drawings: {len(frequency_results)}")
    print(
        f"Evaluation period: "
        f"{frequency_results['target_draw_date'].iloc[0]} to "
        f"{frequency_results['target_draw_date'].iloc[-1]}"
    )

    print_comparison(frequency_results, random_results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    filename = game_name.lower().replace(" ", "_")

    frequency_path = REPORTS_DIR / f"{filename}_backtest.csv"
    random_path = (
        REPORTS_DIR / f"{filename}_random_simulations.csv"
    )

    frequency_results.to_csv(frequency_path, index=False)
    random_results.to_csv(random_path, index=False)

    print(f"\nFrequency predictions saved to: {frequency_path}")
    print(f"Random simulation summaries saved to: {random_path}")

    return frequency_results, random_results


def main():
    """Run the frequency-model comparison for both games."""

    print("Starting frequency-model backtest...")

    evaluate_game(
        game_name="Powerball",
        df=prepare_powerball(),
        special_column="powerball",
        white_pool_max=69,
        special_pool_max=26,
    )

    evaluate_game(
        game_name="Mega Millions",
        df=prepare_mega_millions(),
        special_column="mega_ball",
        white_pool_max=70,
        special_pool_max=24,
    )

    print("\nFrequency-model backtest complete.")


if __name__ == "__main__":
    main()