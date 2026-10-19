from pathlib import Path

import pandas as pd

from src.lottery_data import (
    PROJECT_DIR,
    WHITE_COLUMNS,
    prepare_mega_millions,
    prepare_powerball,
)


# Configuration
PREDICTIONS_DIR = PROJECT_DIR / "reports" / "predictions"
RANDOM_DIR = PREDICTIONS_DIR / "random"
FREQUENCY_DIR = PREDICTIONS_DIR / "frequency"
REPORTS_DIR = PROJECT_DIR / "reports" / "model_comparison"

LOOKBACK_WINDOWS = [10, 25, 50, 75, 100]

PREDICTED_WHITE_COLUMNS = [
    f"predicted_white_{i}" for i in range(1, 6)
]

ACTUAL_WHITE_COLUMNS = [
    f"actual_white_{i}" for i in range(1, 6)
]


def normalize_target_dates(df):
    """Normalize target dates to YYYY-MM-DD strings."""

    if "target_draw_date" not in df.columns:
        raise ValueError("Missing target_draw_date column.")

    result = df.copy()

    result["target_draw_date"] = pd.to_datetime(
        result["target_draw_date"],
        errors="raise",
    ).dt.strftime("%Y-%m-%d")

    return result


def load_common_evaluation_dates(game_prefix):
    """Get target dates produced by the longest lookback model.

    Every lookback model will be evaluated on this same set of dates.
    """

    longest_lookback = max(LOOKBACK_WINDOWS)

    filename = (
        f"{game_prefix}_lookback_{longest_lookback}.csv"
    )
    path = FREQUENCY_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing frequency predictions: {path}. "
            "Run python -m src.frequency_model first."
        )

    predictions = normalize_target_dates(
        pd.read_csv(path)
    )

    if predictions.empty:
        raise ValueError(
            f"No predictions found in {path}."
        )

    if predictions["target_draw_date"].duplicated().any():
        raise ValueError(
            f"Duplicate target dates found in {path}."
        )

    dates = sorted(predictions["target_draw_date"].tolist())

    return dates


def filter_frequency_to_common_dates(
    predictions,
    common_dates,
    lookback,
):
    """Keep frequency predictions for the shared evaluation dates."""

    predictions = normalize_target_dates(predictions)

    if predictions["target_draw_date"].duplicated().any():
        raise ValueError(
            f"Duplicate target dates for lookback {lookback}."
        )

    common_date_set = set(common_dates)
    available_dates = set(predictions["target_draw_date"])

    missing_dates = common_date_set - available_dates

    if missing_dates:
        raise ValueError(
            f"Lookback {lookback} is missing "
            f"{len(missing_dates)} common evaluation dates."
        )

    filtered = predictions[
        predictions["target_draw_date"].isin(common_date_set)
    ].copy()

    return (
        filtered.sort_values("target_draw_date")
        .reset_index(drop=True)
    )


def filter_random_to_common_dates(
    predictions,
    common_dates,
):
    """Keep random tickets for the same dates used by all models."""

    predictions = normalize_target_dates(predictions)

    required_columns = {
        "target_draw_date",
        "simulation",
        *PREDICTED_WHITE_COLUMNS,
        "predicted_special",
    }

    missing_columns = required_columns - set(predictions.columns)

    if missing_columns:
        raise ValueError(
            f"Random predictions are missing columns: "
            f"{sorted(missing_columns)}"
        )

    if predictions.duplicated(
        ["target_draw_date", "simulation"]
    ).any():
        raise ValueError(
            "Random predictions contain duplicate "
            "date/simulation pairs."
        )

    common_date_set = set(common_dates)
    available_dates = set(predictions["target_draw_date"])

    missing_dates = common_date_set - available_dates

    if missing_dates:
        raise ValueError(
            f"Random predictions are missing "
            f"{len(missing_dates)} common evaluation dates."
        )

    filtered = predictions[
        predictions["target_draw_date"].isin(common_date_set)
    ].copy()

    if filtered.empty:
        raise ValueError(
            "No random predictions found for common dates."
        )

    # Every target date must have the same number of simulations.
    simulation_count = filtered["simulation"].nunique()

    counts_per_date = (
        filtered.groupby("target_draw_date")["simulation"]
        .nunique()
    )

    if not counts_per_date.eq(simulation_count).all():
        raise ValueError(
            "Random predictions have inconsistent simulation "
            "counts across target dates."
        )

    return filtered.sort_values(
        ["simulation", "target_draw_date"]
    ).reset_index(drop=True)


def make_actual_results(draws, special_column):
    """Build official results keyed by target drawing date."""

    actual = pd.DataFrame({
        "target_draw_date": (
            draws["draw_date"]
            .pipe(pd.to_datetime)
            .dt.strftime("%Y-%m-%d")
        )
    })

    for index, column in enumerate(WHITE_COLUMNS, start=1):
        actual[f"actual_white_{index}"] = (
            draws[column].to_numpy()
        )

    actual["actual_special"] = draws[
        special_column
    ].to_numpy()

    return actual


def score_predictions(predictions, actual_results):
    """Match each prediction to its actual drawing and score it."""

    joined = predictions.merge(
        actual_results,
        on="target_draw_date",
        how="left",
        validate="many_to_one",
    )

    if joined["actual_special"].isna().any():
        raise ValueError(
            "Some predictions do not have matching actual results."
        )

    predicted_white = joined[
        PREDICTED_WHITE_COLUMNS
    ].to_numpy(dtype=int)

    actual_white = joined[
        ACTUAL_WHITE_COLUMNS
    ].to_numpy(dtype=int)

    # Count how many of the five predicted white balls appear
    # among the five official white balls.
    matches = (
        predicted_white[:, :, None]
        == actual_white[:, None, :]
    ).any(axis=2).sum(axis=1)

    joined["white_matches"] = matches
    joined["has_white_match"] = (
        matches > 0
    ).astype(int)

    joined["special_match"] = (
        joined["predicted_special"].astype(int)
        == joined["actual_special"].astype(int)
    ).astype(int)

    return joined


def summarize_prediction_scores(scored_predictions):
    """Summarize one prediction sequence."""

    return {
        "average_white_matches": (
            scored_predictions["white_matches"].mean()
        ),
        "tickets_with_white_match_pct": (
            scored_predictions["has_white_match"].mean() * 100
        ),
        "special_ball_match_rate_pct": (
            scored_predictions["special_match"].mean() * 100
        ),
    }


def summarize_random_simulations(scored_predictions):
    """Calculate metrics independently for each random simulation."""

    summary = (
        scored_predictions
        .groupby("simulation")
        .agg(
            average_white_matches=("white_matches", "mean"),
            tickets_with_white_match_pct=(
                "has_white_match", "mean"
            ),
            special_ball_match_rate_pct=(
                "special_match", "mean"
            ),
        )
        .reset_index()
    )

    summary["tickets_with_white_match_pct"] *= 100
    summary["special_ball_match_rate_pct"] *= 100

    return summary


def compare_scores(
    game_name,
    lookback,
    frequency_scores,
    random_metrics,
):
    """Compare a frequency model with matched random simulations."""

    frequency = summarize_prediction_scores(
        frequency_scores
    )

    metrics = [
        (
            "average_white_matches",
            "Average white-ball matches per ticket",
        ),
        (
            "tickets_with_white_match_pct",
            "Tickets with at least one white match",
        ),
        (
            "special_ball_match_rate_pct",
            "Special-ball match rate",
        ),
    ]

    summary_rows = []

    print(f"\n{game_name} | Lookback window: {lookback}")
    print("-" * 55)

    for metric, label in metrics:
        frequency_value = frequency[metric]
        random_values = random_metrics[metric]

        random_mean = random_values.mean()
        random_5th = random_values.quantile(0.05)
        random_95th = random_values.quantile(0.95)

        random_at_least_as_good_pct = (
            (random_values >= frequency_value).mean() * 100
        )

        difference = frequency_value - random_mean

        print(f"\n{label}")
        print(f"  Frequency model: {frequency_value:.4f}")
        print(f"  Random mean:     {random_mean:.4f}")
        print(f"  Difference:      {difference:+.4f}")
        print(
            f"  Random 5th–95th percentile: "
            f"{random_5th:.4f}–{random_95th:.4f}"
        )
        print(
            f"  Random runs scoring at least as well: "
            f"{random_at_least_as_good_pct:.1f}%"
        )

        summary_rows.append({
            "game": game_name,
            "lookback_draws": lookback,
            "evaluation_draws": len(frequency_scores),
            "evaluation_start_date": (
                frequency_scores["target_draw_date"].min()
            ),
            "evaluation_end_date": (
                frequency_scores["target_draw_date"].max()
            ),
            "metric": metric,
            "frequency_score": frequency_value,
            "random_mean": random_mean,
            "difference_vs_random_mean": difference,
            "random_5th_percentile": random_5th,
            "random_95th_percentile": random_95th,
            "random_runs_at_least_as_good_pct": (
                random_at_least_as_good_pct
            ),
        })

    return summary_rows


def compare_game(game_name, draws, special_column):
    """Compare all lookback windows on identical evaluation dates."""

    game_prefix = game_name.lower().replace(" ", "_")

    random_path = (
        RANDOM_DIR / f"{game_prefix}_random.csv.gz"
    )

    if not random_path.exists():
        raise FileNotFoundError(
            f"Missing random predictions: {random_path}. "
            "Run python -m src.random_baseline first."
        )

    # The longest window determines the shared evaluation dates.
    common_dates = load_common_evaluation_dates(game_prefix)

    actual_results = make_actual_results(
        draws,
        special_column,
    )

    random_predictions = pd.read_csv(random_path)

    random_predictions = filter_random_to_common_dates(
        random_predictions,
        common_dates,
    )

    random_scores = score_predictions(
        random_predictions,
        actual_results,
    )

    random_metrics = summarize_random_simulations(
        random_scores
    )

    all_summary_rows = []

    print(f"\n{'=' * 55}")
    print(f"{game_name} — Model Comparison")
    print("=" * 55)
    print(f"Common evaluation dates: {len(common_dates)}")
    print(
        f"Evaluation period: "
        f"{common_dates[0]} to {common_dates[-1]}"
    )
    print(
        f"Random simulations: "
        f"{random_metrics['simulation'].nunique()}"
    )

    # Each frequency window uses the same common target dates.
    for lookback in LOOKBACK_WINDOWS:
        frequency_path = (
            FREQUENCY_DIR
            / f"{game_prefix}_lookback_{lookback}.csv"
        )

        if not frequency_path.exists():
            raise FileNotFoundError(
                f"Missing frequency predictions: {frequency_path}. "
                "Run python -m src.frequency_model first."
            )

        frequency_predictions = pd.read_csv(frequency_path)

        frequency_predictions = filter_frequency_to_common_dates(
            frequency_predictions,
            common_dates,
            lookback,
        )

        frequency_scores = score_predictions(
            frequency_predictions,
            actual_results,
        )

        rows = compare_scores(
            game_name,
            lookback,
            frequency_scores,
            random_metrics,
        )

        all_summary_rows.extend(rows)

    # Save one random metrics row per simulation per game.
    random_metrics.insert(0, "game", game_name)

    return all_summary_rows, random_metrics


def main():
    """Create comparison tables for both lottery games."""

    print("Starting model comparison...")

    powerball_summary, powerball_random = compare_game(
        "Powerball",
        prepare_powerball(),
        "powerball",
    )

    mega_summary, mega_random = compare_game(
        "Mega Millions",
        prepare_mega_millions(),
        "mega_ball",
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_summary = pd.DataFrame(
        powerball_summary + mega_summary
    )

    random_simulation_metrics = pd.concat(
        [powerball_random, mega_random],
        ignore_index=True,
    )

    comparison_path = (
        REPORTS_DIR / "comparison_summary.csv"
    )
    random_path = (
        REPORTS_DIR / "random_simulation_metrics.csv"
    )

    comparison_summary.to_csv(
        comparison_path,
        index=False,
    )

    random_simulation_metrics.to_csv(
        random_path,
        index=False,
    )

    print(f"\nComparison summary saved to: {comparison_path}")
    print(f"Random simulation statistics saved to: {random_path}")
    print("\nModel comparison complete.")


if __name__ == "__main__":
    main()