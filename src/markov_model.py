
from pathlib import Path

import pandas as pd

from src.lottery_data import (
    BACKTEST_LOOKBACK_DRAWS,
    PROJECT_DIR,
    WHITE_COLUMNS,
    prepare_mega_millions,
    prepare_powerball,
)

DEFAULT_ALPHA = 10.0


def _prepare_history(
    draws: pd.DataFrame,
    special_column: str,
    white_pool_max: int,
    special_pool_max: int,
) -> pd.DataFrame:
    """Validate and sort historical drawings."""

    if white_pool_max < 5 or special_pool_max < 1:
        raise ValueError("Invalid lottery number ranges.")

    required = ["draw_date", *WHITE_COLUMNS, special_column]
    missing = [column for column in required if column not in draws.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if draws.empty:
        raise ValueError("Drawing history cannot be empty.")

    history = draws[required].copy()
    history["draw_date"] = pd.to_datetime(
        history["draw_date"], errors="coerce"
    )

    if history["draw_date"].isna().any():
        raise ValueError("Drawing history contains invalid dates.")
    if history["draw_date"].duplicated().any():
        raise ValueError("Drawing history contains duplicate dates.")

    for column in [*WHITE_COLUMNS, special_column]:
        history[column] = pd.to_numeric(
            history[column], errors="raise"
        ).astype(int)

    white = history[WHITE_COLUMNS].to_numpy()
    special = history[special_column].to_numpy()

    if (history[WHITE_COLUMNS].nunique(axis=1) != 5).any():
        raise ValueError("Each drawing must contain 5 unique white balls.")
    if ((white < 1) | (white > white_pool_max)).any():
        raise ValueError("A white ball is outside the legal number range.")
    if ((special < 1) | (special > special_pool_max)).any():
        raise ValueError("A special ball is outside the legal number range.")

    return history.sort_values("draw_date").reset_index(drop=True)


def _new_counts(pool_max: int):
    """Create source counts and a transition-count matrix."""

    return (
        [0] * (pool_max + 1),
        [[0] * (pool_max + 1) for _ in range(pool_max + 1)],
    )


def _add_transition(
    source_counts: list,
    transitions: list,
    previous,
    following,
):
    """Record one transition between consecutive drawings."""

    for source in set(previous):
        source_counts[source] += 1
        for target in set(following):
            transitions[source][target] += 1


def _ticket_from_counts(
    white: list,
    special: int,
    white_pool_max: int,
    special_pool_max: int,
    white_counts: tuple,
    special_counts: tuple,
    alpha: float,
) -> dict:
    """Rank numbers using smoothed transition probabilities."""

    white_sources, white_transitions = white_counts
    special_sources, special_transitions = special_counts

    white_base = 5 / white_pool_max
    scores = {}

    for candidate in range(1, white_pool_max + 1):
        scores[candidate] = sum(
            (
                white_transitions[source][candidate]
                + alpha * white_base
            ) / (white_sources[source] + alpha)
            for source in white
        ) / 5

    predicted_white = sorted(
        sorted(scores, key=lambda n: (-scores[n], n))[:5]
    )

    special_base = 1 / special_pool_max
    special_scores = {
        candidate: (
            special_transitions[special][candidate]
            + alpha * special_base
        ) / (special_sources[special] + alpha)
        for candidate in range(1, special_pool_max + 1)
    }

    predicted_special = min(
        special_scores,
        key=lambda n: (-special_scores[n], n),
    )

    return {
        **{
            f"predicted_white_{i}": number
            for i, number in enumerate(predicted_white, start=1)
        },
        "predicted_special": predicted_special,
    }


def _build_counts(
    white_rows,
    special_values,
    end_index: int,
    white_pool_max: int,
    special_pool_max: int,
):
    """Count transitions within draws before end_index."""

    white_counts = _new_counts(white_pool_max)
    special_counts = _new_counts(special_pool_max)

    for i in range(end_index - 1):
        _add_transition(
            *white_counts, white_rows[i], white_rows[i + 1]
        )
        _add_transition(
            *special_counts, [special_values[i]], [special_values[i + 1]]
        )

    return white_counts, special_counts


def generate_markov_ticket(
    draws: pd.DataFrame,
    special_column: str,
    white_pool_max: int,
    special_pool_max: int,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """Generate one ticket from historical transitions."""

    if alpha <= 0:
        raise ValueError("alpha must be greater than 0.")

    history = _prepare_history(
        draws, special_column, white_pool_max, special_pool_max
    )
    white = history[WHITE_COLUMNS].to_numpy(dtype=int)
    special = history[special_column].to_numpy(dtype=int)

    if len(history) < 2:
        raise ValueError("At least two drawings are required.")

    white_counts, special_counts = _build_counts(
        white, special, len(history), white_pool_max, special_pool_max
    )

    return _ticket_from_counts(
        white[-1], int(special[-1]), white_pool_max, special_pool_max,
        white_counts, special_counts, alpha,
    )


def generate_predictions_for_backtest(
    draws: pd.DataFrame,
    special_column: str,
    white_pool_max: int,
    special_pool_max: int,
    min_history_draws: int = BACKTEST_LOOKBACK_DRAWS,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Generate walk-forward predictions without using future draws."""

    if min_history_draws < 2:
        raise ValueError("min_history_draws must be at least 2.")
    if alpha <= 0:
        raise ValueError("alpha must be greater than 0.")

    history = _prepare_history(
        draws, special_column, white_pool_max, special_pool_max
    )

    if len(history) <= min_history_draws:
        raise ValueError("Not enough drawings to generate backtest predictions.")

    white = history[WHITE_COLUMNS].to_numpy(dtype=int)
    special = history[special_column].to_numpy(dtype=int)
    dates = history["draw_date"].tolist()

    white_counts, special_counts = _build_counts(
        white, special, min_history_draws,
        white_pool_max, special_pool_max,
    )

    records = []
    total = len(history) - min_history_draws

    for target in range(min_history_draws, len(history)):
        # Use only drawings before the target date.
        ticket = _ticket_from_counts(
            white[target - 1],
            int(special[target - 1]),
            white_pool_max,
            special_pool_max,
            white_counts,
            special_counts,
            alpha,
        )

        records.append({
            "target_draw_date": dates[target].strftime("%Y-%m-%d"),
            "history_start_date": dates[0].strftime("%Y-%m-%d"),
            "history_end_date": dates[target - 1].strftime("%Y-%m-%d"),
            **ticket,
        })

        completed = target - min_history_draws + 1
        if completed % 100 == 0 or completed == total:
            print(f"Markov progress: {completed}/{total}")

        # Add this transition only after predicting its target drawing.
        # It becomes available to the following prediction.
        if target < len(history) - 1:
            _add_transition(
                *white_counts, white[target - 1], white[target]
            )
            _add_transition(
                *special_counts,
                [special[target - 1]],
                [special[target]],
            )

    return pd.DataFrame(records)


def generate_next_draw_prediction(
    draws: pd.DataFrame,
    special_column: str,
    white_pool_max: int,
    special_pool_max: int,
    target_draw_date: str,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """Generate a ticket for a future drawing date."""

    history = _prepare_history(
        draws, special_column, white_pool_max, special_pool_max
    )
    target_date = pd.to_datetime(target_draw_date, errors="coerce")

    if pd.isna(target_date):
        raise ValueError("target_draw_date is invalid.")
    if target_date.normalize() <= history.iloc[-1]["draw_date"].normalize():
        raise ValueError("Target date must be after the latest historical draw.")

    return {
        "target_draw_date": target_date.strftime("%Y-%m-%d"),
        "history_start_date": history.iloc[0]["draw_date"].strftime("%Y-%m-%d"),
        "history_end_date": history.iloc[-1]["draw_date"].strftime("%Y-%m-%d"),
        "history_draws": len(history),
        **generate_markov_ticket(
            history, special_column, white_pool_max, special_pool_max, alpha
        ),
    }


def save_backtest_predictions(
    predictions: pd.DataFrame,
    game_name: str,
) -> Path:
    """Save predictions to the Markov output directory."""

    output_dir = PROJECT_DIR / "reports" / "predictions" / "markov"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{game_name}_markov.csv"
    predictions.to_csv(path, index=False)
    return path


def run_game_backtest(
    game_name: str,
    draws: pd.DataFrame,
    special_column: str,
    white_pool_max: int,
    special_pool_max: int,
    min_history_draws: int = 100,
) -> Path:
    """Generate and save one game's backtest."""

    print(f"\nGenerating Markov backtest: {game_name}")
    print(f"Historical drawings: {len(draws)}")

    predictions = generate_predictions_for_backtest(
        draws,
        special_column,
        white_pool_max,
        special_pool_max,
        min_history_draws,
    )
    path = save_backtest_predictions(predictions, game_name)

    print(f"Predictions generated: {len(predictions)}")
    print(
        f"Evaluation period: {predictions['target_draw_date'].min()} "
        f"to {predictions['target_draw_date'].max()}"
    )
    print(f"Saved to: {path}")
    return path


def main():
    """Generate backtests for Powerball and Mega Millions."""

    run_game_backtest(
        "powerball", prepare_powerball(), "powerball", 69, 26, 100
    )
    run_game_backtest(
        "mega_millions", prepare_mega_millions(), "mega_ball", 70, 24, 100
    )
    print("\nMarkov backtests complete.")


if __name__ == "__main__":
    main()
