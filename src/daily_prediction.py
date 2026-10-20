import argparse
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from src import data_collector
from src.frequency_model import (
    LOOKBACK_WINDOWS,
    generate_next_draw_prediction,
)
from src.lottery_data import (
    PROJECT_DIR,
    prepare_mega_millions,
    prepare_powerball,
)
from src.random_baseline import generate_next_random_prediction


# The lottery drawing schedules are based on Central Time.
CENTRAL_TIME = ZoneInfo("America/Chicago")

PREDICTION_ARCHIVE = (
    PROJECT_DIR / "reports" / "predictions" / "daily_predictions.csv"
)

# Weekday numbers: Monday=0, Tuesday=1, ..., Sunday=6.
GAME_CONFIGS = {
    "powerball": {
        "game_name": "Powerball",
        "weekdays": {0, 2, 5},
        "draw_time": time(21, 59),
        "special_column": "powerball",
        "white_pool_max": 69,
        "special_pool_max": 26,
        "prepare_draws": prepare_powerball,
    },
    "mega_millions": {
        "game_name": "Mega Millions",
        "weekdays": {1, 4},
        "draw_time": time(22, 0),
        "special_column": "mega_ball",
        "white_pool_max": 70,
        "special_pool_max": 24,
        "prepare_draws": prepare_mega_millions,
    },
}

OUTPUT_COLUMNS = [
    "prediction_generated_at",
    "game",
    "target_draw_date",
    "model",
    "lookback_draws",
    "history_start_date",
    "history_end_date",
    "white_1",
    "white_2",
    "white_3",
    "white_4",
    "white_5",
    "special_ball",
]


def normalize_current_time(now=None):
    """Return a timezone-aware time in Central Time."""

    if now is None:
        return datetime.now(CENTRAL_TIME)

    if now.tzinfo is None:
        # Treat naive test/input datetimes as Central Time.
        return now.replace(tzinfo=CENTRAL_TIME)

    return now.astimezone(CENTRAL_TIME)


def next_drawing_date(game_key, now=None):
    """Find the next drawing date that has not yet passed."""

    if game_key not in GAME_CONFIGS:
        raise ValueError(f"Unknown lottery game: {game_key}")

    current_time = normalize_current_time(now)
    schedule = GAME_CONFIGS[game_key]

    for day_offset in range(8):
        candidate_date = (
            current_time.date() + timedelta(days=day_offset)
        )

        if candidate_date.weekday() not in schedule["weekdays"]:
            continue

        drawing_datetime = datetime.combine(
            candidate_date,
            schedule["draw_time"],
            tzinfo=CENTRAL_TIME,
        )

        if drawing_datetime > current_time:
            return candidate_date.strftime("%Y-%m-%d")

    raise RuntimeError(
        f"Could not find the next drawing date for {game_key}."
    )


def make_prediction_record(
    generated_at,
    game_name,
    model_name,
    prediction,
    lookback_draws=None,
):
    """Convert a model's output to the shared report format."""

    record = {
        "prediction_generated_at": generated_at,
        "game": game_name,
        "target_draw_date": prediction["target_draw_date"],
        "model": model_name,
        "lookback_draws": lookback_draws,
        "history_start_date": prediction.get(
            "history_start_date",
            "",
        ),
        "history_end_date": prediction["history_end_date"],
        "special_ball": prediction["predicted_special"],
    }

    for index in range(1, 6):
        record[f"white_{index}"] = prediction[
            f"predicted_white_{index}"
        ]

    return record


def build_daily_predictions(draws_by_game, now=None):
    """Collect next-draw tickets from the current prediction models."""

    current_time = normalize_current_time(now)
    generated_at = current_time.isoformat(timespec="seconds")

    records = []

    for game_key, config in GAME_CONFIGS.items():
        draws = draws_by_game[game_key]
        target_date = next_drawing_date(
            game_key,
            now=current_time,
        )

        print(
            f"\nPreparing {config['game_name']} "
            f"prediction for {target_date}..."
        )

        # One prediction for each frequency-model lookback window.
        for lookback in LOOKBACK_WINDOWS:
            prediction = generate_next_draw_prediction(
                draws=draws,
                special_column=config["special_column"],
                white_pool_max=config["white_pool_max"],
                special_pool_max=config["special_pool_max"],
                lookback_draws=lookback,
                target_draw_date=target_date,
            )

            records.append(
                make_prediction_record(
                    generated_at=generated_at,
                    game_name=config["game_name"],
                    model_name=f"Frequency (lookback={lookback})",
                    prediction=prediction,
                    lookback_draws=lookback,
                )
            )

        # One random ticket provides a daily comparison control.
        random_prediction = generate_next_random_prediction(
            game_name=config["game_name"],
            draws=draws,
            white_pool_max=config["white_pool_max"],
            special_pool_max=config["special_pool_max"],
            target_draw_date=target_date,
        )

        records.append(
            make_prediction_record(
                generated_at=generated_at,
                game_name=config["game_name"],
                model_name="Random baseline (control)",
                prediction=random_prediction,
            )
        )

    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def display_predictions(predictions):
    """Print all current tickets in a readable table."""

    for (game_name, target_date), group in predictions.groupby(
        ["game", "target_draw_date"],
        sort=False,
    ):
        latest_history_date = group["history_end_date"].max()

        print("\n" + "=" * 75)
        print(f"{game_name} — Target drawing: {target_date}")
        print(f"Latest historical drawing used: {latest_history_date}")
        print("=" * 75)
        print(f"{'Model':<32} {'White balls':<26} Special")
        print("-" * 75)

        for row in group.itertuples(index=False):
            white_numbers = " ".join(
                f"{int(getattr(row, f'white_{i}')):02d}"
                for i in range(1, 6)
            )
            special_number = int(row.special_ball)

            print(
                f"{row.model:<32} "
                f"{white_numbers:<26} "
                f"{special_number:02d}"
            )


def save_predictions(predictions):
    """Append predictions to the archive without overwriting prior runs."""

    PREDICTION_ARCHIVE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if PREDICTION_ARCHIVE.exists():
        previous_predictions = pd.read_csv(PREDICTION_ARCHIVE)

        archived_predictions = pd.concat(
            [previous_predictions, predictions],
            ignore_index=True,
        )
    else:
        archived_predictions = predictions.copy()

    archived_predictions.to_csv(
        PREDICTION_ARCHIVE,
        index=False,
    )

    print(f"\nDaily prediction archive saved to: {PREDICTION_ARCHIVE}")


def run_daily_prediction(refresh_data=True, now=None):
    """Update data, generate tickets, display them, and save the archive."""

    if refresh_data:
        print("Updating historical lottery data...")
        data_collector.main()

    draws_by_game = {
        "powerball": GAME_CONFIGS["powerball"]["prepare_draws"](),
        "mega_millions": GAME_CONFIGS["mega_millions"]["prepare_draws"](),
    }

    predictions = build_daily_predictions(
        draws_by_game=draws_by_game,
        now=now,
    )

    display_predictions(predictions)
    save_predictions(predictions)

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Generate next-draw lottery predictions from current models."
    )
    parser.add_argument(
        "--skip-data-update",
        action="store_true",
        help="Use the existing CSV files without downloading new results.",
    )

    args = parser.parse_args()

    run_daily_prediction(
        refresh_data=not args.skip_data_update,
    )


if __name__ == "__main__":
    main()