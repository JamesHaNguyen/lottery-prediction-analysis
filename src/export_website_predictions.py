
import json

import pandas as pd

from src.lottery_data import PROJECT_DIR


ARCHIVE_PATH = (
    PROJECT_DIR
    / "reports"
    / "predictions"
    / "daily_predictions.csv"
)

WEBSITE_DATA_DIR = PROJECT_DIR / "website" / "data"

OUTPUT_PATH = WEBSITE_DATA_DIR / "predictions.json"


def export_latest_predictions():
    """Export the most recent prediction run for the website."""

    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(
            f"Daily prediction archive not found: {ARCHIVE_PATH}"
        )

    df = pd.read_csv(ARCHIVE_PATH)

    required_columns = {
        "prediction_generated_at",
        "game",
        "target_draw_date",
        "model",
        "white_1",
        "white_2",
        "white_3",
        "white_4",
        "white_5",
        "special_ball",
        "history_end_date",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Daily prediction archive is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    if df.empty:
        raise ValueError("Daily prediction archive is empty.")

    # Convert timestamps to UTC so the latest run is identified
    # correctly even if archived records use different UTC offsets.
    df["_generated_at_utc"] = pd.to_datetime(
        df["prediction_generated_at"],
        errors="coerce",
        utc=True,
    )

    if df["_generated_at_utc"].isna().any():
        raise ValueError(
            "The archive contains invalid prediction timestamps."
        )

    latest_timestamp = df["_generated_at_utc"].max()

    latest = df[
        df["_generated_at_utc"] == latest_timestamp
    ].copy()

    # Protect against repeated runs recorded in the same second.
    latest = latest.drop_duplicates(
        subset=["game", "target_draw_date", "model"],
        keep="last",
    )

    games = {}

    for (game_name, target_date), group in latest.groupby(
        ["game", "target_draw_date"],
        sort=False,
    ):
        predictions = []

        for _, row in group.iterrows():
            predictions.append(
                {
                    "model": str(row["model"]),
                    "lookback_draws": (
                        None
                        if pd.isna(row["lookback_draws"])
                        else int(row["lookback_draws"])
                    ) if "lookback_draws" in row.index else None,
                    "white_balls": [
                        int(row[f"white_{index}"])
                        for index in range(1, 6)
                    ],
                    "special_ball": int(row["special_ball"]),
                }
            )

        games[game_name] = {
            "target_draw_date": str(target_date),
            "latest_history_date": str(
                group["history_end_date"].max()
            ),
            "predictions": predictions,
        }

    expected_games = {"Powerball", "Mega Millions"}
    missing_games = expected_games - set(games)

    if missing_games:
        raise ValueError(
            "Latest prediction run is missing games: "
            + ", ".join(sorted(missing_games))
        )

    for game_name in expected_games:
        if len(games[game_name]["predictions"]) != 7:
            raise ValueError(
                f"{game_name} should have 7 predictions, but found "
                f"{len(games[game_name]['predictions'])}."
            )

    payload = {
        "generated_at": latest_timestamp.isoformat(),
        "games": games,
    }

    WEBSITE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)

    print(f"Website prediction data exported to: {OUTPUT_PATH}")
    print("Powerball predictions: 7")
    print("Mega Millions predictions: 7")


def main():
    export_latest_predictions()


if __name__ == "__main__":
    main()
