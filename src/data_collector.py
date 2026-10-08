import csv
import os
import requests


POWERBALL_API = "https://data.ny.gov/resource/d6yy-54nr.json"
MEGA_MILLIONS_API = "https://data.ny.gov/resource/5xaw-6ayf.json"

DATA_DIR = "data"

POWERBALL_FILE = os.path.join(DATA_DIR, "powerball.csv")
MEGA_MILLIONS_FILE = os.path.join(DATA_DIR, "mega_millions.csv")


def get_all_results(api_url):
    """Download all available lottery results from the API."""
    params = {
        "$order": "draw_date ASC",
        "$limit": 10000
    }

    response = requests.get(api_url, params=params, timeout=30)
    response.raise_for_status()

    return response.json()


def normalize_powerball(results):
    """Convert Powerball API records into a consistent format."""
    normalized = []

    for result in results:
        numbers = result["winning_numbers"].split()

        normalized.append({
            "draw_date": result["draw_date"][:10],
            "white_1": numbers[0],
            "white_2": numbers[1],
            "white_3": numbers[2],
            "white_4": numbers[3],
            "white_5": numbers[4],
            "powerball": numbers[5]
        })

    return normalized


def normalize_mega_millions(results):
    """Convert Mega Millions API records into a consistent format."""
    normalized = []

    for result in results:
        numbers = result["winning_numbers"].split()

        normalized.append({
            "draw_date": result["draw_date"][:10],
            "white_1": numbers[0],
            "white_2": numbers[1],
            "white_3": numbers[2],
            "white_4": numbers[3],
            "white_5": numbers[4],
            "mega_ball": result["mega_ball"]
        })

    return normalized


def save_results(filename, results, fieldnames):
    """Create or update a CSV file without duplicate drawing dates."""

    existing = {}
    new_rows = 0

    # Read existing CSV data if the file already exists
    if os.path.exists(filename):
        with open(filename, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                existing[row["draw_date"]] = row

    # Add new results
    for result in results:
        draw_date = result["draw_date"]

        # Check whether this drawing already exists
        if draw_date not in existing:
            new_rows += 1

        existing[draw_date] = result

    # Sort all results by drawing date
    sorted_results = sorted(
        existing.values(),
        key=lambda row: row["draw_date"]
    )

    # Make sure the data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Write the updated data back to the CSV
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(sorted_results)

    print(f"Added {new_rows} new record(s) to {filename}")
    

def main():
    print("Downloading Powerball results...")
    powerball_results = get_all_results(POWERBALL_API)

    print("Downloading Mega Millions results...")
    mega_millions_results = get_all_results(MEGA_MILLIONS_API)

    powerball_data = normalize_powerball(powerball_results)
    mega_millions_data = normalize_mega_millions(mega_millions_results)

    save_results(
        POWERBALL_FILE,
        powerball_data,
        [
            "draw_date",
            "white_1",
            "white_2",
            "white_3",
            "white_4",
            "white_5",
            "powerball"
        ]
    )

    save_results(
        MEGA_MILLIONS_FILE,
        mega_millions_data,
        [
            "draw_date",
            "white_1",
            "white_2",
            "white_3",
            "white_4",
            "white_5",
            "mega_ball"
        ]
    )


if __name__ == "__main__":
    main()