import csv
from datetime import datetime
import os


POWERBALL_FILE = "data/powerball.csv"
MEGA_MILLIONS_FILE = "data/mega_millions.csv"


def validate_date(date_text):
    """Check that the date is in YYYY-MM-DD format."""

    try:
        datetime.strptime(date_text, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def validate_lottery_file(filename, special_ball_name):
    """Validate the structure and basic data quality of a lottery CSV file."""

    print(f"\nValidating {filename}...")

    # Check that the file exists
    if not os.path.exists(filename):
        print("ERROR: File does not exist.")
        return False

    required_columns = [
        "draw_date",
        "white_1",
        "white_2",
        "white_3",
        "white_4",
        "white_5",
        special_ball_name
    ]

    errors = []
    dates_seen = set()
    row_count = 0

    with open(filename, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        # Check that all expected columns exist
        missing_columns = [
            column
            for column in required_columns
            if column not in reader.fieldnames
        ]

        if missing_columns:
            errors.append(
                f"Missing columns: {', '.join(missing_columns)}"
            )

        # Stop here if the CSV structure is wrong
        if errors:
            print("FAILED")
            for error in errors:
                print(f"  - {error}")
            return False

        # Check every row
        for row_number, row in enumerate(reader, start=2):

            row_count += 1

            # ------------------------------
            # Check for missing values
            # ------------------------------

            for column in required_columns:
                if not row[column].strip():
                    errors.append(
                        f"Row {row_number}: {column} is empty."
                    )

            # ------------------------------
            # Validate date
            # ------------------------------

            draw_date = row["draw_date"]

            if not validate_date(draw_date):
                errors.append(
                    f"Row {row_number}: invalid date '{draw_date}'."
                )

            # ------------------------------
            # Check duplicate dates
            # ------------------------------

            if draw_date in dates_seen:
                errors.append(
                    f"Row {row_number}: duplicate draw date {draw_date}."
                )

            dates_seen.add(draw_date)

            # ------------------------------
            # Convert lottery numbers to integers
            # ------------------------------

            white_numbers = []

            for column in [
                "white_1",
                "white_2",
                "white_3",
                "white_4",
                "white_5"
            ]:

                try:
                    number = int(row[column])
                    white_numbers.append(number)

                except ValueError:
                    errors.append(
                        f"Row {row_number}: {column} is not numeric."
                    )

            # Validate special ball
            try:
                int(row[special_ball_name])

            except ValueError:
                errors.append(
                    f"Row {row_number}: "
                    f"{special_ball_name} is not numeric."
                )

            # ------------------------------
            # Check duplicate white balls
            # ------------------------------

            if len(white_numbers) == 5:

                if len(set(white_numbers)) != 5:
                    errors.append(
                        f"Row {row_number}: "
                        "white-ball numbers contain duplicates."
                    )

                # ------------------------------
                # Check white-ball ordering
                # ------------------------------

                if white_numbers != sorted(white_numbers):
                    errors.append(
                        f"Row {row_number}: "
                        "white-ball numbers are not sorted."
                    )

    # ------------------------------
    # Print final result
    # ------------------------------

    if errors:
        print("FAILED")
        print(f"Found {len(errors)} validation error(s):")

        for error in errors[:20]:
            print(f"  - {error}")

        if len(errors) > 20:
            print(
                f"  ... and {len(errors) - 20} more errors."
            )

        return False

    print("PASSED")
    print(f"Validated {row_count} rows.")
    print(f"Validated {len(dates_seen)} unique drawing dates.")

    return True


def main():
    powerball_valid = validate_lottery_file(
        POWERBALL_FILE,
        "powerball"
    )

    mega_millions_valid = validate_lottery_file(
        MEGA_MILLIONS_FILE,
        "mega_ball"
    )

    print("\n==============================")
    print("Validation Summary")
    print("==============================")

    if powerball_valid and mega_millions_valid:
        print("All lottery data passed validation.")

    else:
        print("One or more files failed validation.")


if __name__ == "__main__":
    main()