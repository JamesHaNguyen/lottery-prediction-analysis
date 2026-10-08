import requests


POWERBALL_API = "https://data.ny.gov/resource/d6yy-54nr.json"
MEGA_MILLIONS_API = "https://data.ny.gov/resource/5xaw-6ayf.json"

def get_latest_result(api_url):
    params = {
        "$order" : "draw_date DESC",
        "$limit" : 1
    }
    response = requests.get(api_url, params=params, timeout=10)

    response.raise_for_status()

    results = response.json()

    if not results:
        raise ValueError("No lottery results were returned.")

    return results[0]


def main():
    powerball = get_latest_result(POWERBALL_API)
    mega_millions = get_latest_result(MEGA_MILLIONS_API)

    print("Powerball:")
    print(powerball)

    print("\nMega Millions:")
    print(mega_millions)


if __name__ == "__main__":
    main()