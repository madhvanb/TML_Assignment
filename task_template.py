"""
Submit outputs/submission.csv to the leaderboard.

Set your API key first:
    export TML_API_KEY='your_key_here'
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import requests


BASE = Path(__file__).parent
OUTPUT_CSV = BASE / "outputs" / "submission.csv"

BASE_URL = "http://34.63.153.158"
TASK_ID = "01-mia"


def die(message):
    print(message, file=sys.stderr)
    sys.exit(1)


def check_submission(path):
    if not path.exists():
        die(f"File not found: {path}")

    df = pd.read_csv(path)

    if list(df.columns) != ["id", "score"]:
        die("CSV must have exactly these columns: id, score")
    if df["id"].duplicated().any():
        die("CSV has duplicate ids")
    if df["score"].isna().any():
        die("CSV has missing scores")
    if not df["score"].between(0, 1).all():
        die("All scores must be in [0, 1]")

    print("Submission file looks valid.")
    print(f"Rows: {len(df)}")
    print(f"Score range: {df['score'].min()} to {df['score'].max()}")


def submit(path, api_key):
    with open(path, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": api_key},
            files={"file": (path.name, f, "application/csv")},
            timeout=(10, 600),
        )

    try:
        body = response.json()
    except Exception:
        body = {"raw_text": response.text}

    if response.status_code == 413:
        die("Upload rejected: file too large (HTTP 413).")

    try:
        response.raise_for_status()
    except requests.exceptions.RequestException as error:
        print(f"Submission error: {error}", file=sys.stderr)
        print("Server response:", body, file=sys.stderr)
        sys.exit(1)

    print("Successfully submitted.")
    print("Server response:", body)


def main():
    parser = argparse.ArgumentParser(description="Submit outputs/submission.csv.")
    parser.add_argument("--file", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    check_submission(args.file)

    if args.check_only:
        return

    api_key = os.environ.get("TML_API_KEY")
    if not api_key:
        die("Set your API key first: export TML_API_KEY='your_key_here'")

    submit(args.file, api_key)


if __name__ == "__main__":
    main()
