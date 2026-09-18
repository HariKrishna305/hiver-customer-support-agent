import pandas as pd
from pathlib import Path


# --------------------------------------------------
# File path
# --------------------------------------------------

FILE = "data/golden/golden_set.csv"


def inspect_golden_set():

    # --------------------------------------------------
    # Check if file exists
    # --------------------------------------------------

    if not Path(FILE).exists():
        print(f"File not found: {FILE}")
        return

    # --------------------------------------------------
    # Load golden set
    # --------------------------------------------------

    df = pd.read_csv(FILE)

    # --------------------------------------------------
    # Display basic information
    # --------------------------------------------------

    print("Rows:", len(df))

    print("\nColumns:")
    print(df.columns.tolist())

    # --------------------------------------------------
    # Check required columns
    # --------------------------------------------------

    required_columns = [
        "customer_tweet_id",
        "customer_message",
        "brand_reply",
        "expected_intent",
        "expected_escalation",
        "label_notes"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print("\nMissing columns:")
        print(missing_columns)
        return

    # --------------------------------------------------
    # Display first 10 examples
    # --------------------------------------------------

    print("\nFirst 10 examples:\n")

    for index, row in df.head(10).iterrows():

        print(f"Example {index + 1}")

        print("Customer:")
        print(row["customer_message"])

        print("\nBrand reply:")
        print(row["brand_reply"])

        print("\nExpected intent:")
        print(row["expected_intent"])

        print("\nExpected escalation:")
        print(row["expected_escalation"])

        print("\nLabel notes:")
        print(row["label_notes"])

        print("-" * 80)


# --------------------------------------------------
# Run the function
# --------------------------------------------------

if __name__ == "__main__":
    inspect_golden_set()