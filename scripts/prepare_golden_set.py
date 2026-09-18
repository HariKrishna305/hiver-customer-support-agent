import pandas as pd
from pathlib import Path


# --------------------------------------------------
# File paths
# --------------------------------------------------

INPUT_FILE = "data/golden/golden_set.csv"
OUTPUT_FILE = "data/processed/labeled_golden_set.csv"


def prepare_dataset():

    # --------------------------------------------------
    # Check input file
    # --------------------------------------------------

    if not Path(INPUT_FILE).exists():
        print(f"Input file not found: {INPUT_FILE}")
        return

    print("Loading golden dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Total rows in original dataset: {len(df):,}")

    # --------------------------------------------------
    # Check required columns
    # --------------------------------------------------

    required_columns = [
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
    # Clean label columns
    # --------------------------------------------------

    df["expected_intent"] = (
        df["expected_intent"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["expected_escalation"] = (
        df["expected_escalation"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------
    # Keep only rows where both labels are available
    # --------------------------------------------------

    labeled_df = df[
        (df["expected_intent"] != "")
        & (df["expected_escalation"] != "")
    ].copy()

    # --------------------------------------------------
    # Remove unnecessary ID columns
    # --------------------------------------------------

    labeled_df = labeled_df[
        [
            "customer_message",
            "brand_reply",
            "expected_intent",
            "expected_escalation",
            "label_notes"
        ]
    ]

    # --------------------------------------------------
    # Create output directory
    # --------------------------------------------------

    Path("data/processed").mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Save processed dataset
    # --------------------------------------------------

    labeled_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print(f"Labeled rows: {len(labeled_df):,}")
    print(f"Unlabeled rows: {len(df) - len(labeled_df):,}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nProcessed dataset columns:")
    print(labeled_df.columns.tolist())

    print("\nSample:")
    print(
        labeled_df.head(5).to_string(index=False)
    )


if __name__ == "__main__":
    prepare_dataset()