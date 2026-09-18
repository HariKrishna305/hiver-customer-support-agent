import pandas as pd

from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# File paths
# --------------------------------------------------

TRAIN_FILE = "data/processed/train.csv"
TEST_FILE = "data/processed/test.csv"


def main():

    # --------------------------------------------------
    # Check files
    # --------------------------------------------------

    if not Path(TRAIN_FILE).exists():
        print(f"Training file not found: {TRAIN_FILE}")
        return

    if not Path(TEST_FILE).exists():
        print(f"Testing file not found: {TEST_FILE}")
        return

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------

    print("Loading datasets...")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print(f"Training examples: {len(train_df):,}")
    print(f"Testing examples: {len(test_df):,}")

    # --------------------------------------------------
    # Check required column
    # --------------------------------------------------

    if "expected_intent" not in train_df.columns:
        print("Missing 'expected_intent' column in training data.")
        return

    if "expected_intent" not in test_df.columns:
        print("Missing 'expected_intent' column in testing data.")
        return

    # --------------------------------------------------
    # Find the most common intent
    # --------------------------------------------------

    majority_intent = train_df["expected_intent"].mode()[0]

    print(f"\nMajority intent: {majority_intent}")

    # --------------------------------------------------
    # Predict majority intent for every test example
    # --------------------------------------------------

    predictions = [majority_intent] * len(test_df)

    actual = test_df["expected_intent"]

    # --------------------------------------------------
    # Calculate accuracy
    # --------------------------------------------------

    accuracy = accuracy_score(
        actual,
        predictions
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TRIVIAL BASELINE RESULTS")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy:.2%}")

    print("\nClassification Report:")

    print(
        classification_report(
            actual,
            predictions,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()