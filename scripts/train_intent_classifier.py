import pandas as pd
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# File paths
# --------------------------------------------------

TRAIN_FILE = "data/processed/train.csv"
TEST_FILE = "data/processed/test.csv"

MODEL_FILE = "models/intent_classifier.pkl"


def train_classifier():

    # --------------------------------------------------
    # Check input files
    # --------------------------------------------------

    if not Path(TRAIN_FILE).exists():
        print(f"Training file not found: {TRAIN_FILE}")
        return

    if not Path(TEST_FILE).exists():
        print(f"Testing file not found: {TEST_FILE}")
        return

    print("Loading training and testing data...")

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows: {len(test_df):,}")

    # --------------------------------------------------
    # Check required columns
    # --------------------------------------------------

    required_columns = [
        "customer_message",
        "expected_intent"
    ]

    for column in required_columns:

        if column not in train_df.columns:
            print(f"Missing column in training data: {column}")
            return

        if column not in test_df.columns:
            print(f"Missing column in testing data: {column}")
            return

    # --------------------------------------------------
    # Remove missing values
    # --------------------------------------------------

    train_df = train_df.dropna(
        subset=[
            "customer_message",
            "expected_intent"
        ]
    ).copy()

    test_df = test_df.dropna(
        subset=[
            "customer_message",
            "expected_intent"
        ]
    ).copy()

    # --------------------------------------------------
    # Clean text and labels
    # --------------------------------------------------

    train_df["customer_message"] = (
        train_df["customer_message"]
        .astype(str)
        .str.strip()
    )

    test_df["customer_message"] = (
        test_df["customer_message"]
        .astype(str)
        .str.strip()
    )

    train_df["expected_intent"] = (
        train_df["expected_intent"]
        .astype(str)
        .str.strip()
    )

    test_df["expected_intent"] = (
        test_df["expected_intent"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------
    # Remove empty messages
    # --------------------------------------------------

    train_df = train_df[
        train_df["customer_message"] != ""
    ]

    test_df = test_df[
        test_df["customer_message"] != ""
    ]

    # --------------------------------------------------
    # Prepare training data
    # --------------------------------------------------

    X_train = train_df["customer_message"]
    y_train = train_df["expected_intent"]

    X_test = test_df["customer_message"]
    y_test = test_df["expected_intent"]

    print(f"\nTraining examples: {len(X_train):,}")
    print(f"Testing examples: {len(X_test):,}")

    # --------------------------------------------------
    # Show intent distribution
    # --------------------------------------------------

    print("\nTraining intent distribution:")
    print(y_train.value_counts())

    # --------------------------------------------------
    # Create ML pipeline
    # --------------------------------------------------

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2)
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    # --------------------------------------------------
    # Train model
    # --------------------------------------------------

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # --------------------------------------------------
    # Make predictions
    # --------------------------------------------------

    predictions = model.predict(X_test)

    # --------------------------------------------------
    # Calculate accuracy
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL RESULTS")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy:.2%}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # --------------------------------------------------
    # Save trained model
    # --------------------------------------------------

    Path("models").mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print("=" * 60)
    print(f"Model saved to: {MODEL_FILE}")
    print("=" * 60)


# --------------------------------------------------
# Run the program
# --------------------------------------------------

if __name__ == "__main__":
    train_classifier()