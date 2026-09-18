import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


GOLDEN_FILE = "data/processed/labeled_golden_set.csv"
MODEL_FILE = "models/intent_classifier.pkl"


def main():

    print("Loading golden dataset...")

    df = pd.read_csv(GOLDEN_FILE)

    print(f"Golden examples: {len(df)}")

    print("\nLoading intent classifier...")

    model = joblib.load(MODEL_FILE)

    print("Model loaded successfully.")

    X = df["customer_message"]
    y_true = df["expected_intent"]

    print("\nMaking predictions...")

    y_pred = model.predict(X)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print("\n" + "=" * 60)
    print("GOLDEN SET INTENT EVALUATION")
    print("=" * 60)

    print(f"\nExamples evaluated: {len(df)}")
    print(f"Accuracy: {accuracy:.2%}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    labels = sorted(df["expected_intent"].unique())

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    )

    print(matrix_df)

    print("=" * 60)


if __name__ == "__main__":
    main()