import pandas as pd
import joblib


GOLDEN_FILE = "data/processed/labeled_golden_set.csv"
MODEL_FILE = "models/intent_classifier.pkl"


def main():

    print("Loading golden dataset...")

    df = pd.read_csv(GOLDEN_FILE)

    print("Loading intent classifier...")

    model = joblib.load(MODEL_FILE)

    predictions = model.predict(
        df["customer_message"]
    )

    df["predicted_intent"] = predictions

    errors = df[
        df["expected_intent"] != df["predicted_intent"]
    ].copy()

    print("\n" + "=" * 70)
    print("GOLDEN SET INTENT ERRORS")
    print("=" * 70)

    print(f"\nTotal examples: {len(df)}")
    print(f"Correct predictions: {len(df) - len(errors)}")
    print(f"Incorrect predictions: {len(errors)}")

    for index, row in errors.iterrows():

        print("\n" + "-" * 70)

        print(f"Example: {index + 1}")

        print("\nCustomer message:")
        print(row["customer_message"])

        print("\nExpected intent:")
        print(row["expected_intent"])

        print("\nPredicted intent:")
        print(row["predicted_intent"])

        print("\nLabel notes:")
        print(row["label_notes"])

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()