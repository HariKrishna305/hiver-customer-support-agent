import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.escalation import EscalationDecider


GOLDEN_FILE = PROJECT_ROOT / "data" / "processed" / "labeled_golden_set.csv"


def main():

    print("Loading golden dataset...")

    df = pd.read_csv(GOLDEN_FILE)

    print(f"Golden examples: {len(df)}")

    # Keep only rows with valid escalation labels
    df = df[
        df["expected_escalation"].isin(["yes", "no"])
    ].copy()

    print(f"Examples evaluated: {len(df)}")

    decider = EscalationDecider()

    predictions = []

    print("\nMaking escalation predictions...")

    for _, row in df.iterrows():

        # Controlled evaluation:
        # Use a sufficiently similar retrieval result
        # so we evaluate the escalation rules themselves.
        retrieved_conversations = [
            {
                "customer_message": row["customer_message"],
                "brand_reply": row["brand_reply"],
                "distance": 0.19
            }
        ]

        result = decider.decide(
            customer_message=row["customer_message"],
            intent=row["expected_intent"],
            retrieved_conversations=retrieved_conversations
        )

        if result["decision"] == "escalate":
            predictions.append("yes")
        else:
            predictions.append("no")

    y_true = df["expected_escalation"]
    y_pred = predictions

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print("\n" + "=" * 60)
    print("ESCALATION EVALUATION")
    print("=" * 60)

    print(f"\nExamples evaluated: {len(df)}")
    print(f"Accuracy: {accuracy:.2%}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=["yes", "no"],
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=["yes", "no"]
        )
    )

    print("\nLabels:")
    print("Rows    = Expected")
    print("Columns = Predicted")
    print("[yes, no]")

    print("=" * 60)


if __name__ == "__main__":
    main()