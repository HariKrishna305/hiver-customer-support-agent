import pandas as pd
from pathlib import Path


# --------------------------------------------------
# File path
# --------------------------------------------------

FILE = "data/golden/golden_set.csv"


# --------------------------------------------------
# Intent definitions
# --------------------------------------------------

INTENTS = {
    "1": "delivery_shipping",
    "2": "billing_payment",
    "3": "account_access",
    "4": "technical_service_issue",
    "5": "refund_cancellation",
    "6": "information_request",
    "7": "complaint_feedback",
    "8": "other",
}


def label_dataset():

    # --------------------------------------------------
    # Check whether file exists
    # --------------------------------------------------

    if not Path(FILE).exists():
        print(f"File not found: {FILE}")
        return

    # --------------------------------------------------
    # Load golden dataset
    # --------------------------------------------------

    df = pd.read_csv(FILE)

    print(f"\nTotal examples: {len(df)}")

    # --------------------------------------------------
    # Check required columns
    # --------------------------------------------------

    required_columns = [
        "customer_tweet_id",
        "customer_message",
        "brand_reply",
        "expected_intent",
        "expected_escalation",
        "label_notes",
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
    # Label each example
    # --------------------------------------------------

    for index in range(len(df)):

        # Skip already labeled rows
        if (
            pd.notna(df.loc[index, "expected_intent"])
            and str(df.loc[index, "expected_intent"]).strip() != ""
        ):
            continue

        print("\n" + "=" * 70)
        print(f"Example {index + 1} / {len(df)}")
        print("=" * 70)

        # --------------------------------------------------
        # Customer message
        # --------------------------------------------------

        print("\nCUSTOMER MESSAGE:")
        print(df.loc[index, "customer_message"])

        # --------------------------------------------------
        # Brand reply
        # --------------------------------------------------

        print("\nBRAND REPLY:")
        print(df.loc[index, "brand_reply"])

        print("\n" + "-" * 70)

        # --------------------------------------------------
        # Select intent
        # --------------------------------------------------

        print("\nChoose the intent:\n")

        for number, intent in INTENTS.items():
            print(f"{number}. {intent}")

        while True:

            choice = input(
                "\nEnter choice (1-8): "
            ).strip()

            if choice in INTENTS:
                intent = INTENTS[choice]
                break

            print(
                "Invalid choice. Please enter a number from 1 to 8."
            )

        # --------------------------------------------------
        # Select escalation
        # --------------------------------------------------

        print("\nShould this message be escalated to a human?")
        print("1. No")
        print("2. Yes")

        while True:

            escalation_choice = input(
                "\nEnter choice (1-2): "
            ).strip()

            if escalation_choice == "1":
                escalation = "no"
                break

            elif escalation_choice == "2":
                escalation = "yes"
                break

            print(
                "Invalid choice. Please enter 1 or 2."
            )

        # --------------------------------------------------
        # Add notes
        # --------------------------------------------------

        notes = input(
            "\nEnter a short reason/notes: "
        ).strip()

        # --------------------------------------------------
        # Save labels
        # --------------------------------------------------

        df.loc[index, "expected_intent"] = intent
        df.loc[index, "expected_escalation"] = escalation
        df.loc[index, "label_notes"] = notes

        # --------------------------------------------------
        # Save after every example
        # --------------------------------------------------

        df.to_csv(
            FILE,
            index=False
        )

        print("\nSaved successfully.")

    # --------------------------------------------------
    # Finished
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("ALL EXAMPLES HAVE BEEN LABELED!")
    print("=" * 70)


if __name__ == "__main__":
    label_dataset()