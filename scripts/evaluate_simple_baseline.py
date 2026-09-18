import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


TEST_FILE = "data/processed/test.csv"


def predict_intent(message):

    message = message.lower()

    # Delivery
    if any(word in message for word in [
        "package",
        "delivery",
        "delivered",
        "shipment",
        "tracking",
        "arrived"
    ]):
        return "delivery_shipping"

    # Refund / cancellation
    if any(word in message for word in [
        "refund",
        "return",
        "cancel",
        "cancellation"
    ]):
        return "refund_cancellation"

    # Billing / payment
    if any(word in message for word in [
        "charged",
        "payment",
        "credit card",
        "debit card",
        "billing",
        "charge"
    ]):
        return "billing_payment"

    # Account
    if any(word in message for word in [
        "login",
        "log in",
        "password",
        "account",
        "sign in"
    ]):
        return "account_access"

    # Technical issue
    if any(word in message for word in [
        "not working",
        "broken",
        "error",
        "technical",
        "problem",
        "issue"
    ]):
        return "technical_service_issue"

    # Information request
    if any(word in message for word in [
        "how",
        "what",
        "where",
        "when",
        "which"
    ]):
        return "information_request"

    # Complaint
    if any(word in message for word in [
        "bad",
        "terrible",
        "unhappy",
        "disappointed",
        "complaint",
        "worst"
    ]):
        return "complaint_feedback"

    return "other"


def main():

    print("Loading test dataset...")

    df = pd.read_csv(TEST_FILE)

    print(f"Testing examples: {len(df)}")

    predictions = []

    for message in df["customer_message"]:
        predictions.append(
            predict_intent(message)
        )

    y_true = df["expected_intent"]
    y_pred = predictions

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print("\n" + "=" * 60)
    print("SIMPLE KEYWORD BASELINE RESULTS")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy:.2%}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    print("=" * 60)


if __name__ == "__main__":
    main()