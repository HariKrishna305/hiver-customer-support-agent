from src.intent_classifier import IntentClassifier


def main():

    classifier = IntentClassifier()

    test_messages = [
        "My Amazon order has not arrived yet.",
        "I was charged twice for my order.",
        "I cannot login to my Amazon account.",
        "I want to cancel my order.",
        "My Fire TV is not working."
    ]

    print("\n" + "=" * 60)
    print("INTENT CLASSIFICATION TEST")
    print("=" * 60)

    for message in test_messages:

        intent = classifier.predict(message)

        print("\nCustomer message:")
        print(message)

        print("Predicted intent:")
        print(intent)

    print("=" * 60)


if __name__ == "__main__":
    main()