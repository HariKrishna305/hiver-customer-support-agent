from src.escalation import EscalationDecider


def main():

    decider = EscalationDecider()

    customer_message = "My Amazon order has not arrived yet. Where is my package?"

    intent = "delivery_shipping"

    retrieved_conversations = [
        {
            "customer_message": "Where's my package, Amazon?",
            "brand_reply": "Please check your delivery date.",
            "distance": 0.19
        }
    ]

    result = decider.decide(
        customer_message,
        intent,
        retrieved_conversations
    )

    print("=" * 60)
    print("ESCALATION TEST")
    print("=" * 60)

    print("\nCustomer message:")
    print(customer_message)

    print("\nIntent:")
    print(intent)

    print("\nDecision:")
    print(result["decision"])

    print("\nReason:")
    print(result["reason"])

    print("=" * 60)


if __name__ == "__main__":
    main()