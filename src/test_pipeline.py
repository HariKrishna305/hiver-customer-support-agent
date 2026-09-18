from src.pipeline import AmazonSupportPipeline


def main():

    pipeline = AmazonSupportPipeline()

    customer_message = "My Amazon order has not arrived yet. Where is my package?"

    intent = "delivery_shipping"

    result = pipeline.process(
        customer_message,
        intent
    )

    print("\n" + "=" * 60)
    print("AMAZON CUSTOMER SUPPORT PIPELINE")
    print("=" * 60)

    print("\nCustomer message:")
    print(result["customer_message"])

    print("\nIntent:")
    print(result["intent"])

    print("\nEscalation decision:")
    print(result["escalation"]["decision"])

    print("\nEscalation reason:")
    print(result["escalation"]["reason"])

    print("\nGenerated reply:")
    print(result["reply"])

    print("=" * 60)


if __name__ == "__main__":
    main()