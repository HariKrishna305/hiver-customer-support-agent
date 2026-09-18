import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import AmazonRetriever


def main():

    retriever = AmazonRetriever()

    query = "My Amazon order has not arrived yet. Where is my package?"

    print("\n" + "=" * 60)
    print("RETRIEVAL TEST")
    print("=" * 60)

    print(f"\nCustomer message:\n{query}")

    results = retriever.search(query, top_k=5)

    print("\n" + "=" * 60)
    print("TOP 5 SIMILAR CONVERSATIONS")
    print("=" * 60)

    for i, result in enumerate(results, 1):

        print(f"\n--- Result {i} ---")

        print("Customer:")
        print(result["customer_message"])

        print("\nAmazon reply:")
        print(result["brand_reply"])

        print(f"\nDistance: {result['distance']}")


if __name__ == "__main__":
    main()