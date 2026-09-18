import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import AmazonRetriever
from src.reply_generator import AmazonReplyGenerator


def main():

    customer_message = (
        "My Amazon order has not arrived yet. Where is my package?"
    )

    print("Loading retriever...")
    retriever = AmazonRetriever()

    print("Retrieving similar conversations...")

    results = retriever.search(
    customer_message,
    top_k=5
    )
    

    print(f"Retrieved {len(results)} conversations.")

    print("\nLoading reply generator...")
    generator = AmazonReplyGenerator()

    print("\nGenerating Amazon support reply...")

    reply = generator.generate_reply(
        customer_message,
        results
    )

    print("\n" + "=" * 60)
    print("GENERATED AMAZON REPLY")
    print("=" * 60)
    print(reply)
    print("=" * 60)


if __name__ == "__main__":
    main()