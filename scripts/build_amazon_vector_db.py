import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path


INPUT_FILE = "data/processed/amazon_clean.csv"
CHROMA_DIR = "data/vectorstore"
COLLECTION_NAME = "amazon_support"


def main():

    print("Loading Amazon data...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Total conversations: {len(df)}")

    # Load embedding model
    print("\nLoading embedding model...")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Embedding model loaded.")

    # Create ChromaDB client
    print("\nCreating ChromaDB...")

    Path(CHROMA_DIR).mkdir(
        parents=True,
        exist_ok=True
    )

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    # Delete old collection if it exists
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Existing collection deleted.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Historical Amazon customer support conversations"
        }
    )

    # Process in batches
    batch_size = 500

    total = len(df)

    print("\nCreating embeddings...")

    for start in range(0, total, batch_size):

        end = min(start + batch_size, total)

        batch = df.iloc[start:end]

        customer_messages = (
            batch["customer_message"]
            .astype(str)
            .tolist()
        )

        embeddings = model.encode(
            customer_messages,
            show_progress_bar=False
        ).tolist()

        documents = customer_messages

        metadatas = []

        ids = []

        for i, (_, row) in enumerate(batch.iterrows()):

            metadatas.append({
                "brand_tweet_id": str(row["brand_tweet_id"]),
                "brand_reply": str(row["brand_reply"])
            })

            ids.append(
             f"{row['customer_tweet_id']}_{row['brand_tweet_id']}"
)
            

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        print(
            f"Processed {end}/{total}"
        )

    print("\n" + "=" * 60)
    print("VECTOR DATABASE CREATED")
    print("=" * 60)

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Documents: {collection.count()}")
    print(f"Location: {CHROMA_DIR}")


if __name__ == "__main__":
    main()