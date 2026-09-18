import chromadb
from sentence_transformers import SentenceTransformer


VECTOR_DB_PATH = "data/vectorstore"
COLLECTION_NAME = "amazon_support"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading ChromaDB...")
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    print("Collection:", collection.name)
    print("Documents:", collection.count())

    query = "My Amazon order has not arrived yet. Where is my package?"

    print("\nCustomer message:")
    print(query)

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    print("\n" + "=" * 60)
    print("TOP 5 SIMILAR CONVERSATIONS")
    print("=" * 60)

    for i in range(len(results["documents"][0])):
        print(f"\n--- Result {i + 1} ---")
        print("Customer:")
        print(results["documents"][0][i])

        print("\nBrand reply:")
        print(results["metadatas"][0][i]["brand_reply"])

        print("\nDistance:")
        print(results["distances"][0][i])


if __name__ == "__main__":
    main()