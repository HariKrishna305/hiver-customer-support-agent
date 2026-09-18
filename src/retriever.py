import chromadb
from sentence_transformers import SentenceTransformer


class AmazonRetriever:

    def __init__(self):
        print("Loading embedding model...")

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Loading ChromaDB...")

        self.client = chromadb.PersistentClient(
            path="data/vectorstore"
        )

        self.collection = self.client.get_collection(
            name="amazon_support"
        )

        print(f"Collection: {self.collection.name}")
        print(f"Documents: {self.collection.count()}")

    def search(self, query, top_k=5):

        query_embedding = self.embedding_model.encode(
            [query]
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        conversations = []

        for i in range(len(results["ids"][0])):

            conversations.append({
                "customer_message": results["documents"][0][i],
                "brand_reply": results["metadatas"][0][i]["brand_reply"],
                "distance": results["distances"][0][i]
            })

        return conversations