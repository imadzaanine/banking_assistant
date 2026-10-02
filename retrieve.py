from sentence_transformers import SentenceTransformer
import chromadb


model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")

COLLECTION_NAME = "banking_knowledge"

def retrieve_embeddings(query, top_k=5):
    """
    Retrieve the top_k most similar documents to the given query from the ChromaDB collection.

    Args:
        query (str): The query string for which to find similar documents.
        top_k (int): The number of top similar documents to retrieve.
    """
    # Get the collection 

    collection = client.get_collection(COLLECTION_NAME)

    # Generate the embedding for the query

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["metadatas", "documents"]
    )
    return results

if __name__ == "__main__":
    query = "What is the process for opening a new bank account?"
    results = retrieve_embeddings(query, top_k=3)

    metadatas = results["metadatas"][0]
    documents = results["documents"][0]

    for i, (metadata, policy) in enumerate(zip(metadatas, documents)):
        print(f"\nDocument {i + 1}:")
        print(f"Title: {metadata['title']}")
        print(f"Content: {policy}")
