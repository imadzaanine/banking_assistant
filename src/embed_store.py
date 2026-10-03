import os

import chromadb 
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="chroma_db")
model = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION_NAME = "banking_knowledge"

def store_embeddings(documents):
    """
    Store the embeddings of the given policy document in the ChromaDB collection.
    
    Args:
        documents (list): A list of strings, each representing a policy document.

    """
    # Create or get the collection 
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # Iterate through each document and store its embedding
    text_to_embed = []
    for policy in documents:
        """
        Each policy has an id, a title, and a content. We will store the embedding of the content.
        """
        text_to_embed.append(policy["content"])

    # Generate embeddings for the documents
    embeddings = model.encode(text_to_embed)

    # Making the ids for each policy
    ids = [ policy["id"] for policy in documents]


    # Crreating the metadata for each policy
    metadatas = [{ "title": policy["title"]} for policy in documents]

    policy = [policy["content"] for policy in documents]

    # Store the embeddings in the collection
    collection.add(
        ids = ids,
        embeddings = embeddings.tolist(),
        metadatas = metadatas,
        documents = policy
    )
    print(f"Stored embeddings for {len(documents)} documents in the '{COLLECTION_NAME}' collection.")


if __name__ == "__main__":
    from src.parse import parse_docs
    documents = parse_docs()
    store_embeddings(documents)


