import os
import groq
from dotenv import load_dotenv

load_dotenv()

client = groq.Client(
    api_key=os.getenv("GROQ_API_KEY"))

def generate_answers(query, retrieved_documents, model="openai/gpt-oss-120b"):
    """
    You are a banking assistant.

    Answer the question ONLY using the information provided in the documents below.
    Do not add information that is not explicitly stated in the documents.
    If the documents do not contain the answer, say:
    "I do not have enough information in the provided documents."

    Args:
        query (str): The input query for the model.
        retrieved_documents (list): The list of retrieved documents.
        model (str): The model to use for generating answers.
    
    Returns:
        str: The generated answer from the model.
    """
    prompt = f"""
    You are a helpful banking assistant. Use the following documents to answer the question.
    Documents: {retrieved_documents}
    Question: {query}
    Answer:
    """
    response = client.chat.completions.create(
        model=model,
         messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print("Running main")

    from retrieve import retrieve_embeddings

    query = "What is the process for opening a new bank account?"

    print("Retrieving documents...")
    retrieved_results = retrieve_embeddings(query, top_k=3)
    documents = retrieved_results["documents"][0][0]
    print(documents)
    

    print("Generating answer...")
    answer = generate_answers(query, documents)

    print("Generated Answer:")
    print(answer)