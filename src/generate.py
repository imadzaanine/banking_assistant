import os
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from langsmith import wrappers

# Groq exposes an OpenAI-compatible API, so we point the OpenAI client at it
# and wrap it so every call is traced in LangSmith.
client = wrappers.wrap_openai(
    OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
)


def generate_answers(query, retrieved_documents, model="openai/gpt-oss-120b"):
    """
    Generate an answer to `query` using only the retrieved documents.

    Args:
        query (str): The input query for the model.
        retrieved_documents (str | list): The retrieved document(s).
        model (str): The model to use for generating answers.

    Returns:
        str: The generated answer from the model.
    """
    prompt = f"""
You are a banking assistant.

Answer the question ONLY using the information in the documents below.
Do not add information that is not explicitly stated in the documents.
If the documents do not contain the answer, say:
"I do not have enough information in the provided documents."

Documents: {retrieved_documents}

Question: {query}

Answer:
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("Running main")

    from src.retrieve import retrieve_embeddings

    query = "What is the process for opening a new bank account?"

    print("Retrieving documents...")
    retrieved_results = retrieve_embeddings(query, top_k=3)
    documents = retrieved_results["documents"][0][0]
    print(documents)

    print("Generating answer...")
    answer = generate_answers(query, documents)

    print("Generated Answer:")
    print(answer)