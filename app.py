from dotenv import load_dotenv
load_dotenv()  

from langsmith import traceable
from src.generate import generate_answers
from src.retrieve import retrieve_embeddings


@traceable(name="retrieve", run_type="retriever")
def retrieve(query, top_k=3):
    results = retrieve_embeddings(query, top_k=top_k)
    return results["documents"][0]


@traceable(name="generate")
def generate(query, document):
    return generate_answers(query, document)


@traceable(name="rag_pipeline")
def rag(query):
    docs = retrieve(query, top_k=3)
    docs = "\n\n".join(docs) 
    return generate(query, docs)  


if __name__ == "__main__":
    while True:  
        query = input("Enter your query: ")
        generated_answer = rag(query)
        print("\nGenerated Answer:")
        print(generated_answer)