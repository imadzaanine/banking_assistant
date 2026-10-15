from src.parse import parse_docs
from app import vectorstore
from langchain_core.documents import Document


if __name__ == "__main__":
    policies = parse_docs()
    docs =[]
    for policy in policies:
        docs.append(Document(
            page_content=policy["content"],
            metadata={
                "title": policy["title"],
                "id": policy["id"]
            }
        ))
    vectorstore.add_documents(docs)
    

    
