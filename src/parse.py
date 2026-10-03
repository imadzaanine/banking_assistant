import os
import json

def parse_docs(path="documents"):
    documents = []

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)

        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                document = json.load(file)   
                documents.append(document)

    return documents

if __name__ == "__main__":
    docs = parse_docs()

    print(f"Parsed {len(docs)} documents.")
    print(docs[0]["content"])
    print(docs[0]["id"])