from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage,BaseMessage

from langchain_groq import ChatGroq
from operator import add

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class AgentState(TypedDict):
    query: str
    chunks: list[str]
    final_message:str


llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    collection_name="banking_knowledge",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)