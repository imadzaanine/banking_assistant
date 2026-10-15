from state import AgentState, vectorstore


def Retrieve_node(state: AgentState, limit: int = 5):
    """
    Retrieves relevant chunks from ChromaDB.

    Args:
        state: Current LangGraph state
        limit: Number of documents to retrieve

    Returns:
        Updated state with retrieved chunks
    """

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": limit}
    )

    docs = retriever.invoke(
        state["query"]
    )

    return {
        "chunks": [
            doc.page_content
            for doc in docs
        ]
    }