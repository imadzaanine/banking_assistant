from state import AgentState,llm
from langchain.agents import create_agent


SYSTEM_PROMPT = """
You are a banking assistant.

Answer using ONLY the information in the documents provided.
You may apply the documents' rules to the customer's situation
(e.g., compare their age to an eligibility range), but do not use
outside knowledge or invent policies.

Only if the documents contain nothing relevant to the question, say:
"I do not have enough information in the provided documents."
"""
from langchain_core.messages import SystemMessage, HumanMessage

def Processing_node(state: AgentState):
    context = "\n\n".join(state["chunks"])
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion:\n{state['query']}"),
    ])
    return {"final_message": response.content}