
from dotenv import load_dotenv
load_dotenv()


from langgraph.graph import START,END, StateGraph
from state import AgentState
from processing import Processing_node
from retrieve import Retrieve_node



graph = StateGraph(AgentState)

graph.add_node("Processing", Processing_node)
graph.add_node("Retreiver",Retrieve_node)

graph.add_edge(START,"Retreiver")
graph.add_edge("Retreiver","Processing")
graph.add_edge("Processing", END)



app = graph.compile()

while True:
    query = input("You: ").strip()

    if not query:
        continue
    if query.lower() in {"exit", "quit", "q"}:
        print("Goodbye!")
        break

    result = app.invoke({
        "query": query,
        "chunks": [],
        "final_message": "",
    })
    print(f"\nAssistant: {result['final_message']}\n")



