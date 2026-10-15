# Banking Assistant

A command-line banking assistant built with [LangGraph](https://github.com/langchain-ai/langgraph). It takes a customer question, retrieves relevant information, and generates an answer in a simple chat loop.

## How it works

The assistant runs a two-step LangGraph pipeline:

```
START  ->  Retriever  ->  Processing  ->  END
```

1. **Retriever** (`Retrieve_node`) takes the user's `query` and fills `chunks` with the most relevant pieces of banking content.
2. **Processing** (`Processing_node`) uses the query and retrieved chunks to produce the `final_message`.

The shared state (`AgentState`) carries three fields through the graph:

| Field           | Description                                  |
| --------------- | -------------------------------------------- |
| `query`         | The user's question                          |
| `chunks`        | Retrieved context passed to the next node    |
| `final_message` | The answer printed back to the user          |

## Project structure

```
banking_assistant/
├── app.py          # Entry point: builds the graph and runs the chat loop
├── src/            # State definition, retrieval and processing nodes
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or newer
- An API key for the LLM provider used in `src/` (see Configuration)

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/imadzaanine/banking_assistant.git
cd banking_assistant

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Configuration

The app loads environment variables from a `.env` file via `python-dotenv`. Create one in the project root:

```env
# Replace with the key(s) your chosen provider needs, for example:
OPENAI_API_KEY=your_api_key_here
```

Never commit your `.env` file. It is already excluded if listed in `.gitignore`.

## Usage

Start the assistant:

```bash
python app.py
```

Then type your question at the prompt:

```
You: How do I report a lost card?

Assistant: ...
```

Type `exit`, `quit`, or `q` to leave.

## Customization

- **Change the knowledge base:** update the data or documents used by the retriever in `src/`.
- **Change the answer style:** edit the prompt or logic in the processing node.
- **Add steps:** register a new node in `app.py` with `graph.add_node(...)` and connect it with `graph.add_edge(...)`, for example a guardrail or intent-classification step.
