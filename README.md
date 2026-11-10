# Employee Policy Assistant

A local document question-answering app for exploring an employee handbook. It combines keyword and semantic retrieval with a locally served language model, then returns an answer with handbook page references.

> **Demo document:** The included handbook is a public template, not a finalized company policy. Bracketed values and optional sections in it may be unfilled or conditional. Treat them as unconfirmed.

## Screenshots

### Home screen

![Policy Assistant home screen](images/home.png)

### Workplace conduct question

![Workplace conduct question and answer](images/Demo1.png)

### Sick leave question

![Sick leave question and answer](images/Demo2.png)

## Features

- Ask handbook questions in a chat interface built with Angular.
- Retrieve passages with BM25 keyword search and Chroma semantic search, combined with rank fusion.
- Use conversation history to rewrite follow-up questions for retrieval.
- Parse, section, and chunk PDF content while preserving page and section metadata.
- Generate context-grounded answers with a local Ollama model.
- Identify unresolved template values and conditional handbook sections.
- Return source document and page metadata through FastAPI endpoints.

## How it works

```mermaid
flowchart LR
    A[Angular chat UI] -->|HTTP / SSE| B[FastAPI]
    B --> C[Question and history processing]
    C --> D[Hybrid retrieval]
    D --> E[BM25 keyword search]
    D --> F[Chroma semantic search]
    E --> G[Rank fusion]
    F --> G
    G --> H[Relevant handbook excerpts]
    H --> I[Ollama answer generation]
    I --> B
    B --> A
```

The hybrid retriever combines BM25 keyword results with Chroma semantic results using rank fusion:

![Reciprocal rank fusion illustration](images/RRF.png)

## Tech stack

| Area | Tools |
|---|---|
| Frontend | Angular, TypeScript, HTML, CSS |
| API | Python, FastAPI, Pydantic |
| Language model | Ollama (`llama3.2:3b` by default), LangChain `ChatOpenAI` for OpenAI-compatible endpoints |
| Retrieval | BM25, semantic search, ensemble rank fusion |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB |
| PDF ingestion | PyPDF, LangChain text splitters |
| Chat transport | REST and Server-Sent Events (SSE) |

## Repository layout

```text
ai-policy-bot/
├── images/
│   ├── home.png               # Chat home screen
│   ├── Demo1.png              # Workplace conduct example
│   ├── Demo2.png              # Sick leave example
│   └── RRF.png                # Rank-fusion illustration
├── documents/
│   └── company_policy.pdf     # Source handbook template
├── chroma_db/                 # Generated Chroma index
├── frontend/                  # Angular application
├── src/
│   ├── ingest.py              # PDF parsing and indexing
│   ├── rag.py                 # Retrieval and answer generation
│   └── main.py                # FastAPI app and chat endpoints
├── .env.example               # Model-provider configuration template
├── test_hybrid.py
├── test_retrieval.py
├── test_stream.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.12
- Node.js and npm
- [Ollama](https://ollama.com/)
- The `llama3.2:3b` model pulled into Ollama

## Run locally

### 1. Create a Python environment

From the repository root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Start the local model

```powershell
ollama pull llama3.2:3b
```

Make sure the Ollama service is running before starting the API.

### Optional: select the OpenAI-compatible client

The default configuration uses LangChain's native `ChatOllama` integration. To run the same local Ollama model through LangChain's `ChatOpenAI` client and Ollama's OpenAI-compatible endpoint, copy `.env.example` to `.env` and set:

```dotenv
LLM_PROVIDER=openai-compatible
LLM_MODEL=llama3.2:3b
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
```

The local Ollama server ignores the key-shaped value, so this mode does not use OpenAI credits. To point the client at a hosted OpenAI-compatible service, set that service's base URL, model, and API key instead; hosted services may charge for use. Do not commit `.env` or real API keys.

### 3. Index the handbook

The included PDF is already at `documents/company_policy.pdf`. To create or refresh the Chroma index, run:

```powershell
python -m src.ingest
```

The first run may download the Sentence Transformers embedding model.

### 4. Start the API

In a terminal at the repository root:

```powershell
uvicorn src.main:app --reload
```

- API: <http://localhost:8000>
- Interactive API docs: <http://localhost:8000/docs>

### 5. Start the Angular app

In a second terminal:

```powershell
cd frontend
npm install
npm start
```

Open <http://localhost:4200>.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | API health message |
| `POST` | `/chat` | Return a complete answer and source pages |
| `POST` | `/chat/stream` | Return answer, source, and completion events over SSE |

Example request:

```json
{
  "question": "What does the handbook say about sick leave?",
  "history": []
}
```

The `history` field is optional and contains prior messages with `user` or `assistant` roles.

## Source handbook

The sample PDF is the [Workable Employee Handbook template](https://resources.workable.com/wp-content/uploads/2017/09/Employee-Handbook.pdf). Its placeholder values, alternatives, and conditional sections are not confirmed policies for any particular company.

## Notes and limitations

- Answers depend on retrieval quality and the local language model; check the cited handbook pages for important details.
- The sample handbook contains incomplete and optional template content, so some policy values are intentionally unspecified.
- The demo API does not include authentication or multi-user data isolation.

