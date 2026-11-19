# AI Real Estate Assistant

A Streamlit chatbot for real estate questions, built with hybrid retrieval
(BM25 + Chroma vector search), metadata filters, an intent guard, query
rewriting, and a Tavily web-search fallback.

## Features

- Bilingual chat (Arabic and English) with conversation memory.
- Metadata filtering by city, district, country, project, unit type, purpose,
  and price range. Language is intentionally not filtered.
- Intent guard: greetings and off-topic messages are answered without retrieval.
- Query rewriting: follow-ups like "what about Riyadh?" become standalone
  questions; questions already answered in the chat are answered from history.
- Hybrid retrieval with reciprocal rank fusion over a persisted Chroma store.
- Tavily fallback when the model APIs are unavailable.

## Project structure

```text
app/app.py               Streamlit chat UI
pipeline/                Build and recommendation pipelines
src/                     Retrieval, filters, rewriter, intent guard, fallback
utils/                   LLM clients, logging, JSON helpers
config/config.py         Settings and secrets
data/chatbot_corpus.jsonl
chroma_db/               Persisted vector store (committed for deployment)
```

## Local setup

Python 3.12 is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```text
OPENROUTER_API_KEY=...
GROQ_API_KEY=...
TAVILY_API_KEY=...
```

Build the vector store only if `chroma_db/` is missing or data changed:

```powershell
python pipeline/build_pipeline.py --overwrite
```

Run the app:

```powershell
streamlit run app/app.py
```

## Streamlit Cloud deployment

1. Push this repository to GitHub.
2. Create a new Streamlit app and set the main file to `app/app.py`.
3. Add the secrets in the app settings:

```toml
OPENROUTER_API_KEY = "..."
GROQ_API_KEY = "..."
TAVILY_API_KEY = "..."
```

Optional overrides: `MODEL_NAME`, `REWRITE_MODEL`, `OPENROUTER_BASE_URL`,
`EMBEDDING_MODEL`.

`config/config.py` reads values from environment variables first and then from
Streamlit secrets, so the same code works locally and on Streamlit Cloud.
