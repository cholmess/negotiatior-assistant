# ShipNegotiate AI (Streamlit Front-end)

A minimal Streamlit front-end for a contract negotiation assistant, modeled after the provided mock. This app will later be embedded into Databricks as a Lakehouse App or hooked to Databricks endpoints.

## Local run

1. Create a virtual environment (optional but recommended)
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app.py --server.port 8501 --server.headless true
```

## Structure
- `app.py`: Streamlit UI with sidebar, chat history, quick actions, and LLM + RAG wiring
- `requirements.txt`: Python dependencies

## Connect the chat to an LLM
The app uses an OpenAI-compatible Chat Completions client. Configure via environment variables or Streamlit secrets (`.streamlit/secrets.toml`).

Required:
- `OPENAI_API_KEY`: API key or token

Optional:
- `OPENAI_BASE_URL`: Defaults to `https://api.openai.com/v1`. For Databricks, set to your endpoint's OpenAI-compatible base, e.g. `https://<workspace-host>/serving-endpoints/<endpoint-name>/v1`.
- `LLM_MODEL`: Model name (e.g., `gpt-4o-mini`, or the model configured on your Databricks endpoint)
- `OPENAI_EMBED_MODEL`: Embedding model for RAG (default `text-embedding-3-small`)
- `LLM_PROVIDER`: Free-form tag for display (e.g., `databricks`)
- `NEGOTIATION_STYLE`: Set to `voss` to enable a Chris Voss–inspired persona (default is `voss`).

Example env vars:
```bash
export OPENAI_API_KEY=********
export OPENAI_BASE_URL=https://<your-dbx-host>/serving-endpoints/<endpoint>/v1
export LLM_MODEL=meta-llama-3.1-70b-instruct
export OPENAI_EMBED_MODEL=text-embedding-3-small
export NEGOTIATION_STYLE=voss
```

Notes for Databricks:
- Use a Personal Access Token (PAT) or configured token as `OPENAI_API_KEY`.
- Ensure your serving endpoint exposes an OpenAI-compatible Chat Completions and Embeddings path.

## Contextual Q&A (RAG)
- Toggle "Use contract context (RAG)" from the sidebar.
- Upload `.txt` or `.md` documents and click "Ingest uploads", or click "Load demo knowledge" for sample clauses.
- The app chunks, embeds, and stores vectors in-memory; queries retrieve top-k chunks and pass them as system context to the LLM. Sources referenced are shown beneath the answer.

Production notes:
- Replace the simple in-memory vector store with Databricks Vector Search or a Delta table powered by Mosaic AI embeddings.
- Swap the `_simple_chunk` and `_embed_texts` implementations with your preferred chunker and embedding model.
- For multi-tenant isolation, persist vectors per user or workspace.
