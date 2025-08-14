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
- `app.py`: Streamlit UI with sidebar, chat history, quick actions, and LLM wiring
- `requirements.txt`: Python dependencies

## Connect the chat to an LLM
The app uses an OpenAI-compatible Chat Completions client. Configure via environment variables or Streamlit secrets (`.streamlit/secrets.toml`).

Required:
- `OPENAI_API_KEY`: API key or token

Optional:
- `OPENAI_BASE_URL`: Defaults to `https://api.openai.com/v1`. For Databricks, set to your endpoint's OpenAI-compatible base, e.g. `https://<workspace-host>/serving-endpoints/<endpoint-name>/v1`.
- `LLM_MODEL`: Model name (e.g., `gpt-4o-mini`, or the model configured on your Databricks endpoint)
- `LLM_PROVIDER`: Free-form tag for display (e.g., `databricks`)

Example env vars:
```bash
export OPENAI_API_KEY=********
export OPENAI_BASE_URL=https://<your-dbx-host>/serving-endpoints/<endpoint>/v1
export LLM_MODEL=meta-llama-3.1-70b-instruct
```

Notes for Databricks:
- Use a Personal Access Token (PAT) or configured token as `OPENAI_API_KEY`.
- Ensure your serving endpoint exposes an OpenAI-compatible Chat Completions path.

## Next steps (Databricks)
- Replace the stubbed data builders with calls to Databricks SQL, Unity Catalog tables, or Lakehouse APIs
- If deploying as a Databricks app, point the launcher to `app.py`
- If connecting to Mosaic AI or Model Serving, keep the OpenAI-compatible config above, or replace `generate_llm_response()` with a direct Databricks SDK call
