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
- `app.py`: Streamlit UI with sidebar, chat history, and quick actions
- `requirements.txt`: Python dependencies

## Next steps (Databricks)
- Replace the stubbed data builders with calls to Databricks SQL, Unity Catalog tables, or Lakehouse APIs
- If deploying as a Databricks app, point the launcher to `app.py`
- If connecting to Mosaic AI or Model Serving, wrap calls behind a function like `generate_response_via_endpoint()` in `app.py`
