import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import os
from typing import List, Dict, Any, Tuple
import numpy as np

st.set_page_config(
    page_title="ShipNegotiate AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "kind": "text",
                "content": (
                    "Hi Christopher! I'm Chris, your AI negotiation assistant. I have access to all your "
                    "contract data, market intelligence, and customer insights. I can help you analyze margins, "
                    "assess customer risks, benchmark rates, and develop negotiation strategies. What would you like to explore today?"
                ),
            }
        ]
    if "pending_user_prompt" not in st.session_state:
        st.session_state.pending_user_prompt = None
    if "use_rag" not in st.session_state:
        st.session_state.use_rag = True
    if "rag_chunks" not in st.session_state:
        st.session_state.rag_chunks: List[str] = []
    if "rag_metas" not in st.session_state:
        st.session_state.rag_metas: List[Dict[str, Any]] = []
    if "rag_matrix" not in st.session_state:
        st.session_state.rag_matrix: np.ndarray | None = None  # shape: (n, d), L2-normalized


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### ShipNegotiate AI")

        st.markdown("**Portfolio Overview**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Active Contracts", value="247")
            st.metric("Renewals Due", value="23")
        with col_b:
            st.metric("Avg Margin", value="16.8%")
            st.metric("High Risk Accounts", value="5")

        st.divider()
        st.markdown("**Data Sources**")
        data_sources = [
            ("Contract Management System", "#22c55e"),
            ("Financial Database", "#22c55e"),
            ("Market Intelligence Feed", "#22c55e"),
            ("Credit Rating Service", "#22c55e"),
            ("Commodity Exchange Data", "#ef4444"),
        ]
        for name, color in data_sources:
            st.markdown(
                f"<span style='color:{color};font-size:18px;'>●</span> {name}",
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown("**Recent Alerts**")
        st.info("Credit Rating Downgrade — GlobalTrade Corp downgraded to BBB" )

        st.divider()
        st.markdown("**Knowledge base**")
        st.checkbox("Use contract context (RAG)", key="use_rag", value=st.session_state.use_rag)
        st.caption("Upload .txt or .md docs. We'll chunk, embed, and retrieve top-k context for Q&A.")
        uploaded = st.file_uploader("Upload documents", accept_multiple_files=True, type=["txt", "md"])
        col_kb1, col_kb2, col_kb3 = st.columns([1,1,1])
        with col_kb1:
            if st.button("Ingest uploads", use_container_width=True, type="primary") and uploaded:
                texts_with_meta = []
                for f in uploaded:
                    try:
                        raw = f.read().decode("utf-8", errors="ignore")
                    except Exception:
                        raw = f.read()
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8", errors="ignore")
                    texts_with_meta.append((raw, {"source": f.name}))
                add_texts_to_kb(texts_with_meta)
                st.success(f"Ingested {len(texts_with_meta)} file(s). KB chunks: {len(st.session_state.rag_chunks)}")
        with col_kb2:
            if st.button("Load demo knowledge", use_container_width=True):
                demo_texts = _demo_contract_texts()
                add_texts_to_kb([(t, {"source": "demo.md"}) for t in demo_texts])
                st.success(f"Demo loaded. KB chunks: {len(st.session_state.rag_chunks)}")
        with col_kb3:
            if st.button("Clear KB", use_container_width=True):
                clear_kb()
                st.info("Knowledge base cleared")

        st.caption(
            f"KB size: {len(st.session_state.rag_chunks)} chunks | Model: {os.getenv('OPENAI_EMBED_MODEL', 'text-embedding-3-small')}"
        )

        # LLM config quick view (uses env vars or Streamlit secrets)
        with st.expander("LLM configuration"):
            provider = os.getenv("LLM_PROVIDER", "openai/databricks-compatible")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
            embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
            st.write({"provider": provider, "base_url": base_url, "chat_model": model_name, "embed_model": embed_model})
            st.caption("Set OPENAI_API_KEY (and OPENAI_BASE_URL for Databricks) in env vars or .streamlit/secrets.toml")


def render_header() -> None:
    st.markdown("### Chris - Your Negotiation Assistant")
    st.caption("Online • Ready to help with contract insights")


def create_margin_dataframe() -> pd.DataFrame:
    data = [
        {"Customer": "GlobalTrade Corp", "Revenue (M)": 8.5, "COGS (M)": 6.6, "Last Renewal": "2024-11-01"},
        {"Customer": "Pacific Logistics", "Revenue (M)": 6.2, "COGS (M)": 5.1, "Last Renewal": "2024-07-15"},
        {"Customer": "Apex Retail Group", "Revenue (M)": 4.9, "COGS (M)": 3.9, "Last Renewal": "2025-01-30"},
        {"Customer": "EuroChem Ltd", "Revenue (M)": 7.3, "COGS (M)": 6.3, "Last Renewal": "2024-12-10"},
        {"Customer": "NorthStar Foods", "Revenue (M)": 5.4, "COGS (M)": 4.4, "Last Renewal": "2025-02-12"},
    ]
    df = pd.DataFrame(data)
    df["Margin %"] = ((df["Revenue (M)"] - df["COGS (M)"]) / df["Revenue (M)"]) * 100
    return df


def create_risk_dataframe() -> pd.DataFrame:
    data = [
        {"Customer": "GlobalTrade Corp", "Credit Rating": "BBB", "DSO": 62, "On-time %": 88, "Risk": "High"},
        {"Customer": "Pacific Logistics", "Credit Rating": "A-", "DSO": 45, "On-time %": 95, "Risk": "Medium"},
        {"Customer": "Apex Retail Group", "Credit Rating": "A", "DSO": 38, "On-time %": 97, "Risk": "Low"},
        {"Customer": "EuroChem Ltd", "Credit Rating": "BBB+", "DSO": 50, "On-time %": 92, "Risk": "Medium"},
        {"Customer": "NorthStar Foods", "Credit Rating": "BBB", "DSO": 58, "On-time %": 90, "Risk": "Medium"},
    ]
    return pd.DataFrame(data)


def create_rate_trend_dataframe(periods: int = 12) -> pd.DataFrame:
    start_date = datetime.today().date().replace(day=1)
    dates = [start_date - pd.offsets.MonthBegin(n) for n in range(periods - 1, -1, -1)]
    market_rate = [1.85, 1.9, 1.95, 1.98, 2.05, 2.1, 2.08, 2.12, 2.2, 2.25, 2.22, 2.28]
    your_rate = [1.95, 1.97, 2.02, 2.04, 2.12, 2.18, 2.13, 2.17, 2.26, 2.3, 2.27, 2.35]
    df = pd.DataFrame({"Month": dates, "Market Benchmark Rate": market_rate, "Your Avg Rate": your_rate})
    return df


def append_assistant_message(message: dict) -> None:
    st.session_state.messages.append(message)


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("kind") == "margin_analysis":
            st.write(message["content"])
            df = pd.DataFrame(message["data"]["rows"])
            st.dataframe(df, use_container_width=True)
            chart_df = df.set_index("Customer")["Margin %"]
            st.bar_chart(chart_df)
        elif message.get("kind") == "risk_assessment":
            st.write(message["content"])
            df = pd.DataFrame(message["data"]["rows"])
            st.dataframe(df, use_container_width=True)
        elif message.get("kind") == "rate_trends":
            st.write(message["content"])
            df = pd.DataFrame(message["data"]["rows"])
            fig = px.line(df, x="Month", y=["Market Benchmark Rate", "Your Avg Rate"], markers=True)
            fig.update_layout(height=360, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        elif message.get("kind") == "sources":
            st.caption("Context used:")
            for s in message.get("data", {}).get("sources", []):
                st.write(f"- {s}")
        else:
            st.write(message["content"])


# ---------- LLM integration ----------

def _get_secret_env(name: str, default: str | None = None) -> str | None:
    if name in st.secrets:
        try:
            return st.secrets[name]
        except Exception:
            pass
    return os.getenv(name, default)


def _build_system_prompt() -> str:
    style = os.getenv("NEGOTIATION_STYLE", "voss").lower()
    base_prompt = (
        "You are Chris, an AI contract negotiation assistant. Be concise, data-driven, and action-oriented. "
        "When appropriate, suggest concrete next steps, thresholds, and negotiation levers (rate, term, indexation, volume, surcharges)."
    )
    if style in {"voss", "chris_voss", "never_split_the_difference", "voss-inspired"}:
        return (
            "You are Chris, an AI contract negotiation assistant inspired by the negotiation style in 'Never Split the Difference'. "
            "Communicate with tactical empathy and focus on uncovering information before proposing numbers.\n\n"
            "Guidelines:\n"
            "- Tactical empathy and labeling: acknowledge feelings with labels like 'It seems…', 'It sounds like…'.\n"
            "- Mirroring: repeat 1–3 key words from the counterpart to encourage elaboration.\n"
            "- Calibrated questions: prefer open 'what' and 'how' questions to make the counterpart solve the problem.\n"
            "- No-oriented questions: make it easy to say 'no' safely (e.g., 'Would it be ridiculous to…?').\n"
            "- Accusation audit: surface and defuse negatives early.\n"
            "- Summaries that earn 'That's right.': paraphrase facts and concerns until alignment.\n"
            "- Ackerman bargaining: set target, anchor strategically, and plan 3 decreasing concessions (use precise numbers).\n"
            "- Seek Black Swans: probe for hidden constraints, decision makers, or non-monetary levers.\n"
            "- Avoid splitting the difference; aim for durable agreements that meet core interests.\n\n"
            "Response format:\n"
            "1) Brief insight and recommended next move (2–4 sentences).\n"
            "2) Calibrated questions (3–5).\n"
            "3) If pricing is involved: an Ackerman ladder with anchor, counter, and concession plan.\n"
            "4) Talk track: 3–5 bullet lines you can say verbatim.\n"
            "Keep answers concise (<= 200 words) unless asked for more."
        )
    return base_prompt


def _convert_history_to_openai_messages(history: List[Dict[str, Any]], user_input: str, context: str | None) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": _build_system_prompt()}]
    if context:
        messages.append({
            "role": "system",
            "content": (
                "Use the following context to answer the user's question. If the answer is not contained, say you don't know.\n\n" + context
            ),
        })
    for msg in history[-8:]:  # last few for context
        if msg.get("kind") == "text" and msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})
    return messages


def _get_openai_client():
    api_key = _get_secret_env("OPENAI_API_KEY")
    base_url = _get_secret_env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        raise RuntimeError(
            "LLM is not configured. Please set OPENAI_API_KEY (and OPENAI_BASE_URL for Databricks)."
        )
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url)


def generate_llm_response(user_input: str, context: str | None = None) -> str:
    model_name = _get_secret_env("LLM_MODEL", "gpt-4o-mini")
    client = _get_openai_client()
    chat_messages = _convert_history_to_openai_messages(st.session_state.messages, user_input, context)

    with st.spinner("Thinking…"):
        response = client.chat.completions.create(
            model=model_name,
            messages=chat_messages,
            temperature=0.2,
            max_tokens=700,
        )
    return (response.choices[0].message.content or "").strip()


# ---------- RAG: ingest, embed, retrieve ----------

def _demo_contract_texts() -> List[str]:
    return [
        (
            "Master Services Agreement (MSA) — Pricing & Indexation. Base transport rate denominated in USD per metric ton. "
            "Annual indexation capped at 3% YoY based on CPI-U. Fuel surcharge applies when Brent > $85/bbl per published table."
        ),
        (
            "Termination & Renewal. Auto-renews for 12 months unless notice is provided 60 days before expiry. "
            "Either party may terminate for material breach with 30 days cure period. Early termination fee equals 2 months average billing."
        ),
        (
            "Service Levels. On-time pickup 96%, delivery 94%. Credits: 2% of monthly fees for each percentage point below threshold up to 10%. "
            "DSO target 45 days; late payments > 60 days may trigger credit review and temporary service limits."
        ),
    ]


def _simple_chunk(text: str, chunk_chars: int = 900, overlap: int = 150) -> List[str]:
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def _embed_texts(texts: List[str]) -> np.ndarray:
    embed_model = _get_secret_env("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    client = _get_openai_client()
    # Batch in chunks of up to 128 texts to be gentle
    vectors: List[np.ndarray] = []
    batch_size = 96
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=embed_model, input=batch)
        arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
        # Normalize for cosine similarity via dot product
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        arr = arr / norms
        vectors.append(arr)
    if not vectors:
        return np.zeros((0, 3), dtype=np.float32)
    return np.vstack(vectors)


def add_texts_to_kb(texts_with_meta: List[Tuple[str, Dict[str, Any]]]) -> None:
    all_chunks: List[str] = []
    all_metas: List[Dict[str, Any]] = []
    for text, meta in texts_with_meta:
        chunks = _simple_chunk(text)
        all_chunks.extend(chunks)
        all_metas.extend([{**meta, "chunk_index": idx} for idx in range(len(chunks))])

    if not all_chunks:
        return

    new_matrix = _embed_texts(all_chunks)

    if st.session_state.rag_matrix is None or len(st.session_state.rag_chunks) == 0:
        st.session_state.rag_chunks = list(all_chunks)
        st.session_state.rag_metas = list(all_metas)
        st.session_state.rag_matrix = new_matrix
    else:
        # Append to existing matrix
        st.session_state.rag_chunks.extend(all_chunks)
        st.session_state.rag_metas.extend(all_metas)
        st.session_state.rag_matrix = np.vstack([st.session_state.rag_matrix, new_matrix])


def clear_kb() -> None:
    st.session_state.rag_chunks = []
    st.session_state.rag_metas = []
    st.session_state.rag_matrix = None


def retrieve_context(query: str, top_k: int = 4, max_chars: int = 2400) -> Tuple[str, List[str]]:
    if st.session_state.rag_matrix is None or len(st.session_state.rag_chunks) == 0:
        return "", []
    q_vec = _embed_texts([query])
    q_vec = q_vec[0]
    sims = st.session_state.rag_matrix @ q_vec
    if top_k >= len(sims):
        idxs = np.argsort(-sims)
    else:
        idxs = np.argpartition(-sims, top_k)[:top_k]
        idxs = idxs[np.argsort(-sims[idxs])]
    sources: List[str] = []
    context_parts: List[str] = []
    total = 0
    for i in idxs:
        meta = st.session_state.rag_metas[int(i)]
        text = st.session_state.rag_chunks[int(i)]
        label = f"Source: {meta.get('source','unknown')} (chunk {meta.get('chunk_index',0)})"
        part = f"{label}\n{text}"
        if total + len(part) > max_chars and context_parts:
            break
        context_parts.append(part)
        total += len(part)
        sources.append(label)
    context = "\n\n".join(context_parts)
    return context, sources


def handle_prompt(prompt_text: str) -> None:
    append_assistant_message({"role": "user", "kind": "text", "content": prompt_text})

    lower = prompt_text.lower()

    if "margin" in lower:
        df = create_margin_dataframe()
        append_assistant_message(
            {
                "role": "assistant",
                "kind": "margin_analysis",
                "content": "Here is a quick margin analysis for top accounts:",
                "data": {"rows": df.to_dict(orient="records")},
            }
        )
    elif "risk" in lower:
        df = create_risk_dataframe()
        append_assistant_message(
            {
                "role": "assistant",
                "kind": "risk_assessment",
                "content": "Customer risk assessment based on credit rating, DSO, and payment behavior:",
                "data": {"rows": df.to_dict(orient="records")},
            }
        )
    elif "rate" in lower or "market" in lower or "trend" in lower:
        df = create_rate_trend_dataframe()
        append_assistant_message(
            {
                "role": "assistant",
                "kind": "rate_trends",
                "content": "Market benchmark rates vs your average realized rates:",
                "data": {"rows": df.to_dict(orient="records")},
            }
        )
    else:
        context = None
        sources: List[str] = []
        if st.session_state.use_rag:
            context, sources = retrieve_context(prompt_text, top_k=4)
        llm_text = generate_llm_response(prompt_text, context=context)
        append_assistant_message(
            {
                "role": "assistant",
                "kind": "text",
                "content": llm_text,
            }
        )
        if sources:
            append_assistant_message(
                {
                    "role": "assistant",
                    "kind": "sources",
                    "content": "",
                    "data": {"sources": sources},
                }
            )


initialize_session_state()
render_sidebar()
render_header()

for message in st.session_state.messages:
    render_message(message)

cta_cols = st.columns([1, 1, 1])
with cta_cols[0]:
    if st.button("Show me margin analysis"):
        st.session_state.pending_user_prompt = "Show me margin analysis"
with cta_cols[1]:
    if st.button("Customer risk assessment"):
        st.session_state.pending_user_prompt = "Customer risk assessment"
with cta_cols[2]:
    if st.button("Market rate trends"):
        st.session_state.pending_user_prompt = "Market rate trends"

user_text = st.chat_input("Ask about customers, margins, rates, negotiations…")
prompt_to_use = user_text or st.session_state.pending_user_prompt
if prompt_to_use:
    st.session_state.pending_user_prompt = None
    handle_prompt(prompt_to_use)
    st.rerun()  # ensure message renders in history